#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

#ifdef USE_OPENCV
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#endif

static int64_t curr_ms() {
  using namespace std::chrono;
  return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}

static std::vector<std::string> load_labels(const std::string& labels_path) {
  std::ifstream ifs(labels_path);
  if (!ifs) throw std::runtime_error("Failed to open labels file: " + labels_path);

  std::vector<std::string> labels;
  std::string line;
  while (std::getline(ifs, line)) {
    // trim right
    while (!line.empty() && (line.back() == '\r' || line.back() == '\n' || line.back() == ' ' || line.back() == '\t'))
      line.pop_back();
    // trim left
    size_t s = 0;
    while (s < line.size() && (line[s] == ' ' || line[s] == '\t')) s++;
    if (s < line.size()) labels.push_back(line.substr(s));
  }
  return labels;
}

#ifdef USE_OPENCV
struct ImageRGB {
  int h = 0, w = 0, c = 0;              // c should be 3
  std::vector<uint8_t> nhwc_u8;         // size = h*w*c
};

static ImageRGB load_rgb_u8_resized(const std::string& path, int out_h, int out_w) {
  cv::Mat bgr = cv::imread(path, cv::IMREAD_COLOR);
  if (bgr.empty()) throw std::runtime_error("Failed to read image: " + path);

  cv::Mat rgb;
  cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

  cv::Mat resized;
  cv::resize(rgb, resized, cv::Size(out_w, out_h), 0, 0, cv::INTER_LINEAR);

  if (resized.type() != CV_8UC3) resized.convertTo(resized, CV_8UC3);

  ImageRGB out;
  out.h = out_h;
  out.w = out_w;
  out.c = 3;
  out.nhwc_u8.resize(static_cast<size_t>(out_h) * out_w * 3);
  std::memcpy(out.nhwc_u8.data(), resized.data, out.nhwc_u8.size());
  return out;
}
#else
static_assert(true, "Build with -DUSE_OPENCV or implement image loading.");
#endif

static std::vector<float> softmax_1d(const std::vector<float>& x) {
  if (x.empty()) return {};
  float maxv = *std::max_element(x.begin(), x.end());

  std::vector<float> ex(x.size());
  for (size_t i = 0; i < x.size(); ++i) ex[i] = std::exp(x[i] - maxv);

  float sum = std::accumulate(ex.begin(), ex.end(), 0.0f);
  if (sum <= 0.0f) return std::vector<float>(x.size(), 0.0f);

  for (auto& v : ex) v /= sum;
  return ex;
}

static size_t tensor_num_elems(const TfLiteTensor* t) {
  size_t n = 1;
  for (int i = 0; i < t->dims->size; ++i) n *= static_cast<size_t>(t->dims->data[i]);
  return n;
}

int main(int argc, char** argv) {
  try {
    std::string model_path = "vit-vit-w8a8.tflite";
    std::string image_path = "boa-constrictor.jpg";
    std::string labels_path = "vit-vit-labels.txt";

    // FP32 preprocessing knobs
    // Default: x = x/255.0
    bool fp32_use_mean_std = false;
    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float stdv[3] = {0.229f, 0.224f, 0.225f};

    // Args:
    //   --model <path> --image <path> --labels <path>
    //   --fp32-mean-std   (use (x/255 - mean)/std per channel)
    for (int i = 1; i < argc; ++i) {
      std::string a = argv[i];
      if (a == "--model" && i + 1 < argc) model_path = argv[++i];
      else if (a == "--image" && i + 1 < argc) image_path = argv[++i];
      else if (a == "--labels" && i + 1 < argc) labels_path = argv[++i];
      else if (a == "--fp32-mean-std") fp32_use_mean_std = true;
      else {
        std::cerr << "Unknown/invalid arg: " << a << "\n";
        return 2;
      }
    }

    auto labels = load_labels(labels_path);

    auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    if (!model) {
      std::cerr << "Failed to load model: " << model_path << "\n";
      return 1;
    }

    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    if (!interpreter) {
      std::cerr << "Failed to create interpreter.\n";
      return 1;
    }

    if (interpreter->AllocateTensors() != kTfLiteOk) {
      std::cerr << "AllocateTensors() failed.\n";
      return 1;
    }

    // Input tensor
    const int input_idx = interpreter->inputs()[0];
    TfLiteTensor* in = interpreter->tensor(input_idx);
    if (!in || !in->dims || in->dims->size != 4) {
      std::cerr << "Expected input tensor with dims=4 (NHWC).\n";
      return 1;
    }
    const int n = in->dims->data[0];
    const int h = in->dims->data[1];
    const int w = in->dims->data[2];
    const int c = in->dims->data[3];

    if (n != 1) std::cerr << "[WARN] batch != 1 (batch=" << n << ")\n";
    if (c != 3) std::cerr << "[WARN] channels != 3 (channels=" << c << ")\n";

#ifdef USE_OPENCV
    ImageRGB img = load_rgb_u8_resized(image_path, h, w);
#else
    throw std::runtime_error("No image backend enabled.");
#endif

    // Feed input depending on tensor type
    if (in->type == kTfLiteUInt8) {
      // Quantized input expects uint8 NHWC
      const size_t bytes = static_cast<size_t>(n) * h * w * c;
      if (img.nhwc_u8.size() != bytes) {
        std::cerr << "Input byte size mismatch.\n";
        return 1;
      }
      uint8_t* dst = interpreter->typed_tensor<uint8_t>(input_idx);
      if (!dst) {
        std::cerr << "typed_tensor<uint8_t> returned null.\n";
        return 1;
      }
      std::memcpy(dst, img.nhwc_u8.data(), bytes);

    } else if (in->type == kTfLiteFloat32) {
      // FP32 input expects float NHWC
      const size_t elems = static_cast<size_t>(n) * h * w * c;
      float* dst = interpreter->typed_tensor<float>(input_idx);
      if (!dst) {
        std::cerr << "typed_tensor<float> returned null.\n";
        return 1;
      }
      if (img.nhwc_u8.size() != elems) {
        std::cerr << "Input element size mismatch.\n";
        return 1;
      }

      // Default: [0,1] scaling; Optional: mean/std normalization.
      for (size_t i = 0; i < elems; i += 3) {
        float r = img.nhwc_u8[i + 0] / 255.0f;
        float g = img.nhwc_u8[i + 1] / 255.0f;
        float b = img.nhwc_u8[i + 2] / 255.0f;

        if (fp32_use_mean_std) {
          r = (r - mean[0]) / stdv[0];
          g = (g - mean[1]) / stdv[1];
          b = (b - mean[2]) / stdv[2];
        }

        dst[i + 0] = r;
        dst[i + 1] = g;
        dst[i + 2] = b;
      }

    } else {
      std::cerr << "Unsupported input tensor type: " << in->type
                << " (this example supports uint8 or float32).\n";
      return 1;
    }

    // Warmup
    if (interpreter->Invoke() != kTfLiteOk) {
      std::cerr << "Invoke() failed on warmup.\n";
      return 1;
    }

    // Timed runs (10x)
    const int runs = 10;
    int64_t start = curr_ms();
    for (int i = 0; i < runs; ++i) {
      if (interpreter->Invoke() != kTfLiteOk) {
        std::cerr << "Invoke() failed on iteration " << i << "\n";
        return 1;
      }
    }
    int64_t end = curr_ms();
    double avg_ms = static_cast<double>(end - start) / runs;

    // Output tensor
    const int output_idx = interpreter->outputs()[0];
    TfLiteTensor* out = interpreter->tensor(output_idx);
    if (!out) {
      std::cerr << "Output tensor is null.\n";
      return 1;
    }

    const size_t out_elems = tensor_num_elems(out);
    std::vector<float> logits(out_elems, 0.0f);

    // If output is quantized, dequantize. If float, copy directly.
    if (out->type == kTfLiteUInt8) {
      const uint8_t* q = interpreter->typed_tensor<uint8_t>(output_idx);

      float scale = out->params.scale;
      int32_t zp = out->params.zero_point;

      for (size_t i = 0; i < out_elems; ++i) {
        logits[i] = (static_cast<int32_t>(q[i]) - zp) * scale;
      }

    } else if (out->type == kTfLiteInt8) {
      const int8_t* q = interpreter->typed_tensor<int8_t>(output_idx);

      float scale = out->params.scale;
      int32_t zp = out->params.zero_point;

      for (size_t i = 0; i < out_elems; ++i) {
        logits[i] = (static_cast<int32_t>(q[i]) - zp) * scale;
      }

    } else if (out->type == kTfLiteFloat32) {
      const float* f = interpreter->typed_tensor<float>(output_idx);
      std::memcpy(logits.data(), f, out_elems * sizeof(float));

    } else {
      std::cerr << "Unsupported output type: " << out->type << "\n";
      return 1;
    }

    // Softmax + top-5
    std::vector<float> probs = softmax_1d(logits);

    std::vector<int> idx(probs.size());
    std::iota(idx.begin(), idx.end(), 0);

    const int topk = 5;
    std::partial_sort(
        idx.begin(), idx.begin() + std::min<int>(topk, idx.size()), idx.end(),
        [&](int a, int b) { return probs[a] > probs[b]; });

    std::cout << "\nTop-5 predictions:\n";
    for (int i = 0; i < std::min<int>(topk, idx.size()); ++i) {
      int cls = idx[i];
      std::string name = (cls < static_cast<int>(labels.size()))
                             ? labels[cls]
                             : ("<class " + std::to_string(cls) + ">");
      std::cout << "Class " << name << ": score=" << probs[cls] << "\n";
    }

    std::cout << "\nInference took (on average): " << avg_ms << " ms per image\n";
    return 0;

  } catch (const std::exception& e) {
    std::cerr << "Fatal error: " << e.what() << "\n";
    return 1;
  }
}
