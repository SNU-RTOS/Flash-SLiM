import tensorflow as tf


model_path="/home/rtos/workspace/ghpark/export/llama-3.2-3b-it-q8/llama_q8_ekv1024.tflite"
tf.lite.experimental.Analyzer.analyze(model_path=model_path,
                                      model_content=None,
                                      gpu_compatibility=False)


