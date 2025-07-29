# Model Directory Structure

This document describes the standard directory structure for storing **Large Language Models (LLMs)** in this project. It is important to follow this structure so that scripts can correctly locate and use the models.

## Directory Naming Convention

Each model should be in its own directory within the `models/` directory. The directory name should follow this format:

`[ModelFamilyName][Version]-[Size]`

- **[ModelFamilyName]**: The name of the model family (e.g., `Llama`, `Gemma`).
- **[Version]**: The model version (e.g., `3.2`).
- **[Size]**: The size of the model (e.g., `1B`, `3B`, `7B`).

**Example:**

```bash
models/Llama3.2-1B/
models/Gemma3-1B/
```

## Directory Contents

Each model directory must contain the following files:

- **TFLite Model File**: The main model file with a `.tflite` extension.
- **Tokenizer Model**: A file, typically named `tokenizer.model`.

Optionally, a weight cache file (e.g., `.xnnpack_cache`) can be included.

## Model File Naming Convention

The `.tflite` model file should be named according to the following convention to clearly indicate its properties:

`[ModelName]_[QuantizationInfo]_[MaxTokens]`

- **[ModelName]**: The base name of the model (e.g., `llama`, `gemma`).
- **[QuantizationInfo]**: Information about the quantization applied (e.g., `q8` for 8-bit quantization, `f16` for 16-bit float). This part can be omitted if the model is not quantized.
- **[MaxTokens]**: The maximum token length the model supports or is configured for (e.g., `ekv1024`).

**Note:** In the example, the `[MaxTokens]` part is shown as `ekv1024`, which may relate to model configuration (e.g., expert key-value cache size). Use an identifier that best describes the model's context length or a similar key characteristic.

## Example

The following is an example of the complete model directory structure for a quantized Llama 3.2 1B model:

```bash
models/
└── Llama3.2-1B/
    ├── llama_q8_ekv1024.tflite
    ├── llama_q8_ekv1024.xnnpack_cache
    └── tokenizer.model
```

## Non-LLM Models

The structure described above is specifically for Large Language Models (LLMs). Other types of `.tflite` models, used for different tasks (e.g., image classification), should be placed in the `test/` directory for validation and testing.

For example, the following models are currently in the workspace and can be used for testing within the `test` directory:

- `mobileone_s0.tflite`
- `mobilenetv3_small_100.lamb_in1k.tflite`
