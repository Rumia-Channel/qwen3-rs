# Description

**qwen3-rs** is an educational Rust project for exploring and running Qwen3 language family models. It is designed to be clear, modular, and approachable for learners, with minimal dependencies and many core algorithms reimplemented from scratch for transparency.

> **Note:** Parts of this codebase, including documentation and core algorithms, were generated or assisted by large language models (LLMs) to accelerate development and improve educational clarity. As a starting reference, the project [qwen3.c](https://github.com/adriancable/qwen3.c) was used for understanding model internals and file formats.

## Limitations

- **Text-only**: Image and video inputs are not supported.
- **CPU-only**: No GPU or hardware acceleration.
- **Q8 quantization**: Weights use 8-bit quantized linear layers; small accuracy trade-off.
- **Qwen3.5 prefill**: Initial prefill processes tokens one-by-one (same as decode), but skips
  the LM head for non-final prompt tokens. Chunked prefill is a future optimization.
- **No streaming API**: Output is line-buffered generation to stdout.

## Project Goals

- **Educational:** Learn how transformer architectures, quantization, and efficient inference work in Rust.
- **Minimal Dependencies:** Most algorithms (tokenization, quantization, sampling, etc.) are implemented from scratch—no heavy ML or Python bindings.
- **Modular:** Core library logic is separated from CLI tools for clarity and maintainability.
- **Efficiency:** Uses memory mapping and zero-copy techniques for handling large model files.

## Supported Models

| Model | Architecture | Version | Status |
|-------|-------------|---------|--------|
| Qwen3 (0.6B, 4B, 8B) | `Qwen3ForCausalLM` | v1 | Full support |
| Qwen3.5-2B (text only) | `Qwen3_5ForConditionalGeneration` | v2 | Experimental; export/inference smoke-tested* |
| DeepSeek-R1-0528-Qwen3-8B | LoRA + Qwen3 | v1 | Supported |

> *Qwen3.5 vision and MoE variants are explicitly unsupported. Only the dense language
> model inside `Qwen3_5ForConditionalGeneration` is exported and run. Numerical parity
> with Transformers must be verified before production use.

## Workspace Structure

```
qwen3-rs/
├── docs                # LLM generated docs for key components
├── Cargo.toml          # Workspace configuration
├── qwen3-cli/          # Command-line interface crate
├── qwen3-export/       # Model export crate
├── qwen3-inference/    # LLM inference crate
```

## How to Use

### 1. Get a Hugging Face model

`Qwen3ForCausalLM` and the text submodel of `Qwen3_5ForConditionalGeneration` are recognized.

```bash
git clone https://huggingface.co/Qwen/Qwen3-0.6B
# Or try larger/alternative models:
# git clone https://huggingface.co/Qwen/Qwen3-4B
# git clone https://huggingface.co/Qwen/Qwen3-8B
# git clone https://huggingface.co/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B
```

**NOTE**: `Low Rank Adaptation (LoRA)` is supported: copy `adapter_config.json` and safe tensor files to the same folder,
where base model is located, and it will be automatically detected (tested with `usloth` output).

### 2. Build and run the exporter

```bash
cargo build --release -p qwen3-cli

# Export a Hugging Face model to a quantized checkpoint
cargo run --release -p qwen3-cli -- export /path/to/model /path/to/output.bin --group-size 64

# Qwen3.5-2B example (vision/MTP weights are skipped)
cargo run --release -p qwen3-cli -- export sample/Qwen/Qwen3.5-2B qwen3.5-2b-q8.bin --group-size 64
```

### 3. Run inference

In chat mode with default parameters:

```bash
# Qwen3 chat (default context = model max)
cargo run --release -p qwen3-cli -- inference /path/to/qwen3.bin -m chat

# Qwen3.5 chat (default context = 4096)
cargo run --release -p qwen3-cli -- inference /path/to/qwen3_5.bin -m chat

# Generate mode with a prompt
cargo run --release -p qwen3-cli -- inference /path/to/model.bin -m generate -i "Once upon a time"

# Enable thinking mode (Qwen3.5)
cargo run --release -p qwen3-cli -- inference /path/to/qwen3_5.bin -m chat -r 1

# Override context window
cargo run --release -p qwen3-cli -- inference /path/to/model.bin -c 8192
```

**Note:** Inference currently runs on CPU with scalar Q8 kernels. Performance is sufficient
for educational purposes but will be slower than GPU-based inference.

## CLI Commands and Options

### `export`

Exports a HuggingFace Qwen3 model to a custom binary format for efficient Rust inference.

**Usage:**

```bash
qwen3 export <MODEL_PATH> <OUTPUT_PATH> [--group-size <SIZE>]
```

- `MODEL_PATH`: Path to HuggingFace model directory (must contain config.json, \*.safetensors, tokenizer.json)
- `OUTPUT_PATH`: Output path for the binary model file
- `--group-size`, `-g`: Quantization group size (default: 64)

### `inference`

Runs inference on a binary Qwen3 model.

**Usage:**

```bash
qwen3 inference <checkpoint> [options]
```

**Options:**

- `--temperature`, `-t <FLOAT>`: Sampling temperature (default: 1.0)
- `--topp`, `-p <FLOAT>`: Top-p nucleus sampling (default: 0.9)
- `--seed`, `-s <INT>`: Random seed
- `--context`, `-c <INT>`: Context window size (default: model max for Qwen3, 4096 for Qwen3.5)
- `--mode`, `-m <STRING>`: Mode: `generate` or `chat` (default: chat)
- `--input`, `-i <STRING>`: Input prompt
- `--system`, `-y <STRING>`: System prompt (for chat mode)
- `--reasoning`, `-r <INT>`: Reasoning mode: 0=no thinking, 1=thinking (default: 0)

  - Thinking disabled: assistant begins with an empty `<think>\n\n</think>\n\n` block.
  - Thinking enabled: assistant begins with `<think>\n` for chain-of-thought reasoning
    (supported by Qwen3 and Qwen3.5 chat templates).
