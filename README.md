# LLM Test

Android test app for running local Qwen models via llama.cpp. It provides a simple UI to load a `.gguf` model, enter a user prompt, and stream the response incrementally.


## Requirements
- A Qwen/Qwen3 `.gguf` model file (e.g., `Qwen3-0.6B-Q4_K_M.gguf`, `Qwen3-0.6B-Q8_0.gguf`).


## Using the app

1. Tap the folder icon (Browse) and select a `.gguf` model. The app will copy the file into private storage for reliable access.
2. Tap **Load Model**. When loading succeeds, **Generate Response** becomes enabled.
3. Enter a user prompt and tap **Generate Response**. Output streams into the card; a timing line shows token count, decode time, tok/s, and ms/token.


## Model support

- Designed for Qwen`.gguf` models converted for llama.cpp.
- Works with common quantizations like Q4_K_M, Q8_0.

### Quick download links

- Qwen2.5-0.5B-Instruct-GGUF (Q8_0):
  https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q8_0.gguf
- Qwen3-0.6B-GGUF (Q8_0):
  https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf


## Project structure (high level)

- `app/src/main/java/com/samsung/llmtest/`
  - `MainActivity.kt`: UI + model load + streaming inference.
  - `QwenBridge.kt`: JNI bindings and convenience wrappers.
  - `QwenPromptFormatter.kt`: builds user prompts and message bundles.
- `app/src/main/cpp/`
  - `qwen_bridge.cpp`: JNI implementation bridging to llama.cpp (load/generate/stream).
  - `llama/`: synced headers from pinned llama.cpp and ggml.
- `app/src/main/jniLibs/arm64-v8a/`: prebuilt `libllama.so`, `libggml.so`, `libggml-base.so`, `libggml-cpu.so`.
