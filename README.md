# LLM Test

Android app that runs the Qwen model locally via llama.cpp and exposes a lightweight interface for experimentation on device.

## Prerequisites

- Android Studio Giraffe (or newer) with NDK support enabled.
- Android device or emulator that supports `arm64-v8a`.
- Prebuilt llama.cpp shared libraries (already provided under `llama cpp Code/main/jniLibs`).
- A Qwen2.5-0.5B Instruct `.gguf` model file copied onto the device or exposed via a document provider (nothing is bundled by default).

## Project layout

```
app/                       # Android application module
  src/main/java/com/samsung/llmtest
    MainActivity.kt        # UI + model lifecycle management
    QwenBridge.kt          # Kotlin wrapper around the native llama.cpp bridge
    QwenPromptFormatter.kt # Helper for building Qwen chat prompts
  src/main/cpp             # JNI bridge powered by llama.cpp
llama cpp Code/            # Prebuilt llama.cpp headers + shared libraries (provided)
```

## Getting started

1. Launch Android Studio and open this folder (`LLMTest`).
2. When prompted, allow Android Studio to download the Gradle/NDK components.
3. Build & run the app on an `arm64-v8a` device. The first launch may take a moment while the native libraries are extracted.
4. In the UI:
    - The model path field starts empty; use **Browse** to select a `.gguf` file or paste an absolute path/`content://` URI. Models are never auto-loaded from bundled assets.
    - Tap **Load Model** to initialize llama.cpp. Status messages appear under the toolbar; the **Generate Response** button is enabled once loading succeeds.
    - Provide an optional system prompt and the user prompt you want to send to Qwen. The default system prompt is a generic "helpful assistant" instruction.
    - Tap **Generate Response** to run inference. The spinner indicates progress, the decoded text appears in the card below, and a timing line shows token count, decode time, tokens/sec, and ms/token.
    - Status updates (success/failure, permission hints, etc.) remain visible in the status label above the prompt fields.

To swap to a different GGUF model, repeat the **Browse**/**Load Model** flow. The last-used document URI/path is remembered between sessions, but models are only loaded when you explicitly tap **Load Model**.
