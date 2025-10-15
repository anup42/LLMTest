package com.samsung.llmtest

import android.os.Build
import android.util.Log
import androidx.annotation.Keep

object QwenBridge {
    private const val TAG = "QwenBridge"
    private val loadedLibs = mutableListOf<String>()
    @Volatile private var vulkanActive = false

    @Keep
    fun interface TokenConsumer {
        fun onToken(chunk: String)
    }

    init {
        Log.i(TAG, "Init: device=${Build.DEVICE}, hardware=${Build.HARDWARE}")

        tryLoad("ggml", required = true)
        tryLoad("llama", required = true)

        kotlin.runCatching { System.loadLibrary("ggml-cpu") }
            .onSuccess { loadedLibs += "ggml-cpu" }
            .onFailure { Log.d(TAG, "Optional ggml-cpu not loaded: ${it.localizedMessage}") }

        System.loadLibrary("native-lib")
        loadedLibs += "native-lib"

        Log.i(TAG, "Loaded native libs: ${loadedLibs.joinToString()}")
        Log.i(TAG, "Vulkan active=$vulkanActive")
    }

    private fun tryLoad(lib: String, required: Boolean): Boolean {
        val result = runCatching {
            System.loadLibrary(lib)
            loadedLibs += lib
            true
        }

        result.exceptionOrNull()?.let { error ->
            val message = "Library $lib failed: ${error.localizedMessage}"
            if (required) {
                Log.e(TAG, message)
            } else {
                Log.d(TAG, message)
            }
        }

        return result.getOrElse { error ->
            if (required) throw error else false
        }
    }

    fun load(modelPath: String, threads: Int): Boolean {
        Log.i(TAG, "nativeInit threads=$threads vulkan=$vulkanActive")
        return nativeInit(modelPath, threads)
    }

    fun generate(prompt: String, maxTokens: Int): String = nativeGenerate(prompt, maxTokens)

    fun generateStreaming(
        prompt: String,
        maxTokens: Int,
        onToken: (String) -> Unit
    ): String {
        return nativeGenerateStreaming(prompt, maxTokens, TokenConsumer { chunk ->
            onToken(chunk)
        })
    }
    fun release() = nativeRelease()

    fun isVulkanActive(): Boolean = vulkanActive
    fun loadedLibraries(): List<String> = loadedLibs.toList()
    fun lastTokenCount(): Int = nativeLastTokenCount()
    fun lastDecodeMs(): Double = nativeLastDecodeMs()

    private external fun nativeInit(modelPath: String, nThreads: Int): Boolean
    private external fun nativeGenerate(prompt: String, maxTokens: Int): String
    private external fun nativeGenerateStreaming(
        prompt: String,
        maxTokens: Int,
        consumer: TokenConsumer
    ): String
    private external fun nativeRelease()
    private external fun nativeLastTokenCount(): Int
    private external fun nativeLastDecodeMs(): Double
}

