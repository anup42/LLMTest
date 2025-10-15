package com.samsung.llmtest

object QwenPromptFormatter {
    const val MAX_TOKENS = 1024
    private const val DEFAULT_SYSTEM_PROMPT = "You are a helpful AI assistant."

    fun defaultSystemPrompt(): String = DEFAULT_SYSTEM_PROMPT

    fun buildPrompt(systemPrompt: String, userPrompt: String): String {
        if (userPrompt.contains("<|im_start|>") && userPrompt.contains("<|im_end|>")) {
            return userPrompt
        }

        val systemSection = systemPrompt.ifBlank { DEFAULT_SYSTEM_PROMPT }
        val userSection = userPrompt.trim().ifBlank { "Please respond." }

        return buildString {
            append("<|im_start|>system\n")
            append(systemSection)
            append("\n<|im_end|>\n<|im_start|>user\n")
            append(userSection)
            append("\n<|im_end|>\n<|im_start|>assistant\n")
        }
    }
}
