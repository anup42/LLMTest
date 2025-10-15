package com.samsung.llmtest

object QwenPromptFormatter {
    const val MAX_TOKENS = 1024
    private const val DEFAULT_SYSTEM_PROMPT = "You are a helpful AI assistant."

    data class ChatMessage(
        val role: String,
        val content: String
    )

    data class PromptBundle(
        val fallbackPrompt: String,
        val messages: List<ChatMessage>
    ) {
        val useChatTemplate: Boolean get() = messages.isNotEmpty()
    }

    fun defaultSystemPrompt(): String = DEFAULT_SYSTEM_PROMPT

    fun buildPrompt(systemPrompt: String, userPrompt: String): PromptBundle {
        if (userPrompt.contains("<|im_start|>") && userPrompt.contains("<|im_end|>")) {
            val trimmed = userPrompt.trim()
            return PromptBundle(
                fallbackPrompt = trimmed,
                messages = emptyList()
            )
        }

        val systemSection = systemPrompt.ifBlank { DEFAULT_SYSTEM_PROMPT }
        val userSection = userPrompt.trim().ifBlank { "Please respond." }

        val fallbackPrompt = buildString {
            append("<|im_start|>system\n")
            append(systemSection)
            append("\n<|im_end|>\n<|im_start|>user\n")
            append(userSection)
            append("\n<|im_end|>\n<|im_start|>assistant\n")
        }

        val messages = listOf(
            ChatMessage(role = "system", content = systemSection),
            ChatMessage(role = "user", content = userSection)
        )

        return PromptBundle(
            fallbackPrompt = fallbackPrompt,
            messages = messages
        )
    }
}
