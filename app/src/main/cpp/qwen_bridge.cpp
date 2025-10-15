#include <jni.h>
#include <android/log.h>

#include <algorithm>
#include <atomic>
#include <mutex>
#include <string>
#include <vector>
#include <chrono>
#include <deque>
#include <sys/stat.h>

#include "llama.h"

#define LOG_TAG "QwenBridge"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

static std::mutex g_mutex;
static llama_model *g_model = nullptr;
static llama_context *g_ctx = nullptr;
static bool g_backend_initialized = false;
static std::atomic<int> g_last_generated_tokens{0};
static std::atomic<double> g_last_decode_ms{0.0};
static std::string g_last_error;
static std::deque<std::string> g_llama_log_tail;
static const size_t kLogTailMax = 32;
static bool g_capture_llama_errors = false;

static void ggml_log_to_last_error(enum ggml_log_level level, const char * text, void * /*user_data*/) {
    if (!g_capture_llama_errors || text == nullptr) return;
    std::string line = text ? std::string(text) : std::string();
    if (!line.empty()) {
        g_llama_log_tail.push_back(line);
        if (g_llama_log_tail.size() > kLogTailMax) {
            g_llama_log_tail.pop_front();
        }
    }
    if (level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR) {
        g_last_error = line;
    }
}

namespace {

constexpr int32_t kDefaultContext = 4096;
constexpr int32_t kDefaultBatch = 128;

// Removed legacy default system instruction and chat templating.

static std::string jstring_to_utf8(JNIEnv *env, jstring value) {
    if (!value) {
        return {};
    }
    const char *chars = env->GetStringUTFChars(value, nullptr);
    if (!chars) {
        return {};
    }
    std::string result(chars);
    env->ReleaseStringUTFChars(value, chars);
    return result;
}

static std::vector<std::string> jstring_array_to_vector(JNIEnv *env, jobjectArray array) {
    std::vector<std::string> result;
    if (!array) {
        return result;
    }
    const jsize length = env->GetArrayLength(array);
    result.reserve(length);
    for (jsize i = 0; i < length; ++i) {
        auto element = static_cast<jstring>(env->GetObjectArrayElement(array, i));
        if (!element) {
            result.emplace_back();
        } else {
            const char *chars = env->GetStringUTFChars(element, nullptr);
            if (chars) {
                result.emplace_back(chars);
                env->ReleaseStringUTFChars(element, chars);
            } else {
                result.emplace_back();
            }
            env->DeleteLocalRef(element);
        }
    }
    return result;
}

static bool apply_model_chat_template(const std::vector<std::string> &roles,
                                      const std::vector<std::string> &contents,
                                      std::string &out) {
    if (!g_model || roles.empty() || roles.size() != contents.size()) {
        return false;
    }

    std::vector<llama_chat_message> messages(roles.size());
    std::vector<std::string> sanitized_roles = roles;
    std::vector<std::string> sanitized_contents = contents;

    for (size_t i = 0; i < roles.size(); ++i) {
        messages[i].role = sanitized_roles[i].c_str();
        messages[i].content = sanitized_contents[i].c_str();
    }

    size_t estimated = 0;
    for (const auto &piece : sanitized_contents) {
        estimated += piece.size();
    }
    estimated = std::max<size_t>(estimated * 2 + 256, 1024);

    std::vector<char> buffer(estimated);
    int32_t formatted = llama_chat_apply_template(
            nullptr,
            messages.data(),
            messages.size(),
            true,
            buffer.data(),
            static_cast<int32_t>(buffer.size()));

    if (formatted < 0) {
        LOGE("llama_chat_apply_template failed with code %d", formatted);
        return false;
    }

    if (static_cast<size_t>(formatted) >= buffer.size()) {
        buffer.resize(static_cast<size_t>(formatted) + 1);
        formatted = llama_chat_apply_template(
                nullptr,
                messages.data(),
                messages.size(),
                true,
                buffer.data(),
                static_cast<int32_t>(buffer.size()));
        if (formatted < 0) {
            LOGE("llama_chat_apply_template failed on retry with code %d", formatted);
            return false;
        }
    }

    buffer[std::min<size_t>(buffer.size() - 1, static_cast<size_t>(formatted))] = '\0';
    out.assign(buffer.data(), static_cast<size_t>(formatted));
    return true;
}

static bool build_prompt_from_components(const std::string &fallback,
                                         const std::vector<std::string> &roles,
                                         const std::vector<std::string> &contents,
                                         std::string &out) {
    if (apply_model_chat_template(roles, contents, out)) {
        return true;
    }
    if (!fallback.empty()) {
        // If a fallback prompt was provided, use it as-is
        out = fallback;
        return true;
    }
    LOGE("No prompt data available for generation");
    return false;
}

static void release_locked() {
    if (g_ctx) {
        LOGI("Releasing llama context");
        llama_free(g_ctx);
        g_ctx = nullptr;
    }
    if (g_model) {
        LOGI("Releasing llama model");
        llama_free_model(g_model);
        g_model = nullptr;
    }
    if (g_backend_initialized) {
        LOGI("Releasing llama backend");
        llama_backend_free();
        g_backend_initialized = false;
    }
    g_last_generated_tokens.store(0);
    g_last_decode_ms.store(0.0);
    g_last_error.clear();
}

static bool progress_logger(float p, void * /*user*/) {
    LOGI("Model load progress: %.1f%%", p * 100.0f);
    return true;
}

static bool decode_one(llama_context *ctx, llama_token tok, llama_pos pos) {
    llama_batch batch = llama_batch_init(1, 0, 1);
    batch.n_tokens = 1;
    batch.token[0] = tok;
    batch.pos[0] = pos;
    batch.seq_id[0][0] = 0;
    batch.n_seq_id[0] = 1;
    batch.logits[0] = true;
    const int rc = llama_decode(ctx, batch);
    llama_batch_free(batch);
    return rc == 0;
}

static llama_token greedy_from_logits(llama_context *ctx, const llama_model *model) {
    const float *logits = llama_get_logits(ctx);
    if (!logits || !model) {
        return -1;
    }
    const struct llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    if (n_vocab <= 0) {
        return -1;
    }

    int best = 0;
    float best_val = logits[0];
    for (int i = 1; i < n_vocab; ++i) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best = i;
        }
    }
    return static_cast<llama_token>(best);
}

static bool append_clean_piece(std::string &dst, const llama_model *model, llama_token tok) {
    char tmp[64];
    const struct llama_vocab * vocab = llama_model_get_vocab(model);
    int n = llama_token_to_piece(vocab, tok, tmp, static_cast<int>(sizeof(tmp)), /*lstrip*/ 0, /*special*/ true);
    if (n > 0) {
        std::string tag(tmp, n);
        if (tag == "<|im_end|>" || tag == "<|im_start|>" || tag == "<|assistant|>" ||
            tag == "<|user|>" || tag == "<|system|>") {
            return false;
        }
    }

    char buf[256];
    n = llama_token_to_piece(vocab, tok, buf, static_cast<int>(sizeof(buf)), /*lstrip*/ 0, /*special*/ false);
    if (n >= 0) {
        if (n) dst.append(buf, n);
        return true;
    }

    std::string wide;
    wide.resize(static_cast<size_t>(-n));
    n = llama_token_to_piece(vocab, tok, wide.data(), static_cast<int>(wide.size()), /*lstrip*/ 0, /*special*/ false);
    if (n > 0) dst.append(wide.data(), n);
    return true;
}

static std::vector<llama_token> tokenize_prompt(const llama_model *model, const std::string &prompt) {
    if (!model) {
        return {};
    }
    const struct llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> tokens(prompt.size() + 16);
    int32_t count = llama_tokenize(vocab, prompt.c_str(), static_cast<int32_t>(prompt.size()), tokens.data(),
                                   static_cast<int32_t>(tokens.size()), /*add_special*/ true, /*parse_special*/ true);
    if (count < 0) {
        tokens.resize(static_cast<size_t>(-count));
        count = llama_tokenize(vocab, prompt.c_str(), static_cast<int32_t>(prompt.size()), tokens.data(),
                               static_cast<int32_t>(tokens.size()), true, true);
    }
    if (count < 0) {
        return {};
    }
    tokens.resize(static_cast<size_t>(count));
    return tokens;
}

static bool prefill_prompt(const std::vector<llama_token> &tokens, llama_pos &n_past) {
    const size_t total = tokens.size();
    if (total == 0) {
        LOGE("Prefill requested with zero tokens");
        return false;
    }

    const auto start = std::chrono::steady_clock::now();
    size_t iterations = 0;

    const int batch_cap = std::max<int>(kDefaultBatch, 32);
    llama_batch batch = llama_batch_init(batch_cap, 0, 1);

    size_t consumed = 0;
    while (consumed < total) {
        ++iterations;
        const int cur = std::min<int>(batch_cap, (int) (total - consumed));
        batch.n_tokens = cur;
        for (int i = 0; i < cur; ++i) {
            batch.token[i] = tokens[consumed + i];
            batch.pos[i] = n_past + i;
            batch.seq_id[i][0] = 0;
            batch.n_seq_id[i] = 1;
            batch.logits[i] = (consumed + i == total - 1);
        }
        if (llama_decode(g_ctx, batch) != 0) {
            llama_batch_free(batch);
            LOGE("llama_decode failed during prefill on iteration %zu", iterations);
            return false;
        }
        n_past += cur;
        consumed += (size_t) cur;
    }
    llama_batch_free(batch);

    const auto end = std::chrono::steady_clock::now();
    const double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    LOGI("Prefill complete: tokens=%zu batches=%zu elapsed=%.2f ms", total, iterations, elapsed_ms);
    return true;
}

static std::string generate_text(const std::vector<llama_token> &prompt_tokens,
                                 int max_tokens,
                                 JNIEnv *env = nullptr,
                                 jobject callback = nullptr,
                                 jmethodID on_token = nullptr) {
    g_last_generated_tokens.store(0);
    g_last_decode_ms.store(0.0);

    llama_pos n_past = 0;
    if (!prefill_prompt(prompt_tokens, n_past)) {
        g_last_generated_tokens.store(0);
        g_last_decode_ms.store(0.0);
        return "[error] Failed to prefill prompt.";
    }

    const llama_model *model = g_model;
    if (!model) {
        g_last_generated_tokens.store(0);
        return "[error] Model is not available.";
    }
    const struct llama_vocab * vocab = llama_model_get_vocab(model);
    const llama_token eos = llama_vocab_eos(vocab);
    std::string output;
    output.reserve(static_cast<size_t>(std::max(128, max_tokens * 4)));

    const int to_generate = std::max(1, max_tokens);
    const auto decode_start = std::chrono::steady_clock::now();
    int generated = 0;
    for (int i = 0; i < to_generate; ++i) {
        llama_token next = greedy_from_logits(g_ctx, model);
        if (next < 0) {
            g_last_generated_tokens.store(0);
            g_last_decode_ms.store(0.0);
            return "[error] Failed to sample token.";
        }
        if (next == eos) {
            LOGI("Reached EOS after %d tokens", generated);
            break;
        }
        const size_t previous_size = output.size();
        if (!append_clean_piece(output, model, next)) {
            LOGI("Stopped generation because append_clean_piece rejected token %d", next);
            break;
        }
        if (!decode_one(g_ctx, next, n_past)) {
            g_last_generated_tokens.store(0);
            g_last_decode_ms.store(0.0);
            return "[error] Failed to decode token.";
        }

        if (env != nullptr && callback != nullptr && on_token != nullptr) {
            const size_t new_size = output.size();
            if (new_size > previous_size) {
                const std::string chunk(output.data() + previous_size, new_size - previous_size);
                if (!chunk.empty()) {
                    jstring j_chunk = env->NewStringUTF(chunk.c_str());
                    if (j_chunk != nullptr) {
                        env->CallVoidMethod(callback, on_token, j_chunk);
                        env->DeleteLocalRef(j_chunk);
                        if (env->ExceptionCheck()) {
                            LOGE("Token callback threw an exception; stopping stream.");
                            env->ExceptionClear();
                            break;
                        }
                    }
                }
            }
        }

        ++n_past;
        ++generated;
    }
    const auto decode_end = std::chrono::steady_clock::now();
    const double decode_ms = std::chrono::duration<double, std::milli>(decode_end - decode_start).count();
    const double tok_per_sec = decode_ms > 0.0 ? generated / (decode_ms / 1000.0) : 0.0;
    g_last_decode_ms.store(decode_ms);
    LOGI("Decode timings: tokens=%d elapsed=%.2f ms (%.2f tok/s)", generated, decode_ms, tok_per_sec);

    if (output.empty()) {
        output = "[error] Model returned empty response.";
        g_last_generated_tokens.store(0);
        g_last_decode_ms.store(0.0);
    } else {
        g_last_generated_tokens.store(generated);
    }
    return output;
}

}  // namespace


extern "C" JNIEXPORT jint JNICALL
Java_com_samsung_llmtest_QwenBridge_nativeLastTokenCount(
        JNIEnv * /*env*/, jobject /*thiz*/) {
    return g_last_generated_tokens.load();
}

extern "C" JNIEXPORT jdouble JNICALL
Java_com_samsung_llmtest_QwenBridge_nativeLastDecodeMs(
        JNIEnv * /*env*/, jobject /*thiz*/) {
    return g_last_decode_ms.load();
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_samsung_llmtest_QwenBridge_nativeInit(
        JNIEnv *env, jobject /*thiz*/, jstring jModelPath, jint jThreads) {
    if (!jModelPath) {
        return JNI_FALSE;
    }

    const char *model_path = env->GetStringUTFChars(jModelPath, nullptr);
    if (!model_path) {
        return JNI_FALSE;
    }

    const int threads = std::max(1, jThreads);

    std::lock_guard<std::mutex> lock(g_mutex);

    if (!g_backend_initialized) {
        llama_backend_init();
        g_backend_initialized = true;
    }

    if (g_ctx || g_model) {
        release_locked();
        llama_backend_init();
        g_backend_initialized = true;
    }

    llama_model_params mparams = llama_model_default_params();
    // Force non-mmap path for stability on some devices
    mparams.use_mmap = false;
    mparams.use_mlock = false;
    mparams.n_gpu_layers = -1;
    mparams.check_tensors = true; // validate tensor headers/data to avoid segfaults on bad/corrupt GGUF
    mparams.progress_callback = progress_logger;
    mparams.progress_callback_user_data = nullptr;
    LOGI("GPU offload support=%d (requested layers=%d)", llama_supports_gpu_offload(), mparams.n_gpu_layers);

    // Before loading, perform quick file checks
    {
        struct stat st{};
        if (stat(model_path, &st) == 0) {
            LOGI("Model file size: %lld bytes", (long long) st.st_size);
            if (st.st_size <= 0) {
                g_last_error = "model file is empty";
            }
        } else {
            LOGE("stat() failed on model path: %s", model_path);
        }
    }

    // Hook llama logs to capture precise load errors
    g_last_error.clear();
    g_llama_log_tail.clear();
    g_capture_llama_errors = true;
    llama_log_set(ggml_log_to_last_error, nullptr);
    g_model = llama_model_load_from_file(model_path, mparams);
    if (!g_model) {
        LOGE("Failed to load model at %s", model_path);
        if (!g_last_error.empty()) {
            LOGE("llama last error: %s", g_last_error.c_str());
        }
        if (!g_llama_log_tail.empty()) {
            std::string joined;
            for (const auto &s : g_llama_log_tail) {
                if (!joined.empty()) joined += " | ";
                joined += s;
            }
            // keep it short
            if (joined.size() > 1024) joined.resize(1024);
            g_last_error = g_last_error.empty() ? joined : (g_last_error + " | " + joined);
        }
        if (g_last_error.empty()) g_last_error = "load failed";
        // Stop capturing logs
        g_capture_llama_errors = false;
        llama_log_set(nullptr, nullptr);
        env->ReleaseStringUTFChars(jModelPath, model_path);
        release_locked();
        return JNI_FALSE;
    }
    // Stop capturing logs now that model is ready
    g_capture_llama_errors = false;
    llama_log_set(nullptr, nullptr);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = kDefaultContext;
    cparams.n_batch = kDefaultBatch;
    cparams.n_threads = threads;
    cparams.n_threads_batch = threads;

    g_ctx = llama_init_from_model(g_model, cparams);
    if (!g_ctx) {
        LOGE("Failed to create context for %s", model_path);
        g_last_error = "context init failed";
        env->ReleaseStringUTFChars(jModelPath, model_path);
        release_locked();
        return JNI_FALSE;
    }

    llama_set_n_threads(g_ctx, threads, threads);
    LOGI("Context ready: n_ctx=%d batch=%d threads=%d", llama_n_ctx(g_ctx), cparams.n_batch, threads);

    env->ReleaseStringUTFChars(jModelPath, model_path);
    LOGI("Loaded Qwen model using %d threads", threads);
    return JNI_TRUE;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_samsung_llmtest_QwenBridge_nativeLastError(
        JNIEnv *env, jobject /*thiz*/) {
    if (g_last_error.empty()) {
        return env->NewStringUTF("");
    }
    return env->NewStringUTF(g_last_error.c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_samsung_llmtest_QwenBridge_nativeGenerate(
        JNIEnv *env,
        jobject /*thiz*/,
        jstring jFallbackPrompt,
        jobjectArray jRoles,
        jobjectArray jContents,
        jint jMaxTokens) {
    const std::string fallback = jstring_to_utf8(env, jFallbackPrompt);
    const std::vector<std::string> roles = jstring_array_to_vector(env, jRoles);
    const std::vector<std::string> contents = jstring_array_to_vector(env, jContents);

    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_model || !g_ctx) {
        return env->NewStringUTF("[error] Model is not initialized.");
    }

    std::string prompt;
    if (!build_prompt_from_components(fallback, roles, contents, prompt)) {
        return env->NewStringUTF("[error] Unable to build prompt.");
    }

    std::vector<llama_token> tokens = tokenize_prompt(g_model, prompt);

    if (tokens.empty()) {
        return env->NewStringUTF("[error] Failed to tokenize prompt.");
    }

    const int n_ctx = llama_n_ctx(g_ctx);
    if ((int) tokens.size() >= n_ctx) {
        return env->NewStringUTF("[error] Prompt is longer than the context window.");
    }

    const int requested = jMaxTokens > 0 ? jMaxTokens : 512;
    const int available = n_ctx - (int) tokens.size();
    const int capped = std::max(16, std::min(requested, available));

    std::string result = generate_text(tokens, capped);
    return env->NewStringUTF(result.c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_samsung_llmtest_QwenBridge_nativeGenerateStreaming(
        JNIEnv *env,
        jobject /*thiz*/,
        jstring jFallbackPrompt,
        jobjectArray jRoles,
        jobjectArray jContents,
        jint jMaxTokens,
        jobject jCallback) {
    const std::string fallback = jstring_to_utf8(env, jFallbackPrompt);
    const std::vector<std::string> roles = jstring_array_to_vector(env, jRoles);
    const std::vector<std::string> contents = jstring_array_to_vector(env, jContents);

    jmethodID on_token_method = nullptr;
    if (jCallback) {
        jclass callback_class = env->GetObjectClass(jCallback);
        if (!callback_class) {
            return env->NewStringUTF("[error] Failed to inspect token consumer.");
        }
        on_token_method = env->GetMethodID(callback_class, "onToken", "(Ljava/lang/String;)V");
        env->DeleteLocalRef(callback_class);
        if (!on_token_method) {
            return env->NewStringUTF("[error] Token consumer missing onToken callback.");
        }
    }

    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_model || !g_ctx) {
        return env->NewStringUTF("[error] Model is not initialized.");
    }

    std::string prompt;
    if (!build_prompt_from_components(fallback, roles, contents, prompt)) {
        return env->NewStringUTF("[error] Unable to build prompt.");
    }

    std::vector<llama_token> tokens = tokenize_prompt(g_model, prompt);
    if (tokens.empty()) {
        return env->NewStringUTF("[error] Failed to tokenize prompt.");
    }

    const int n_ctx = llama_n_ctx(g_ctx);
    if ((int) tokens.size() >= n_ctx) {
        return env->NewStringUTF("[error] Prompt is longer than the context window.");
    }

    const int requested = jMaxTokens > 0 ? jMaxTokens : 512;
    const int available = n_ctx - (int) tokens.size();
    const int capped = std::max(16, std::min(requested, available));

    std::string result = generate_text(
            tokens,
            capped,
            (jCallback != nullptr) ? env : nullptr,
            jCallback,
            on_token_method);
    return env->NewStringUTF(result.c_str());
}

extern "C" JNIEXPORT void JNICALL
Java_com_samsung_llmtest_QwenBridge_nativeRelease(
        JNIEnv * /*env*/, jobject /*thiz*/) {
    std::lock_guard<std::mutex> lock(g_mutex);
    release_locked();
}

















