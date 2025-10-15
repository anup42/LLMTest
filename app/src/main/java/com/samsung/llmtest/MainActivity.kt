package com.samsung.llmtest

import android.Manifest
import android.content.ContentResolver
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.os.SystemClock
import android.provider.DocumentsContract
import android.util.Log
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.view.isVisible
import androidx.lifecycle.lifecycleScope
import com.samsung.llmtest.databinding.ActivityMainBinding
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream

class MainActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMainBinding
    private var isModelReady = false
    private var pendingModelPath: String? = null

    private val tokenRegex = Regex("\\S+")

    private val pickModelFile = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        uri?.let { handleModelUri(it) } ?: updateStatus(getString(R.string.status_no_file))
    }

    private val storagePermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        val granted = permissions.entries.all { it.value }
        val pathToRetry = pendingModelPath
        pendingModelPath = null
        if (granted && pathToRetry != null) {
            loadModelInternal(pathToRetry)
        } else if (!granted) {
            updateStatus(getString(R.string.status_permission_denied))
            resetLoadingUi()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        restoreLastModelPath()
        binding.systemPromptInput.setText(QwenPromptFormatter.defaultSystemPrompt())
        binding.userPromptInput.setText(getString(R.string.sample_user_prompt))

        binding.generateButton.isEnabled = false
        binding.modelPathLayout.setEndIconOnClickListener { openModelPicker() }
        binding.loadModelButton.setOnClickListener { loadModel() }
        binding.generateButton.setOnClickListener { generateResponse() }

        updateStatus(getString(R.string.status_idle))

        // TEMP: auto-load model on launch for debugging crashes on Load button
        val autoPath = binding.modelPathInput.text?.toString()?.trim().orEmpty()
        if (autoPath.isNotEmpty()) {
            // Post to ensure views are initialized
            binding.root.post {
                loadModel()
            }
        }
    }

    private fun restoreLastModelPath() {
        val prefs = getPreferences(MODE_PRIVATE)
        val lastPath = prefs.getString(KEY_MODEL_PATH, null)
        binding.modelPathInput.setText(lastPath ?: "")
        lastPath?.let { binding.modelPathInput.setSelection(it.length) }
    }

    private fun loadModel() {
        val modelPath = binding.modelPathInput.text?.toString()?.trim()
        if (modelPath.isNullOrEmpty()) {
            updateStatus(getString(R.string.status_no_file))
            return
        }

        if (requiresLegacyStoragePermission(modelPath) && !hasStoragePermission()) {
            pendingModelPath = modelPath
            storagePermissionLauncher.launch(requiredStoragePermissions())
            return
        }

        loadModelInternal(modelPath)
    }

    private fun loadModelInternal(requestedPath: String) {
        binding.progressBar.isVisible = true
        binding.loadModelButton.isEnabled = false
        binding.generateButton.isEnabled = false
        updateStatus(getString(R.string.status_loading_model))

        val cpuCount = Runtime.getRuntime().availableProcessors().coerceAtLeast(2)
        val threads = when {
            cpuCount >= 10 -> cpuCount - 2
            cpuCount >= 8 -> cpuCount - 1
            cpuCount >= 6 -> cpuCount - 1
            else -> cpuCount
        }.coerceAtLeast(4).coerceAtMost(8).coerceAtMost(cpuCount)

        Log.i(
            "MainActivity",
            "Loading model threads=$threads vulkan=${QwenBridge.isVulkanActive()} libs=${QwenBridge.loadedLibraries().joinToString()}"
        )

        lifecycleScope.launch {
            val preparedPath = withContext(Dispatchers.IO) { prepareModelFile(requestedPath) }
            if (preparedPath == null) {
                binding.progressBar.isVisible = false
                binding.loadModelButton.isEnabled = true
                val hint = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                    "Use Browse to pick the file so the app can access it."
                } else ""
                updateStatus(getString(R.string.status_model_failed) +
                        (if (hint.isNotEmpty()) "\n$hint" else ""))
                return@launch
            }

            val success = withContext(Dispatchers.IO) {
                runCatching { QwenBridge.load(preparedPath, threads) }.getOrElse { false }
            }

            binding.progressBar.isVisible = false
            binding.loadModelButton.isEnabled = true

            if (success) {
                isModelReady = true
                binding.generateButton.isEnabled = true
                getPreferences(MODE_PRIVATE).edit()
                    .putString(KEY_MODEL_PATH, requestedPath)
                    .putString(KEY_MODEL_LOCAL_PATH, preparedPath)
                    .apply()
                updateStatus(getString(R.string.status_model_ready, threads))
            } else {
                isModelReady = false
                val detail = QwenBridge.lastError().takeIf { it.isNotBlank() } ?: ""
                val message = if (detail.isNotEmpty())
                    getString(R.string.status_model_failed) + "\nReason: $detail"
                else getString(R.string.status_model_failed)
                updateStatus(message)
            }
        }
    }

    private fun generateResponse() {
        if (!isModelReady) {
            updateStatus(getString(R.string.status_model_required))
            return
        }

        val systemPrompt = binding.systemPromptInput.text?.toString().orEmpty()
        val userPrompt = binding.userPromptInput.text?.toString()?.trim().orEmpty()
        if (userPrompt.isEmpty()) {
            updateStatus(getString(R.string.status_prompt_required))
            return
        }

        binding.progressBar.isVisible = true
        binding.generateButton.isEnabled = false
        binding.timingText.isVisible = false
        binding.outputCard.isVisible = true
        binding.outputText.text = ""
        updateStatus(getString(R.string.status_generating))

        lifecycleScope.launch {
            val promptBundle = QwenPromptFormatter.buildPrompt(systemPrompt, userPrompt)
            val tokenStream = Channel<String>(Channel.UNLIMITED)
            val assistantBuilder = StringBuilder()

            val collectorJob = launch {
                for (chunk in tokenStream) {
                    if (chunk.isEmpty()) continue
                    assistantBuilder.append(chunk)
                    binding.outputText.text = assistantBuilder.toString()
                }
            }

            val startTime = SystemClock.elapsedRealtime()
            val output = try {
                withContext(Dispatchers.IO) {
                    QwenBridge.generateStreaming(promptBundle, QwenPromptFormatter.MAX_TOKENS) { chunk ->
                        tokenStream.trySend(chunk).isSuccess
                    }
                }
            } catch (throwable: Throwable) {
                "[error] ${throwable.localizedMessage}"
            } finally {
                tokenStream.close()
            }
            val elapsedMs = SystemClock.elapsedRealtime() - startTime
            collectorJob.join()

            binding.progressBar.isVisible = false
            binding.generateButton.isEnabled = isModelReady

            if (output.startsWith("[error]")) {
                val message = output.removePrefix("[error]").trim().ifEmpty { "Unknown error" }
                binding.outputText.text = getString(R.string.inference_failed, message)
                updateStatus(getString(R.string.inference_failed, message))
                return@launch
            }

            val sanitizedOutput = output.trim()
            binding.outputText.text = sanitizedOutput

            val bridgeTokens = QwenBridge.lastTokenCount().takeIf { it > 0 }
            val tokenCount = bridgeTokens ?: approximateTokenCount(sanitizedOutput)
            val decodeMs = QwenBridge.lastDecodeMs().takeIf { it > 0.0 } ?: elapsedMs.toDouble()
            val tokensPerSecond = if (decodeMs > 0.0 && tokenCount > 0) tokenCount * 1000.0 / decodeMs else 0.0
            val msPerToken = if (tokenCount > 0) decodeMs / tokenCount else 0.0

            binding.timingText.text = getString(
                R.string.inference_metrics,
                tokenCount,
                decodeMs,
                tokensPerSecond,
                msPerToken
            )
            binding.timingText.isVisible = true
            updateStatus(getString(R.string.status_generation_complete))
        }
    }

    private fun approximateTokenCount(text: String): Int {
        if (text.isBlank()) {
            return 0
        }
        return tokenRegex.findAll(text).count()
    }

    private fun updateStatus(message: String) {
        binding.statusText.text = message
    }

    private fun resetLoadingUi() {
        binding.progressBar.isVisible = false
        binding.loadModelButton.isEnabled = true
        binding.generateButton.isEnabled = isModelReady
    }

    override fun onDestroy() {
        if (isModelReady) {
            runCatching { QwenBridge.release() }
        }
        super.onDestroy()
    }

    private fun openModelPicker() {
        pickModelFile.launch(arrayOf("*/*"))
    }

    private fun handleModelUri(uri: Uri) {
        val resolvedPath = resolveDocumentPath(uri)
        val displayText = resolvedPath ?: uri.toString()
        val fileName = resolvedPath?.let { File(it).name }
            ?: uri.lastPathSegment?.substringAfter('/')
            ?: "model.gguf"

        runCatching {
            contentResolver.takePersistableUriPermission(
                uri,
                Intent.FLAG_GRANT_READ_URI_PERMISSION
            )
        }

        getPreferences(MODE_PRIVATE).edit()
            .putString(KEY_MODEL_URI, uri.toString())
            .putString(KEY_MODEL_PATH, displayText)
            .remove(KEY_MODEL_LOCAL_PATH)
            .apply()

        binding.modelPathInput.setText(displayText)
        binding.modelPathInput.setSelection(displayText.length)
        updateStatus("Selected model: $fileName")
    }

    private fun resolveDocumentPath(uri: Uri): String? {
        if (uri.scheme == ContentResolver.SCHEME_FILE) {
            return uri.path
        }

        if (uri.scheme != ContentResolver.SCHEME_CONTENT) {
            return null
        }

        if (!DocumentsContract.isDocumentUri(this, uri)) {
            return null
        }

        val documentId = runCatching { DocumentsContract.getDocumentId(uri) }.getOrNull() ?: return null
        if (documentId.startsWith("raw:")) {
            return documentId.removePrefix("raw:")
        }

        val parts = documentId.split(":")
        if (parts.size == 2) {
            val type = parts[0]
            val relativePath = parts[1]
            val base = when (type.lowercase()) {
                "primary" -> Environment.getExternalStorageDirectory().absolutePath
                "home" -> Environment.getExternalStorageDirectory().absolutePath + "/Documents"
                else -> null
            }
            if (base != null) {
                return "$base/$relativePath"
            }
        }

        return null
    }

    private fun requiresLegacyStoragePermission(path: String): Boolean {
        if (assetNameFromPath(path) != null) {
            return false
        }
        if (Build.VERSION.SDK_INT > Build.VERSION_CODES.S_V2) {
            return false
        }
        if (path.startsWith("content://")) {
            return false
        }
        val externalRoot = Environment.getExternalStorageDirectory().absolutePath
        return path.startsWith(externalRoot)
    }

    private fun hasStoragePermission(): Boolean {
        if (Build.VERSION.SDK_INT > Build.VERSION_CODES.S_V2) {
            return true
        }
        return ContextCompat.checkSelfPermission(
            this,
            Manifest.permission.READ_EXTERNAL_STORAGE
        ) == PackageManager.PERMISSION_GRANTED
    }

    private fun requiredStoragePermissions(): Array<String> {
        return if (Build.VERSION.SDK_INT > Build.VERSION_CODES.S_V2) {
            emptyArray()
        } else {
            arrayOf(Manifest.permission.READ_EXTERNAL_STORAGE)
        }
    }

    private fun prepareModelFile(requestedPath: String): String? {
        assetNameFromPath(requestedPath)?.let { assetName ->
            return copyAssetModelToPrivateStorage(assetName)
        }
        val directFile = File(requestedPath)
        if (directFile.exists() && directFile.canRead()) {
            return directFile.absolutePath
        }

        val prefs = getPreferences(MODE_PRIVATE)
        val cachedLocal = prefs.getString(KEY_MODEL_LOCAL_PATH, null)?.takeIf { it.isNotBlank() }
        val cachedFile = cachedLocal?.let(::File)
        if (cachedFile != null && cachedFile.exists() && cachedFile.canRead()) {
            return cachedFile.absolutePath
        }

        val uriString = prefs.getString(KEY_MODEL_URI, null)?.takeIf { it.isNotBlank() }
            ?: requestedPath.takeIf { it.startsWith("content://") }
        val uri = uriString?.let(Uri::parse) ?: return null

        return copyModelToPrivateStorage(uri)
    }

    private fun copyModelToPrivateStorage(uri: Uri): String? {
        val modelsDir = File(filesDir, "models").apply { if (!exists()) mkdirs() }
        val destination = File(modelsDir, guessFileName(uri))

        return runCatching {
            contentResolver.openInputStream(uri)?.use { input ->
                FileOutputStream(destination).use { output ->
                    input.copyTo(output)
                }
            } ?: throw IllegalStateException("Unable to open model URI")
            destination.absolutePath
        }.onFailure {
            destination.delete()
        }.getOrNull()
    }

    private fun guessFileName(uri: Uri): String {
        val prefs = getPreferences(MODE_PRIVATE)
        val recordedPathName = prefs.getString(KEY_MODEL_PATH, null)
            ?.let { File(it).name }
            ?.takeIf { it.isNotBlank() }
        val uriName = uri.lastPathSegment?.substringAfter('/')?.takeIf { it.isNotBlank() }
        return uriName ?: recordedPathName ?: "model.gguf"
    }

    private fun copyAssetModelToPrivateStorage(assetName: String): String? {
        val modelsDir = File(filesDir, "models").apply { if (!exists()) mkdirs() }
        val destination = File(modelsDir, assetName)

        if (destination.exists() && destination.length() > 0) {
            return destination.absolutePath
        }

        return runCatching {
            assets.open(assetName).use { input ->
                FileOutputStream(destination).use { output ->
                    input.copyTo(output)
                }
            }
            destination.absolutePath
        }.onFailure {
            destination.delete()
        }.getOrNull()
    }

    private fun assetNameFromPath(path: String): String? {
        return path.takeIf { it.startsWith(ASSET_MODEL_PREFIX) }
            ?.removePrefix(ASSET_MODEL_PREFIX)
            ?.takeIf { it.isNotBlank() }
    }

    companion object {
        private const val KEY_MODEL_PATH = "model_path"
        private const val KEY_MODEL_URI = "model_uri"
        private const val KEY_MODEL_LOCAL_PATH = "model_local_path"
        private const val ASSET_MODEL_PREFIX = "asset://"
    }
}
