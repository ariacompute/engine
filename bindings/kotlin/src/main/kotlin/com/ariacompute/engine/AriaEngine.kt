package com.ariacompute.engine

/** Kotlin/JVM + Android JNI wrapper over libaria_ffi. */
class AriaEngine(bundlePath: String) : AutoCloseable {
    private var handle: Long = nativeInit(bundlePath)

    init {
        if (handle == 0L) throw IllegalStateException(nativeLastError() ?: "init failed")
    }

    fun complete(messagesJson: String, optionsJson: String = """{"max_tokens":16}""", toolsJson: String = "[]"): String {
        return nativeComplete(handle, messagesJson, optionsJson, toolsJson)
            ?: throw IllegalStateException(nativeLastError() ?: "complete failed")
    }

    fun embed(inputJson: String): String =
        nativeEmbed(handle, inputJson) ?: throw IllegalStateException(nativeLastError() ?: "embed failed")

    fun transcribe(pcm: ByteArray): String =
        nativeTranscribe(handle, pcm) ?: throw IllegalStateException(nativeLastError() ?: "transcribe failed")

    override fun close() {
        if (handle != 0L) {
            nativeDestroy(handle)
            handle = 0L
        }
    }

    private external fun nativeInit(path: String): Long
    private external fun nativeDestroy(handle: Long)
    private external fun nativeComplete(handle: Long, messages: String, options: String, tools: String): String?
    private external fun nativeEmbed(handle: Long, input: String): String?
    private external fun nativeTranscribe(handle: Long, pcm: ByteArray): String?
    private external fun nativeLastError(): String?

    companion object {
        init {
            try {
                System.loadLibrary("aria_ffi")
            } catch (_: UnsatisfiedLinkError) {
                // Host tests may System.load(ARIA_FFI_LIB) before constructing.
            }
        }
    }
}
