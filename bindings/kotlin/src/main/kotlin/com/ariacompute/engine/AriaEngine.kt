package com.ariacompute.engine

import org.json.JSONObject
import java.io.File

/**
 * JVM binding placeholder: load libaria-engine_ffi via JNA/JNI in a follow-up.
 * API shape matches System One decide.
 */
class AriaEngine(private val checkpoint: String, private val track: String = "encoder") {
    fun systemone(request: JSONObject): JSONObject {
        throw UnsupportedOperationException(
            "Link libaria-engine_ffi and call aria_systemone (checkpoint=$checkpoint track=$track)"
        )
    }

    fun destroy() {}

    companion object {
        fun libCandidates(): List<File> {
            val env = System.getenv("ARIA_FFI_LIB")
            if (env != null) return listOf(File(env))
            val home = System.getenv("ARIA_COMPUTE_HOME")
                ?: (System.getProperty("user.home") + "/.ariacompute")
            return listOf(
                File("$home/lib/libaria-engine_ffi.so"),
                File("$home/lib/libaria-engine_ffi.dylib"),
                File("$home/lib/aria-engine_ffi.dll"),
            )
        }
    }
}
