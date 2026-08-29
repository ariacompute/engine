package com.ariacompute.engine

import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse

class AuthTest {
    @Test
    fun defaults() {
        val cfg = defaultAuthConfig()
        assertEquals("auto", cfg.compute)
        assertEquals("", cfg.router)
    }

    @Test
    fun invalidEnum() {
        assertFailsWith<IllegalArgumentException> {
            applyAuthFields(defaultAuthConfig(), compute = "gpu")
        }
    }

    @Test
    fun fillUrlsFromCnSite() {
        val got = fillAuthUrls(AuthConfig(siteUrl = CN_SITE))
        assertEquals(CN_UPGRADE, got.upgradeUrl)
    }

    @Test
    fun allFields() {
        val st = applyAuthFields(
            defaultAuthConfig(),
            router = "http://127.0.0.1:8080",
            siteUrl = CN_SITE,
            upgradeUrl = CN_UPGRADE,
            compute = "cpu",
            hfToken = "hf_abc",
            modelscopeApiToken = "ms_xyz",
        )
        assertEquals("http://127.0.0.1:8080", st.router)
        assertEquals("cpu", st.compute)
        assertEquals("hf_abc", st.hfToken)
    }

    @Test
    fun doesNotWriteConfigYml() {
        val home = File.createTempFile("aria-kt", "home")
        home.delete()
        home.mkdirs()
        System.setProperty("user.home", home.absolutePath)
        applyAuthFields(defaultAuthConfig(), router = "http://127.0.0.1:8080", hfToken = "hf_x")
        assertFalse(File(home, "config.yml").isFile)
    }
}
