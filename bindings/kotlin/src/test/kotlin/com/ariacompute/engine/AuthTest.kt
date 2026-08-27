package com.ariacompute.engine

import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class AuthTest {
    @Test
    fun defaults() {
        val cfg = defaultAuthConfig()
        assertEquals("balance", cfg.hybridMode)
        assertEquals("hybrid", cfg.hybridExecution)
        assertTrue(cfg.hybridSemantic)
        assertEquals(800, cfg.hybridSemanticTimeoutMs)
        assertEquals(512, cfg.hybridSemanticCacheSize)
        assertEquals("auto", cfg.compute)
    }

    @Test
    fun invalidEnum() {
        assertFailsWith<IllegalArgumentException> {
            applyAuthFields(defaultAuthConfig(), hybridMode = "fast")
        }
        assertFailsWith<IllegalArgumentException> {
            applyAuthFields(defaultAuthConfig(), hybridExecution = "local")
        }
        assertFailsWith<IllegalArgumentException> {
            applyAuthFields(defaultAuthConfig(), compute = "gpu")
        }
    }

    @Test
    fun fillUrlsFromCnSite() {
        val got = fillAuthUrls(AuthConfig(siteUrl = CN_SITE))
        assertEquals(CN_CLOUD, got.cloudUrl)
        assertEquals(CN_UPGRADE, got.upgradeUrl)
    }

    @Test
    fun instanceAllFields() {
        val eng = AriaEngine()
        eng.auth(
            cloudApiKey = "sk-test",
            cloudUrl = CN_CLOUD,
            siteUrl = CN_SITE,
            upgradeUrl = CN_UPGRADE,
            hybridMode = "cost",
            hybridExecution = "device",
            hybridSemantic = false,
            hybridSemanticTimeoutMs = 250,
            hybridSemanticCacheSize = 16,
            compute = "cpu",
            hfToken = "hf_abc",
            modelscopeApiToken = "ms_xyz",
        )
        val st = eng.authStatus()
        assertEquals("sk-test", st.cloudApiKey)
        assertEquals("cost", st.hybridMode)
        assertEquals("device", st.hybridExecution)
        assertFalse(st.hybridSemantic)
        assertEquals("cpu", st.compute)
        assertEquals("hf_abc", st.hfToken)
        assertEquals(CN_SITE, st.siteUrl)
    }

    @Test
    fun partialMerge() {
        val eng = AriaEngine()
        eng.auth(hfToken = "hf_one", hybridMode = "intelligence")
        eng.auth(compute = "cuda")
        val st = eng.authStatus()
        assertEquals("hf_one", st.hfToken)
        assertEquals("intelligence", st.hybridMode)
        assertEquals("cuda", st.compute)
    }

    @Test
    fun invalidEnumLeavesState() {
        val eng = AriaEngine()
        eng.auth(hybridMode = "cost")
        assertFailsWith<IllegalArgumentException> { eng.auth(hybridMode = "nope") }
        assertEquals("cost", eng.authStatus().hybridMode)
    }

    @Test
    fun clearResetsInstance() {
        val eng = AriaEngine()
        eng.auth(hfToken = "hf_x", hybridMode = "cost")
        eng.authClear()
        val st = eng.authStatus()
        assertEquals("", st.hfToken)
        assertEquals("balance", st.hybridMode)
    }

    @Test
    fun fillsUrlsFromSiteTld() {
        val eng = AriaEngine()
        eng.auth(siteUrl = "https://ariacompute.cn")
        val st = eng.authStatus()
        assertEquals(CN_CLOUD, st.cloudUrl)
        assertEquals(CN_UPGRADE, st.upgradeUrl)
    }

    @Test
    fun doesNotWriteConfigYml() {
        val home = File(System.getProperty("java.io.tmpdir"), "aria-auth-${System.nanoTime()}")
        home.mkdirs()
        try {
            val eng = AriaEngine()
            eng.auth(cloudApiKey = "sk-test", siteUrl = "https://ariacompute.com", hfToken = "hf_x")
            assertFalse(File(home, "config.yml").isFile)
        } finally {
            home.deleteRecursively()
        }
    }

    @Test
    fun detectUrlsFromKeyMocked() {
        val prev = probeDashboard
        probeDashboard = { site, _ -> site.contains("ariacompute.cn") }
        try {
            val eng = AriaEngine()
            eng.auth(cloudApiKey = "sk-region")
            val st = eng.authStatus()
            assertEquals(CN_SITE, st.siteUrl)
            assertEquals(CN_CLOUD, st.cloudUrl)
            assertEquals(CN_UPGRADE, st.upgradeUrl)
        } finally {
            probeDashboard = prev
        }
    }
}
