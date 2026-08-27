import Foundation

public let intlCloud = "https://gateway.ariacompute.com"
public let intlSite = "https://ariacompute.com"
public let intlUpgrade = "https://github.com/ariacompute"
public let cnCloud = "https://gateway.ariacompute.cn"
public let cnSite = "https://ariacompute.cn"
public let cnUpgrade = "https://gitee.com/ariacompute"

public struct AuthConfig: Equatable {
    public var cloudApiKey: String = ""
    public var cloudUrl: String = ""
    public var siteUrl: String = ""
    public var upgradeUrl: String = ""
    public var hybridMode: String = "balance"
    public var hybridExecution: String = "hybrid"
    public var hybridSemantic: Bool = true
    public var hybridSemanticTimeoutMs: Int = 800
    public var hybridSemanticCacheSize: Int = 512
    public var compute: String = "auto"
    public var hfToken: String = ""
    public var modelscopeApiToken: String = ""

    public init() {}
}

public struct AuthUpdates {
    public var cloudApiKey: String?
    public var cloudUrl: String?
    public var siteUrl: String?
    public var upgradeUrl: String?
    public var hybridMode: String?
    public var hybridExecution: String?
    public var hybridSemantic: Bool?
    public var hybridSemanticTimeoutMs: Int?
    public var hybridSemanticCacheSize: Int?
    public var compute: String?
    public var hfToken: String?
    public var modelscopeApiToken: String?

    public init(
        cloudApiKey: String? = nil,
        cloudUrl: String? = nil,
        siteUrl: String? = nil,
        upgradeUrl: String? = nil,
        hybridMode: String? = nil,
        hybridExecution: String? = nil,
        hybridSemantic: Bool? = nil,
        hybridSemanticTimeoutMs: Int? = nil,
        hybridSemanticCacheSize: Int? = nil,
        compute: String? = nil,
        hfToken: String? = nil,
        modelscopeApiToken: String? = nil
    ) {
        self.cloudApiKey = cloudApiKey
        self.cloudUrl = cloudUrl
        self.siteUrl = siteUrl
        self.upgradeUrl = upgradeUrl
        self.hybridMode = hybridMode
        self.hybridExecution = hybridExecution
        self.hybridSemantic = hybridSemantic
        self.hybridSemanticTimeoutMs = hybridSemanticTimeoutMs
        self.hybridSemanticCacheSize = hybridSemanticCacheSize
        self.compute = compute
        self.hfToken = hfToken
        self.modelscopeApiToken = modelscopeApiToken
    }
}

public enum AriaAuthError: Error, Equatable {
    case invalidHybridMode(String)
    case invalidHybridExecution(String)
    case invalidCompute(String)
    case invalidTimeoutOrCache
}

func gatewayRegion(_ url: String) -> String? {
    let lower = url.lowercased()
    if lower.contains("ariacompute.cn") || lower.contains("gitee.com/ariacompute") { return "cn" }
    if lower.contains("ariacompute.com") || lower.contains("github.com/ariacompute") { return "intl" }
    return nil
}

func pairUrls(_ region: String) -> (String, String, String) {
    region == "cn" ? (cnCloud, cnSite, cnUpgrade) : (intlCloud, intlSite, intlUpgrade)
}

public func fillAuthUrls(_ cfg: AuthConfig) -> AuthConfig {
    var out = cfg
    guard let region = gatewayRegion(out.siteUrl) ?? gatewayRegion(out.cloudUrl) ?? gatewayRegion(out.upgradeUrl) else {
        return out
    }
    let (cloud, site, upgrade) = pairUrls(region)
    if out.cloudUrl.isEmpty { out.cloudUrl = cloud }
    if out.siteUrl.isEmpty { out.siteUrl = site }
    if out.upgradeUrl.isEmpty { out.upgradeUrl = upgrade }
    return out
}

func localePrefersCn() -> Bool {
    let lang = ((ProcessInfo.processInfo.environment["LANG"] ?? "") + (ProcessInfo.processInfo.environment["LC_ALL"] ?? "")).lowercased()
    return lang.contains("zh") || lang.contains(".cn") || lang.hasPrefix("cn")
}

/// Replace in tests to avoid a real Dashboard probe.
public var probeDashboard: (String, String) -> Bool = defaultProbeDashboard

func defaultProbeDashboard(siteUrl: String, apiKey: String) -> Bool {
    let urlString = siteUrl.trimmingCharacters(in: CharacterSet(charactersIn: "/")) + "/api/dashboard/models"
    guard let url = URL(string: urlString) else { return false }
    var req = URLRequest(url: url, timeoutInterval: 10)
    req.setValue("aria-engine-sdk/0.1.0", forHTTPHeaderField: "User-Agent")
    req.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
    var ok = false
    let sem = DispatchSemaphore(value: 0)
    URLSession.shared.dataTask(with: req) { _, resp, _ in
        if let http = resp as? HTTPURLResponse, (200..<300).contains(http.statusCode) {
            ok = true
        }
        sem.signal()
    }.resume()
    _ = sem.wait(timeout: .now() + 10)
    return ok
}

public func detectGatewayPair(_ apiKey: String) -> (String, String, String) {
    let key = apiKey.trimmingCharacters(in: .whitespacesAndNewlines)
    let first = localePrefersCn() ? "cn" : "intl"
    let second = first == "cn" ? "intl" : "cn"
    for region in [first, second] {
        let (cloud, site, upgrade) = pairUrls(region)
        if !key.isEmpty && probeDashboard(site, key) {
            return (cloud, site, upgrade)
        }
    }
    return pairUrls(first)
}

public func applyAuth(_ existing: AuthConfig, _ updates: AuthUpdates) throws -> AuthConfig {
    var out = existing
    if let v = updates.cloudApiKey { out.cloudApiKey = v }
    if let v = updates.cloudUrl { out.cloudUrl = v }
    if let v = updates.siteUrl { out.siteUrl = v }
    if let v = updates.upgradeUrl { out.upgradeUrl = v }
    if let v = updates.hybridMode { out.hybridMode = v }
    if let v = updates.hybridExecution { out.hybridExecution = v }
    if let v = updates.hybridSemantic { out.hybridSemantic = v }
    if let v = updates.hybridSemanticTimeoutMs { out.hybridSemanticTimeoutMs = v }
    if let v = updates.hybridSemanticCacheSize { out.hybridSemanticCacheSize = v }
    if let v = updates.compute { out.compute = v }
    if let v = updates.hfToken { out.hfToken = v }
    if let v = updates.modelscopeApiToken { out.modelscopeApiToken = v }
    let modes = ["cost", "balance", "intelligence"]
    if !modes.contains(out.hybridMode) { throw AriaAuthError.invalidHybridMode(out.hybridMode) }
    let execs = ["hybrid", "device", "cloud"]
    if !execs.contains(out.hybridExecution) { throw AriaAuthError.invalidHybridExecution(out.hybridExecution) }
    let computes = ["auto", "cpu", "cuda"]
    if !computes.contains(out.compute) { throw AriaAuthError.invalidCompute(out.compute) }
    if out.hybridSemanticTimeoutMs <= 0 || out.hybridSemanticCacheSize <= 0 {
        throw AriaAuthError.invalidTimeoutOrCache
    }
    out = fillAuthUrls(out)
    if !out.cloudApiKey.isEmpty && (out.cloudUrl.isEmpty || out.siteUrl.isEmpty || out.upgradeUrl.isEmpty) {
        let (cloud, site, upgrade) = detectGatewayPair(out.cloudApiKey)
        if out.cloudUrl.isEmpty { out.cloudUrl = cloud }
        if out.siteUrl.isEmpty { out.siteUrl = site }
        if out.upgradeUrl.isEmpty { out.upgradeUrl = upgrade }
    }
    return out
}
