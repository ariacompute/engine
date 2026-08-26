import Foundation

public enum AriaDownloadError: Error {
    case invalidModelName(String)
    case requestFailed(Int, String)
    case invalidBundle(String)
}

public final class AriaEngine {
    private var handle: OpaquePointer?

    // MARK: - Model name resolution

    /// Parse `slug`/`quant` from a model name such as `gemma-4-e2b-it_q4`.
    public static func parseBundleName(_ model: String) -> (slug: String, quant: String)? {
        guard !model.isEmpty, !model.contains("/"), !model.contains("\\") else { return nil }
        if let idx = model.lastIndex(of: "_q") {
            let slug = String(model[..<idx])
            let suffix = String(model[model.index(idx, offsetBy: 2)...])
            let quant: String
            switch suffix {
            case "4": quant = "int4"
            case "8": quant = "int8"
            case "326", "3.26": quant = "int326"
            default: return nil
            }
            guard !slug.isEmpty else { return nil }
            return (slug, quant)
        }
        return (model, "int4")
    }

    private static func ariaHome() -> String {
        if let override = ProcessInfo.processInfo.environment["ARIA_COMPUTE_HOME"],
           !override.isEmpty {
            return override
        }
        let home = NSHomeDirectory()
        return (home as NSString).appendingPathComponent(".ariacompute")
    }

    private static func cacheDir(for model: String) -> String {
        return (ariaHome() as NSString).appendingPathComponent("models").appendingPathComponent(model)
    }

    private static func isLocalRef(_ ref: String) -> Bool {
        return ref.contains("/") || ref.contains("\\") || FileManager.default.fileExists(atPath: ref)
    }

    private static func isValidBundle(_ dir: String) -> Bool {
        let fm = FileManager.default
        let weight = (dir as NSString).appendingPathComponent("weight.bin")
        let config = (dir as NSString).appendingPathComponent("config.json")
        guard fm.fileExists(atPath: weight), let data = fm.contents(atPath: config) else { return false }
        guard let meta = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let format = meta["format"] as? String else { return false }
        return format == "aria-quant-bundle"
    }

    // MARK: - Download (regional public hub)

    private static let defaultSDK = "v1.0"
    private static let hubRequired = ["config.json", "weight.bin"]
    private static let hubOptional = [
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
    ]

    private static func preferredPublicHub(_ site: String) -> String {
        site.lowercased().contains("ariacompute.cn") ? "modelscope" : "huggingface"
    }

    private static func hubBearer(_ token: String) -> String? {
        let t = token.trimmingCharacters(in: .whitespacesAndNewlines)
        if t.isEmpty { return nil }
        let low = t.lowercased()
        if low.hasPrefix("sk-") || low.hasPrefix("bfvk-") { return nil }
        return t
    }

    private static func unquoteYAML(_ v: String) -> String {
        let t = v.trimmingCharacters(in: .whitespacesAndNewlines)
        if t.count >= 2 {
            if (t.hasPrefix("\"") && t.hasSuffix("\"")) || (t.hasPrefix("'") && t.hasSuffix("'")) {
                return String(t.dropFirst().dropLast())
            }
        }
        return t
    }

    private static func configYMLScalar(_ key: String) -> String? {
        let path = (ariaHome() as NSString).appendingPathComponent("config.yml")
        guard let raw = try? String(contentsOfFile: path, encoding: .utf8) else { return nil }
        for line in raw.components(separatedBy: "\n") {
            if line.hasPrefix(" ") || line.hasPrefix("\t") { continue }
            let s = line.trimmingCharacters(in: .whitespacesAndNewlines)
            if s.isEmpty || s.hasPrefix("#") { continue }
            guard let idx = s.firstIndex(of: ":") else { continue }
            if String(s[..<idx]).trimmingCharacters(in: .whitespaces) != key { continue }
            let v = unquoteYAML(String(s[s.index(after: idx)...]))
            return v.isEmpty ? nil : v
        }
        return nil
    }

    private static func resolveHubToken(source: String, token: String, hfToken: String, modelscopeApiToken: String) -> String? {
        let named = source == "modelscope" ? modelscopeApiToken : hfToken
        let field = source == "modelscope" ? "modelscope_api_token" : "hf_token"
        for cand in [named, token, configYMLScalar(field) ?? ""] {
            if let b = hubBearer(cand) { return b }
        }
        return nil
    }

    private static func hubPathNames(_ model: String) -> [String] {
        var names = [model]
        var lower = model.lowercased()
        var core = model
        for suf in ["_channel", "_group"] where lower.hasSuffix(suf) {
            core = String(model.dropLast(suf.count))
            lower = core.lowercased()
            break
        }
        var stems = [core]
        if lower.hasSuffix("_q326") {
            stems.append(String(core.dropLast(5)) + "q3.26")
        } else if lower.hasSuffix("_q3.26") {
            stems.append(String(core.dropLast(6)) + "q326")
        }
        for stem in stems {
            for share in ["", "_channel", "_group"] {
                let cand = stem + share
                if !names.contains(cand) { names.append(cand) }
            }
        }
        return names
    }

    private static func hubFileURLs(source: String, model: String, file: String) -> [String] {
        var urls: [String] = []
        for name in hubPathNames(model) {
            if source == "modelscope" {
                for repo in ["AriaCompute/\(name)", "AriaCompute/model"] {
                    urls.append("https://www.modelscope.cn/models/\(repo)/resolve/master/\(defaultSDK)/\(name)/\(file)")
                    urls.append("https://modelscope.cn/models/\(repo)/resolve/master/\(defaultSDK)/\(name)/\(file)")
                }
            } else {
                for repo in ["ariacompute/\(name)", "ariacompute/model"] {
                    urls.append("https://huggingface.co/\(repo)/resolve/main/\(defaultSDK)/\(name)/\(file)")
                }
            }
        }
        return urls
    }

    private static func fetchURLToFile(_ urlStr: String, dest: String, token: String?) throws {
        guard let url = URL(string: urlStr) else {
            throw AriaDownloadError.requestFailed(0, "bad url")
        }
        var req = URLRequest(url: url)
        if let token, !token.isEmpty {
            req.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        let (data, code): (Data, Int) = try awaitWith { completion in
            URLSession.shared.dataTask(with: req) { data, resp, err in
                if let err = err { completion(.failure(err)); return }
                let code = (resp as? HTTPURLResponse)?.statusCode ?? 0
                guard let data = data else {
                    completion(.failure(AriaDownloadError.requestFailed(code, "empty")))
                    return
                }
                completion(.success((data, code)))
            }.resume()
        }
        if code == 401 || code == 403 {
            throw AriaDownloadError.requestFailed(code, "auth")
        }
        if code != 200 {
            throw AriaDownloadError.requestFailed(code, "http")
        }
        let destURL = URL(fileURLWithPath: dest)
        try FileManager.default.createDirectory(at: destURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        try data.write(to: destURL)
    }

    private static func fetchHubFile(source: String, model: String, file: String, dest: String, token: String?, required: Bool) throws {
        var last: Error?
        for url in hubFileURLs(source: source, model: model, file: file) {
            do {
                try fetchURLToFile(url, dest: dest, token: token)
                return
            } catch let AriaDownloadError.requestFailed(code, _) where code == 401 || code == 403 {
                let field = source == "modelscope" ? "modelscope_api_token" : "hf_token"
                throw AriaDownloadError.requestFailed(
                    code,
                    "auth failed HTTP \(code); set \(field) via aria-engine auth (do not pass a Dashboard sk-/bfvk- key as the hub token)"
                )
            } catch {
                last = error
            }
        }
        if required {
            throw AriaDownloadError.requestFailed(0, "\(source): missing \(file)\(last.map { ": \($0)" } ?? "")")
        }
    }

    /// Download `model` from the regional public hub into `~/.ariacompute/models/{model}`.
    /// Dashboard is not used. Skips the download when a valid bundle is already cached.
    @discardableResult
    public static func downloadModel(
        _ model: String,
        token: String = "",
        site: String = "https://ariacompute.com",
        hfToken: String = "",
        modelscopeApiToken: String = ""
    ) throws -> String {
        guard parseBundleName(model) != nil else {
            throw AriaDownloadError.invalidModelName(model)
        }
        let cache = cacheDir(for: model)
        if FileManager.default.fileExists(atPath: cache), isValidBundle(cache) {
            return cache
        }
        let source = preferredPublicHub(site)
        let hubToken = resolveHubToken(source: source, token: token, hfToken: hfToken, modelscopeApiToken: modelscopeApiToken)
        let staging = (ariaHome() as NSString).appendingPathComponent("models").appendingPathComponent(".\(model).partial")
        try? FileManager.default.removeItem(atPath: staging)
        try FileManager.default.createDirectory(atPath: staging, withIntermediateDirectories: true)
        do {
            for file in hubRequired {
                try fetchHubFile(source: source, model: model, file: file, dest: (staging as NSString).appendingPathComponent(file), token: hubToken, required: true)
            }
            for extra in hubOptional {
                try? fetchHubFile(source: source, model: model, file: extra, dest: (staging as NSString).appendingPathComponent(extra), token: hubToken, required: false)
            }
            if !isValidBundle(staging) {
                throw AriaDownloadError.invalidBundle("need weight.bin + aria-quant-bundle config.json")
            }
            try? FileManager.default.removeItem(atPath: cache)
            try FileManager.default.moveItem(atPath: staging, toPath: cache)
            return cache
        } catch {
            try? FileManager.default.removeItem(atPath: staging)
            throw error
        }
    }

    // MARK: - Init

    public init(bundlePath: String) throws {
        // Link libaria_ffi and call aria_model_init via bridging header / module map.
        // Stub for host documentation; wire C calls when XCFramework is linked.
        self.handle = nil
        if bundlePath.isEmpty { throw NSError(domain: "Aria", code: 1) }
    }

    /// Open a model by reference. A value containing a separator or already on
    /// disk is a local path (loaded directly); otherwise it is a model name
    /// downloaded from the regional public hub then loaded.
    public static func open(
        _ ref: String,
        token: String = "",
        site: String = "https://ariacompute.com",
        hfToken: String = "",
        modelscopeApiToken: String = ""
    ) throws -> AriaEngine {
        if isLocalRef(ref) {
            return try AriaEngine(bundlePath: ref)
        }
        let bundle = try downloadModel(ref, token: token, site: site, hfToken: hfToken, modelscopeApiToken: modelscopeApiToken)
        return try AriaEngine(bundlePath: bundle)
    }

    public func complete(messagesJson: String, optionsJson: String, toolsJson: String = "[]") throws -> String {
        // aria_complete(...)
        return #"{"success":true,"response":"","function_calls":[]}"#
    }

    deinit {
        // aria_model_destroy(handle)
    }
}

/// Minimal synchronous wrapper around an async URLSession completion handler.
private func awaitWith<T>(_ body: (@escaping (Result<T, Error>) -> Void) -> Void) throws -> T {
    var result: Result<T, Error>?
    let semaphore = DispatchSemaphore(value: 0)
    body { res in
        result = res
        semaphore.signal()
    }
    semaphore.wait()
    return try result!.get()
}
