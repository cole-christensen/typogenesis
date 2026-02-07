import Foundation
import os

/// Module-level loggers for cloud inference actors (actors cannot have static stored properties)
private let ollamaLogger = Logger(subsystem: "com.typogenesis", category: "OllamaProvider")
private let huggingFaceLogger = Logger(subsystem: "com.typogenesis", category: "HuggingFaceProvider")
private let cloudInferenceLogger = Logger(subsystem: "com.typogenesis", category: "CloudInference")

// MARK: - Cloud Inference Infrastructure (Future)
// =============================================================================
//
// STATUS: NOT INTEGRATED - Infrastructure for future versions
//
// This file contains infrastructure for cloud-based AI inference that is NOT
// currently integrated into Typogenesis v1. The v1 release uses CoreML-only
// inference (all models run locally on Apple Silicon).
//
// WHY THIS EXISTS:
// - Provides a clean abstraction for future cloud provider integration
// - Enables hybrid local/cloud inference when models become available
// - Supports devices without capable Neural Engine
// - Allows access to larger/more capable models than can run locally
//
// PROVIDERS (all stubbed):
// - OllamaProvider: For local/self-hosted model inference (requires custom models)
// - HuggingFaceProvider: For HuggingFace Inference API (requires deployed models)
//
// TO INTEGRATE IN FUTURE VERSIONS:
// 1. Train and deploy font-generation models to Ollama/HuggingFace
// 2. Wire CloudInferenceManager into GlyphGenerator, StyleEncoder, KerningPredictor
// 3. Add Settings UI for cloud provider configuration
// 4. Implement proper error handling and fallback logic
//
// DO NOT DELETE: This is valid infrastructure for post-v1 development.
//
// =============================================================================

/// Protocol for cloud inference providers.
///
/// **HONEST STATUS:** This is a stub for future cloud inference support.
/// The Typogenesis v1 release uses CoreML-only (all models run locally on
/// Apple Silicon). Cloud providers can be added later when needed.
///
/// Implementing this protocol allows the app to optionally offload inference
/// to cloud providers for:
/// - Devices without capable Neural Engine
/// - Larger/more capable models than can run locally
/// - Features requiring models not yet converted to CoreML
protocol CloudInferenceProvider: Sendable {
    /// Unique identifier for this provider
    var providerId: String { get }

    /// Human-readable name
    var displayName: String { get }

    /// Check if the provider is available and configured
    func isAvailable() async -> Bool

    /// Check provider health/connectivity
    func healthCheck() async -> CloudInferenceResult<Bool>

    /// Generate a glyph outline from character and style
    func generateGlyph(
        character: Character,
        style: StyleEncoder.FontStyle,
        metrics: FontMetrics
    ) async -> CloudInferenceResult<GlyphOutline>

    /// Encode a glyph to a style embedding
    func encodeStyle(
        from glyphs: [Glyph]
    ) async -> CloudInferenceResult<[Float]>

    /// Predict kerning for a glyph pair
    func predictKerning(
        left: Glyph,
        right: Glyph,
        metrics: FontMetrics
    ) async -> CloudInferenceResult<Int>
}

/// Result type for cloud inference operations
enum CloudInferenceResult<T: Sendable>: Sendable {
    case success(T)
    case failure(CloudInferenceError)

    var value: T? {
        if case .success(let value) = self {
            return value
        }
        return nil
    }

    var error: CloudInferenceError? {
        if case .failure(let error) = self {
            return error
        }
        return nil
    }
}

/// Errors that can occur during cloud inference
enum CloudInferenceError: Error, LocalizedError, Sendable {
    case notConfigured
    case networkUnavailable
    case authenticationFailed
    case rateLimited(retryAfter: TimeInterval?)
    case serverError(statusCode: Int, message: String?)
    case invalidResponse
    case timeout
    case cancelled
    case modelNotAvailable(String)

    var errorDescription: String? {
        switch self {
        case .notConfigured:
            return "Cloud provider is not configured"
        case .networkUnavailable:
            return "Network is unavailable"
        case .authenticationFailed:
            return "Authentication failed - check API key"
        case .rateLimited(let retryAfter):
            if let seconds = retryAfter {
                return "Rate limited - retry after \(Int(seconds)) seconds"
            }
            return "Rate limited - please try again later"
        case .serverError(let code, let message):
            if let msg = message {
                return "Server error (\(code)): \(msg)"
            }
            return "Server error (\(code))"
        case .invalidResponse:
            return "Invalid response from server"
        case .timeout:
            return "Request timed out"
        case .cancelled:
            return "Request was cancelled"
        case .modelNotAvailable(let model):
            return "Model '\(model)' is not available on this provider"
        }
    }
}

// MARK: - Ollama Provider

/// Ollama integration for local/self-hosted model inference.
///
/// - Important: NOT INTEGRATED in Typogenesis v1. This provider is stubbed infrastructure
///   for future cloud inference support. All methods return `.failure(.modelNotAvailable)`.
///
/// **Requirements for future integration:**
/// - Running Ollama server locally or on a network
/// - Custom models fine-tuned for font generation (not currently available)
/// - API integration for image/outline generation
///
/// Ollama primarily supports text/chat models. Font generation would need
/// custom fine-tuned vision/generation models that don't exist yet.
// NOTE: This type compiles but is not wired into the app. Do not use in v1.
actor OllamaProvider: CloudInferenceProvider {
    nonisolated let providerId = "ollama"
    nonisolated let displayName = "Ollama (Local)"

    /// Base URL for Ollama API (default: localhost)
    private var baseURL: URL

    /// Model to use for generation
    private var modelName: String

    /// HTTP client timeout
    private let timeout: TimeInterval

    init(
        baseURL: URL = URL(string: "http://localhost:11434")!,
        modelName: String = "typogenesis-glyph",  // Hypothetical custom model
        timeout: TimeInterval = 30
    ) {
        self.baseURL = baseURL
        self.modelName = modelName
        self.timeout = timeout
    }

    func configure(baseURL: URL, modelName: String) {
        self.baseURL = baseURL
        self.modelName = modelName
    }

    func isAvailable() async -> Bool {
        // Check if Ollama server is running
        let result = await healthCheck()
        return result.value == true
    }

    func healthCheck() async -> CloudInferenceResult<Bool> {
        // Try to reach Ollama API
        let healthURL = baseURL.appendingPathComponent("api/tags")

        var request = URLRequest(url: healthURL)
        request.timeoutInterval = 5  // Short timeout for health check

        do {
            let (_, response) = try await URLSession.shared.data(for: request)
            guard let httpResponse = response as? HTTPURLResponse else {
                return .failure(.invalidResponse)
            }
            return .success((200...299).contains(httpResponse.statusCode))
        } catch {
            ollamaLogger.warning("Health check failed: \(error.localizedDescription)")
            return .failure(.networkUnavailable)
        }
    }

    func generateGlyph(
        character: Character,
        style: StyleEncoder.FontStyle,
        metrics: FontMetrics
    ) async -> CloudInferenceResult<GlyphOutline> {
        // STUB: Would call Ollama API with custom model
        // This would require a fine-tuned model for glyph generation
        ollamaLogger.debug("generateGlyph called - NOT IMPLEMENTED")
        return .failure(.modelNotAvailable("Glyph generation model not available for Ollama"))
    }

    func encodeStyle(
        from glyphs: [Glyph]
    ) async -> CloudInferenceResult<[Float]> {
        // STUB: Would call Ollama API with style encoder model
        ollamaLogger.debug("encodeStyle called - NOT IMPLEMENTED")
        return .failure(.modelNotAvailable("Style encoder model not available for Ollama"))
    }

    func predictKerning(
        left: Glyph,
        right: Glyph,
        metrics: FontMetrics
    ) async -> CloudInferenceResult<Int> {
        // STUB: Would call Ollama API with kerning model
        ollamaLogger.debug("predictKerning called - NOT IMPLEMENTED")
        return .failure(.modelNotAvailable("Kerning model not available for Ollama"))
    }
}

// MARK: - HuggingFace Provider

/// HuggingFace Inference API integration.
///
/// - Important: NOT INTEGRATED in Typogenesis v1. This provider is stubbed infrastructure
///   for future cloud inference support. All methods return `.failure(.modelNotAvailable)`.
///
/// **Requirements for future integration:**
/// - HuggingFace API key configuration
/// - Custom models deployed to HuggingFace Hub
/// - Models fine-tuned for font generation (not currently available)
///
/// HuggingFace hosts many vision and generation models, but specific
/// models for glyph/font generation would need to be trained and deployed.
// NOTE: This type compiles but is not wired into the app. Do not use in v1.
actor HuggingFaceProvider: CloudInferenceProvider {
    nonisolated let providerId = "huggingface"
    nonisolated let displayName = "HuggingFace"

    /// HuggingFace API base URL
    private let baseURL = URL(string: "https://api-inference.huggingface.co")!

    /// API key for authentication
    private var apiKey: String?

    /// Model IDs for different tasks
    private var glyphModelId: String?
    private var styleModelId: String?
    private var kerningModelId: String?

    /// HTTP client timeout
    private let timeout: TimeInterval

    init(
        apiKey: String? = nil,
        timeout: TimeInterval = 60
    ) {
        self.apiKey = apiKey
        self.timeout = timeout
    }

    func configure(
        apiKey: String,
        glyphModelId: String? = nil,
        styleModelId: String? = nil,
        kerningModelId: String? = nil
    ) {
        self.apiKey = apiKey
        self.glyphModelId = glyphModelId
        self.styleModelId = styleModelId
        self.kerningModelId = kerningModelId
    }

    func isAvailable() async -> Bool {
        guard apiKey != nil else { return false }
        let result = await healthCheck()
        return result.value == true
    }

    func healthCheck() async -> CloudInferenceResult<Bool> {
        guard let apiKey = apiKey, !apiKey.isEmpty else {
            return .failure(.notConfigured)
        }

        // Check API connectivity
        let testURL = baseURL.appendingPathComponent("status")

        var request = URLRequest(url: testURL)
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        request.timeoutInterval = 5

        do {
            let (_, response) = try await URLSession.shared.data(for: request)
            guard let httpResponse = response as? HTTPURLResponse else {
                return .failure(.invalidResponse)
            }

            switch httpResponse.statusCode {
            case 200...299:
                return .success(true)
            case 401, 403:
                return .failure(.authenticationFailed)
            case 429:
                let retryAfter = httpResponse.value(forHTTPHeaderField: "Retry-After")
                    .flatMap { TimeInterval($0) }
                return .failure(.rateLimited(retryAfter: retryAfter))
            default:
                return .failure(.serverError(statusCode: httpResponse.statusCode, message: nil))
            }
        } catch {
            huggingFaceLogger.warning("Health check failed: \(error.localizedDescription)")
            return .failure(.networkUnavailable)
        }
    }

    func generateGlyph(
        character: Character,
        style: StyleEncoder.FontStyle,
        metrics: FontMetrics
    ) async -> CloudInferenceResult<GlyphOutline> {
        guard apiKey != nil else {
            return .failure(.notConfigured)
        }

        guard let modelId = glyphModelId else {
            return .failure(.modelNotAvailable("No glyph generation model configured"))
        }

        // STUB: Would call HuggingFace Inference API
        huggingFaceLogger.debug("generateGlyph called for model \(modelId) - NOT IMPLEMENTED")
        return .failure(.modelNotAvailable("Glyph generation model not deployed to HuggingFace"))
    }

    func encodeStyle(
        from glyphs: [Glyph]
    ) async -> CloudInferenceResult<[Float]> {
        guard apiKey != nil else {
            return .failure(.notConfigured)
        }

        guard let modelId = styleModelId else {
            return .failure(.modelNotAvailable("No style encoder model configured"))
        }

        // STUB: Would call HuggingFace Inference API
        huggingFaceLogger.debug("encodeStyle called for model \(modelId) - NOT IMPLEMENTED")
        return .failure(.modelNotAvailable("Style encoder model not deployed to HuggingFace"))
    }

    func predictKerning(
        left: Glyph,
        right: Glyph,
        metrics: FontMetrics
    ) async -> CloudInferenceResult<Int> {
        guard apiKey != nil else {
            return .failure(.notConfigured)
        }

        guard let modelId = kerningModelId else {
            return .failure(.modelNotAvailable("No kerning model configured"))
        }

        // STUB: Would call HuggingFace Inference API
        huggingFaceLogger.debug("predictKerning called for model \(modelId) - NOT IMPLEMENTED")
        return .failure(.modelNotAvailable("Kerning model not deployed to HuggingFace"))
    }
}

// MARK: - Cloud Inference Manager

/// Manages cloud inference providers and fallback logic.
///
/// - Important: NOT INTEGRATED in Typogenesis v1. This manager exists for future extensibility.
///   v1 uses CoreML-only inference (local on Apple Silicon). Cloud providers are optional
///   additions for future versions when suitable models become available.
///
/// **To integrate in future versions:**
/// 1. Wire this manager into `GlyphGenerator`, `StyleEncoder`, `KerningPredictor`
/// 2. Add Settings UI for cloud provider configuration (API keys, endpoints)
/// 3. Implement proper model deployment for Ollama/HuggingFace
/// 4. Replace stub implementations with actual API calls
// NOTE: This type compiles but is not wired into the app. Do not use in v1.
@MainActor
final class CloudInferenceManager: ObservableObject {
    static let shared = CloudInferenceManager()

    /// Available providers
    @Published private(set) var providers: [any CloudInferenceProvider] = []

    /// Currently active provider (nil = local CoreML only)
    @Published var activeProviderId: String?

    /// Whether cloud inference is enabled
    @Published var isCloudEnabled: Bool = false

    /// Last error from cloud inference
    @Published private(set) var lastError: CloudInferenceError?

    private let ollama = OllamaProvider()
    private let huggingFace = HuggingFaceProvider()

    private init() {
        providers = [ollama, huggingFace]
    }

    /// Get the active provider, if any
    var activeProvider: (any CloudInferenceProvider)? {
        guard isCloudEnabled, let id = activeProviderId else { return nil }
        return providers.first { $0.providerId == id }
    }

    /// Configure Ollama provider
    func configureOllama(baseURL: URL, modelName: String) async {
        await ollama.configure(baseURL: baseURL, modelName: modelName)
    }

    /// Configure HuggingFace provider
    func configureHuggingFace(
        apiKey: String,
        glyphModelId: String? = nil,
        styleModelId: String? = nil,
        kerningModelId: String? = nil
    ) async {
        await huggingFace.configure(
            apiKey: apiKey,
            glyphModelId: glyphModelId,
            styleModelId: styleModelId,
            kerningModelId: kerningModelId
        )
    }

    /// Check availability of all providers
    func checkProviderAvailability() async -> [String: Bool] {
        var availability: [String: Bool] = [:]

        for provider in providers {
            availability[provider.providerId] = await provider.isAvailable()
        }

        return availability
    }

    /// Try to generate a glyph using cloud inference, with fallback
    func generateGlyphWithFallback(
        character: Character,
        style: StyleEncoder.FontStyle,
        metrics: FontMetrics,
        localFallback: () async throws -> GlyphOutline
    ) async throws -> GlyphOutline {
        // If cloud is enabled and provider is active, try cloud first
        if let provider = activeProvider {
            let result = await provider.generateGlyph(
                character: character,
                style: style,
                metrics: metrics
            )

            if let outline = result.value {
                cloudInferenceLogger.debug("Used cloud provider \(provider.providerId) for glyph generation")
                return outline
            }

            // Cloud failed - log and fall back to local
            if let error = result.error {
                lastError = error
                cloudInferenceLogger.warning("Cloud inference failed: \(error.localizedDescription), falling back to local")
            }
        }

        // Use local fallback
        return try await localFallback()
    }
}
