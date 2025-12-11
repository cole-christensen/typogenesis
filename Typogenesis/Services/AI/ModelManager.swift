import Foundation
import CoreML
import Combine

/// Manages AI model loading, caching, and lifecycle
@MainActor
final class ModelManager: ObservableObject, Sendable {
    static let shared = ModelManager()

    // MARK: - Model Status

    enum ModelStatus: Equatable, Sendable {
        case notDownloaded
        case downloading(progress: Double)
        case downloaded
        case loading
        case loaded
        case error(String)

        var isAvailable: Bool {
            self == .loaded
        }

        var displayText: String {
            switch self {
            case .notDownloaded:
                return "Not Downloaded"
            case .downloading(let progress):
                return "Downloading \(Int(progress * 100))%"
            case .downloaded:
                return "Ready to Load"
            case .loading:
                return "Loading..."
            case .loaded:
                return "Ready"
            case .error(let message):
                return "Error: \(message)"
            }
        }
    }

    // MARK: - Model Types

    enum ModelType: String, CaseIterable, Identifiable, Sendable {
        case glyphDiffusion = "GlyphDiffusion"
        case styleEncoder = "StyleEncoder"
        case kerningNet = "KerningNet"

        var id: String { rawValue }

        var displayName: String {
            switch self {
            case .glyphDiffusion:
                return "Glyph Generator"
            case .styleEncoder:
                return "Style Encoder"
            case .kerningNet:
                return "Kerning Predictor"
            }
        }

        var description: String {
            switch self {
            case .glyphDiffusion:
                return "Generates new glyph shapes using diffusion"
            case .styleEncoder:
                return "Extracts style features from existing fonts"
            case .kerningNet:
                return "Predicts optimal kerning values"
            }
        }

        var estimatedSize: String {
            switch self {
            case .glyphDiffusion:
                return "~150 MB"
            case .styleEncoder:
                return "~50 MB"
            case .kerningNet:
                return "~25 MB"
            }
        }

        var fileName: String {
            "\(rawValue).mlmodelc"
        }
    }

    // MARK: - Published State

    @Published private(set) var glyphDiffusionStatus: ModelStatus = .notDownloaded
    @Published private(set) var styleEncoderStatus: ModelStatus = .notDownloaded
    @Published private(set) var kerningNetStatus: ModelStatus = .notDownloaded

    // MARK: - Private State

    private var glyphDiffusionModel: MLModel?
    private var styleEncoderModel: MLModel?
    private var kerningNetModel: MLModel?

    private let modelsDirectory: URL
    private var downloadTasks: [ModelType: URLSessionDownloadTask] = [:]

    // MARK: - Initialization

    private init() {
        // Set up models directory in Application Support
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        modelsDirectory = appSupport.appendingPathComponent("Typogenesis/Models", isDirectory: true)

        // Create directory if needed
        try? FileManager.default.createDirectory(at: modelsDirectory, withIntermediateDirectories: true)

        // Check which models are already downloaded
        checkDownloadedModels()
    }

    // MARK: - Public API

    /// Get status for a specific model type
    func status(for modelType: ModelType) -> ModelStatus {
        switch modelType {
        case .glyphDiffusion:
            return glyphDiffusionStatus
        case .styleEncoder:
            return styleEncoderStatus
        case .kerningNet:
            return kerningNetStatus
        }
    }

    /// Check if all models are loaded and ready
    var allModelsReady: Bool {
        glyphDiffusionStatus == .loaded &&
        styleEncoderStatus == .loaded &&
        kerningNetStatus == .loaded
    }

    /// Check if any model is currently loading or downloading
    var isLoading: Bool {
        switch glyphDiffusionStatus {
        case .downloading, .loading: return true
        default: break
        }
        switch styleEncoderStatus {
        case .downloading, .loading: return true
        default: break
        }
        switch kerningNetStatus {
        case .downloading, .loading: return true
        default: break
        }
        return false
    }

    /// Load all downloaded models
    func loadAllModels() async {
        for modelType in ModelType.allCases {
            if isModelDownloaded(modelType) {
                await loadModel(modelType)
            }
        }
    }

    /// Load a specific model
    func loadModel(_ modelType: ModelType) async {
        let modelURL = modelsDirectory.appendingPathComponent(modelType.fileName)

        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            setStatus(.notDownloaded, for: modelType)
            return
        }

        setStatus(.loading, for: modelType)

        do {
            let config = MLModelConfiguration()
            config.computeUnits = .cpuAndNeuralEngine

            let model = try await MLModel.load(contentsOf: modelURL, configuration: config)

            switch modelType {
            case .glyphDiffusion:
                glyphDiffusionModel = model
            case .styleEncoder:
                styleEncoderModel = model
            case .kerningNet:
                kerningNetModel = model
            }

            setStatus(.loaded, for: modelType)
        } catch {
            setStatus(.error(error.localizedDescription), for: modelType)
        }
    }

    /// Download a model (placeholder - would connect to model hosting)
    func downloadModel(_ modelType: ModelType) async {
        // In a real implementation, this would download from a server
        // For now, we'll simulate the download process

        setStatus(.downloading(progress: 0), for: modelType)

        // Simulate download progress
        for i in 1...10 {
            try? await Task.sleep(nanoseconds: 200_000_000)
            setStatus(.downloading(progress: Double(i) / 10.0), for: modelType)
        }

        // Mark as downloaded (in reality, would write file)
        setStatus(.downloaded, for: modelType)

        // Automatically load after download
        await loadModel(modelType)
    }

    /// Cancel an in-progress download
    func cancelDownload(_ modelType: ModelType) {
        downloadTasks[modelType]?.cancel()
        downloadTasks[modelType] = nil
        setStatus(.notDownloaded, for: modelType)
    }

    /// Unload a model to free memory
    func unloadModel(_ modelType: ModelType) {
        switch modelType {
        case .glyphDiffusion:
            glyphDiffusionModel = nil
        case .styleEncoder:
            styleEncoderModel = nil
        case .kerningNet:
            kerningNetModel = nil
        }

        if isModelDownloaded(modelType) {
            setStatus(.downloaded, for: modelType)
        } else {
            setStatus(.notDownloaded, for: modelType)
        }
    }

    /// Delete a downloaded model
    func deleteModel(_ modelType: ModelType) {
        // Unload first
        unloadModel(modelType)

        // Delete file
        let modelURL = modelsDirectory.appendingPathComponent(modelType.fileName)
        try? FileManager.default.removeItem(at: modelURL)

        setStatus(.notDownloaded, for: modelType)
    }

    // MARK: - Model Access

    /// Get the glyph diffusion model (if loaded)
    var glyphDiffusion: MLModel? {
        glyphDiffusionModel
    }

    /// Get the style encoder model (if loaded)
    var styleEncoder: MLModel? {
        styleEncoderModel
    }

    /// Get the kerning net model (if loaded)
    var kerningNet: MLModel? {
        kerningNetModel
    }

    // MARK: - Private Helpers

    private func checkDownloadedModels() {
        for modelType in ModelType.allCases {
            if isModelDownloaded(modelType) {
                setStatus(.downloaded, for: modelType)
            }
        }
    }

    private func isModelDownloaded(_ modelType: ModelType) -> Bool {
        let modelURL = modelsDirectory.appendingPathComponent(modelType.fileName)
        return FileManager.default.fileExists(atPath: modelURL.path)
    }

    private func setStatus(_ status: ModelStatus, for modelType: ModelType) {
        switch modelType {
        case .glyphDiffusion:
            glyphDiffusionStatus = status
        case .styleEncoder:
            styleEncoderStatus = status
        case .kerningNet:
            kerningNetStatus = status
        }
    }
}

// MARK: - Model Info Display

extension ModelManager {
    /// Summary of all model statuses for UI display
    var statusSummary: String {
        let total = ModelType.allCases.count
        let loaded = ModelType.allCases.filter { status(for: $0) == .loaded }.count

        if loaded == total {
            return "All models ready"
        } else if loaded > 0 {
            return "\(loaded)/\(total) models loaded"
        } else {
            return "No models loaded"
        }
    }

    /// Total estimated size of all models
    nonisolated static var totalModelSize: String {
        "~225 MB"
    }
}
