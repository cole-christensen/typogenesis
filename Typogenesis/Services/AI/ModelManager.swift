import Foundation
@preconcurrency import CoreML
import Combine
import CryptoKit
import os

/// Manages AI model loading, caching, and lifecycle.
///
/// **HONEST STATUS:** Model downloads are not yet implemented. The download
/// infrastructure exists but there is no server to download from. When models
/// become available, this will be updated to download from a real server.
///
/// Currently, all AI features fall back to geometric/placeholder generation.
@MainActor
final class ModelManager: ObservableObject {
    static let shared = ModelManager()

    private let logger = Logger(subsystem: "com.typogenesis", category: "ModelManager")

    // MARK: - Model Status

    enum ModelStatus: Equatable, Sendable {
        case notDownloaded
        case downloading(progress: Double)
        case downloaded
        case validating
        case loading
        case loaded
        case updateAvailable(currentVersion: String, newVersion: String)
        case error(String)

        var isAvailable: Bool {
            self == .loaded
        }

        var displayText: String {
            switch self {
            case .notDownloaded:
                return "Not Downloaded"
            case .downloading(let progress):
                // Clamp progress to valid range and handle NaN/Infinity
                let safeProgress: Double
                if progress.isNaN || progress.isInfinite {
                    safeProgress = 0
                } else {
                    safeProgress = min(max(progress, 0), 1)
                }
                return "Downloading \(Int(safeProgress * 100))%"
            case .downloaded:
                return "Ready to Load"
            case .validating:
                return "Validating..."
            case .loading:
                return "Loading..."
            case .loaded:
                return "Ready"
            case .updateAvailable(let current, let new):
                return "Update available: v\(current) -> v\(new)"
            case .error(let message):
                return "Error: \(message)"
            }
        }
    }

    // MARK: - Model Version Info

    /// Information about a model version
    struct ModelVersionInfo: Codable, Equatable, Sendable {
        let version: String
        let checksum: String  // SHA256 hash of the model file
        let size: Int64       // Size in bytes
        let minAppVersion: String?  // Minimum app version required
        let releaseNotes: String?
        let downloadURL: URL?

        static func current(for modelType: ModelType) -> ModelVersionInfo? {
            // Return info for bundled/downloaded models
            // In a real implementation, this would read from a manifest file
            ModelVersionInfo(
                version: "1.0.0",
                checksum: "",
                size: 0,
                minAppVersion: nil,
                releaseNotes: nil,
                downloadURL: nil
            )
        }
    }

    /// Model manifest from remote server
    struct ModelManifest: Codable, Sendable {
        let schemaVersion: Int
        let models: [String: ModelVersionInfo]
        let lastUpdated: Date
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

        var estimatedSizeBytes: Int64 {
            switch self {
            case .glyphDiffusion:
                return 150_000_000
            case .styleEncoder:
                return 50_000_000
            case .kerningNet:
                return 25_000_000
            }
        }

        var fileName: String {
            "\(rawValue).mlmodelc"
        }

        var versionFileName: String {
            "\(rawValue).version.json"
        }

        /// Capabilities provided by this model
        var capabilities: [String] {
            switch self {
            case .glyphDiffusion:
                return ["glyph_generation", "style_variation", "interpolation"]
            case .styleEncoder:
                return ["style_extraction", "style_embedding", "similarity_scoring"]
            case .kerningNet:
                return ["kerning_prediction", "optical_spacing"]
            }
        }
    }

    // MARK: - Published State

    @Published private(set) var glyphDiffusionStatus: ModelStatus = .notDownloaded
    @Published private(set) var styleEncoderStatus: ModelStatus = .notDownloaded
    @Published private(set) var kerningNetStatus: ModelStatus = .notDownloaded

    /// Versions of currently loaded/downloaded models
    @Published private(set) var modelVersions: [ModelType: ModelVersionInfo] = [:]

    /// Latest available versions from server (if checked)
    @Published private(set) var availableVersions: [ModelType: ModelVersionInfo] = [:]

    /// Combined download progress for all models (0-1)
    @Published private(set) var totalDownloadProgress: Double = 0

    /// Whether any model operation is in progress
    @Published private(set) var isPerformingOperation: Bool = false

    // MARK: - Private State

    private var glyphDiffusionModel: MLModel?
    private var styleEncoderModel: MLModel?
    private var kerningNetModel: MLModel?

    private let modelsDirectory: URL
    private var downloadTasks: [ModelType: Task<Void, Never>] = [:]
    private let downloadTimeout: TimeInterval = 300  // 5 minutes
    private let maxRetries = 3
    private var initializationError: Error?

    /// Base URL for model downloads (when available)
    private let modelServerBaseURL: URL? = nil  // Set when model hosting is available

    /// Cache for model validation results
    private var validationCache: [ModelType: Bool] = [:]

    // MARK: - Initialization

    private init() {
        // Set up models directory in Application Support
        let appSupportURL: URL
        if let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first {
            appSupportURL = appSupport
        } else {
            appSupportURL = FileManager.default.temporaryDirectory
            print("[ModelManager] WARNING: Application Support directory unavailable, falling back to temp directory")
        }
        modelsDirectory = appSupportURL.appendingPathComponent("Typogenesis/Models", isDirectory: true)

        // Create directory if needed - handle errors explicitly
        do {
            try FileManager.default.createDirectory(at: modelsDirectory, withIntermediateDirectories: true)
        } catch {
            // Log error and store for later - don't silently swallow
            logger.error("Failed to create models directory at \(self.modelsDirectory.path): \(error.localizedDescription)")
            initializationError = error
            // Set all statuses to error so UI can display the problem
            glyphDiffusionStatus = .error("Storage unavailable")
            styleEncoderStatus = .error("Storage unavailable")
            kerningNetStatus = .error("Storage unavailable")
            return
        }

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
    var areAllModelsReady: Bool {
        glyphDiffusionStatus == .loaded &&
        styleEncoderStatus == .loaded &&
        kerningNetStatus == .loaded
    }

    /// Check if any model is currently loading or downloading
    var isLoading: Bool {
        for modelType in ModelType.allCases {
            let status = status(for: modelType)
            switch status {
            case .downloading, .loading, .validating:
                return true
            default:
                continue
            }
        }
        return false
    }

    /// Check if any model has an update available
    var hasUpdatesAvailable: Bool {
        for modelType in ModelType.allCases {
            if case .updateAvailable = status(for: modelType) {
                return true
            }
        }
        return false
    }

    /// Get detailed status report for all models
    var detailedStatusReport: ModelStatusReport {
        ModelStatusReport(
            glyphDiffusion: ModelDetailedStatus(
                type: .glyphDiffusion,
                status: glyphDiffusionStatus,
                version: modelVersions[.glyphDiffusion],
                availableVersion: availableVersions[.glyphDiffusion],
                isModelLoaded: glyphDiffusionModel != nil
            ),
            styleEncoder: ModelDetailedStatus(
                type: .styleEncoder,
                status: styleEncoderStatus,
                version: modelVersions[.styleEncoder],
                availableVersion: availableVersions[.styleEncoder],
                isModelLoaded: styleEncoderModel != nil
            ),
            kerningNet: ModelDetailedStatus(
                type: .kerningNet,
                status: kerningNetStatus,
                version: modelVersions[.kerningNet],
                availableVersion: availableVersions[.kerningNet],
                isModelLoaded: kerningNetModel != nil
            )
        )
    }

    /// Detailed status for a single model
    struct ModelDetailedStatus: Sendable {
        let type: ModelType
        let status: ModelStatus
        let version: ModelVersionInfo?
        let availableVersion: ModelVersionInfo?
        let isModelLoaded: Bool

        var capabilities: [String] {
            isModelLoaded ? type.capabilities : []
        }

        var usingFallback: Bool {
            !isModelLoaded
        }
    }

    /// Complete status report for all models
    struct ModelStatusReport: Sendable {
        let glyphDiffusion: ModelDetailedStatus
        let styleEncoder: ModelDetailedStatus
        let kerningNet: ModelDetailedStatus

        var allModelsLoaded: Bool {
            glyphDiffusion.isModelLoaded &&
            styleEncoder.isModelLoaded &&
            kerningNet.isModelLoaded
        }

        var loadedCount: Int {
            [glyphDiffusion, styleEncoder, kerningNet]
                .filter { $0.isModelLoaded }
                .count
        }

        var totalCount: Int { 3 }

        /// All capabilities currently available (from loaded models)
        var availableCapabilities: Set<String> {
            var caps = Set<String>()
            if glyphDiffusion.isModelLoaded {
                caps.formUnion(glyphDiffusion.capabilities)
            }
            if styleEncoder.isModelLoaded {
                caps.formUnion(styleEncoder.capabilities)
            }
            if kerningNet.isModelLoaded {
                caps.formUnion(kerningNet.capabilities)
            }
            return caps
        }
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
        // Cancel any existing download for this model type
        downloadTasks[modelType]?.cancel()

        isPerformingOperation = true

        // Create and track the download task
        // No [weak self] needed: ModelManager.shared is a singleton that is never deallocated
        let downloadTask = Task { @MainActor in
            var lastError: Error?

            for attempt in 1...self.maxRetries {
                do {
                    try await self.performDownload(modelType, attempt: attempt)
                    // Success - validate and clean up
                    self.setStatus(.validating, for: modelType)
                    let isValid = await self.validateModel(modelType)
                    if isValid {
                        self.validationCache[modelType] = true
                        self.setStatus(.downloaded, for: modelType)
                        self.logger.info("Model \(modelType.displayName) downloaded and validated successfully")
                    } else {
                        self.setStatus(.error("Model validation failed - file may be corrupted"), for: modelType)
                        // Delete the corrupted file
                        self.deleteModel(modelType)
                    }
                    self.downloadTasks.removeValue(forKey: modelType)
                    self.isPerformingOperation = false
                    return
                } catch is CancellationError {
                    // Download was cancelled, don't retry
                    self.setStatus(.notDownloaded, for: modelType)
                    self.downloadTasks.removeValue(forKey: modelType)
                    self.isPerformingOperation = false
                    return
                } catch {
                    lastError = error
                    // Retry after a delay (except on last attempt)
                    if attempt < self.maxRetries {
                        do {
                            try await Task.sleep(nanoseconds: UInt64(attempt) * 1_000_000_000)
                        } catch {
                            // Sleep was interrupted (likely task cancellation) - propagate
                            self.logger.debug("Download retry sleep interrupted: \(error.localizedDescription)")
                            self.setStatus(.notDownloaded, for: modelType)
                            self.downloadTasks.removeValue(forKey: modelType)
                            self.isPerformingOperation = false
                            return
                        }
                    }
                }
            }

            // All retries failed
            self.setStatus(.error(lastError?.localizedDescription ?? "Download failed after \(self.maxRetries) attempts"), for: modelType)
            self.downloadTasks.removeValue(forKey: modelType)
            self.isPerformingOperation = false
        }

        downloadTasks[modelType] = downloadTask
        await downloadTask.value
    }

    /// Download all models that are not yet downloaded
    func downloadAllModels(onProgress: ((ModelType, Double) -> Void)? = nil) async {
        isPerformingOperation = true
        totalDownloadProgress = 0

        let modelsToDownload = ModelType.allCases.filter { !isModelDownloaded($0) }
        let totalModels = modelsToDownload.count
        guard totalModels > 0 else {
            isPerformingOperation = false
            return
        }

        for (index, modelType) in modelsToDownload.enumerated() {
            await downloadModel(modelType)
            let progress = Double(index + 1) / Double(totalModels)
            totalDownloadProgress = progress
            onProgress?(modelType, progress)
        }

        isPerformingOperation = false
    }

    /// Update a model to a new version
    func updateModel(_ modelType: ModelType) async throws {
        guard let newVersion = availableVersions[modelType] else {
            throw DownloadError.noUpdateAvailable
        }

        // Backup current model before updating
        let backupURL = modelsDirectory.appendingPathComponent("\(modelType.fileName).backup")
        let modelURL = modelsDirectory.appendingPathComponent(modelType.fileName)

        if FileManager.default.fileExists(atPath: modelURL.path) {
            do {
                // Remove old backup if exists
                if FileManager.default.fileExists(atPath: backupURL.path) {
                    try FileManager.default.removeItem(at: backupURL)
                }
                try FileManager.default.copyItem(at: modelURL, to: backupURL)
            } catch {
                logger.error("Could not backup model before update: \(error.localizedDescription)")
            }
        }

        // Unload current model
        unloadModel(modelType)

        // Download new version
        await downloadModel(modelType)

        // If download succeeded, update version info
        if isModelDownloaded(modelType) {
            modelVersions[modelType] = newVersion
            saveVersionInfo(newVersion, for: modelType)

            // Clean up backup
            try? FileManager.default.removeItem(at: backupURL)
        } else {
            // Restore from backup if update failed
            if FileManager.default.fileExists(atPath: backupURL.path) {
                try? FileManager.default.moveItem(at: backupURL, to: modelURL)
                setStatus(.downloaded, for: modelType)
            }
        }
    }

    /// Check for model updates from remote server
    func checkForUpdates() async {
        logger.info("Checking for model updates...")

        // HONEST STATUS: No model server exists yet
        // When a server is available, this will fetch the manifest and compare versions
        guard let _ = modelServerBaseURL else {
            logger.info("No model server configured - updates not available")
            return
        }

        // Future implementation would:
        // 1. Fetch manifest from server
        // 2. Compare versions with installed models
        // 3. Update availableVersions and status for models with updates
    }

    /// Internal download implementation
    ///
    /// **HONEST STATUS:** Downloads are not yet implemented. This function
    /// immediately returns an error explaining that model downloads are not
    /// available. When a model hosting server is set up, this will perform
    /// real downloads.
    private func performDownload(_ modelType: ModelType, attempt: Int) async throws {
        logger.info("Download attempt \(attempt) for \(modelType.displayName)")

        // Check if model server is configured
        guard let baseURL = modelServerBaseURL else {
            // HONEST: Downloads are not implemented yet
            setStatus(.error("Model downloads not yet available. AI features use geometric fallback."), for: modelType)
            throw DownloadError.notImplemented
        }

        // When server is available, this would:
        // 1. Build download URL
        let downloadURL = baseURL.appendingPathComponent(modelType.fileName)

        // 2. Create URLSession download task with progress
        setStatus(.downloading(progress: 0), for: modelType)

        let session = URLSession.shared
        let (tempURL, response) = try await session.download(from: downloadURL, delegate: nil)

        guard let httpResponse = response as? HTTPURLResponse,
              (200...299).contains(httpResponse.statusCode) else {
            throw DownloadError.serverUnavailable
        }

        // 3. Move downloaded file to models directory
        let destinationURL = modelsDirectory.appendingPathComponent(modelType.fileName)
        try FileManager.default.moveItem(at: tempURL, to: destinationURL)

        setStatus(.downloading(progress: 1.0), for: modelType)
    }

    /// Validate model integrity after download
    func validateModel(_ modelType: ModelType) async -> Bool {
        let modelURL = modelsDirectory.appendingPathComponent(modelType.fileName)

        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            logger.error("Validation failed: model file does not exist at \(modelURL.path)")
            return false
        }

        // Check file size
        do {
            let attributes = try FileManager.default.attributesOfItem(atPath: modelURL.path)
            let fileSize = attributes[.size] as? Int64 ?? 0

            if fileSize == 0 {
                logger.error("Validation failed: model file is empty")
                return false
            }

            // If we have expected version info with size, verify it
            if let versionInfo = modelVersions[modelType], versionInfo.size > 0 {
                if fileSize != versionInfo.size {
                    logger.error("Validation failed: file size mismatch (expected \(versionInfo.size), got \(fileSize))")
                    return false
                }
            }
        } catch {
            logger.error("Validation failed: could not read file attributes: \(error.localizedDescription)")
            return false
        }

        // Verify checksum if available
        if let versionInfo = modelVersions[modelType], !versionInfo.checksum.isEmpty {
            let computedChecksum = await computeChecksum(for: modelURL)
            if computedChecksum != versionInfo.checksum {
                logger.error("Validation failed: checksum mismatch")
                return false
            }
        }

        // Try to compile/load the model to verify it's valid
        do {
            let config = MLModelConfiguration()
            config.computeUnits = .cpuOnly  // Use CPU for validation to be faster
            _ = try await MLModel.load(contentsOf: modelURL, configuration: config)
            logger.info("Validation passed: model \(modelType.displayName) is valid CoreML model")
            return true
        } catch {
            logger.error("Validation failed: could not load model: \(error.localizedDescription)")
            return false
        }
    }

    /// Compute SHA256 checksum of a file using streaming reads.
    /// Uses a detached task to avoid blocking the MainActor with potentially large file reads.
    /// Reads in 64KB chunks to avoid loading entire model files into memory.
    private func computeChecksum(for url: URL) async -> String {
        let capturedURL = url
        return await Task.detached {
            do {
                let bufferSize = 65_536  // 64KB chunks
                let fileHandle = try FileHandle(forReadingFrom: capturedURL)
                defer { fileHandle.closeFile() }

                var hasher = SHA256()
                while true {
                    let chunk = fileHandle.readData(ofLength: bufferSize)
                    if chunk.isEmpty { break }
                    hasher.update(data: chunk)
                }
                let hash = hasher.finalize()
                return hash.compactMap { String(format: "%02x", $0) }.joined()
            } catch {
                print("[ModelManager] Could not compute checksum: \(error.localizedDescription)")
                return ""
            }
        }.value
    }

    /// Save version info for a model
    private func saveVersionInfo(_ info: ModelVersionInfo, for modelType: ModelType) {
        let versionURL = modelsDirectory.appendingPathComponent(modelType.versionFileName)
        do {
            let data = try JSONEncoder().encode(info)
            try data.write(to: versionURL)
        } catch {
            logger.error("Could not save version info: \(error.localizedDescription)")
        }
    }

    /// Load version info for a model
    private func loadVersionInfo(for modelType: ModelType) -> ModelVersionInfo? {
        let versionURL = modelsDirectory.appendingPathComponent(modelType.versionFileName)
        do {
            let data = try Data(contentsOf: versionURL)
            return try JSONDecoder().decode(ModelVersionInfo.self, from: data)
        } catch {
            return nil
        }
    }

    enum DownloadError: Error, LocalizedError {
        case notImplemented
        case serverUnavailable
        case networkError(String)
        case noUpdateAvailable
        case validationFailed
        case checksumMismatch

        var errorDescription: String? {
            switch self {
            case .notImplemented:
                return "Model downloads are not yet implemented. The app uses geometric placeholder generation instead."
            case .serverUnavailable:
                return "Model server is not available"
            case .networkError(let message):
                return "Network error: \(message)"
            case .noUpdateAvailable:
                return "No update available for this model"
            case .validationFailed:
                return "Downloaded model failed validation"
            case .checksumMismatch:
                return "Downloaded file checksum does not match expected value"
            }
        }
    }

    /// Cancel an in-progress download
    func cancelDownload(_ modelType: ModelType) {
        if let task = downloadTasks[modelType] {
            task.cancel()
            downloadTasks.removeValue(forKey: modelType)
        }
        setStatus(.notDownloaded, for: modelType)
    }

    /// Cancel all in-progress downloads
    func cancelAllDownloads() {
        for (modelType, task) in downloadTasks {
            task.cancel()
            setStatus(.notDownloaded, for: modelType)
        }
        downloadTasks.removeAll()
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
        do {
            try FileManager.default.removeItem(at: modelURL)
            setStatus(.notDownloaded, for: modelType)
        } catch let error as NSError {
            // File not found is expected if model was never downloaded
            if error.domain == NSCocoaErrorDomain && error.code == NSFileNoSuchFileError {
                setStatus(.notDownloaded, for: modelType)
            } else {
                // Actual error - log it and set error status
                logger.error("Failed to delete model \(modelType.displayName): \(error.localizedDescription)")
                setStatus(.error("Delete failed: \(error.localizedDescription)"), for: modelType)
            }
        }
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
                // Load version info if available
                if let versionInfo = loadVersionInfo(for: modelType) {
                    modelVersions[modelType] = versionInfo
                }
            }
        }
    }

    func isModelDownloaded(_ modelType: ModelType) -> Bool {
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

    /// Get the URL for a model's file
    func modelURL(for modelType: ModelType) -> URL {
        modelsDirectory.appendingPathComponent(modelType.fileName)
    }

    /// Get total storage used by downloaded models
    var totalStorageUsed: Int64 {
        var total: Int64 = 0
        for modelType in ModelType.allCases {
            let url = modelURL(for: modelType)
            if let attributes = try? FileManager.default.attributesOfItem(atPath: url.path),
               let size = attributes[.size] as? Int64 {
                total += size
            }
        }
        return total
    }

    /// Format bytes as human-readable string
    static func formatBytes(_ bytes: Int64) -> String {
        let formatter = ByteCountFormatter()
        formatter.countStyle = .file
        return formatter.string(fromByteCount: bytes)
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
            return "No models loaded (using fallback)"
        }
    }

    /// Total estimated size of all models
    nonisolated static var totalModelSize: String {
        "~225 MB"
    }

    /// Get a summary of fallback usage
    var fallbackSummary: String {
        let usingFallback = ModelType.allCases.filter { status(for: $0) != .loaded }

        if usingFallback.isEmpty {
            return "All features using AI models"
        } else if usingFallback.count == ModelType.allCases.count {
            return "All features using geometric fallback"
        } else {
            let names = usingFallback.map { $0.displayName }.joined(separator: ", ")
            return "Fallback active for: \(names)"
        }
    }

    /// Check if a specific capability is available from loaded models
    func isCapabilityAvailable(_ capability: String) -> Bool {
        detailedStatusReport.availableCapabilities.contains(capability)
    }

    /// Get which models provide a specific capability
    func modelsProviding(capability: String) -> [ModelType] {
        ModelType.allCases.filter { $0.capabilities.contains(capability) }
    }
}

// MARK: - Logging Support

extension ModelManager {
    /// Log current model status for debugging
    func logCurrentStatus() {
        logger.info("=== Model Status Report ===")
        for modelType in ModelType.allCases {
            let status = status(for: modelType)
            let version = modelVersions[modelType]?.version ?? "unknown"
            logger.info("\(modelType.displayName): \(status.displayText) (v\(version))")
        }
        logger.info("Storage used: \(Self.formatBytes(self.totalStorageUsed))")
        logger.info("\(self.fallbackSummary)")
        logger.info("===========================")
    }
}
