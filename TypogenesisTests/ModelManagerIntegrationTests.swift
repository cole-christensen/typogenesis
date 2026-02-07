import Testing
import Foundation
@testable import Typogenesis

@Suite("ModelManager Integration Tests")
@MainActor
struct ModelManagerIntegrationTests {

    // MARK: - Test Isolation

    /// Reset ModelManager singleton state before each test.
    /// Each @Test method creates a new struct instance, so this init
    /// ensures consistent starting state (no model loaded).
    init() {
        let manager = ModelManager.shared
        for modelType in ModelManager.ModelType.allCases {
            manager.unloadModel(modelType)
        }
    }

    // MARK: - Helper Functions

    /// Check if a status is an error status
    private func isErrorStatus(_ status: ModelManager.ModelStatus) -> Bool {
        if case .error = status {
            return true
        }
        return false
    }

    /// Check if a status is downloading
    private func isDownloadingStatus(_ status: ModelManager.ModelStatus) -> Bool {
        if case .downloading = status {
            return true
        }
        return false
    }

    // isValidProcessedStatus removed (B21): was dead code, never called, always returned true

    // MARK: - Model Discovery Tests

    @Test("ModelManager singleton is consistent")
    func testSingletonConsistent() {
        let manager1 = ModelManager.shared
        let manager2 = ModelManager.shared
        #expect(manager1 === manager2, "ModelManager.shared should return the same instance")
    }

    @Test("All model types are defined")
    func testAllModelTypesDefined() {
        let allTypes = ModelManager.ModelType.allCases
        #expect(allTypes.count == 3, "Should have 3 model types")
        #expect(allTypes.contains(.glyphDiffusion))
        #expect(allTypes.contains(.styleEncoder))
        #expect(allTypes.contains(.kerningNet))
    }

    @Test("Model types have display names")
    func testModelDisplayNames() {
        for modelType in ModelManager.ModelType.allCases {
            #expect(!modelType.displayName.isEmpty, "\(modelType) should have display name")
            #expect(!modelType.description.isEmpty, "\(modelType) should have description")
            #expect(!modelType.estimatedSize.isEmpty, "\(modelType) should have size estimate")
        }
    }

    @Test("Model types have valid file names")
    func testModelFileNames() {
        for modelType in ModelManager.ModelType.allCases {
            #expect(modelType.fileName.hasSuffix(".mlmodelc"), "Model file should have .mlmodelc extension")
            #expect(modelType.fileName.contains(modelType.rawValue), "File name should contain model type")
        }
    }

    // MARK: - Model Status Tests

    @Test("Initial status is notDownloaded without models")
    func testInitialStatus() {
        let manager = ModelManager.shared

        for modelType in ModelManager.ModelType.allCases {
            let status = manager.status(for: modelType)

            // Without loaded models, initial status should be notDownloaded
            #expect(status == .notDownloaded, "Initial status without models should be notDownloaded, got \(status) for \(modelType)")
        }
    }

    @Test("Status has display text")
    func testStatusDisplayText() {
        let statuses: [ModelManager.ModelStatus] = [
            .notDownloaded,
            .downloading(progress: 0.5),
            .downloaded,
            .loading,
            .loaded,
            .error("Test error")
        ]

        for status in statuses {
            #expect(!status.displayText.isEmpty, "Status \(status) should have display text")
        }
    }

    @Test("isAvailable returns correct value")
    func testStatusIsAvailable() {
        #expect(ModelManager.ModelStatus.loaded.isAvailable)
        #expect(!ModelManager.ModelStatus.notDownloaded.isAvailable)
        #expect(!ModelManager.ModelStatus.downloading(progress: 0.5).isAvailable)
        #expect(!ModelManager.ModelStatus.downloaded.isAvailable)
        #expect(!ModelManager.ModelStatus.loading.isAvailable)
        #expect(!ModelManager.ModelStatus.error("test").isAvailable)
    }

    @Test("Downloading progress is clamped")
    func testDownloadingProgressClamped() {
        // Test various progress values
        let status1 = ModelManager.ModelStatus.downloading(progress: 0.5)
        #expect(status1.displayText.contains("50%"))

        let status2 = ModelManager.ModelStatus.downloading(progress: 1.5)
        #expect(status2.displayText.contains("100%"), "Progress > 1 should be clamped to 100%")

        let status3 = ModelManager.ModelStatus.downloading(progress: -0.5)
        #expect(status3.displayText.contains("0%"), "Negative progress should be clamped to 0%")

        let status4 = ModelManager.ModelStatus.downloading(progress: Double.nan)
        #expect(status4.displayText.contains("0%"), "NaN progress should default to 0%")
    }

    // MARK: - Model Loading Tests

    @Test("Load model sets status correctly")
    func testLoadModelStatus() async {
        let manager = ModelManager.shared

        // Since models aren't downloaded, loading should set notDownloaded
        await manager.loadModel(.glyphDiffusion)

        let status = manager.status(for: .glyphDiffusion)
        // Should either be notDownloaded (file doesn't exist) or error (load failed)
        let isValidStatus = status == .notDownloaded || isErrorStatus(status)
        #expect(isValidStatus, "Status after loading non-existent model should be notDownloaded or error")
    }

    @Test("Load all models without model files stays notDownloaded or errors")
    func testLoadAllModels() async {
        let manager = ModelManager.shared

        await manager.loadAllModels()

        // Without model files, loadAllModels should leave status at notDownloaded or error
        let validPostLoad: [ModelManager.ModelStatus] = [.notDownloaded]
        for modelType in ModelManager.ModelType.allCases {
            let status = manager.status(for: modelType)
            #expect(validPostLoad.contains(status) || status.displayText.contains("Error"),
                    "After loadAllModels without model files, status should be notDownloaded or error, got \(status) for \(modelType)")
        }
    }

    // MARK: - Model Access Tests

    @Test("Model accessors return nil when not loaded")
    func testModelAccessorsReturnNil() {
        let manager = ModelManager.shared

        // Without loaded models (no .mlmodelc files in test environment),
        // all model accessors should return nil.
        // We verify the status first to ensure the precondition holds.
        #expect(manager.status(for: .glyphDiffusion) != .loaded,
                "Test environment should not have loaded glyphDiffusion model")
        #expect(manager.glyphDiffusion == nil,
                "glyphDiffusion accessor should return nil when model is not loaded")

        #expect(manager.status(for: .styleEncoder) != .loaded,
                "Test environment should not have loaded styleEncoder model")
        #expect(manager.styleEncoder == nil,
                "styleEncoder accessor should return nil when model is not loaded")

        #expect(manager.status(for: .kerningNet) != .loaded,
                "Test environment should not have loaded kerningNet model")
        #expect(manager.kerningNet == nil,
                "kerningNet accessor should return nil when model is not loaded")
    }

    // MARK: - Download Tests

    @Test("Download sets error status (not implemented)")
    func testDownloadSetsError() async {
        let manager = ModelManager.shared

        // Downloads are not implemented, should set error
        await manager.downloadModel(.styleEncoder)

        let status = manager.status(for: .styleEncoder)
        if case .error(let message) = status {
            #expect(!message.isEmpty, "Error status should have a descriptive message")
        } else {
            // May also be notDownloaded if download was cancelled
            #expect(status == .notDownloaded)
        }
    }

    @Test("Cancel download resets status")
    func testCancelDownload() async throws {
        let manager = ModelManager.shared

        // Start a download
        Task {
            await manager.downloadModel(.kerningNet)
        }

        // Give the download task time to start
        try await Task.sleep(for: .milliseconds(100))

        // Cancel it
        manager.cancelDownload(.kerningNet)

        // Give time for state to settle
        try await Task.sleep(for: .milliseconds(100))

        // Status should be notDownloaded or error - not stuck in downloading
        let status = manager.status(for: .kerningNet)
        #expect(status == .notDownloaded || isErrorStatus(status),
                "Status should be notDownloaded or error after cancel, got \(status)")
    }

    @Test("Cancel all downloads works")
    func testCancelAllDownloads() async throws {
        let manager = ModelManager.shared

        // Start downloads
        for modelType in ModelManager.ModelType.allCases {
            Task {
                await manager.downloadModel(modelType)
            }
        }

        // Give download tasks time to start
        try await Task.sleep(for: .milliseconds(100))

        // Cancel all
        manager.cancelAllDownloads()

        // Give time for state to settle
        try await Task.sleep(for: .milliseconds(100))

        // All should be cancelled - not stuck in downloading
        #expect(!manager.isLoading, "Should not be loading after cancel all")
        for modelType in ModelManager.ModelType.allCases {
            let status = manager.status(for: modelType)
            #expect(status == .notDownloaded || isErrorStatus(status),
                    "Status should be notDownloaded or error after cancel all, got \(status) for \(modelType)")
        }
    }

    // MARK: - Unload Tests

    @Test("Unload model updates status")
    func testUnloadModel() {
        let manager = ModelManager.shared

        manager.unloadModel(.glyphDiffusion)

        let status = manager.status(for: .glyphDiffusion)
        #expect(status == .notDownloaded || status == .downloaded,
                "Unloaded model should be notDownloaded or downloaded")
    }

    // MARK: - Delete Tests

    @Test("Delete model updates status")
    func testDeleteModel() {
        let manager = ModelManager.shared

        manager.deleteModel(.styleEncoder)

        let status = manager.status(for: .styleEncoder)
        // Should be notDownloaded (or error if delete failed for other reason)
        let isValidStatus = status == .notDownloaded || isErrorStatus(status)
        #expect(isValidStatus, "Status should be notDownloaded or error")
    }

    // MARK: - Aggregate Status Tests

    @Test("areAllModelsReady returns false when not loaded")
    func testAreAllModelsReady() {
        let manager = ModelManager.shared

        // Without loaded models (no .mlmodelc files exist), should be false
        #expect(!manager.areAllModelsReady,
                "areAllModelsReady should be false when models are not loaded")
    }

    @Test("isLoading reflects current state")
    func testIsLoading() {
        let manager = ModelManager.shared

        // Before any async operations, should not be loading
        #expect(!manager.isLoading, "Should not be loading in idle state")
    }

    @Test("statusSummary returns valid string")
    func testStatusSummary() {
        let manager = ModelManager.shared

        let summary = manager.statusSummary
        #expect(!summary.isEmpty, "Status summary should not be empty")
        #expect(summary.contains("model"), "Summary should mention models")
    }

    @Test("totalModelSize is defined")
    func testTotalModelSize() {
        let size = ModelManager.totalModelSize
        #expect(!size.isEmpty)
        #expect(size.contains("MB"), "Size should be in MB")
    }

    // MARK: - Error Handling Tests

    @Test("Error status contains message")
    func testErrorStatusMessage() {
        let errorStatus = ModelManager.ModelStatus.error("Test error message")

        #expect(errorStatus.displayText.contains("Error"))
        #expect(errorStatus.displayText.contains("Test error message"))
    }

    @Test("Model status equality works")
    func testModelStatusEquality() {
        #expect(ModelManager.ModelStatus.loaded == ModelManager.ModelStatus.loaded)
        #expect(ModelManager.ModelStatus.notDownloaded == ModelManager.ModelStatus.notDownloaded)
        #expect(ModelManager.ModelStatus.loaded != ModelManager.ModelStatus.notDownloaded)

        // Downloading with same progress
        #expect(ModelManager.ModelStatus.downloading(progress: 0.5) ==
                ModelManager.ModelStatus.downloading(progress: 0.5))
    }

    // MARK: - Concurrent Access Tests

    @Test("Concurrent status access is safe")
    func testConcurrentAccess() async {
        let manager = ModelManager.shared

        // Access status from multiple concurrent tasks
        await withTaskGroup(of: Void.self) { group in
            for _ in 0..<10 {
                group.addTask {
                    for modelType in ModelManager.ModelType.allCases {
                        _ = await MainActor.run {
                            manager.status(for: modelType)
                        }
                    }
                }
            }
        }

        // Should complete without crash and return valid statuses
        for modelType in ModelManager.ModelType.allCases {
            let status = manager.status(for: modelType)
            #expect(!status.displayText.isEmpty, "Status should have display text after concurrent access")
        }
    }
}
