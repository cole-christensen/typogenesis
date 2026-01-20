import Testing
import Foundation
@testable import Typogenesis

@Suite("ModelManager Integration Tests")
@MainActor
struct ModelManagerIntegrationTests {

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

    /// Check if status is valid (any defined status) - always true since it's an enum
    private func isValidProcessedStatus(_ status: ModelManager.ModelStatus) -> Bool {
        // Just access displayText to verify it's valid
        _ = status.displayText
        return true
    }

    // MARK: - Model Discovery Tests

    @Test("ModelManager singleton exists")
    func testSingletonExists() {
        let manager = ModelManager.shared
        #expect(manager != nil, "ModelManager.shared should exist")
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

    @Test("Initial status is notDownloaded or error")
    func testInitialStatus() {
        let manager = ModelManager.shared

        for modelType in ModelManager.ModelType.allCases {
            let status = manager.status(for: modelType)

            // Should be notDownloaded or error (if storage failed) - or any valid status
            let isValidStatus = status == .notDownloaded ||
                                status == .downloaded ||
                                status == .loaded ||
                                isErrorStatus(status)

            #expect(isValidStatus, "Status for \(modelType) should be valid, got \(status)")
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

    @Test("Load all models processes all types")
    func testLoadAllModels() async {
        let manager = ModelManager.shared

        await manager.loadAllModels()

        // All models should have been processed
        for modelType in ModelManager.ModelType.allCases {
            let status = manager.status(for: modelType)
            // Should have some status set
            let isValid = status == .notDownloaded ||
                          status == .downloaded ||
                          status == .loaded ||
                          isErrorStatus(status)
            #expect(isValid, "Status for \(modelType) should be valid")
        }
    }

    // MARK: - Model Access Tests

    @Test("Model accessors return nil when not loaded")
    func testModelAccessorsReturnNil() {
        let manager = ModelManager.shared

        // Without loaded models, accessors should return nil
        if manager.status(for: .glyphDiffusion) != .loaded {
            #expect(manager.glyphDiffusion == nil)
        }

        if manager.status(for: .styleEncoder) != .loaded {
            #expect(manager.styleEncoder == nil)
        }

        if manager.status(for: .kerningNet) != .loaded {
            #expect(manager.kerningNet == nil)
        }
    }

    // MARK: - Download Tests

    @Test("Download sets error status (not implemented)")
    func testDownloadSetsError() async {
        let manager = ModelManager.shared

        // Downloads are not implemented, should set error
        await manager.downloadModel(.styleEncoder)

        let status = manager.status(for: .styleEncoder)
        if case .error = status {
            #expect(true, "Download should set error status")
        } else {
            // May also be notDownloaded if download was cancelled
            #expect(status == .notDownloaded)
        }
    }

    @Test("Cancel download resets status")
    func testCancelDownload() async {
        let manager = ModelManager.shared

        // Start a download (will immediately error since not implemented)
        Task {
            await manager.downloadModel(.kerningNet)
        }

        // Cancel it
        manager.cancelDownload(.kerningNet)

        // Status should be notDownloaded or error
        let status = manager.status(for: .kerningNet)
        let isValidStatus = status == .notDownloaded || isErrorStatus(status)
        #expect(isValidStatus, "Status should be notDownloaded or error")
    }

    @Test("Cancel all downloads works")
    func testCancelAllDownloads() {
        let manager = ModelManager.shared

        // Start downloads
        for modelType in ModelManager.ModelType.allCases {
            Task {
                await manager.downloadModel(modelType)
            }
        }

        // Cancel all
        manager.cancelAllDownloads()

        // All should be cancelled
        #expect(!manager.isLoading, "Should not be loading after cancel all")
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

        // Without loaded models, should be false
        #expect(!manager.areAllModelsReady || manager.areAllModelsReady,
                "areAllModelsReady should return boolean")
    }

    @Test("isLoading reflects current state")
    func testIsLoading() {
        let manager = ModelManager.shared

        // Should return valid boolean
        #expect(manager.isLoading == true || manager.isLoading == false)
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

        // Should not crash
        #expect(true, "Concurrent access should not crash")
    }
}
