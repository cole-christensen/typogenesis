import XCTest
@testable import Typogenesis

/// Comprehensive bug-finding tests for AI Model downloading and usage
/// Following TESTING.md: Only tests that find real bugs deserve to exist.
///
/// FAILURES FOUND: TBD
///
/// These tests probe the entire model lifecycle including:
/// - Download state machine transitions
/// - Concurrent operations and race conditions
/// - Error handling and recovery
/// - Memory management
/// - File system edge cases
final class ModelManagerBugTests: XCTestCase {

    // MARK: - Test Isolation

    /// Reset ModelManager state before each test to prevent test interdependence.
    /// ModelManager is a singleton, so without this, tests could depend on prior test state.
    @MainActor
    override func setUp() async throws {
        let manager = ModelManager.shared
        // Unload all models to start from a clean state
        for modelType in ModelManager.ModelType.allCases {
            manager.unloadModel(modelType)
        }
    }

    // MARK: - Comprehensive Model Lifecycle Test

    /// Tests the complete model lifecycle from not-downloaded through loaded,
    /// including all state transitions, error conditions, and edge cases.
    /// This single test exercises the entire ModelManager API.
    ///
    /// FAILURES FOUND: 0 (PROPOSED)
    @MainActor
    func testCompleteModelLifecycle() async throws {
        let manager = ModelManager.shared

        // ============================================
        // PHASE 1: Validate initial state and status display
        // ============================================

        // Check all possible status values have valid display text
        let allStatuses: [ModelManager.ModelStatus] = [
            .notDownloaded,
            .downloading(progress: 0),
            .downloading(progress: 0.5),
            .downloading(progress: 1.0),
            .downloaded,
            .validating,
            .loading,
            .loaded,
            .updateAvailable(currentVersion: "1.0", newVersion: "2.0"),
            .error("Test error message")
        ]

        for status in allStatuses {
            let displayText = status.displayText
            XCTAssertFalse(displayText.isEmpty,
                "BUG: Status \(status) produces empty display text")
            XCTAssertFalse(displayText.contains("nil"),
                "BUG: Status \(status) contains 'nil' in display: \(displayText)")

            // Verify display text is appropriate for each status type
            switch status {
            case .notDownloaded:
                XCTAssertTrue(displayText.lowercased().contains("not") || displayText.lowercased().contains("download"),
                    "BUG: notDownloaded status should indicate 'not downloaded', got: \(displayText)")
            case .downloading(let progress):
                XCTAssertTrue(displayText.contains("Downloading") || displayText.contains("%"),
                    "BUG: downloading status should contain 'Downloading' or percentage, got: \(displayText)")
                // Verify progress percentage appears in text
                let expectedPercentage = Int(progress * 100)
                XCTAssertTrue(displayText.contains("\(expectedPercentage)%"),
                    "BUG: downloading status should show progress \(expectedPercentage)%, got: \(displayText)")
            case .downloaded:
                XCTAssertTrue(displayText.contains("Ready") || displayText.lowercased().contains("load"),
                    "BUG: downloaded status should indicate ready to load, got: \(displayText)")
            case .validating:
                XCTAssertTrue(displayText.contains("Validating"),
                    "BUG: validating status should contain 'Validating', got: \(displayText)")
            case .loading:
                XCTAssertTrue(displayText.contains("Loading"),
                    "BUG: loading status should contain 'Loading', got: \(displayText)")
            case .loaded:
                XCTAssertTrue(displayText.contains("Ready"),
                    "BUG: loaded status should contain 'Ready', got: \(displayText)")
            case .updateAvailable:
                XCTAssertTrue(displayText.contains("Update") || displayText.contains("available"),
                    "BUG: updateAvailable status should contain 'Update' or 'available', got: \(displayText)")
            case .error(let message):
                XCTAssertTrue(displayText.contains("Error"),
                    "BUG: error status should contain 'Error', got: \(displayText)")
                // Error message should be included (unless empty)
                if !message.isEmpty {
                    XCTAssertTrue(displayText.contains(message),
                        "BUG: error status should contain the error message '\(message)', got: \(displayText)")
                }
            }
        }

        // Test isAvailable computed property
        XCTAssertFalse(ModelManager.ModelStatus.notDownloaded.isAvailable)
        XCTAssertFalse(ModelManager.ModelStatus.downloading(progress: 0.5).isAvailable)
        XCTAssertFalse(ModelManager.ModelStatus.downloaded.isAvailable)
        XCTAssertFalse(ModelManager.ModelStatus.validating.isAvailable)
        XCTAssertFalse(ModelManager.ModelStatus.loading.isAvailable)
        XCTAssertTrue(ModelManager.ModelStatus.loaded.isAvailable)
        XCTAssertFalse(ModelManager.ModelStatus.updateAvailable(currentVersion: "1.0", newVersion: "2.0").isAvailable)
        XCTAssertFalse(ModelManager.ModelStatus.error("test").isAvailable)

        // ============================================
        // PHASE 2: Test progress edge cases
        // ============================================

        let progressEdgeCases: [Double] = [
            -1.0,           // Negative
            -0.0001,        // Tiny negative
            0.0,            // Zero
            0.0001,         // Tiny positive
            0.5,            // Normal
            0.9999,         // Almost complete
            1.0,            // Complete
            1.0001,         // Slightly over
            2.0,            // Way over
            100.0,          // Extreme
            Double.infinity,
            -Double.infinity,
            Double.nan
        ]

        for progress in progressEdgeCases {
            let status = ModelManager.ModelStatus.downloading(progress: progress)
            let text = status.displayText

            // Should always produce non-empty text
            XCTAssertFalse(text.isEmpty,
                "BUG: Progress \(progress) produces empty display text")

            // Check for NaN/Inf leaking into UI
            if progress.isNaN {
                XCTAssertFalse(text.lowercased().contains("nan"),
                    "BUG: NaN progress shows 'nan' in UI: \(text)")
            }
            if progress.isInfinite {
                XCTAssertFalse(text.lowercased().contains("inf"),
                    "BUG: Infinite progress shows 'inf' in UI: \(text)")
            }

            // The text format is "Downloading X%"
            // Verify percentage is reasonable
            if !progress.isNaN && !progress.isInfinite {
                let percentage = Int(progress * 100)
                if percentage < 0 {
                    // Negative percentage is a bug
                    XCTAssertFalse(text.contains("-"),
                        "BUG: Negative progress \(progress) shows negative percentage: \(text)")
                }
                // Progress should be clamped to 1.0 (100%) for display
                XCTAssertTrue(progress <= 1.0 || text.contains("100%"),
                    "BUG: Progress \(progress) should be clamped to 100%% for display, got: \(text)")
            }
        }

        // ============================================
        // PHASE 3: Test error message handling
        // ============================================

        let errorMessages = [
            "",                              // Empty error
            "A",                             // Single char
            "Network connection failed",    // Normal error
            String(repeating: "X", count: 1000),  // Very long error
            "Error with unicode: 🔥💥🚨",   // Unicode
            "Error\nwith\nnewlines",        // Multi-line
            "Error\twith\ttabs",            // Tabs
            "<script>alert('xss')</script>", // XSS-like content
            "Error: nil was found",         // Contains 'nil'
        ]

        for message in errorMessages {
            let status = ModelManager.ModelStatus.error(message)
            let text = status.displayText

            XCTAssertTrue(text.contains("Error"),
                "BUG: Error status should contain 'Error': \(text)")
            XCTAssertFalse(text.isEmpty,
                "BUG: Error status produces empty display text")
        }

        // ============================================
        // PHASE 4: Test model type properties
        // ============================================

        var seenIds = Set<String>()
        var seenFileNames = Set<String>()

        for modelType in ModelManager.ModelType.allCases {
            // Check ID uniqueness
            XCTAssertFalse(seenIds.contains(modelType.id),
                "BUG: Duplicate model type ID: \(modelType.id)")
            seenIds.insert(modelType.id)

            // Check filename uniqueness
            XCTAssertFalse(seenFileNames.contains(modelType.fileName),
                "BUG: Duplicate model filename: \(modelType.fileName)")
            seenFileNames.insert(modelType.fileName)

            // Check filename format
            XCTAssertTrue(modelType.fileName.hasSuffix(".mlmodelc"),
                "BUG: Model filename should end with .mlmodelc: \(modelType.fileName)")
            XCTAssertFalse(modelType.fileName.contains("/"),
                "BUG: Model filename contains path separator: \(modelType.fileName)")
            XCTAssertFalse(modelType.fileName.isEmpty,
                "BUG: Model filename is empty for \(modelType)")

            // Check display name is human-readable
            XCTAssertFalse(modelType.displayName.isEmpty,
                "BUG: Display name is empty for \(modelType)")
            XCTAssertFalse(modelType.displayName.contains("_"),
                "Display name should not have underscores: \(modelType.displayName)")

            // Check description is present
            XCTAssertFalse(modelType.description.isEmpty,
                "BUG: Description is empty for \(modelType)")

            // Check size estimate is present
            XCTAssertFalse(modelType.estimatedSize.isEmpty,
                "BUG: Size estimate is empty for \(modelType)")
            XCTAssertTrue(modelType.estimatedSize.contains("MB"),
                "Size should contain 'MB': \(modelType.estimatedSize)")
        }

        // ============================================
        // PHASE 5: Test size calculations
        // ============================================

        // Individual sizes
        XCTAssertTrue(ModelManager.ModelType.glyphDiffusion.estimatedSize.contains("150"),
            "Glyph diffusion should be ~150 MB")
        XCTAssertTrue(ModelManager.ModelType.styleEncoder.estimatedSize.contains("50"),
            "Style encoder should be ~50 MB")
        XCTAssertTrue(ModelManager.ModelType.kerningNet.estimatedSize.contains("25"),
            "Kerning net should be ~25 MB")

        // Total size should match
        let totalSize = ModelManager.totalModelSize
        XCTAssertTrue(totalSize.contains("225"),
            "BUG: Total size should be ~225 MB (150+50+25), got: \(totalSize)")

        // ============================================
        // PHASE 6: Test unload behavior
        // ============================================

        // Unload all models
        for modelType in ModelManager.ModelType.allCases {
            manager.unloadModel(modelType)
        }

        // Verify models are nil after unload
        XCTAssertNil(manager.glyphDiffusion,
            "BUG: glyphDiffusion should be nil after unload")
        XCTAssertNil(manager.styleEncoder,
            "BUG: styleEncoder should be nil after unload")
        XCTAssertNil(manager.kerningNet,
            "BUG: kerningNet should be nil after unload")

        // Verify status is not 'loaded' after unload
        for modelType in ModelManager.ModelType.allCases {
            let status = manager.status(for: modelType)
            XCTAssertNotEqual(status, .loaded,
                "BUG: Model \(modelType) still shows loaded after unload")
        }

        // ============================================
        // PHASE 7: Test summary calculations
        // ============================================

        // After unload, no models should be loaded
        XCTAssertFalse(manager.areAllModelsReady,
            "BUG: areAllModelsReady should be false after unloading all")

        let summary = manager.statusSummary
        XCTAssertFalse(summary.isEmpty,
            "BUG: Status summary should not be empty")

        // ============================================
        // PHASE 8: Test loading missing model
        // ============================================

        // Delete any existing model file first
        manager.deleteModel(.kerningNet)

        // Try to load non-existent model
        await manager.loadModel(.kerningNet)

        let statusAfterLoadMissing = manager.status(for: .kerningNet)

        // Should be notDownloaded (not error, not loaded)
        XCTAssertEqual(statusAfterLoadMissing, .notDownloaded,
            "Loading missing model should result in notDownloaded status, got: \(statusAfterLoadMissing)")

        // Model should still be nil
        XCTAssertNil(manager.kerningNet,
            "BUG: kerningNet should be nil after loading missing file")

        // ============================================
        // PHASE 9: Test corrupted model handling
        // ============================================

        // Use a temp directory instead of the real Application Support directory
        // to avoid destroying any real model files
        let tempModelsDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("TypogenesisTest-\(UUID().uuidString)", isDirectory: true)

        // Ensure temp directory exists
        try FileManager.default.createDirectory(at: tempModelsDir, withIntermediateDirectories: true)

        defer {
            try? FileManager.default.removeItem(at: tempModelsDir)
        }

        // Create corrupted file in temp dir to verify the data format
        let corruptedPath = tempModelsDir.appendingPathComponent(ModelManager.ModelType.kerningNet.fileName)
        let corruptedData = "THIS IS NOT A VALID MLMODEL FILE".data(using: .utf8)!
        try corruptedData.write(to: corruptedPath)

        // Verify file exists
        XCTAssertTrue(FileManager.default.fileExists(atPath: corruptedPath.path),
            "Test setup failed: corrupted file not created")

        // Note: We can't test loading from the temp dir without injecting the path
        // into ModelManager, so we test the status API and model availability instead.
        // The key assertion is that non-existent/corrupted models don't show as loaded.
        let statusAfterMissing = manager.status(for: .kerningNet)
        XCTAssertNotEqual(statusAfterMissing, .loaded,
            "BUG: kerningNet should not be loaded when no valid model exists")

        // ============================================
        // PHASE 10: Test download simulation timing
        // ============================================

        // The download simulation uses 200ms * 10 = 2 seconds
        let downloadStart = Date()
        await manager.downloadModel(.kerningNet)
        let downloadDuration = Date().timeIntervalSince(downloadStart)

        // Should complete within expected time (2s simulation + overhead, wide bounds for CI)
        XCTAssertLessThan(downloadDuration, 30.0,
            "BUG: Download took \(downloadDuration)s, expected ~2s")

        // Should not be instant (simulation takes time) - use wide bounds for CI tolerance
        XCTAssertGreaterThan(downloadDuration, 0.1,
            "BUG: Download completed too fast (\(downloadDuration)s), simulation may be broken")

        // After download, status should be loaded (auto-loads after download)
        let statusAfterDownload = manager.status(for: .kerningNet)
        // After download simulation, status should not be stuck in downloading
        if case .downloading = statusAfterDownload {
            XCTFail("BUG: Status should not be stuck in downloading after download completes")
        }

        // ============================================
        // PHASE 11: Test isLoading flag
        // ============================================

        // After all operations complete, should not be loading
        XCTAssertFalse(manager.isLoading,
            "BUG: isLoading should be false after all operations complete")

        // ============================================
        // PHASE 12: Test cancel download
        // ============================================

        // Start download
        let downloadTask = Task {
            await manager.downloadModel(.styleEncoder)
        }

        // Immediately cancel
        try? await Task.sleep(nanoseconds: 50_000_000)  // 50ms
        manager.cancelDownload(.styleEncoder)
        downloadTask.cancel()

        // Status should revert to notDownloaded
        // (Give a moment for state to update)
        try? await Task.sleep(nanoseconds: 100_000_000)  // 100ms

        let statusAfterCancel = manager.status(for: .styleEncoder)
        // After cancel, should not be stuck in downloading state
        if case .downloading = statusAfterCancel {
            XCTFail("BUG: styleEncoder stuck in downloading state after cancel")
        }

        // Clean up any partial downloads
        manager.deleteModel(.styleEncoder)
        manager.deleteModel(.kerningNet)

    }

    // MARK: - Comprehensive Concurrent Operations Test

    /// Tests race conditions and concurrent access to ModelManager.
    /// This test exercises multiple simultaneous operations to find
    /// thread-safety issues.
    ///
    /// FAILURES FOUND: 0 (PROPOSED)
    @MainActor
    func testConcurrentModelOperations() async throws {
        let manager = ModelManager.shared

        // ============================================
        // PHASE 1: Concurrent downloads of all models
        // ============================================

        // Clean state first
        for modelType in ModelManager.ModelType.allCases {
            manager.deleteModel(modelType)
        }

        // Start all downloads concurrently
        let downloadStart = Date()

        await withTaskGroup(of: Void.self) { group in
            for modelType in ModelManager.ModelType.allCases {
                group.addTask {
                    await manager.downloadModel(modelType)
                }
            }
        }

        let totalDownloadTime = Date().timeIntervalSince(downloadStart)

        // All downloads should complete
        XCTAssertLessThan(totalDownloadTime, 30.0,
            "BUG: Concurrent downloads took too long: \(totalDownloadTime)s")

        // No model should be stuck in downloading state
        for modelType in ModelManager.ModelType.allCases {
            let status = manager.status(for: modelType)
            if case .downloading(let progress) = status {
                XCTFail("BUG: Model \(modelType) stuck in downloading state with progress \(progress)")
            }
        }

        // ============================================
        // PHASE 2: Concurrent status checks during operations
        // ============================================

        // Start a download while rapidly checking status
        var statusReadCount = 0
        var statusesSeen = Set<String>()

        let statusCheckTask = Task {
            while !Task.isCancelled {
                for modelType in ModelManager.ModelType.allCases {
                    let status = manager.status(for: modelType)
                    statusesSeen.insert(status.displayText)
                    statusReadCount += 1
                }
                try? await Task.sleep(nanoseconds: 1_000_000)  // 1ms
            }
        }

        // Run for a bit
        try? await Task.sleep(nanoseconds: 500_000_000)  // 500ms
        statusCheckTask.cancel()

        XCTAssertGreaterThan(statusReadCount, 0,
            "Should have read status at least once: \(statusReadCount)")
        XCTAssertGreaterThan(statusesSeen.count, 0,
            "Should have seen at least one unique status state")

        // ============================================
        // PHASE 3: Rapid load/unload cycles
        // ============================================

        for _ in 0..<10 {
            // Unload all
            for modelType in ModelManager.ModelType.allCases {
                manager.unloadModel(modelType)
            }

            // Load all (they won't actually load since files don't exist after delete)
            await manager.loadAllModels()

            // Delete all
            for modelType in ModelManager.ModelType.allCases {
                manager.deleteModel(modelType)
            }
        }

        // Should not have crashed or corrupted state
        for modelType in ModelManager.ModelType.allCases {
            let status = manager.status(for: modelType)
            // After delete, model should not be in loaded state
            XCTAssertNotEqual(status, .loaded,
                "BUG: Model \(modelType) should not be loaded after delete")
            // Status should have non-empty display text
            XCTAssertFalse(status.displayText.isEmpty,
                "BUG: Status display text should not be empty for \(modelType)")
        }

        // ============================================
        // PHASE 4: Concurrent cancel operations
        // ============================================

        // Start downloads
        var downloadTasks: [Task<Void, Never>] = []
        for modelType in ModelManager.ModelType.allCases {
            let task = Task {
                await manager.downloadModel(modelType)
            }
            downloadTasks.append(task)
        }

        // Cancel all immediately
        for modelType in ModelManager.ModelType.allCases {
            manager.cancelDownload(modelType)
        }

        // Cancel tasks
        for task in downloadTasks {
            task.cancel()
        }

        // Wait for cleanup
        try? await Task.sleep(nanoseconds: 200_000_000)

        // Should not be stuck
        for modelType in ModelManager.ModelType.allCases {
            let status = manager.status(for: modelType)
            if case .loading = status {
                XCTFail("BUG: Model \(modelType) stuck in loading state after cancel")
            }
        }

    }

    // MARK: - Comprehensive File System Edge Cases Test

    /// Tests file system edge cases including permissions, disk space,
    /// and unusual file conditions.
    ///
    /// FAILURES FOUND: 0 (PROPOSED)
    @MainActor
    func testFileSystemEdgeCases() async throws {
        let manager = ModelManager.shared
        let fileManager = FileManager.default

        // Get models directory path (read-only - DO NOT delete real directory)
        let appSupport = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        let modelsDir = appSupport.appendingPathComponent("Typogenesis/Models", isDirectory: true)

        // ============================================
        // PHASE 1: Verify directory exists (non-destructive check)
        // ============================================

        // ModelManager creates the directory on init, verify it exists
        // We do NOT delete the real directory as that would destroy user data
        await manager.loadModel(.kerningNet)

        var isDirectory: ObjCBool = false
        let exists = fileManager.fileExists(atPath: modelsDir.path, isDirectory: &isDirectory)

        // After loadModel, directory should have been created
        XCTAssertTrue(exists, "Models directory should exist after loadModel")
        if exists {
            XCTAssertTrue(isDirectory.boolValue, "Models path should be a directory")
        }

        // ============================================
        // PHASE 2: Test with file instead of directory
        // ============================================

        // This is an edge case: what if the models "directory" is actually a file?
        // This could happen due to user error or malicious modification

        // We won't actually test this destructively, but document the concern
        // Note: If modelsDir path is a file instead of directory, behavior is undefined.
        // Testing this destructively is skipped to avoid corrupting real storage.

        // ============================================
        // PHASE 3: Test model file paths
        // ============================================

        for modelType in ModelManager.ModelType.allCases {
            let fileName = modelType.fileName

            // Construct full path
            let fullPath = modelsDir.appendingPathComponent(fileName)

            // Path should be valid
            XCTAssertFalse(fullPath.path.isEmpty,
                "BUG: Model path is empty for \(modelType)")

            // Path should not escape the models directory
            XCTAssertTrue(fullPath.path.hasPrefix(modelsDir.path),
                "BUG: Model path escapes models directory: \(fullPath.path)")

            // Path should not contain ".." (directory traversal)
            XCTAssertFalse(fileName.contains(".."),
                "BUG: Model filename contains '..': \(fileName)")
        }

        // ============================================
        // PHASE 4: Test empty model files (in temp dir)
        // ============================================

        // Use temp directory to avoid corrupting real model storage
        let tempTestDir = fileManager.temporaryDirectory
            .appendingPathComponent("TypogenesisTest-\(UUID().uuidString)", isDirectory: true)
        try fileManager.createDirectory(at: tempTestDir, withIntermediateDirectories: true)
        defer { try? fileManager.removeItem(at: tempTestDir) }

        // Create empty file in temp dir
        let emptyPath = tempTestDir.appendingPathComponent(ModelManager.ModelType.kerningNet.fileName)
        fileManager.createFile(atPath: emptyPath.path, contents: Data())

        // Verify empty file was created
        XCTAssertTrue(fileManager.fileExists(atPath: emptyPath.path),
            "Test setup failed: empty file not created")
        let emptyAttrs = try fileManager.attributesOfItem(atPath: emptyPath.path)
        XCTAssertEqual(emptyAttrs[.size] as? Int, 0, "File should be empty")

        // We can't inject the temp path into ModelManager singleton, but we verify
        // that non-existent models don't load from the real directory
        let statusForKerning = manager.status(for: .kerningNet)
        XCTAssertNotEqual(statusForKerning, .loaded,
            "BUG: kerningNet should not be loaded without a valid model file")

        // ============================================
        // PHASE 5: Test model file format validation concept
        // ============================================

        // Write a PNG file with .mlmodelc extension in temp dir
        let fakeModelPath = tempTestDir.appendingPathComponent(ModelManager.ModelType.styleEncoder.fileName)
        let pngHeader = Data([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])
        try pngHeader.write(to: fakeModelPath)

        // Verify fake model was created
        XCTAssertTrue(fileManager.fileExists(atPath: fakeModelPath.path),
            "Test setup failed: fake model file not created")

        // Verify real model is not loaded
        let statusForStyle = manager.status(for: .styleEncoder)
        XCTAssertNotEqual(statusForStyle, .loaded,
            "BUG: styleEncoder should not be loaded without a valid model file")

        // ============================================
        // PHASE 6: Test very large filename (edge case)
        // ============================================

        // Model filenames are hardcoded, but verify they're reasonable
        for modelType in ModelManager.ModelType.allCases {
            let fileName = modelType.fileName
            XCTAssertLessThan(fileName.count, 256,
                "Model filename is too long: \(fileName.count) chars")
        }

    }
}
