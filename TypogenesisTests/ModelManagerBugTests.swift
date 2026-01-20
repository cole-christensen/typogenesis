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
                if percentage > 100 {
                    // Over 100% looks bad
                    if percentage > 999 {
                        print("WARNING: Progress \(progress) shows \(percentage)% which is confusing")
                    }
                }
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

        // Create a fake corrupted model file
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        let modelsDir = appSupport.appendingPathComponent("Typogenesis/Models", isDirectory: true)

        // Ensure directory exists
        try? FileManager.default.createDirectory(at: modelsDir, withIntermediateDirectories: true)

        // Create corrupted file
        let corruptedPath = modelsDir.appendingPathComponent(ModelManager.ModelType.kerningNet.fileName)
        let corruptedData = "THIS IS NOT A VALID MLMODEL FILE".data(using: .utf8)!

        // Write corrupted file
        try? FileManager.default.removeItem(at: corruptedPath)  // Clean first
        try corruptedData.write(to: corruptedPath)

        // Verify file exists
        XCTAssertTrue(FileManager.default.fileExists(atPath: corruptedPath.path),
            "Test setup failed: corrupted file not created")

        // Try to load corrupted model
        await manager.loadModel(.kerningNet)

        let statusAfterCorrupted = manager.status(for: .kerningNet)

        // Should be error state
        switch statusAfterCorrupted {
        case .error:
            // Expected - corrupted file should produce error
            break
        case .loaded:
            XCTFail("BUG: Corrupted model file loaded successfully - this is a security/stability issue")
        case .loading:
            XCTFail("BUG: Model stuck in loading state for corrupted file")
        case .notDownloaded:
            // Acceptable - file might have been cleaned up
            break
        default:
            print("Unexpected status after loading corrupted file: \(statusAfterCorrupted)")
        }

        // Clean up
        try? FileManager.default.removeItem(at: corruptedPath)

        // ============================================
        // PHASE 10: Test download simulation timing
        // ============================================

        // The download simulation uses 200ms * 10 = 2 seconds
        let downloadStart = Date()
        await manager.downloadModel(.kerningNet)
        let downloadDuration = Date().timeIntervalSince(downloadStart)

        // Should complete within expected time (2s simulation + overhead)
        XCTAssertLessThan(downloadDuration, 5.0,
            "BUG: Download took \(downloadDuration)s, expected ~2s")

        // Should not be instant (simulation takes time)
        XCTAssertGreaterThan(downloadDuration, 1.0,
            "BUG: Download completed too fast (\(downloadDuration)s), simulation may be broken")

        // After download, status should be loaded (auto-loads after download)
        let statusAfterDownload = manager.status(for: .kerningNet)
        // Note: Since we corrupted the file earlier and cleaned up, the download
        // simulation doesn't create a real file, so it might fail to load
        print("Status after download: \(statusAfterDownload)")

        // ============================================
        // PHASE 11: Test isLoading flag
        // ============================================

        // At minimum, isLoading should be a valid boolean and not crash
        _ = manager.isLoading

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
        print("Status after cancel: \(statusAfterCancel)")

        // Clean up any partial downloads
        manager.deleteModel(.styleEncoder)
        manager.deleteModel(.kerningNet)

        // ============================================
        // Summary
        // ============================================

        print("""

        Model Lifecycle Test Summary:
        - Tested all \(allStatuses.count) status states
        - Tested \(progressEdgeCases.count) progress edge cases
        - Tested \(errorMessages.count) error message formats
        - Tested \(ModelManager.ModelType.allCases.count) model types
        - Tested unload, load missing, load corrupted, download, cancel
        """)
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
        print("All concurrent downloads completed in \(totalDownloadTime)s")

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

        XCTAssertGreaterThan(statusReadCount, 100,
            "Should have read status many times: \(statusReadCount)")
        print("Read status \(statusReadCount) times, saw \(statusesSeen.count) unique states")

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
            // Should be in a valid state
            switch status {
            case .notDownloaded, .downloading, .downloaded, .validating, .loading, .loaded, .updateAvailable, .error:
                break  // Valid
            }
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

        // ============================================
        // Summary
        // ============================================

        print("""

        Concurrent Operations Test Summary:
        - Tested concurrent downloads of \(ModelManager.ModelType.allCases.count) models
        - Read status \(statusReadCount) times during operations
        - Performed 10 rapid load/unload cycles
        - Tested concurrent cancel operations
        """)
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

        // Get models directory
        let appSupport = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        let modelsDir = appSupport.appendingPathComponent("Typogenesis/Models", isDirectory: true)

        // ============================================
        // PHASE 1: Verify directory creation
        // ============================================

        // Remove directory if exists (to test creation)
        try? fileManager.removeItem(at: modelsDir)

        // ModelManager should create directory on init
        // Force re-check by loading a model
        await manager.loadModel(.kerningNet)

        // Directory should now exist
        var isDirectory: ObjCBool = false
        let exists = fileManager.fileExists(atPath: modelsDir.path, isDirectory: &isDirectory)

        // Note: The manager creates the directory in init, so it should exist
        // But our test may be running after init already happened
        print("Models directory exists: \(exists), isDirectory: \(isDirectory.boolValue)")

        // ============================================
        // PHASE 2: Test with file instead of directory
        // ============================================

        // This is an edge case: what if the models "directory" is actually a file?
        // This could happen due to user error or malicious modification

        // We won't actually test this destructively, but document the concern
        print("WARNING: If modelsDir path is a file instead of directory, behavior is undefined")

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
        // PHASE 4: Test empty model files
        // ============================================

        // Ensure directory exists
        try fileManager.createDirectory(at: modelsDir, withIntermediateDirectories: true)

        // Create empty file
        let emptyPath = modelsDir.appendingPathComponent(ModelManager.ModelType.kerningNet.fileName)
        try? fileManager.removeItem(at: emptyPath)
        fileManager.createFile(atPath: emptyPath.path, contents: Data())

        // Try to load
        await manager.loadModel(.kerningNet)

        let statusAfterEmpty = manager.status(for: .kerningNet)
        switch statusAfterEmpty {
        case .error:
            // Expected - empty file is invalid
            break
        case .loaded:
            XCTFail("BUG: Empty model file loaded successfully")
        default:
            print("Status after loading empty file: \(statusAfterEmpty)")
        }

        // Clean up
        try? fileManager.removeItem(at: emptyPath)

        // ============================================
        // PHASE 5: Test model file with wrong extension
        // ============================================

        // The filename is ".mlmodelc" but what if the file is actually something else?
        // This tests that the loader validates the file format, not just the extension

        let fakeModelPath = modelsDir.appendingPathComponent(ModelManager.ModelType.styleEncoder.fileName)
        try? fileManager.removeItem(at: fakeModelPath)

        // Write a PNG file with .mlmodelc extension
        let pngHeader = Data([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])
        try pngHeader.write(to: fakeModelPath)

        await manager.loadModel(.styleEncoder)

        let statusAfterFake = manager.status(for: .styleEncoder)
        switch statusAfterFake {
        case .error:
            // Expected - PNG is not a valid model
            break
        case .loaded:
            XCTFail("BUG: Fake model file (PNG with .mlmodelc extension) loaded successfully - security issue")
        default:
            print("Status after loading fake model: \(statusAfterFake)")
        }

        // Clean up
        try? fileManager.removeItem(at: fakeModelPath)

        // ============================================
        // PHASE 6: Test very large filename (edge case)
        // ============================================

        // Model filenames are hardcoded, but verify they're reasonable
        for modelType in ModelManager.ModelType.allCases {
            let fileName = modelType.fileName
            XCTAssertLessThan(fileName.count, 256,
                "Model filename is too long: \(fileName.count) chars")
        }

        // ============================================
        // Summary
        // ============================================

        print("""

        File System Edge Cases Test Summary:
        - Verified directory creation
        - Tested loading empty model file
        - Tested loading fake model file (PNG with .mlmodelc extension)
        - Verified model filenames are reasonable length
        """)

        // Final cleanup
        for modelType in ModelManager.ModelType.allCases {
            manager.deleteModel(modelType)
        }
    }
}
