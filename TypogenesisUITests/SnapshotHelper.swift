//
//  SnapshotHelper.swift
//  Typogenesis
//
//  Created for Fastlane snapshot support
//

import XCTest

/// Helper class for capturing screenshots with Fastlane
/// Based on Fastlane's SnapshotHelper for macOS
enum Snapshot {
    static var screenshotsDirectory: URL? {
        // Check for Fastlane's environment variable
        if let path = ProcessInfo.processInfo.environment["SNAPSHOT_ARTIFACTS_FOLDER"] {
            return URL(fileURLWithPath: path)
        }

        // Use project root's fastlane/screenshots directory
        // Navigate from source file location to project root
        let sourceFile = URL(fileURLWithPath: #file)
        let projectRoot = sourceFile
            .deletingLastPathComponent()  // Remove SnapshotHelper.swift
            .deletingLastPathComponent()  // Remove TypogenesisUITests
        return projectRoot.appendingPathComponent("fastlane/screenshots")
    }

    /// Capture a screenshot with the given name
    /// - Parameters:
    ///   - name: The name for the screenshot file
    ///   - waitForLoadingIndicator: Whether to wait for loading indicators to disappear
    static func snapshot(_ name: String, waitForLoadingIndicator: Bool = true) {
        let app = XCUIApplication()

        if waitForLoadingIndicator {
            Thread.sleep(forTimeInterval: 0.5) // Give UI time to settle
        }

        // Take screenshot using XCTest's built-in method
        // Note: XCUITest runs on main thread, so this is safe
        let screenshot = app.windows.firstMatch.screenshot()

        // Save to screenshots directory if available
        if let directory = screenshotsDirectory {
            let fileManager = FileManager.default

            // Create directory if needed
            try? fileManager.createDirectory(at: directory, withIntermediateDirectories: true)

            let fileName = "\(name).png"
            let fileURL = directory.appendingPathComponent(fileName)

            do {
                try screenshot.pngRepresentation.write(to: fileURL)
                print("📸 Screenshot saved: \(fileName) -> \(fileURL.path)")
            } catch {
                print("⚠️ Failed to save screenshot: \(error)")
            }
        } else {
            print("⚠️ Screenshots directory not available")
        }

        // Also attach to test results for Xcode viewing
        let attachment = XCTAttachment(screenshot: screenshot)
        attachment.name = name
        attachment.lifetime = .keepAlways
        XCTContext.runActivity(named: "Screenshot: \(name)") { activity in
            activity.add(attachment)
        }
    }

    /// Take a screenshot of a specific element
    static func snapshot(_ name: String, element: XCUIElement) {
        guard element.exists else {
            print("⚠️ Element does not exist for screenshot: \(name)")
            return
        }

        let screenshot = element.screenshot()

        if let directory = screenshotsDirectory {
            let fileManager = FileManager.default
            try? fileManager.createDirectory(at: directory, withIntermediateDirectories: true)

            let fileName = "\(name).png"
            let fileURL = directory.appendingPathComponent(fileName)

            do {
                try screenshot.pngRepresentation.write(to: fileURL)
                print("📸 Element screenshot saved: \(fileName) -> \(fileURL.path)")
            } catch {
                print("⚠️ Failed to save screenshot: \(error)")
            }
        }

        let attachment = XCTAttachment(screenshot: screenshot)
        attachment.name = name
        attachment.lifetime = .keepAlways
        XCTContext.runActivity(named: "Screenshot: \(name)") { activity in
            activity.add(attachment)
        }
    }
}

// MARK: - XCUIApplication Extension

extension XCUIApplication {
    /// Take a screenshot of the entire app window
    func snapshot(_ name: String) {
        Snapshot.snapshot(name)
    }
}
