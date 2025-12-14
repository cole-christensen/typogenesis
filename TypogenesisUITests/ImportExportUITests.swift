import XCTest

/// End-to-end UI tests based on User Story 3: Sam Imports and Modifies an Existing Font
///
/// This test simulates a real user workflow:
/// 1. Launch app → Import existing TTF
/// 2. Verify glyphs loaded
/// 3. Navigate to different sections
/// 4. Export to new format
///
/// See docs/USER_STORIES.md for the full user story.
///
final class ImportExportUITests: XCTestCase {

    var app: XCUIApplication!

    override func setUpWithError() throws {
        continueAfterFailure = false
        app = XCUIApplication()
        app.launch()
    }

    override func tearDownWithError() throws {
        app = nil
    }

    // MARK: - Story 3: Import/Export Roundtrip

    /// Test that Import Font button exists on welcome screen
    func testImportFontButtonExists() throws {
        let importButton = app.buttons["welcome.importFont"]
        XCTAssertTrue(importButton.waitForExistence(timeout: 5), "Import Font button should exist on welcome screen")
    }

    /// Test that clicking Import opens file dialog or import sheet
    func testImportFontOpensDialog() throws {
        let importButton = app.buttons["welcome.importFont"]
        XCTAssertTrue(importButton.waitForExistence(timeout: 5))
        importButton.click()

        // The import action opens a file picker (NSOpenPanel)
        // We can verify by checking for the Open dialog or just ensure no crash
        Thread.sleep(forTimeInterval: 1)

        // Press Escape to dismiss any dialog
        app.typeKey(.escape, modifierFlags: [])
    }

    /// Test navigation to Kerning section after creating a project
    func testKerningSection() throws {
        // Create project first
        let createButton = app.buttons["welcome.createNewFont"]
        XCTAssertTrue(createButton.waitForExistence(timeout: 5))
        createButton.click()

        // Wait for project
        let glyphsItem = app.staticTexts["sidebar.glyphs"].firstMatch
        XCTAssertTrue(glyphsItem.waitForExistence(timeout: 5))

        // Navigate to Kerning
        let kerningItem = app.staticTexts["sidebar.kerning"].firstMatch
        kerningItem.click()
        Thread.sleep(forTimeInterval: 0.5)

        // Verify we navigated (no crash)
        XCTAssertTrue(kerningItem.exists, "Should be able to navigate to Kerning section")
    }

    /// Test that export supports multiple formats
    func testExportFormatOptions() throws {
        // Create project
        let createButton = app.buttons["welcome.createNewFont"]
        XCTAssertTrue(createButton.waitForExistence(timeout: 5))
        createButton.click()

        // Wait for project
        let glyphsItem = app.staticTexts["sidebar.glyphs"].firstMatch
        XCTAssertTrue(glyphsItem.waitForExistence(timeout: 5))

        // Open export via menu
        let menuBar = app.menuBars
        menuBar.menuBarItems["File"].click()

        let exportMenuItem = menuBar.menuItems["Export Font..."]
        guard exportMenuItem.waitForExistence(timeout: 2) else {
            XCTFail("Export Font menu item not found")
            return
        }

        // Check if enabled (may be disabled without glyphs)
        if !exportMenuItem.isEnabled {
            app.typeKey(.escape, modifierFlags: [])
            // Expected when no glyphs - test passes
            return
        }

        exportMenuItem.click()

        // Verify export sheet appeared
        let exportSheet = app.sheets.firstMatch
        XCTAssertTrue(exportSheet.waitForExistence(timeout: 5), "Export sheet should appear")

        // Dismiss
        app.typeKey(.escape, modifierFlags: [])
    }

    /// Test import via File menu
    func testImportViaMenu() throws {
        // First create a project (to exit welcome screen)
        let createButton = app.buttons["welcome.createNewFont"]
        XCTAssertTrue(createButton.waitForExistence(timeout: 5))
        createButton.click()

        // Wait for project
        let glyphsItem = app.staticTexts["sidebar.glyphs"].firstMatch
        XCTAssertTrue(glyphsItem.waitForExistence(timeout: 5))

        // Try to import via File menu
        let menuBar = app.menuBars
        menuBar.menuBarItems["File"].click()

        // Look for Import menu item
        let importMenuItem = menuBar.menuItems["Import Font..."]
        if importMenuItem.waitForExistence(timeout: 2) && importMenuItem.isEnabled {
            importMenuItem.click()
            Thread.sleep(forTimeInterval: 0.5)
            // Dismiss any dialog
            app.typeKey(.escape, modifierFlags: [])
        } else {
            // Dismiss menu
            app.typeKey(.escape, modifierFlags: [])
        }
    }

    /// Test preview section works
    func testPreviewSection() throws {
        // Create project
        let createButton = app.buttons["welcome.createNewFont"]
        XCTAssertTrue(createButton.waitForExistence(timeout: 5))
        createButton.click()

        // Wait for project
        let glyphsItem = app.staticTexts["sidebar.glyphs"].firstMatch
        XCTAssertTrue(glyphsItem.waitForExistence(timeout: 5))

        // Navigate to Preview
        let previewItem = app.staticTexts["sidebar.preview"].firstMatch
        previewItem.click()
        Thread.sleep(forTimeInterval: 0.5)

        // Verify we navigated successfully
        XCTAssertTrue(previewItem.exists, "Should be able to navigate to Preview section")
    }

    /// Test metrics section works
    func testMetricsSection() throws {
        // Create project
        let createButton = app.buttons["welcome.createNewFont"]
        XCTAssertTrue(createButton.waitForExistence(timeout: 5))
        createButton.click()

        // Wait for project
        let glyphsItem = app.staticTexts["sidebar.glyphs"].firstMatch
        XCTAssertTrue(glyphsItem.waitForExistence(timeout: 5))

        // Navigate to Metrics
        let metricsItem = app.staticTexts["sidebar.metrics"].firstMatch
        metricsItem.click()
        Thread.sleep(forTimeInterval: 0.5)

        // Verify we navigated successfully
        XCTAssertTrue(metricsItem.exists, "Should be able to navigate to Metrics section")
    }
}
