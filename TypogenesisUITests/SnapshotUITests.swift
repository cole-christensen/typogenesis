import XCTest

/// UI Tests specifically for capturing screenshots with Fastlane
/// Run with: `fastlane screenshots` or `fastlane mac screenshots`
///
/// These tests navigate through key areas of the app and capture
/// screenshots for App Store listings, documentation, and marketing.
///
final class SnapshotUITests: XCTestCase {

    var app: XCUIApplication!

    override func setUpWithError() throws {
        continueAfterFailure = true  // Continue to capture all screenshots even if one fails
        app = XCUIApplication()
        app.launch()
    }

    override func tearDownWithError() throws {
        app = nil
    }

    // MARK: - Welcome Screen

    func test01_WelcomeScreen() throws {
        // Wait for welcome screen to appear
        let createButton = app.buttons["welcome.createNewFont"]
        XCTAssertTrue(createButton.waitForExistence(timeout: 5))

        // Capture the welcome screen
        Snapshot.snapshot("01_WelcomeScreen")
    }

    // MARK: - Main Window with Empty Project

    func test02_EmptyProject() throws {
        // Create new project
        let createButton = app.buttons["welcome.createNewFont"]
        XCTAssertTrue(createButton.waitForExistence(timeout: 5))
        createButton.click()

        // Wait for main window
        let glyphsItem = app.staticTexts["sidebar.glyphs"].firstMatch
        XCTAssertTrue(glyphsItem.waitForExistence(timeout: 5))

        Thread.sleep(forTimeInterval: 0.5)
        Snapshot.snapshot("02_EmptyProject")
    }

    // MARK: - Glyph Grid with Glyphs

    func test03_GlyphGrid() throws {
        // Create new project
        let createButton = app.buttons["welcome.createNewFont"]
        XCTAssertTrue(createButton.waitForExistence(timeout: 5))
        createButton.click()

        // Wait for main window
        let glyphsItem = app.staticTexts["sidebar.glyphs"].firstMatch
        XCTAssertTrue(glyphsItem.waitForExistence(timeout: 5))

        // Click Add New Glyph button
        let addGlyphButton = app.buttons["glyphGrid.addGlyph"]
        if addGlyphButton.waitForExistence(timeout: 3) {
            addGlyphButton.click()
            Thread.sleep(forTimeInterval: 0.5)

            // Capture the Add Glyph sheet
            Snapshot.snapshot("03a_AddGlyphSheet")

            // Dismiss the sheet
            app.typeKey(.escape, modifierFlags: [])
            Thread.sleep(forTimeInterval: 0.3)
        }

        Snapshot.snapshot("03_GlyphGrid")
    }

    // MARK: - Metrics Editor

    func test04_MetricsEditor() throws {
        // Create new project
        let createButton = app.buttons["welcome.createNewFont"]
        XCTAssertTrue(createButton.waitForExistence(timeout: 5))
        createButton.click()

        // Wait for main window
        let glyphsItem = app.staticTexts["sidebar.glyphs"].firstMatch
        XCTAssertTrue(glyphsItem.waitForExistence(timeout: 5))

        // Navigate to Metrics
        let metricsItem = app.staticTexts["sidebar.metrics"].firstMatch
        metricsItem.click()
        Thread.sleep(forTimeInterval: 0.5)

        Snapshot.snapshot("04_MetricsEditor")
    }

    // MARK: - Kerning Editor

    func test05_KerningEditor() throws {
        // Create new project
        let createButton = app.buttons["welcome.createNewFont"]
        XCTAssertTrue(createButton.waitForExistence(timeout: 5))
        createButton.click()

        // Wait for main window
        let glyphsItem = app.staticTexts["sidebar.glyphs"].firstMatch
        XCTAssertTrue(glyphsItem.waitForExistence(timeout: 5))

        // Navigate to Kerning
        let kerningItem = app.staticTexts["sidebar.kerning"].firstMatch
        kerningItem.click()
        Thread.sleep(forTimeInterval: 0.5)

        Snapshot.snapshot("05_KerningEditor")
    }

    // MARK: - Preview Panel

    func test06_PreviewPanel() throws {
        // Create new project
        let createButton = app.buttons["welcome.createNewFont"]
        XCTAssertTrue(createButton.waitForExistence(timeout: 5))
        createButton.click()

        // Wait for main window
        let glyphsItem = app.staticTexts["sidebar.glyphs"].firstMatch
        XCTAssertTrue(glyphsItem.waitForExistence(timeout: 5))

        // Navigate to Preview
        let previewItem = app.staticTexts["sidebar.preview"].firstMatch
        previewItem.click()
        Thread.sleep(forTimeInterval: 0.5)

        Snapshot.snapshot("06_PreviewPanel")
    }

    // MARK: - Variable Font Editor

    func test07_VariableFontEditor() throws {
        // Create new project
        let createButton = app.buttons["welcome.createNewFont"]
        XCTAssertTrue(createButton.waitForExistence(timeout: 5))
        createButton.click()

        // Wait for main window
        let glyphsItem = app.staticTexts["sidebar.glyphs"].firstMatch
        XCTAssertTrue(glyphsItem.waitForExistence(timeout: 5))

        // Navigate to Variable Font
        let variableItem = app.staticTexts["sidebar.variable"].firstMatch
        variableItem.click()
        Thread.sleep(forTimeInterval: 0.5)

        Snapshot.snapshot("07a_VariableFontDisabled")

        // Try to enable variable font
        let enableToggle = app.checkBoxes["variable.enableToggle"]
        if enableToggle.waitForExistence(timeout: 2) {
            enableToggle.click()
            Thread.sleep(forTimeInterval: 0.5)
            Snapshot.snapshot("07b_VariableFontEnabled")
        }
    }

    // MARK: - AI Generate View

    func test08_AIGenerateView() throws {
        // Create new project
        let createButton = app.buttons["welcome.createNewFont"]
        XCTAssertTrue(createButton.waitForExistence(timeout: 5))
        createButton.click()

        // Wait for main window
        let glyphsItem = app.staticTexts["sidebar.glyphs"].firstMatch
        XCTAssertTrue(glyphsItem.waitForExistence(timeout: 5))

        // Navigate to AI Generate
        let generateItem = app.staticTexts["sidebar.generate"].firstMatch
        generateItem.click()
        Thread.sleep(forTimeInterval: 0.5)

        Snapshot.snapshot("08_AIGenerate")
    }

    // MARK: - Handwriting Scanner

    func test09_HandwritingScanner() throws {
        // Create new project
        let createButton = app.buttons["welcome.createNewFont"]
        XCTAssertTrue(createButton.waitForExistence(timeout: 5))
        createButton.click()

        // Wait for main window
        let glyphsItem = app.staticTexts["sidebar.glyphs"].firstMatch
        XCTAssertTrue(glyphsItem.waitForExistence(timeout: 5))

        // Navigate to Handwriting
        let handwritingItem = app.staticTexts["sidebar.handwriting"].firstMatch
        handwritingItem.click()
        Thread.sleep(forTimeInterval: 0.5)

        Snapshot.snapshot("09_HandwritingScanner")
    }

    // MARK: - Export Sheet

    func test10_ExportSheet() throws {
        // Create new project
        let createButton = app.buttons["welcome.createNewFont"]
        XCTAssertTrue(createButton.waitForExistence(timeout: 5))
        createButton.click()

        // Wait for main window
        let glyphsItem = app.staticTexts["sidebar.glyphs"].firstMatch
        XCTAssertTrue(glyphsItem.waitForExistence(timeout: 5))

        // Open export via menu
        let menuBar = app.menuBars
        menuBar.menuBarItems["File"].click()

        let exportMenuItem = menuBar.menuItems["Export Font..."]
        if exportMenuItem.waitForExistence(timeout: 2) {
            if exportMenuItem.isEnabled {
                exportMenuItem.click()
                Thread.sleep(forTimeInterval: 0.5)

                Snapshot.snapshot("10_ExportSheet")

                // Dismiss
                app.typeKey(.escape, modifierFlags: [])
            } else {
                // Menu disabled, dismiss menu
                app.typeKey(.escape, modifierFlags: [])
            }
        }
    }

    // MARK: - All Screens Summary Test

    func testAllScreensSequential() throws {
        // This test captures all screens in sequence without restarting the app
        // Useful for a quick full capture

        // 1. Welcome Screen
        let createButton = app.buttons["welcome.createNewFont"]
        XCTAssertTrue(createButton.waitForExistence(timeout: 5))
        Snapshot.snapshot("Summary_01_Welcome")

        // 2. Create project
        createButton.click()
        let glyphsItem = app.staticTexts["sidebar.glyphs"].firstMatch
        XCTAssertTrue(glyphsItem.waitForExistence(timeout: 5))
        Thread.sleep(forTimeInterval: 0.3)
        Snapshot.snapshot("Summary_02_Glyphs")

        // 3. Metrics
        let metricsItem = app.staticTexts["sidebar.metrics"].firstMatch
        metricsItem.click()
        Thread.sleep(forTimeInterval: 0.3)
        Snapshot.snapshot("Summary_03_Metrics")

        // 4. Kerning
        let kerningItem = app.staticTexts["sidebar.kerning"].firstMatch
        kerningItem.click()
        Thread.sleep(forTimeInterval: 0.3)
        Snapshot.snapshot("Summary_04_Kerning")

        // 5. Preview
        let previewItem = app.staticTexts["sidebar.preview"].firstMatch
        previewItem.click()
        Thread.sleep(forTimeInterval: 0.3)
        Snapshot.snapshot("Summary_05_Preview")

        // 6. Variable Font
        let variableItem = app.staticTexts["sidebar.variable"].firstMatch
        variableItem.click()
        Thread.sleep(forTimeInterval: 0.3)
        Snapshot.snapshot("Summary_06_Variable")

        // 7. AI Generate
        let generateItem = app.staticTexts["sidebar.generate"].firstMatch
        generateItem.click()
        Thread.sleep(forTimeInterval: 0.3)
        Snapshot.snapshot("Summary_07_Generate")

        // 8. Handwriting
        let handwritingItem = app.staticTexts["sidebar.handwriting"].firstMatch
        handwritingItem.click()
        Thread.sleep(forTimeInterval: 0.3)
        Snapshot.snapshot("Summary_08_Handwriting")
    }
}
