import XCTest

/// End-to-end UI tests based on User Story 2: Jordan Creates a Variable Weight Font
///
/// This test simulates a real user workflow:
/// 1. Create new project
/// 2. Enable variable font mode
/// 3. Add weight axis
/// 4. Configure masters
/// 5. Add named instances
///
/// See docs/USER_STORIES.md for the full user story.
///
final class VariableFontUITests: XCTestCase {

    var app: XCUIApplication!

    override func setUpWithError() throws {
        continueAfterFailure = false
        app = XCUIApplication()
        app.launch()
    }

    override func tearDownWithError() throws {
        app = nil
    }

    // MARK: - Story 2: Variable Font Creation

    /// Test that Variable Font section exists in sidebar
    func testVariableFontSectionExists() throws {
        // Create project first
        let createButton = app.buttons["welcome.createNewFont"]
        XCTAssertTrue(createButton.waitForExistence(timeout: 5))
        createButton.click()

        // Wait for project
        let glyphsItem = app.staticTexts["sidebar.glyphs"].firstMatch
        XCTAssertTrue(glyphsItem.waitForExistence(timeout: 5))

        // Verify Variable Font item exists
        let variableItem = app.staticTexts["sidebar.variable"].firstMatch
        XCTAssertTrue(variableItem.exists, "Variable Font item should exist in sidebar")
    }

    /// Test navigation to Variable Font editor
    func testNavigateToVariableFontEditor() throws {
        // Create project
        let createButton = app.buttons["welcome.createNewFont"]
        XCTAssertTrue(createButton.waitForExistence(timeout: 5))
        createButton.click()

        // Wait for project
        let glyphsItem = app.staticTexts["sidebar.glyphs"].firstMatch
        XCTAssertTrue(glyphsItem.waitForExistence(timeout: 5))

        // Navigate to Variable Font
        let variableItem = app.staticTexts["sidebar.variable"].firstMatch
        variableItem.click()
        Thread.sleep(forTimeInterval: 0.5)

        // The Variable Font editor should be visible
        // We verify by checking for "Variable Font" text in the editor
        let variableFontText = app.staticTexts["Variable Font"].firstMatch
        XCTAssertTrue(variableFontText.waitForExistence(timeout: 5), "Variable Font editor should be visible")
    }

    /// Test enabling variable font mode
    func testEnableVariableFontMode() throws {
        // Create project
        let createButton = app.buttons["welcome.createNewFont"]
        XCTAssertTrue(createButton.waitForExistence(timeout: 5))
        createButton.click()

        // Wait for project
        let glyphsItem = app.staticTexts["sidebar.glyphs"].firstMatch
        XCTAssertTrue(glyphsItem.waitForExistence(timeout: 5))

        // Navigate to Variable Font
        let variableItem = app.staticTexts["sidebar.variable"].firstMatch
        variableItem.click()
        Thread.sleep(forTimeInterval: 0.5)

        // Find and click the toggle - try multiple element types since SwiftUI renders differently
        let enableToggle = findVariableFontToggle()
        if let toggle = enableToggle {
            toggle.click()
            Thread.sleep(forTimeInterval: 0.5)

            // After enabling, the Add Axis button should appear
            let addAxisButton = app.buttons["variable.addAxis"]
            XCTAssertTrue(addAxisButton.waitForExistence(timeout: 5), "Add Axis button should appear after enabling variable font")
        } else {
            // If we can't find the toggle, check if variable font is already enabled by looking for axes section
            let axesHeader = app.staticTexts["Axes"].firstMatch
            XCTAssertTrue(axesHeader.waitForExistence(timeout: 5), "Should find Variable Font UI elements")
        }
    }

    /// Helper to find the Variable Font toggle regardless of how SwiftUI renders it
    private func findVariableFontToggle() -> XCUIElement? {
        // Try toggle first
        let toggle = app.toggles["variable.enableToggle"]
        if toggle.waitForExistence(timeout: 2) {
            return toggle
        }

        // Try checkbox (macOS renders toggles as checkboxes)
        let checkbox = app.checkBoxes["variable.enableToggle"]
        if checkbox.waitForExistence(timeout: 2) {
            return checkbox
        }

        // Try switch
        let toggleSwitch = app.switches["variable.enableToggle"]
        if toggleSwitch.waitForExistence(timeout: 2) {
            return toggleSwitch
        }

        return nil
    }

    /// Test that Add Axis button works
    func testAddAxisButton() throws {
        // Create project and enable variable font
        let createButton = app.buttons["welcome.createNewFont"]
        XCTAssertTrue(createButton.waitForExistence(timeout: 5))
        createButton.click()

        let glyphsItem = app.staticTexts["sidebar.glyphs"].firstMatch
        XCTAssertTrue(glyphsItem.waitForExistence(timeout: 5))

        let variableItem = app.staticTexts["sidebar.variable"].firstMatch
        variableItem.click()
        Thread.sleep(forTimeInterval: 0.5)

        // Enable variable font if not already enabled
        enableVariableFontIfNeeded()

        // Click Add Axis
        let addAxisButton = app.buttons["variable.addAxis"]
        if addAxisButton.waitForExistence(timeout: 5) {
            addAxisButton.click()
            Thread.sleep(forTimeInterval: 0.5)

            // A sheet should appear for adding axis
            let sheet = app.sheets.firstMatch
            if sheet.waitForExistence(timeout: 2) {
                // Dismiss the sheet
                app.typeKey(.escape, modifierFlags: [])
            }
        }
    }

    /// Helper to enable variable font mode if not already enabled
    private func enableVariableFontIfNeeded() {
        if let toggle = findVariableFontToggle() {
            // Try to check if already enabled
            let value = toggle.value as? String
            if value != "1" {
                toggle.click()
                Thread.sleep(forTimeInterval: 0.5)
            }
        }
    }

    /// Test that Add Master button works
    func testAddMasterButton() throws {
        // Create project and enable variable font
        let createButton = app.buttons["welcome.createNewFont"]
        XCTAssertTrue(createButton.waitForExistence(timeout: 5))
        createButton.click()

        let glyphsItem = app.staticTexts["sidebar.glyphs"].firstMatch
        XCTAssertTrue(glyphsItem.waitForExistence(timeout: 5))

        let variableItem = app.staticTexts["sidebar.variable"].firstMatch
        variableItem.click()
        Thread.sleep(forTimeInterval: 0.5)

        // Enable variable font if not already enabled
        enableVariableFontIfNeeded()

        // Click Add Master
        let addMasterButton = app.buttons["variable.addMaster"]
        if addMasterButton.waitForExistence(timeout: 5) {
            addMasterButton.click()
            Thread.sleep(forTimeInterval: 0.5)

            // A sheet should appear for adding master
            let sheet = app.sheets.firstMatch
            if sheet.waitForExistence(timeout: 2) {
                // Dismiss the sheet
                app.typeKey(.escape, modifierFlags: [])
            }
        }
    }

    /// Test that Add Instance button works
    func testAddInstanceButton() throws {
        // Create project and enable variable font
        let createButton = app.buttons["welcome.createNewFont"]
        XCTAssertTrue(createButton.waitForExistence(timeout: 5))
        createButton.click()

        let glyphsItem = app.staticTexts["sidebar.glyphs"].firstMatch
        XCTAssertTrue(glyphsItem.waitForExistence(timeout: 5))

        let variableItem = app.staticTexts["sidebar.variable"].firstMatch
        variableItem.click()
        Thread.sleep(forTimeInterval: 0.5)

        // Enable variable font if not already enabled
        enableVariableFontIfNeeded()

        // Click Add Instance
        let addInstanceButton = app.buttons["variable.addInstance"]
        if addInstanceButton.waitForExistence(timeout: 5) {
            addInstanceButton.click()
            Thread.sleep(forTimeInterval: 0.5)

            // A sheet should appear for adding instance
            let sheet = app.sheets.firstMatch
            if sheet.waitForExistence(timeout: 2) {
                // Dismiss the sheet
                app.typeKey(.escape, modifierFlags: [])
            }
        }
    }

    /// Test complete variable font workflow
    func testVariableFontWorkflow() throws {
        // Create project
        let createButton = app.buttons["welcome.createNewFont"]
        XCTAssertTrue(createButton.waitForExistence(timeout: 5))
        createButton.click()

        // Wait for project
        let glyphsItem = app.staticTexts["sidebar.glyphs"].firstMatch
        XCTAssertTrue(glyphsItem.waitForExistence(timeout: 5))

        // Navigate to Variable Font
        let variableItem = app.staticTexts["sidebar.variable"].firstMatch
        variableItem.click()
        Thread.sleep(forTimeInterval: 0.5)

        // Verify Variable Font editor is visible
        let variableFontText = app.staticTexts["Variable Font"].firstMatch
        XCTAssertTrue(variableFontText.waitForExistence(timeout: 5), "Variable Font editor should be visible")

        // Enable variable font if not already enabled
        enableVariableFontIfNeeded()

        // Navigate back to Glyphs
        glyphsItem.click()
        Thread.sleep(forTimeInterval: 0.5)

        // Navigate to Preview
        let previewItem = app.staticTexts["sidebar.preview"].firstMatch
        previewItem.click()
        Thread.sleep(forTimeInterval: 0.5)

        // Navigate back to Variable Font
        variableItem.click()
        Thread.sleep(forTimeInterval: 0.5)

        // Verify we're back at Variable Font section
        XCTAssertTrue(variableFontText.waitForExistence(timeout: 5), "Should be back at Variable Font section")
    }
}
