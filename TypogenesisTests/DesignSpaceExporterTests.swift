import Testing
import Foundation
import CoreGraphics
@testable import Typogenesis

@Suite("DesignSpace Exporter Tests")
struct DesignSpaceExporterTests {

    // MARK: - Helper Methods

    private func createVariableFontProject() -> FontProject {
        var project = FontProject(name: "TestVarFont", family: "TestVarFont", style: "Regular")
        project.metadata = FontMetadata(
            copyright: "Copyright 2024 Test",
            designer: "Test Designer",
            description: "A test variable font"
        )

        // Add some test glyphs
        let aOutline = GlyphOutline(contours: [
            Contour(
                points: [
                    PathPoint(position: CGPoint(x: 250, y: 700), type: .corner),
                    PathPoint(position: CGPoint(x: 0, y: 0), type: .corner),
                    PathPoint(position: CGPoint(x: 100, y: 0), type: .corner),
                    PathPoint(position: CGPoint(x: 250, y: 500), type: .corner),
                    PathPoint(position: CGPoint(x: 400, y: 0), type: .corner),
                    PathPoint(position: CGPoint(x: 500, y: 0), type: .corner)
                ],
                isClosed: true
            )
        ])
        project.glyphs["A"] = Glyph(character: "A", outline: aOutline, advanceWidth: 500, leftSideBearing: 0)

        let bOutline = GlyphOutline(contours: [
            Contour(
                points: [
                    PathPoint(position: CGPoint(x: 50, y: 0), type: .corner),
                    PathPoint(position: CGPoint(x: 50, y: 700), type: .corner),
                    PathPoint(position: CGPoint(x: 300, y: 700), type: .smooth),
                    PathPoint(position: CGPoint(x: 400, y: 600), type: .smooth),
                    PathPoint(position: CGPoint(x: 300, y: 500), type: .smooth),
                    PathPoint(position: CGPoint(x: 50, y: 500), type: .corner)
                ],
                isClosed: true
            )
        ])
        project.glyphs["B"] = Glyph(character: "B", outline: bOutline, advanceWidth: 450, leftSideBearing: 50)

        // Add kerning
        project.kerning = [
            KerningPair(left: "A", right: "V", value: -50),
            KerningPair(left: "V", right: "A", value: -50)
        ]

        // Configure as variable font with weight axis
        var config = VariableFontConfig.weightOnly()

        // Add glyphs to masters
        config.masters[0].glyphs = project.glyphs  // Light master
        config.masters[1].glyphs = project.glyphs  // Bold master

        project.variableConfig = config

        return project
    }

    private func createWeightAndWidthProject() -> FontProject {
        var project = createVariableFontProject()

        // Configure with weight + width axes
        var config = VariableFontConfig.weightAndWidth()

        // Add glyphs to all masters
        for i in 0..<config.masters.count {
            config.masters[i].glyphs = project.glyphs
        }

        project.variableConfig = config
        return project
    }

    private func createTemporaryDirectory() throws -> URL {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        return tempDir
    }

    // MARK: - Directory Structure Tests

    @Test("Export creates DesignSpace directory structure")
    func testExportCreatesDirectoryStructure() async throws {
        let project = createVariableFontProject()
        let tempDir = try createTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let dsURL = tempDir.appendingPathComponent("TestVarFont.designspace")
        let exporter = DesignSpaceExporter()

        try await exporter.export(project: project, to: dsURL)

        // Check main directory exists
        var isDirectory: ObjCBool = false
        #expect(FileManager.default.fileExists(atPath: dsURL.path, isDirectory: &isDirectory))
        #expect(isDirectory.boolValue)

        // Check .designspace XML file exists
        let xmlFile = dsURL.appendingPathComponent("TestVarFont.designspace")
        #expect(FileManager.default.fileExists(atPath: xmlFile.path))
    }

    @Test("Export creates master UFO directories")
    func testExportCreatesMasterUFOs() async throws {
        let project = createVariableFontProject()
        let tempDir = try createTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let dsURL = tempDir.appendingPathComponent("TestVarFont.designspace")
        let exporter = DesignSpaceExporter()

        try await exporter.export(project: project, to: dsURL)

        // Check that master UFO directories were created
        // Default config has "Light Master" and "Bold Master"
        var isDirectory: ObjCBool = false

        let lightUFO = dsURL.appendingPathComponent("Light_Master.ufo")
        #expect(FileManager.default.fileExists(atPath: lightUFO.path, isDirectory: &isDirectory))
        #expect(isDirectory.boolValue)

        let boldUFO = dsURL.appendingPathComponent("Bold_Master.ufo")
        #expect(FileManager.default.fileExists(atPath: boldUFO.path, isDirectory: &isDirectory))
        #expect(isDirectory.boolValue)
    }

    // MARK: - XML Format Tests

    @Test("Export XML has correct format version")
    func testXMLFormatVersion() async throws {
        let project = createVariableFontProject()
        let tempDir = try createTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let dsURL = tempDir.appendingPathComponent("TestVarFont.designspace")
        let exporter = DesignSpaceExporter()

        try await exporter.export(project: project, to: dsURL)

        let xmlFile = dsURL.appendingPathComponent("TestVarFont.designspace")
        let content = try String(contentsOf: xmlFile, encoding: .utf8)

        #expect(content.contains("<?xml version=\"1.0\" encoding=\"UTF-8\"?>"))
        #expect(content.contains("<designspace format=\"5.0\">"))
    }

    @Test("Export XML contains axes section")
    func testXMLContainsAxes() async throws {
        let project = createVariableFontProject()
        let tempDir = try createTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let dsURL = tempDir.appendingPathComponent("TestVarFont.designspace")
        let exporter = DesignSpaceExporter()

        try await exporter.export(project: project, to: dsURL)

        let xmlFile = dsURL.appendingPathComponent("TestVarFont.designspace")
        let content = try String(contentsOf: xmlFile, encoding: .utf8)

        #expect(content.contains("<axes>"))
        #expect(content.contains("</axes>"))
        #expect(content.contains("tag=\"wght\""))
        #expect(content.contains("name=\"Weight\""))
    }

    @Test("Export XML has correct axis values")
    func testXMLAxisValues() async throws {
        let project = createVariableFontProject()
        let tempDir = try createTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let dsURL = tempDir.appendingPathComponent("TestVarFont.designspace")
        let exporter = DesignSpaceExporter()

        try await exporter.export(project: project, to: dsURL)

        let xmlFile = dsURL.appendingPathComponent("TestVarFont.designspace")
        let xmlData = try Data(contentsOf: xmlFile)
        let xmlDoc = try XMLDocument(data: xmlData, options: [])

        // Find the weight axis element by its tag attribute
        let axisNodes = try xmlDoc.nodes(forXPath: "//axis[@tag='wght']")
        #expect(axisNodes.count == 1, "Expected exactly 1 weight axis, got \(axisNodes.count)")

        let weightAxis = try #require(axisNodes.first as? XMLElement, "Weight axis element not found")

        // Verify all axis attributes on the correct element
        #expect(weightAxis.attribute(forName: "minimum")?.stringValue == "100",
                "Weight axis minimum should be 100")
        #expect(weightAxis.attribute(forName: "default")?.stringValue == "400",
                "Weight axis default should be 400")
        #expect(weightAxis.attribute(forName: "maximum")?.stringValue == "900",
                "Weight axis maximum should be 900")
        #expect(weightAxis.attribute(forName: "name")?.stringValue == "Weight",
                "Weight axis name should be 'Weight'")
    }

    @Test("Export XML contains sources section")
    func testXMLContainsSources() async throws {
        let project = createVariableFontProject()
        let tempDir = try createTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let dsURL = tempDir.appendingPathComponent("TestVarFont.designspace")
        let exporter = DesignSpaceExporter()

        try await exporter.export(project: project, to: dsURL)

        let xmlFile = dsURL.appendingPathComponent("TestVarFont.designspace")
        let content = try String(contentsOf: xmlFile, encoding: .utf8)

        #expect(content.contains("<sources>"))
        #expect(content.contains("</sources>"))
        #expect(content.contains("<source filename="))
        #expect(content.contains("<location>"))
        #expect(content.contains("<dimension name="))
    }

    @Test("Export XML sources have correct master locations")
    func testXMLSourceLocations() async throws {
        let project = createVariableFontProject()
        let tempDir = try createTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let dsURL = tempDir.appendingPathComponent("TestVarFont.designspace")
        let exporter = DesignSpaceExporter()

        try await exporter.export(project: project, to: dsURL)

        let xmlFile = dsURL.appendingPathComponent("TestVarFont.designspace")
        let content = try String(contentsOf: xmlFile, encoding: .utf8)

        // Light master at weight 300, Bold at 700
        #expect(content.contains("xvalue=\"300\""))
        #expect(content.contains("xvalue=\"700\""))
    }

    @Test("Export XML contains instances section")
    func testXMLContainsInstances() async throws {
        let project = createVariableFontProject()
        let tempDir = try createTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let dsURL = tempDir.appendingPathComponent("TestVarFont.designspace")
        let exporter = DesignSpaceExporter()

        try await exporter.export(project: project, to: dsURL)

        let xmlFile = dsURL.appendingPathComponent("TestVarFont.designspace")
        let content = try String(contentsOf: xmlFile, encoding: .utf8)

        #expect(content.contains("<instances>"))
        #expect(content.contains("</instances>"))
        #expect(content.contains("<instance familyname="))
        #expect(content.contains("stylename=\"Regular\""))
        #expect(content.contains("stylename=\"Bold\""))
    }

    // MARK: - Master UFO Content Tests

    @Test("Master UFOs contain glyph files")
    func testMasterUFOsContainGlyphs() async throws {
        let project = createVariableFontProject()
        let tempDir = try createTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let dsURL = tempDir.appendingPathComponent("TestVarFont.designspace")
        let exporter = DesignSpaceExporter()

        try await exporter.export(project: project, to: dsURL)

        // Check Light master UFO has glyphs
        let lightGlyphsDir = dsURL
            .appendingPathComponent("Light_Master.ufo")
            .appendingPathComponent("glyphs")

        #expect(FileManager.default.fileExists(atPath: lightGlyphsDir.appendingPathComponent("A.glif").path))
        #expect(FileManager.default.fileExists(atPath: lightGlyphsDir.appendingPathComponent("B.glif").path))

        // Check Bold master UFO has glyphs
        let boldGlyphsDir = dsURL
            .appendingPathComponent("Bold_Master.ufo")
            .appendingPathComponent("glyphs")

        #expect(FileManager.default.fileExists(atPath: boldGlyphsDir.appendingPathComponent("A.glif").path))
        #expect(FileManager.default.fileExists(atPath: boldGlyphsDir.appendingPathComponent("B.glif").path))
    }

    @Test("Master UFOs have valid fontinfo")
    func testMasterUFOsFontinfo() async throws {
        let project = createVariableFontProject()
        let tempDir = try createTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let dsURL = tempDir.appendingPathComponent("TestVarFont.designspace")
        let exporter = DesignSpaceExporter()

        try await exporter.export(project: project, to: dsURL)

        let lightFontinfo = dsURL
            .appendingPathComponent("Light_Master.ufo")
            .appendingPathComponent("fontinfo.plist")

        let content = try String(contentsOf: lightFontinfo, encoding: .utf8)

        #expect(content.contains("<key>familyName</key>"))
        #expect(content.contains("<string>TestVarFont</string>"))
    }

    // MARK: - Error Handling Tests

    @Test("Export throws error for non-variable font")
    func testExportThrowsForNonVariableFont() async {
        var project = FontProject(name: "Static", family: "Static", style: "Regular")
        project.glyphs["A"] = Glyph(
            character: "A",
            outline: GlyphOutline(contours: []),
            advanceWidth: 500,
            leftSideBearing: 0
        )
        // variableConfig.isVariableFont defaults to false

        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let dsURL = tempDir.appendingPathComponent("Static.designspace")
        let exporter = DesignSpaceExporter()

        await #expect(throws: DesignSpaceExporter.DesignSpaceError.self) {
            try await exporter.export(project: project, to: dsURL)
        }

        try? FileManager.default.removeItem(at: tempDir)
    }

    @Test("Export throws error for no axes")
    func testExportThrowsForNoAxes() async {
        var project = FontProject(name: "NoAxes", family: "NoAxes", style: "Regular")
        project.glyphs["A"] = Glyph(
            character: "A",
            outline: GlyphOutline(contours: []),
            advanceWidth: 500,
            leftSideBearing: 0
        )

        // Variable font with empty axes
        project.variableConfig = VariableFontConfig(
            isVariableFont: true,
            axes: [],
            masters: [
                FontMaster(name: "Light", location: [:]),
                FontMaster(name: "Bold", location: [:])
            ],
            instances: []
        )

        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let dsURL = tempDir.appendingPathComponent("NoAxes.designspace")
        let exporter = DesignSpaceExporter()

        await #expect(throws: DesignSpaceExporter.DesignSpaceError.self) {
            try await exporter.export(project: project, to: dsURL)
        }

        try? FileManager.default.removeItem(at: tempDir)
    }

    @Test("Export throws error for insufficient masters")
    func testExportThrowsForInsufficientMasters() async {
        var project = FontProject(name: "OneMaster", family: "OneMaster", style: "Regular")
        project.glyphs["A"] = Glyph(
            character: "A",
            outline: GlyphOutline(contours: []),
            advanceWidth: 500,
            leftSideBearing: 0
        )

        // Variable font with only 1 master
        project.variableConfig = VariableFontConfig(
            isVariableFont: true,
            axes: [.weight],
            masters: [
                FontMaster(name: "Regular", location: [VariationAxis.weightTag: 400], glyphs: project.glyphs)
            ],
            instances: []
        )

        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let dsURL = tempDir.appendingPathComponent("OneMaster.designspace")
        let exporter = DesignSpaceExporter()

        await #expect(throws: DesignSpaceExporter.DesignSpaceError.self) {
            try await exporter.export(project: project, to: dsURL)
        }

        try? FileManager.default.removeItem(at: tempDir)
    }

    // MARK: - Multi-Axis Tests

    @Test("Export handles weight and width axes")
    func testExportMultiAxis() async throws {
        let project = createWeightAndWidthProject()
        let tempDir = try createTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let dsURL = tempDir.appendingPathComponent("TestVarFont.designspace")
        let exporter = DesignSpaceExporter()

        try await exporter.export(project: project, to: dsURL)

        let xmlFile = dsURL.appendingPathComponent("TestVarFont.designspace")
        let content = try String(contentsOf: xmlFile, encoding: .utf8)

        // Check both axes are present
        #expect(content.contains("tag=\"wght\""))
        #expect(content.contains("tag=\"wdth\""))
        #expect(content.contains("name=\"Weight\""))
        #expect(content.contains("name=\"Width\""))
    }

    @Test("Export multi-axis creates all master UFOs")
    func testExportMultiAxisMasters() async throws {
        let project = createWeightAndWidthProject()
        let tempDir = try createTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let dsURL = tempDir.appendingPathComponent("TestVarFont.designspace")
        let exporter = DesignSpaceExporter()

        try await exporter.export(project: project, to: dsURL)

        // weightAndWidth config has 4 masters
        var isDirectory: ObjCBool = false

        let lightCondensed = dsURL.appendingPathComponent("Light_Condensed.ufo")
        #expect(FileManager.default.fileExists(atPath: lightCondensed.path, isDirectory: &isDirectory))

        let lightExpanded = dsURL.appendingPathComponent("Light_Expanded.ufo")
        #expect(FileManager.default.fileExists(atPath: lightExpanded.path, isDirectory: &isDirectory))

        let boldCondensed = dsURL.appendingPathComponent("Bold_Condensed.ufo")
        #expect(FileManager.default.fileExists(atPath: boldCondensed.path, isDirectory: &isDirectory))

        let boldExpanded = dsURL.appendingPathComponent("Bold_Expanded.ufo")
        #expect(FileManager.default.fileExists(atPath: boldExpanded.path, isDirectory: &isDirectory))
    }

    // MARK: - Options Tests

    @Test("Export with byIndex naming strategy")
    func testExportWithIndexNaming() async throws {
        let project = createVariableFontProject()
        let tempDir = try createTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let dsURL = tempDir.appendingPathComponent("TestVarFont.designspace")
        let exporter = DesignSpaceExporter()
        let options = DesignSpaceExporter.ExportOptions(
            includeKerning: true,
            masterNamingStrategy: .byIndex
        )

        try await exporter.export(project: project, to: dsURL, options: options)

        // Check masters are named by index
        var isDirectory: ObjCBool = false

        let master0 = dsURL.appendingPathComponent("Master_0.ufo")
        #expect(FileManager.default.fileExists(atPath: master0.path, isDirectory: &isDirectory))

        let master1 = dsURL.appendingPathComponent("Master_1.ufo")
        #expect(FileManager.default.fileExists(atPath: master1.path, isDirectory: &isDirectory))
    }

    @Test("Export includes kerning when requested")
    func testExportIncludesKerning() async throws {
        let project = createVariableFontProject()
        let tempDir = try createTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let dsURL = tempDir.appendingPathComponent("TestVarFont.designspace")
        let exporter = DesignSpaceExporter()
        let options = DesignSpaceExporter.ExportOptions(includeKerning: true)

        try await exporter.export(project: project, to: dsURL, options: options)

        // Check kerning in a master UFO
        let kerningPath = dsURL
            .appendingPathComponent("Light_Master.ufo")
            .appendingPathComponent("kerning.plist")

        #expect(FileManager.default.fileExists(atPath: kerningPath.path))

        // Parse the kerning plist as a proper dictionary, not just string matching
        let kerningData = try Data(contentsOf: kerningPath)
        let plistObject = try PropertyListSerialization.propertyList(
            from: kerningData,
            options: [],
            format: nil
        )

        guard let kerningDict = plistObject as? [String: [String: Int]] else {
            Issue.record("kerning.plist did not parse as [String: [String: Int]]")
            return
        }

        // The project has 2 kerning pairs: A->V=-50 and V->A=-50
        // This produces 2 top-level keys (one per left glyph)
        #expect(kerningDict.count == 2, "Expected 2 left-hand kerning entries, got \(kerningDict.count)")

        // Verify A -> V = -50
        let aKerning = try #require(kerningDict["A"], "Missing kerning entry for left glyph 'A'")
        #expect(aKerning["V"] == -50, "Expected A->V kerning of -50, got \(String(describing: aKerning["V"]))")

        // Verify V -> A = -50
        let vKerning = try #require(kerningDict["V"], "Missing kerning entry for left glyph 'V'")
        #expect(vKerning["A"] == -50, "Expected V->A kerning of -50, got \(String(describing: vKerning["A"]))")
    }

    @Test("Export excludes kerning when not requested")
    func testExportExcludesKerning() async throws {
        let project = createVariableFontProject()
        let tempDir = try createTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let dsURL = tempDir.appendingPathComponent("TestVarFont.designspace")
        let exporter = DesignSpaceExporter()
        let options = DesignSpaceExporter.ExportOptions(includeKerning: false)

        try await exporter.export(project: project, to: dsURL, options: options)

        // Verify kerning.plist does NOT exist in any master UFO
        let lightKerningPath = dsURL
            .appendingPathComponent("Light_Master.ufo")
            .appendingPathComponent("kerning.plist")
        #expect(!FileManager.default.fileExists(atPath: lightKerningPath.path),
                "kerning.plist should not exist in Light master when kerning is excluded")

        let boldKerningPath = dsURL
            .appendingPathComponent("Bold_Master.ufo")
            .appendingPathComponent("kerning.plist")
        #expect(!FileManager.default.fileExists(atPath: boldKerningPath.path),
                "kerning.plist should not exist in Bold master when kerning is excluded")
    }
}
