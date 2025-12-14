import Testing
import Foundation
import CoreGraphics
@testable import Typogenesis

@Suite("UFO Exporter Tests")
struct UFOExporterTests {

    // MARK: - Helper Methods

    private func createTestProject() -> FontProject {
        var project = FontProject(name: "Test Font", family: "Test", style: "Regular")
        project.metadata = FontMetadata(
            copyright: "Copyright 2024 Test",
            designer: "Test Designer",
            description: "A test font"
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

        return project
    }

    private func createTemporaryDirectory() throws -> URL {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        return tempDir
    }

    // MARK: - Tests

    @Test("Export creates UFO directory structure")
    func testExportCreatesDirectoryStructure() async throws {
        let project = createTestProject()
        let tempDir = try createTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let ufoURL = tempDir.appendingPathComponent("TestFont.ufo")
        let exporter = UFOExporter()

        try await exporter.export(project: project, to: ufoURL)

        // Check main directory exists
        var isDirectory: ObjCBool = false
        #expect(FileManager.default.fileExists(atPath: ufoURL.path, isDirectory: &isDirectory))
        #expect(isDirectory.boolValue)

        // Check required files exist
        #expect(FileManager.default.fileExists(atPath: ufoURL.appendingPathComponent("metainfo.plist").path))
        #expect(FileManager.default.fileExists(atPath: ufoURL.appendingPathComponent("fontinfo.plist").path))
        #expect(FileManager.default.fileExists(atPath: ufoURL.appendingPathComponent("layercontents.plist").path))

        // Check glyphs directory
        let glyphsDir = ufoURL.appendingPathComponent("glyphs")
        #expect(FileManager.default.fileExists(atPath: glyphsDir.path, isDirectory: &isDirectory))
        #expect(isDirectory.boolValue)
        #expect(FileManager.default.fileExists(atPath: glyphsDir.appendingPathComponent("contents.plist").path))
    }

    @Test("Export metainfo.plist contains correct version")
    func testMetainfoVersion() async throws {
        let project = createTestProject()
        let tempDir = try createTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let ufoURL = tempDir.appendingPathComponent("TestFont.ufo")
        let exporter = UFOExporter()

        try await exporter.export(project: project, to: ufoURL)

        let metainfoPath = ufoURL.appendingPathComponent("metainfo.plist")
        let content = try String(contentsOf: metainfoPath, encoding: .utf8)

        #expect(content.contains("<key>formatVersion</key>"))
        #expect(content.contains("<integer>3</integer>"))
        #expect(content.contains("<key>creator</key>"))
        #expect(content.contains("<string>Typogenesis</string>"))
    }

    @Test("Export fontinfo.plist contains font metadata")
    func testFontinfoContent() async throws {
        let project = createTestProject()
        let tempDir = try createTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let ufoURL = tempDir.appendingPathComponent("TestFont.ufo")
        let exporter = UFOExporter()

        try await exporter.export(project: project, to: ufoURL)

        let fontinfoPath = ufoURL.appendingPathComponent("fontinfo.plist")
        let content = try String(contentsOf: fontinfoPath, encoding: .utf8)

        #expect(content.contains("<key>familyName</key>"))
        #expect(content.contains("<string>Test</string>"))
        #expect(content.contains("<key>styleName</key>"))
        #expect(content.contains("<string>Regular</string>"))
        #expect(content.contains("<key>unitsPerEm</key>"))
    }

    @Test("Export creates glyph files")
    func testGlyphFilesCreated() async throws {
        let project = createTestProject()
        let tempDir = try createTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let ufoURL = tempDir.appendingPathComponent("TestFont.ufo")
        let exporter = UFOExporter()

        try await exporter.export(project: project, to: ufoURL)

        let glyphsDir = ufoURL.appendingPathComponent("glyphs")

        // Check glyph files exist
        #expect(FileManager.default.fileExists(atPath: glyphsDir.appendingPathComponent("A.glif").path))
        #expect(FileManager.default.fileExists(atPath: glyphsDir.appendingPathComponent("B.glif").path))
    }

    @Test("Export glyph file contains valid GLIF XML")
    func testGlifXMLValid() async throws {
        let project = createTestProject()
        let tempDir = try createTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let ufoURL = tempDir.appendingPathComponent("TestFont.ufo")
        let exporter = UFOExporter()

        try await exporter.export(project: project, to: ufoURL)

        let glifPath = ufoURL.appendingPathComponent("glyphs").appendingPathComponent("A.glif")
        let content = try String(contentsOf: glifPath, encoding: .utf8)

        // Check XML structure
        #expect(content.contains("<?xml version=\"1.0\""))
        #expect(content.contains("<glyph name=\"A\""))
        #expect(content.contains("<advance width=\"500\""))
        #expect(content.contains("<unicode hex=\"0041\""))
        #expect(content.contains("<outline>"))
        #expect(content.contains("<contour>"))
        #expect(content.contains("<point"))
    }

    @Test("Export includes kerning when requested")
    func testKerningIncluded() async throws {
        let project = createTestProject()
        let tempDir = try createTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let ufoURL = tempDir.appendingPathComponent("TestFont.ufo")
        let exporter = UFOExporter()
        let options = UFOExporter.ExportOptions(includeKerning: true)

        try await exporter.export(project: project, to: ufoURL, options: options)

        let kerningPath = ufoURL.appendingPathComponent("kerning.plist")
        #expect(FileManager.default.fileExists(atPath: kerningPath.path))

        let content = try String(contentsOf: kerningPath, encoding: .utf8)
        #expect(content.contains("<key>A</key>"))
    }

    @Test("Export excludes kerning when not requested")
    func testKerningExcluded() async throws {
        let project = createTestProject()
        let tempDir = try createTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let ufoURL = tempDir.appendingPathComponent("TestFont.ufo")
        let exporter = UFOExporter()
        let options = UFOExporter.ExportOptions(includeKerning: false)

        try await exporter.export(project: project, to: ufoURL, options: options)

        let kerningPath = ufoURL.appendingPathComponent("kerning.plist")
        // kerning.plist should not exist when kerning is excluded
        #expect(!FileManager.default.fileExists(atPath: kerningPath.path))
    }

    @Test("Export fails with empty project")
    func testExportFailsWithEmptyProject() async {
        let project = FontProject(name: "Empty", family: "Empty", style: "Regular")
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)

        let ufoURL = tempDir.appendingPathComponent("Empty.ufo")
        let exporter = UFOExporter()

        do {
            try await exporter.export(project: project, to: ufoURL)
            Issue.record("Should have thrown noGlyphs error")
        } catch {
            #expect(error is UFOExporter.UFOError)
        }

        // Cleanup
        try? FileManager.default.removeItem(at: tempDir)
    }

    @Test("Export contents.plist maps glyph names to files")
    func testContentsMapping() async throws {
        let project = createTestProject()
        let tempDir = try createTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let ufoURL = tempDir.appendingPathComponent("TestFont.ufo")
        let exporter = UFOExporter()

        try await exporter.export(project: project, to: ufoURL)

        let contentsPath = ufoURL.appendingPathComponent("glyphs").appendingPathComponent("contents.plist")
        let content = try String(contentsOf: contentsPath, encoding: .utf8)

        // Should contain mapping entries
        #expect(content.contains("<key>A</key>"))
        #expect(content.contains("<string>A.glif</string>"))
        #expect(content.contains("<key>B</key>"))
        #expect(content.contains("<string>B.glif</string>"))
    }
}
