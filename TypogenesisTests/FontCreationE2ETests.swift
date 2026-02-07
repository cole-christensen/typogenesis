import Testing
import Foundation
import CoreGraphics
@testable import Typogenesis

// MARK: - Font Creation to Export E2E Tests

/// End-to-end tests for the complete font creation and export workflow.
/// These tests simulate the entire user journey:
/// 1. Create a new font project
/// 2. Add glyphs with outlines
/// 3. Configure metrics
/// 4. Add kerning pairs
/// 5. Export to various formats
/// 6. Verify exported files are valid

@Suite("Font Creation E2E Tests")
struct FontCreationE2ETests {

    // MARK: - Helper Functions

    /// Creates a simple letter glyph outline (rectangle with notch for 'A' shape)
    func createLetterAOutline() -> GlyphOutline {
        // Create an A-like shape with triangular outline
        let contour = Contour(
            points: [
                // Left leg bottom
                PathPoint(position: CGPoint(x: 0, y: 0), type: .corner),
                // Peak
                PathPoint(position: CGPoint(x: 250, y: 700), type: .corner),
                // Right leg bottom
                PathPoint(position: CGPoint(x: 500, y: 0), type: .corner),
                // Right leg inner
                PathPoint(position: CGPoint(x: 400, y: 0), type: .corner),
                // Crossbar right
                PathPoint(position: CGPoint(x: 350, y: 250), type: .corner),
                // Crossbar left
                PathPoint(position: CGPoint(x: 150, y: 250), type: .corner),
                // Left leg inner
                PathPoint(position: CGPoint(x: 100, y: 0), type: .corner)
            ],
            isClosed: true
        )
        return GlyphOutline(contours: [contour])
    }

    /// Creates a simple circular glyph outline for 'O'
    func createLetterOOutline() -> GlyphOutline {
        // Create an O shape with outer and inner contours
        let outerContour = Contour(
            points: [
                PathPoint(position: CGPoint(x: 50, y: 350), type: .smooth,
                         controlIn: CGPoint(x: 50, y: 150), controlOut: CGPoint(x: 50, y: 550)),
                PathPoint(position: CGPoint(x: 250, y: 700), type: .smooth,
                         controlIn: CGPoint(x: 100, y: 700), controlOut: CGPoint(x: 400, y: 700)),
                PathPoint(position: CGPoint(x: 450, y: 350), type: .smooth,
                         controlIn: CGPoint(x: 450, y: 550), controlOut: CGPoint(x: 450, y: 150)),
                PathPoint(position: CGPoint(x: 250, y: 0), type: .smooth,
                         controlIn: CGPoint(x: 400, y: 0), controlOut: CGPoint(x: 100, y: 0))
            ],
            isClosed: true
        )

        let innerContour = Contour(
            points: [
                PathPoint(position: CGPoint(x: 150, y: 350), type: .smooth,
                         controlIn: CGPoint(x: 150, y: 200), controlOut: CGPoint(x: 150, y: 500)),
                PathPoint(position: CGPoint(x: 250, y: 600), type: .smooth,
                         controlIn: CGPoint(x: 180, y: 600), controlOut: CGPoint(x: 320, y: 600)),
                PathPoint(position: CGPoint(x: 350, y: 350), type: .smooth,
                         controlIn: CGPoint(x: 350, y: 500), controlOut: CGPoint(x: 350, y: 200)),
                PathPoint(position: CGPoint(x: 250, y: 100), type: .smooth,
                         controlIn: CGPoint(x: 320, y: 100), controlOut: CGPoint(x: 180, y: 100))
            ],
            isClosed: true
        )

        return GlyphOutline(contours: [outerContour, innerContour])
    }

    /// Creates a simple rectangular glyph for testing
    func createRectangleGlyph(char: Character, width: Int = 500) -> Glyph {
        let contour = Contour(
            points: [
                PathPoint(position: CGPoint(x: 50, y: 0), type: .corner),
                PathPoint(position: CGPoint(x: CGFloat(width - 50), y: 0), type: .corner),
                PathPoint(position: CGPoint(x: CGFloat(width - 50), y: 700), type: .corner),
                PathPoint(position: CGPoint(x: 50, y: 700), type: .corner)
            ],
            isClosed: true
        )
        let outline = GlyphOutline(contours: [contour])
        return Glyph(character: char, outline: outline, advanceWidth: width, leftSideBearing: 50)
    }

    // MARK: - Basic Font Creation Tests

    @Test("Create font project with basic metadata")
    func createFontProject() {
        let project = FontProject(name: "TestFont", family: "Test Family", style: "Regular")

        #expect(project.name == "TestFont")
        #expect(project.family == "Test Family")
        #expect(project.style == "Regular")
        #expect(project.glyphs.isEmpty)
        #expect(project.kerning.isEmpty)
    }

    @Test("Add single glyph to project")
    func addSingleGlyph() {
        var project = FontProject(name: "TestFont", family: "Test Family", style: "Regular")

        let glyph = Glyph(
            character: "A",
            outline: createLetterAOutline(),
            advanceWidth: 600,
            leftSideBearing: 50
        )

        project.glyphs["A"] = glyph

        #expect(project.glyphs.count == 1)
        #expect(project.glyphs["A"] != nil)
        #expect(project.glyphs["A"]?.advanceWidth == 600)
    }

    @Test("Add multiple glyphs to project")
    func addMultipleGlyphs() {
        var project = FontProject(name: "TestFont", family: "Test Family", style: "Regular")

        // Add uppercase letters
        for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" {
            project.glyphs[char] = createRectangleGlyph(char: char)
        }

        // Add lowercase letters
        for char in "abcdefghijklmnopqrstuvwxyz" {
            project.glyphs[char] = createRectangleGlyph(char: char, width: 400)
        }

        // Add numbers
        for char in "0123456789" {
            project.glyphs[char] = createRectangleGlyph(char: char, width: 450)
        }

        #expect(project.glyphs.count == 62)
    }

    // MARK: - Metrics Configuration Tests

    @Test("Configure font metrics")
    func configureFontMetrics() {
        var project = FontProject(name: "TestFont", family: "Test Family", style: "Regular")

        project.metrics.unitsPerEm = 1000
        project.metrics.ascender = 800
        project.metrics.descender = -200
        project.metrics.xHeight = 500
        project.metrics.capHeight = 700
        project.metrics.lineGap = 100

        #expect(project.metrics.unitsPerEm == 1000)
        #expect(project.metrics.ascender == 800)
        #expect(project.metrics.descender == -200)
        #expect(project.metrics.xHeight == 500)
        #expect(project.metrics.capHeight == 700)
        #expect(project.metrics.lineGap == 100)
    }

    // MARK: - Kerning Configuration Tests

    @Test("Add kerning pairs to project")
    func addKerningPairs() {
        var project = FontProject(name: "TestFont", family: "Test Family", style: "Regular")

        // Add glyphs first
        project.glyphs["A"] = createRectangleGlyph(char: "A")
        project.glyphs["V"] = createRectangleGlyph(char: "V")
        project.glyphs["T"] = createRectangleGlyph(char: "T")
        project.glyphs["o"] = createRectangleGlyph(char: "o", width: 400)

        // Add kerning pairs
        project.kerning = [
            KerningPair(left: "A", right: "V", value: -50),
            KerningPair(left: "T", right: "o", value: -30),
            KerningPair(left: "V", right: "A", value: -50)
        ]

        #expect(project.kerning.count == 3)

        let avPair = project.kerning.first { $0.left == "A" && $0.right == "V" }
        #expect(avPair?.value == -50)
    }

    // MARK: - Export Tests

    @Test("Export font project to TTF format")
    func exportToTTF() async throws {
        var project = FontProject(name: "E2ETestFont", family: "E2E Test", style: "Regular")

        // Configure metrics
        project.metrics.unitsPerEm = 1000
        project.metrics.ascender = 800
        project.metrics.descender = -200

        // Add some glyphs
        project.glyphs["A"] = Glyph(
            character: "A",
            outline: createLetterAOutline(),
            advanceWidth: 600,
            leftSideBearing: 50
        )

        project.glyphs["O"] = Glyph(
            character: "O",
            outline: createLetterOOutline(),
            advanceWidth: 600,
            leftSideBearing: 50
        )

        // Add a few more letters
        for char in "BCDEFGHIJ" {
            project.glyphs[char] = createRectangleGlyph(char: char)
        }

        // Export
        let exporter = FontExporter()
        let data = try await exporter.export(project: project)

        // Verify export produced data
        #expect(data.count > 0)

        // Verify TTF signature (first 4 bytes should be 0x00010000 for TrueType)
        let signature = data.prefix(4)
        let signatureValue = signature.withUnsafeBytes { $0.load(as: UInt32.self) }
        let bigEndianSignature = UInt32(bigEndian: signatureValue)

        // Valid signatures: TrueType (0x00010000) or OpenType CFF (0x4F54544F "OTTO")
        let validSignatures: [UInt32] = [0x00010000, 0x4F54544F]
        #expect(validSignatures.contains(bigEndianSignature), "Should have valid font signature")
    }

    @Test("Export font project to UFO format")
    func exportToUFO() async throws {
        var project = FontProject(name: "E2ETestFont", family: "E2E Test", style: "Regular")

        project.metrics.unitsPerEm = 1000
        project.metrics.ascender = 800
        project.metrics.descender = -200

        // Add a glyph
        project.glyphs["A"] = Glyph(
            character: "A",
            outline: createLetterAOutline(),
            advanceWidth: 600,
            leftSideBearing: 50
        )

        // Create temporary directory for UFO export
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)

        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer {
            try? FileManager.default.removeItem(at: tempDir)
        }

        let ufoURL = tempDir.appendingPathComponent("TestFont.ufo")

        // Export to UFO
        let ufoExporter = UFOExporter()
        try await ufoExporter.export(project: project, to: ufoURL)

        // Verify UFO structure exists
        let fileManager = FileManager.default
        #expect(fileManager.fileExists(atPath: ufoURL.path))
        #expect(fileManager.fileExists(atPath: ufoURL.appendingPathComponent("metainfo.plist").path))
        #expect(fileManager.fileExists(atPath: ufoURL.appendingPathComponent("fontinfo.plist").path))
        #expect(fileManager.fileExists(atPath: ufoURL.appendingPathComponent("glyphs").path))
    }

    @Test("Export font with WOFF web font format")
    func exportToWOFF() async throws {
        var project = FontProject(name: "WebTestFont", family: "Web Test", style: "Regular")

        project.metrics.unitsPerEm = 1000
        project.metrics.ascender = 800
        project.metrics.descender = -200

        // Add minimal glyphs
        project.glyphs["A"] = createRectangleGlyph(char: "A")
        project.glyphs["B"] = createRectangleGlyph(char: "B")

        // First export to TTF
        let fontExporter = FontExporter()
        let ttfData = try await fontExporter.export(project: project)

        // Then convert to WOFF
        let webExporter = WebFontExporter()
        let woffData = try await webExporter.exportWOFF(ttfData: ttfData)

        #expect(woffData.count > 0)

        // WOFF signature is 'wOFF' (0x774F4646)
        let signature = woffData.prefix(4)
        let signatureValue = signature.withUnsafeBytes { $0.load(as: UInt32.self) }
        let bigEndianSignature = UInt32(bigEndian: signatureValue)
        #expect(bigEndianSignature == 0x774F4646, "Should have WOFF signature")
    }

    // MARK: - Complete Workflow E2E Tests

    @Test("Complete workflow: Create font, add glyphs, configure, export")
    func completeWorkflow() async throws {
        // Step 1: Create project
        var project = FontProject(name: "CompleteE2EFont", family: "Complete E2E", style: "Regular")

        // Step 2: Configure metrics
        project.metrics.unitsPerEm = 1000
        project.metrics.ascender = 800
        project.metrics.descender = -200
        project.metrics.xHeight = 500
        project.metrics.capHeight = 700
        project.metrics.lineGap = 90

        // Step 3: Add uppercase glyphs
        for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" {
            let width = 500 + (char.asciiValue.map { Int($0) % 100 } ?? 0)
            project.glyphs[char] = createRectangleGlyph(char: char, width: width)
        }

        // Step 4: Add lowercase glyphs
        for char in "abcdefghijklmnopqrstuvwxyz" {
            let width = 400 + (char.asciiValue.map { Int($0) % 80 } ?? 0)
            project.glyphs[char] = createRectangleGlyph(char: char, width: width)
        }

        // Step 5: Add numbers
        for char in "0123456789" {
            project.glyphs[char] = createRectangleGlyph(char: char, width: 500)
        }

        // Step 6: Add basic punctuation
        for char in ".,;:!?'-" {
            let width = 200 + (char.asciiValue.map { Int($0) % 50 } ?? 0)
            project.glyphs[char] = createRectangleGlyph(char: char, width: width)
        }

        // Step 7: Add kerning pairs
        project.kerning = [
            KerningPair(left: "A", right: "V", value: -50),
            KerningPair(left: "A", right: "W", value: -40),
            KerningPair(left: "A", right: "Y", value: -50),
            KerningPair(left: "V", right: "A", value: -50),
            KerningPair(left: "W", right: "A", value: -40),
            KerningPair(left: "Y", right: "A", value: -50),
            KerningPair(left: "T", right: "o", value: -30),
            KerningPair(left: "T", right: "a", value: -30),
            KerningPair(left: "f", right: "i", value: -10)
        ]

        // Step 8: Verify project state
        #expect(project.glyphs.count == 70)  // 26 + 26 + 10 + 8
        #expect(project.kerning.count == 9)

        // Step 9: Export to TTF
        let exporter = FontExporter()
        let ttfData = try await exporter.export(project: project)

        // Step 10: Verify export
        #expect(ttfData.count > 0)
        #expect(ttfData.count > 1000)  // Should be substantial with 70 glyphs
    }

    @Test("Font with special characters exports correctly")
    func specialCharactersExport() async throws {
        var project = FontProject(name: "SpecialCharsFont", family: "Special", style: "Regular")

        project.metrics.unitsPerEm = 1000
        project.metrics.ascender = 800
        project.metrics.descender = -200

        // Add special characters
        let specialChars = "€£¥©®™°±×÷"
        for char in specialChars {
            project.glyphs[char] = createRectangleGlyph(char: char)
        }

        // Add some extended Latin
        let extendedLatin = "ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞß"
        for char in extendedLatin {
            project.glyphs[char] = createRectangleGlyph(char: char)
        }

        let exporter = FontExporter()
        let data = try await exporter.export(project: project)

        #expect(data.count > 0)
    }

    @Test("Empty font project throws noGlyphs error")
    func emptyFontExport() async throws {
        let project = FontProject(name: "EmptyFont", family: "Empty", style: "Regular")

        // FontExporter.export() guards against empty glyph sets and throws .noGlyphs
        let exporter = FontExporter()
        var thrownError: (any Error)?
        do {
            let _ = try await exporter.export(project: project)
            Issue.record("Expected FontExporterError.noGlyphs to be thrown, but export succeeded")
        } catch {
            thrownError = error
        }

        // Verify the specific error case, not just any error
        let exporterError = try #require(thrownError as? FontExporter.FontExporterError)
        switch exporterError {
        case .noGlyphs:
            break  // This is the expected error
        case .invalidGlyphData:
            Issue.record("Expected .noGlyphs but got .invalidGlyphData")
        case .exportFailed(let reason):
            Issue.record("Expected .noGlyphs but got .exportFailed(\(reason))")
        }
    }

    @Test("Font metrics affect export correctly")
    func metricsAffectExport() async throws {
        // Create two fonts with different metrics
        var project1 = FontProject(name: "Font1", family: "Test", style: "Regular")
        var project2 = FontProject(name: "Font2", family: "Test", style: "Regular")

        // Same glyph
        project1.glyphs["A"] = createRectangleGlyph(char: "A")
        project2.glyphs["A"] = createRectangleGlyph(char: "A")

        // Different metrics
        project1.metrics.unitsPerEm = 1000
        project2.metrics.unitsPerEm = 2048

        project1.metrics.ascender = 800
        project2.metrics.ascender = 1638

        let exporter = FontExporter()
        let data1 = try await exporter.export(project: project1)
        let data2 = try await exporter.export(project: project2)

        // Both must be valid TrueType fonts (signature = 0x00010000)
        #expect(data1.count >= 12, "Font data1 too small to be valid")
        #expect(data2.count >= 12, "Font data2 too small to be valid")

        let sig1 = data1.readUInt32(at: 0)
        let sig2 = data2.readUInt32(at: 0)
        #expect(sig1 == 0x00010000, "data1 should have TrueType signature, got \(String(format: "0x%08X", sig1))")
        #expect(sig2 == 0x00010000, "data2 should have TrueType signature, got \(String(format: "0x%08X", sig2))")

        // Files should be different due to different metrics (same structure, different values)
        #expect(data1 != data2, "Different metrics must produce different font data")

        // Parse the head table to verify unitsPerEm is actually embedded
        let headOffset1 = try #require(findTableOffset(in: data1, tag: "head"), "head table not found in data1")
        let headOffset2 = try #require(findTableOffset(in: data2, tag: "head"), "head table not found in data2")

        // unitsPerEm is at offset 18 within the head table
        let upm1 = data1.readUInt16(at: headOffset1 + 18)
        let upm2 = data2.readUInt16(at: headOffset2 + 18)
        #expect(upm1 == 1000, "data1 head table should contain unitsPerEm=1000, got \(upm1)")
        #expect(upm2 == 2048, "data2 head table should contain unitsPerEm=2048, got \(upm2)")

        // Parse the hhea table to verify ascender is actually embedded
        let hheaOffset1 = try #require(findTableOffset(in: data1, tag: "hhea"), "hhea table not found in data1")
        let hheaOffset2 = try #require(findTableOffset(in: data2, tag: "hhea"), "hhea table not found in data2")

        // ascender is at offset 4 within the hhea table
        let asc1 = data1.readInt16(at: hheaOffset1 + 4)
        let asc2 = data2.readInt16(at: hheaOffset2 + 4)
        #expect(asc1 == 800, "data1 hhea table should contain ascender=800, got \(asc1)")
        #expect(asc2 == 1638, "data2 hhea table should contain ascender=1638, got \(asc2)")
    }

    /// Finds the byte offset of a table within TTF/OTF font data by searching the table directory.
    private func findTableOffset(in data: Data, tag: String) -> Int? {
        guard data.count >= 12 else { return nil }
        let numTables = Int(data.readUInt16(at: 4))
        for i in 0..<numTables {
            let recordOffset = 12 + i * 16
            guard recordOffset + 16 <= data.count else { return nil }
            let tableTag = data.readTag(at: recordOffset)
            if tableTag == tag {
                return Int(data.readUInt32(at: recordOffset + 8))
            }
        }
        return nil
    }

    // MARK: - Error Handling Tests

    @Test("Export handles invalid glyph outline gracefully")
    func invalidGlyphOutline() async throws {
        var project = FontProject(name: "InvalidFont", family: "Invalid", style: "Regular")

        // Add glyph with empty outline
        let emptyOutline = GlyphOutline(contours: [])
        project.glyphs["A"] = Glyph(
            character: "A",
            outline: emptyOutline,
            advanceWidth: 500,
            leftSideBearing: 50
        )

        // Add a valid glyph too
        project.glyphs["B"] = createRectangleGlyph(char: "B")

        let exporter = FontExporter()
        // Should either succeed (ignoring empty outline) or throw gracefully
        do {
            let data = try await exporter.export(project: project)
            #expect(data.count > 0)
        } catch {
            // An error is acceptable here - verify it's a FontExporter error, not a crash
            #expect(error is FontExporter.FontExporterError || error is CFFBuilder.CFFError,
                    "Should throw a font export error, got \(type(of: error)): \(error)")
        }
    }
}

// MARK: - Font Import → Edit → Export Tests

@Suite("Font Import Edit Export E2E Tests")
struct FontImportEditExportE2ETests {

    @Test("Round-trip: Export then parse project")
    func roundTripExportParse() async throws {
        // Create original project
        var original = FontProject(name: "RoundTripFont", family: "RoundTrip", style: "Regular")

        original.metrics.unitsPerEm = 1000
        original.metrics.ascender = 800
        original.metrics.descender = -200

        // Add glyphs
        for char in "ABC" {
            let outline = GlyphOutline(contours: [
                Contour(points: [
                    PathPoint(position: CGPoint(x: 50, y: 0), type: .corner),
                    PathPoint(position: CGPoint(x: 450, y: 0), type: .corner),
                    PathPoint(position: CGPoint(x: 450, y: 700), type: .corner),
                    PathPoint(position: CGPoint(x: 50, y: 700), type: .corner)
                ], isClosed: true)
            ])
            original.glyphs[char] = Glyph(
                character: char,
                outline: outline,
                advanceWidth: 500,
                leftSideBearing: 50
            )
        }

        // Export
        let exporter = FontExporter()
        let exportedData = try await exporter.export(project: original)

        // Parse
        let parser = FontParser()
        let parsed = try await parser.parse(data: exportedData)

        // Verify core properties preserved
        #expect(parsed.glyphs.count == original.glyphs.count)
        #expect(parsed.metrics.unitsPerEm == original.metrics.unitsPerEm)
    }

    @Test("Modify imported font and re-export")
    func modifyAndReExport() async throws {
        // Create base project
        var project = FontProject(name: "ModifyFont", family: "Modify", style: "Regular")

        project.metrics.unitsPerEm = 1000
        project.glyphs["A"] = Glyph(
            character: "A",
            outline: GlyphOutline(contours: [
                Contour(points: [
                    PathPoint(position: CGPoint(x: 50, y: 0), type: .corner),
                    PathPoint(position: CGPoint(x: 450, y: 0), type: .corner),
                    PathPoint(position: CGPoint(x: 450, y: 700), type: .corner),
                    PathPoint(position: CGPoint(x: 50, y: 700), type: .corner)
                ], isClosed: true)
            ]),
            advanceWidth: 500,
            leftSideBearing: 50
        )

        // Export first version
        let exporter = FontExporter()
        let data1 = try await exporter.export(project: project)

        // Modify: add new glyph
        project.glyphs["B"] = Glyph(
            character: "B",
            outline: GlyphOutline(contours: [
                Contour(points: [
                    PathPoint(position: CGPoint(x: 50, y: 0), type: .corner),
                    PathPoint(position: CGPoint(x: 400, y: 0), type: .corner),
                    PathPoint(position: CGPoint(x: 400, y: 700), type: .corner),
                    PathPoint(position: CGPoint(x: 50, y: 700), type: .corner)
                ], isClosed: true)
            ]),
            advanceWidth: 450,
            leftSideBearing: 50
        )

        // Re-export
        let data2 = try await exporter.export(project: project)

        // Modified version should be different (larger)
        #expect(data2.count > data1.count)
    }
}
