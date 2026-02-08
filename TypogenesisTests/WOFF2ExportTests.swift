import Foundation
import Testing
import CoreGraphics
@testable import Typogenesis

/// WOFF2 Export Tests
/// Tests for Web Open Font Format 2.0 (WOFF2) export functionality.
/// WOFF2 uses Brotli compression for better compression ratios than WOFF1's zlib.

@Suite("WOFF2 Export Tests")
struct WOFF2ExportTests {

    // MARK: - Test Helpers

    /// Creates a test font project with basic glyphs
    func createTestProject() -> FontProject {
        var project = FontProject(
            name: "WOFF2 Test Font",
            family: "WOFF2 Test",
            style: "Regular",
            metrics: FontMetrics(
                unitsPerEm: 1000,
                ascender: 800,
                descender: -200,
                xHeight: 500,
                capHeight: 700,
                lineGap: 90
            )
        )

        // Add test glyphs
        project.setGlyph(createRectangleGlyph(char: "A", width: 600), for: "A")
        project.setGlyph(createRectangleGlyph(char: "B", width: 650), for: "B")
        project.setGlyph(createRectangleGlyph(char: "C", width: 620), for: "C")
        project.setGlyph(createSpaceGlyph(), for: " ")

        return project
    }

    /// Creates a minimal test project with just one glyph
    func createMinimalProject() -> FontProject {
        var project = FontProject(
            name: "Minimal WOFF2 Test",
            family: "Minimal Test",
            style: "Regular"
        )

        project.setGlyph(createRectangleGlyph(char: "A", width: 500), for: "A")

        return project
    }

    /// Creates a rectangular glyph for testing
    func createRectangleGlyph(char: Character, width: Int) -> Glyph {
        let points = [
            PathPoint(position: CGPoint(x: 50, y: 0), type: .corner),
            PathPoint(position: CGPoint(x: 50, y: 700), type: .corner),
            PathPoint(position: CGPoint(x: CGFloat(width - 50), y: 700), type: .corner),
            PathPoint(position: CGPoint(x: CGFloat(width - 50), y: 0), type: .corner)
        ]
        let contour = Contour(points: points, isClosed: true)
        let outline = GlyphOutline(contours: [contour])

        return Glyph(
            character: char,
            outline: outline,
            advanceWidth: width,
            leftSideBearing: 50
        )
    }

    /// Creates a space glyph (no outline)
    func createSpaceGlyph() -> Glyph {
        return Glyph(
            character: " ",
            outline: GlyphOutline(),
            advanceWidth: 250,
            leftSideBearing: 0
        )
    }

    // MARK: - WOFF2 Signature Tests

    @Test("WOFF2 export produces file with 'wOF2' signature")
    func woff2Signature() async throws {
        let project = createTestProject()
        let fontExporter = FontExporter()
        let ttfData = try await fontExporter.export(project: project)

        let webExporter = WebFontExporter()
        let woff2Data = try await webExporter.exportWOFF2(ttfData: ttfData)

        // WOFF2 signature is "wOF2" (0x774F4632)
        #expect(woff2Data.count >= 4, "WOFF2 data should have at least 4 bytes for signature")

        let signature = woff2Data.readTag(at: 0)
        #expect(signature == "wOF2", "WOFF2 file should start with 'wOF2' signature, got '\(signature)'")
    }

    @Test("WOFF2 signature bytes match expected values")
    func woff2SignatureBytes() async throws {
        let project = createTestProject()
        let fontExporter = FontExporter()
        let ttfData = try await fontExporter.export(project: project)

        let webExporter = WebFontExporter()
        let woff2Data = try await webExporter.exportWOFF2(ttfData: ttfData)

        // Check individual signature bytes: 'w' 'O' 'F' '2'
        #expect(woff2Data[0] == 0x77, "First byte should be 'w' (0x77)")
        #expect(woff2Data[1] == 0x4F, "Second byte should be 'O' (0x4F)")
        #expect(woff2Data[2] == 0x46, "Third byte should be 'F' (0x46)")
        #expect(woff2Data[3] == 0x32, "Fourth byte should be '2' (0x32)")
    }

    // MARK: - WOFF2 Size Comparison Tests

    @Test("WOFF2 is smaller than original TTF")
    func woff2SmallerThanTTF() async throws {
        let project = createTestProject()
        let fontExporter = FontExporter()
        let ttfData = try await fontExporter.export(project: project)

        let webExporter = WebFontExporter()
        let woff2Data = try await webExporter.exportWOFF2(ttfData: ttfData)

        #expect(woff2Data.count < ttfData.count,
                "WOFF2 (\(woff2Data.count) bytes) should be smaller than TTF (\(ttfData.count) bytes)")
    }

    @Test("WOFF2 achieves significant compression")
    func woff2SignificantCompression() async throws {
        // Create a larger project to get better compression ratios
        var project = createTestProject()

        // Add more glyphs for better compression testing
        for char in "DEFGHIJKLMNOPQRSTUVWXYZ" {
            let width = 500 + Int(char.asciiValue! % 100)
            project.setGlyph(createRectangleGlyph(char: char, width: width), for: char)
        }

        let fontExporter = FontExporter()
        let ttfData = try await fontExporter.export(project: project)

        let webExporter = WebFontExporter()
        let woff2Data = try await webExporter.exportWOFF2(ttfData: ttfData)

        // WOFF2 with Brotli should achieve at least 10% compression
        let compressionRatio = Double(woff2Data.count) / Double(ttfData.count)
        #expect(compressionRatio < 0.9,
                "WOFF2 should achieve at least 10% compression, got \(Int((1 - compressionRatio) * 100))%")
    }

    // MARK: - WOFF2 Header Structure Tests

    @Test("WOFF2 header has correct numTables")
    func woff2NumTables() async throws {
        let project = createTestProject()
        let fontExporter = FontExporter()
        let ttfData = try await fontExporter.export(project: project)

        // Get numTables from original TTF
        let ttfNumTables = ttfData.readUInt16(at: 4)

        let webExporter = WebFontExporter()
        let woff2Data = try await webExporter.exportWOFF2(ttfData: ttfData)

        // numTables is at offset 12 in WOFF2 header
        let woff2NumTables = woff2Data.readUInt16(at: 12)

        #expect(woff2NumTables == ttfNumTables,
                "WOFF2 numTables (\(woff2NumTables)) should match TTF numTables (\(ttfNumTables))")
    }

    @Test("WOFF2 header has correct totalSfntSize")
    func woff2TotalSfntSize() async throws {
        let project = createTestProject()
        let fontExporter = FontExporter()
        let ttfData = try await fontExporter.export(project: project)

        let webExporter = WebFontExporter()
        let woff2Data = try await webExporter.exportWOFF2(ttfData: ttfData)

        // totalSfntSize is at offset 16 in WOFF2 header (after numTables and reserved)
        let totalSfntSize = woff2Data.readUInt32(at: 16)

        #expect(totalSfntSize == UInt32(ttfData.count),
                "WOFF2 totalSfntSize (\(totalSfntSize)) should equal original TTF size (\(ttfData.count))")
    }

    @Test("WOFF2 header length field matches actual file size")
    func woff2LengthField() async throws {
        let project = createTestProject()
        let fontExporter = FontExporter()
        let ttfData = try await fontExporter.export(project: project)

        let webExporter = WebFontExporter()
        let woff2Data = try await webExporter.exportWOFF2(ttfData: ttfData)

        // length field is at offset 8 in WOFF2 header
        let lengthField = woff2Data.readUInt32(at: 8)

        #expect(lengthField == UInt32(woff2Data.count),
                "WOFF2 length field (\(lengthField)) should match actual file size (\(woff2Data.count))")
    }

    @Test("WOFF2 preserves original font flavor in header")
    func woff2PreservesFlavor() async throws {
        let project = createTestProject()
        let fontExporter = FontExporter()
        let ttfData = try await fontExporter.export(project: project)

        // Get flavor from original TTF (first 4 bytes)
        let ttfFlavor = ttfData.readUInt32(at: 0)

        let webExporter = WebFontExporter()
        let woff2Data = try await webExporter.exportWOFF2(ttfData: ttfData)

        // flavor is at offset 4 in WOFF2 header (after signature)
        let woff2Flavor = woff2Data.readUInt32(at: 4)

        #expect(woff2Flavor == ttfFlavor,
                "WOFF2 flavor (0x\(String(woff2Flavor, radix: 16))) should match TTF flavor (0x\(String(ttfFlavor, radix: 16)))")
    }

    // MARK: - Round-trip and Validity Tests

    @Test("WOFF2 export produces non-empty valid data")
    func woff2ProducesValidData() async throws {
        let project = createTestProject()
        let fontExporter = FontExporter()
        let ttfData = try await fontExporter.export(project: project)

        let webExporter = WebFontExporter()
        let woff2Data = try await webExporter.exportWOFF2(ttfData: ttfData)

        #expect(woff2Data.count > 0, "WOFF2 data should not be empty")

        // Minimum WOFF2 header is 48 bytes
        #expect(woff2Data.count >= 48, "WOFF2 data should be at least 48 bytes (header size)")
    }

    @Test("WOFF2 export has valid header structure")
    func woff2ValidHeaderStructure() async throws {
        let project = createTestProject()
        let fontExporter = FontExporter()
        let ttfData = try await fontExporter.export(project: project)

        let webExporter = WebFontExporter()
        let woff2Data = try await webExporter.exportWOFF2(ttfData: ttfData)

        // Check header fields are reasonable
        let signature = woff2Data.readTag(at: 0)
        let length = woff2Data.readUInt32(at: 8)
        let numTables = woff2Data.readUInt16(at: 12)
        let reserved = woff2Data.readUInt16(at: 14)
        let totalSfntSize = woff2Data.readUInt32(at: 16)
        let totalCompressedSize = woff2Data.readUInt32(at: 20)
        let majorVersion = woff2Data.readUInt16(at: 24)
        let minorVersion = woff2Data.readUInt16(at: 26)

        #expect(signature == "wOF2")
        #expect(length == UInt32(woff2Data.count))
        #expect(numTables > 0, "Should have at least one table")
        #expect(numTables <= 100, "Unreasonable number of tables")
        #expect(reserved == 0, "Reserved field should be 0")
        #expect(totalSfntSize == UInt32(ttfData.count))
        #expect(totalCompressedSize > 0, "Should have compressed data")
        #expect(totalCompressedSize < totalSfntSize, "Compressed size should be smaller than original")
        #expect(majorVersion == 1, "Major version should be 1")
        #expect(minorVersion == 0, "Minor version should be 0")
    }

    // MARK: - Minimal and Edge Case Tests

    @Test("WOFF2 handles minimal valid TTF without crash")
    func woff2HandlesMinimalTTF() async throws {
        let project = createMinimalProject()
        let fontExporter = FontExporter()
        let ttfData = try await fontExporter.export(project: project)

        let webExporter = WebFontExporter()

        // Should not crash on minimal input
        let woff2Data = try await webExporter.exportWOFF2(ttfData: ttfData)

        #expect(woff2Data.count > 0, "Should produce output for minimal TTF")
        #expect(woff2Data.readTag(at: 0) == "wOF2", "Should have valid signature")
    }

    @Test("WOFF2 rejects too-small input data")
    func woff2RejectsTooSmallInput() async throws {
        let webExporter = WebFontExporter()

        // Input smaller than 12 bytes (minimum TTF header size)
        let smallData = Data([0x00, 0x01, 0x00, 0x00])

        await #expect(throws: WebFontExporter.WebFontError.self) {
            try await webExporter.exportWOFF2(ttfData: smallData)
        }
    }

    @Test("WOFF2 rejects empty input data")
    func woff2RejectsEmptyInput() async throws {
        let webExporter = WebFontExporter()
        let emptyData = Data()

        await #expect(throws: WebFontExporter.WebFontError.self) {
            try await webExporter.exportWOFF2(ttfData: emptyData)
        }
    }

    // MARK: - Comparison with WOFF1 Tests

    @Test("WOFF2 is smaller than WOFF1 for same font")
    func woff2SmallerThanWOFF1() async throws {
        // Create a larger project for meaningful comparison
        var project = createTestProject()

        for char in "DEFGHIJKLMNOPQRSTUVWXYZ" {
            project.setGlyph(createRectangleGlyph(char: char, width: 500), for: char)
        }
        for char in "abcdefghijklmnopqrstuvwxyz" {
            project.setGlyph(createRectangleGlyph(char: char, width: 400), for: char)
        }

        let fontExporter = FontExporter()
        let ttfData = try await fontExporter.export(project: project)

        let webExporter = WebFontExporter()
        let woff1Data = try await webExporter.exportWOFF(ttfData: ttfData)
        let woff2Data = try await webExporter.exportWOFF2(ttfData: ttfData)

        // Brotli (WOFF2) typically compresses better than zlib (WOFF1)
        #expect(woff2Data.count <= woff1Data.count,
                "WOFF2 (\(woff2Data.count) bytes) should be <= WOFF1 (\(woff1Data.count) bytes)")
    }

    // MARK: - Metadata Tests

    @Test("WOFF2 header metadata fields are zero when no metadata")
    func woff2NoMetadata() async throws {
        let project = createTestProject()
        let fontExporter = FontExporter()
        let ttfData = try await fontExporter.export(project: project)

        let webExporter = WebFontExporter()
        let woff2Data = try await webExporter.exportWOFF2(ttfData: ttfData)

        // Metadata fields start at offset 28
        let metaOffset = woff2Data.readUInt32(at: 28)
        let metaLength = woff2Data.readUInt32(at: 32)
        let metaOrigLength = woff2Data.readUInt32(at: 36)

        #expect(metaOffset == 0, "metaOffset should be 0 when no metadata")
        #expect(metaLength == 0, "metaLength should be 0 when no metadata")
        #expect(metaOrigLength == 0, "metaOrigLength should be 0 when no metadata")
    }

    @Test("WOFF2 header private data fields are zero when no private data")
    func woff2NoPrivateData() async throws {
        let project = createTestProject()
        let fontExporter = FontExporter()
        let ttfData = try await fontExporter.export(project: project)

        let webExporter = WebFontExporter()
        let woff2Data = try await webExporter.exportWOFF2(ttfData: ttfData)

        // Private data fields start at offset 40
        let privOffset = woff2Data.readUInt32(at: 40)
        let privLength = woff2Data.readUInt32(at: 44)

        #expect(privOffset == 0, "privOffset should be 0 when no private data")
        #expect(privLength == 0, "privLength should be 0 when no private data")
    }
}

// MARK: - OTF to WOFF2 Tests

@Suite("OTF to WOFF2 Export Tests")
struct OTFToWOFF2ExportTests {

    func createTestProject() -> FontProject {
        var project = FontProject(
            name: "OTF WOFF2 Test",
            family: "OTF Test",
            style: "Regular"
        )

        let points = [
            PathPoint(position: CGPoint(x: 50, y: 0), type: .corner),
            PathPoint(position: CGPoint(x: 50, y: 700), type: .corner),
            PathPoint(position: CGPoint(x: 450, y: 700), type: .corner),
            PathPoint(position: CGPoint(x: 450, y: 0), type: .corner)
        ]
        let contour = Contour(points: points, isClosed: true)
        let outline = GlyphOutline(contours: [contour])

        project.setGlyph(Glyph(
            character: "A",
            outline: outline,
            advanceWidth: 500,
            leftSideBearing: 50
        ), for: "A")

        project.setGlyph(Glyph(
            character: " ",
            outline: GlyphOutline(),
            advanceWidth: 250,
            leftSideBearing: 0
        ), for: " ")

        return project
    }

    @Test("WOFF2 from OTF has correct signature")
    func woff2FromOTFSignature() async throws {
        let project = createTestProject()
        let fontExporter = FontExporter()
        let options = FontExporter.ExportOptions(format: .otf)
        let otfData = try await fontExporter.export(project: project, options: options)

        let webExporter = WebFontExporter()
        let woff2Data = try await webExporter.exportWOFF2(ttfData: otfData)

        let signature = woff2Data.readTag(at: 0)
        #expect(signature == "wOF2", "WOFF2 from OTF should have 'wOF2' signature")
    }

    @Test("WOFF2 from OTF preserves OTTO flavor")
    func woff2FromOTFPreservesOTTOFlavor() async throws {
        let project = createTestProject()
        let fontExporter = FontExporter()
        let options = FontExporter.ExportOptions(format: .otf)
        let otfData = try await fontExporter.export(project: project, options: options)

        // OTF has OTTO signature (0x4F54544F)
        let otfFlavor = otfData.readUInt32(at: 0)
        #expect(otfFlavor == 0x4F54544F, "OTF should have OTTO signature")

        let webExporter = WebFontExporter()
        let woff2Data = try await webExporter.exportWOFF2(ttfData: otfData)

        // WOFF2 flavor should preserve OTTO
        let woff2Flavor = woff2Data.readUInt32(at: 4)
        #expect(woff2Flavor == 0x4F54544F, "WOFF2 from OTF should preserve OTTO flavor")
    }
}
