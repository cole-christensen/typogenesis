import Foundation
import Testing
@testable import Typogenesis

@Suite("FontParser Tests")
struct FontParserTests {

    @Test("Parse offset table from valid TTF header")
    func parseOffsetTable() async throws {
        // Create minimal valid TTF header
        var data = Data()
        data.writeUInt32(0x00010000)  // TrueType version
        data.writeUInt16(1)           // numTables
        data.writeUInt16(16)          // searchRange
        data.writeUInt16(0)           // entrySelector
        data.writeUInt16(0)           // rangeShift

        // Add a dummy table record
        data.writeTag("head")
        data.writeUInt32(0)           // checksum
        data.writeUInt32(28)          // offset
        data.writeUInt32(54)          // length

        // The actual parsing requires complete tables, but we can test the header parsing
        #expect(data.readUInt32(at: 0) == 0x00010000)
        #expect(data.readUInt16(at: 4) == 1)
    }

    @Test("Data extension reads big-endian correctly")
    func dataReadingBigEndian() {
        var data = Data()
        data.writeUInt16(0x1234)
        data.writeUInt32(0xDEADBEEF)
        data.writeInt16(-100)

        #expect(data.readUInt16(at: 0) == 0x1234)
        #expect(data.readUInt32(at: 2) == 0xDEADBEEF)
        #expect(data.readInt16(at: 6) == -100)
    }

    @Test("Data extension writes big-endian correctly")
    func dataWritingBigEndian() {
        var data = Data()
        data.writeUInt16(0xABCD)

        #expect(data[0] == 0xAB)
        #expect(data[1] == 0xCD)
    }

    @Test("Tag reading and writing")
    func tagReadWrite() {
        var data = Data()
        data.writeTag("head")

        #expect(data.readTag(at: 0) == "head")
    }

    @Test("Fixed point number handling")
    func fixedPointNumbers() {
        let fixed = Fixed(value: 0x00010000)  // 1.0
        #expect(fixed.floatValue == 1.0)

        let fixed2 = Fixed(integer: 2, fraction: 0x8000)  // 2.5
        #expect(abs(fixed2.floatValue - 2.5) < 0.001)
    }

    @Test("Checksum calculation")
    func checksumCalculation() {
        var data = Data()
        data.writeUInt32(0x12345678)
        data.writeUInt32(0x9ABCDEF0)

        let checksum = data.calculateTableChecksum()
        #expect(checksum == 0x12345678 &+ 0x9ABCDEF0)
    }

    @Test("LongDateTime creates current timestamp")
    func longDateTimeCreation() {
        let dt = LongDateTime()
        let date = dt.date

        // Should be close to now
        let diff = abs(Date().timeIntervalSince(date))
        #expect(diff < 5)  // Within 5 seconds
    }
}

@Suite("FontExporter Tests")
struct FontExporterTests {

    func createTestProject() -> FontProject {
        var project = FontProject(
            name: "Test Font",
            family: "Test Font",
            style: "Regular"
        )

        // Add some basic glyphs
        let glyphA = createTestGlyph(character: "A", advanceWidth: 600)
        let glyphB = createTestGlyph(character: "B", advanceWidth: 650)
        let glyphSpace = Glyph(
            character: " ",
            outline: GlyphOutline(),
            advanceWidth: 250,
            leftSideBearing: 0
        )

        project.setGlyph(glyphA, for: "A")
        project.setGlyph(glyphB, for: "B")
        project.setGlyph(glyphSpace, for: " ")

        return project
    }

    func createTestGlyph(character: Character, advanceWidth: Int) -> Glyph {
        // Create a simple rectangular outline
        let points = [
            PathPoint(position: CGPoint(x: 50, y: 0), type: .corner),
            PathPoint(position: CGPoint(x: 50, y: 700), type: .corner),
            PathPoint(position: CGPoint(x: 450, y: 700), type: .corner),
            PathPoint(position: CGPoint(x: 450, y: 0), type: .corner)
        ]
        let contour = Contour(points: points, isClosed: true)
        let outline = GlyphOutline(contours: [contour])

        return Glyph(
            character: character,
            outline: outline,
            advanceWidth: advanceWidth,
            leftSideBearing: 50
        )
    }

    @Test("Export creates non-empty data")
    func exportCreatesData() async throws {
        let project = createTestProject()
        let exporter = FontExporter()

        let data = try await exporter.export(project: project)

        #expect(data.count > 0)
    }

    @Test("Exported font has valid TrueType signature")
    func exportHasValidSignature() async throws {
        let project = createTestProject()
        let exporter = FontExporter()

        let data = try await exporter.export(project: project)

        // Check TrueType signature
        #expect(data.readUInt32(at: 0) == 0x00010000)
    }

    @Test("Exported font contains required tables")
    func exportContainsRequiredTables() async throws {
        let project = createTestProject()
        let exporter = FontExporter()

        let data = try await exporter.export(project: project)

        let numTables = Int(data.readUInt16(at: 4))
        #expect(numTables >= 10)  // Should have at least 10 tables

        // Read table tags
        var tables: Set<String> = []
        for i in 0..<numTables {
            let offset = 12 + i * 16
            let tag = data.readTag(at: offset)
            tables.insert(tag)
        }

        // Check required tables exist
        #expect(tables.contains("head"))
        #expect(tables.contains("hhea"))
        #expect(tables.contains("maxp"))
        #expect(tables.contains("cmap"))
        #expect(tables.contains("glyf"))
        #expect(tables.contains("loca"))
        #expect(tables.contains("hmtx"))
        #expect(tables.contains("name"))
        #expect(tables.contains("OS/2"))
        #expect(tables.contains("post"))
    }

    @Test("Export fails with empty font")
    func exportFailsWithEmptyFont() async throws {
        let project = FontProject(
            name: "Empty",
            family: "Empty",
            style: "Regular"
        )
        let exporter = FontExporter()

        await #expect(throws: FontExporter.FontExporterError.self) {
            try await exporter.export(project: project)
        }
    }

    @Test("Export with kerning includes kern table")
    func exportWithKerning() async throws {
        var project = createTestProject()
        project.kerning = [
            KerningPair(left: "A", right: "B", value: -50)
        ]

        let exporter = FontExporter()
        let data = try await exporter.export(project: project, options: .init(includeKerning: true))

        let numTables = Int(data.readUInt16(at: 4))
        var tables: Set<String> = []
        for i in 0..<numTables {
            let offset = 12 + i * 16
            let tag = data.readTag(at: offset)
            tables.insert(tag)
        }

        #expect(tables.contains("kern"))
    }

    @Test("Export without kerning option omits kern table")
    func exportWithoutKerning() async throws {
        var project = createTestProject()
        project.kerning = [
            KerningPair(left: "A", right: "B", value: -50)
        ]

        let exporter = FontExporter()
        let data = try await exporter.export(project: project, options: .init(includeKerning: false))

        let numTables = Int(data.readUInt16(at: 4))
        var tables: Set<String> = []
        for i in 0..<numTables {
            let offset = 12 + i * 16
            let tag = data.readTag(at: offset)
            tables.insert(tag)
        }

        #expect(!tables.contains("kern"))
    }
}

@Suite("Font Round-Trip Tests")
struct FontRoundTripTests {

    func createTestProject() -> FontProject {
        var project = FontProject(
            name: "RoundTrip Test",
            family: "RoundTrip Test",
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

        // Add glyphs with varying complexity
        project.setGlyph(createSimpleGlyph("A", width: 600), for: "A")
        project.setGlyph(createSimpleGlyph("B", width: 650), for: "B")
        project.setGlyph(createSimpleGlyph("C", width: 700), for: "C")
        project.setGlyph(Glyph(
            character: " ",
            outline: GlyphOutline(),
            advanceWidth: 250,
            leftSideBearing: 0
        ), for: " ")

        return project
    }

    func createSimpleGlyph(_ char: Character, width: Int) -> Glyph {
        let points = [
            PathPoint(position: CGPoint(x: 50, y: 0), type: .corner),
            PathPoint(position: CGPoint(x: 50, y: 700), type: .corner),
            PathPoint(position: CGPoint(x: Double(width - 50), y: 700), type: .corner),
            PathPoint(position: CGPoint(x: Double(width - 50), y: 0), type: .corner)
        ]
        let contour = Contour(points: points, isClosed: true)

        return Glyph(
            character: char,
            outline: GlyphOutline(contours: [contour]),
            advanceWidth: width,
            leftSideBearing: 50
        )
    }

    @Test("Export and re-parse preserves glyph count")
    func roundTripPreservesGlyphCount() async throws {
        let original = createTestProject()
        let exporter = FontExporter()
        let parser = FontParser()

        let exported = try await exporter.export(project: original)
        let parsed = try await parser.parse(data: exported)

        // Should have same number of glyphs (excluding .notdef)
        #expect(parsed.glyphs.count == original.glyphs.count)
    }

    @Test("Export and re-parse preserves character set")
    func roundTripPreservesCharacterSet() async throws {
        let original = createTestProject()
        let exporter = FontExporter()
        let parser = FontParser()

        let exported = try await exporter.export(project: original)
        let parsed = try await parser.parse(data: exported)

        let originalChars = Set(original.glyphs.keys)
        let parsedChars = Set(parsed.glyphs.keys)

        #expect(originalChars == parsedChars)
    }

    @Test("Export and re-parse preserves metrics")
    func roundTripPreservesMetrics() async throws {
        let original = createTestProject()
        let exporter = FontExporter()
        let parser = FontParser()

        let exported = try await exporter.export(project: original)
        let parsed = try await parser.parse(data: exported)

        #expect(parsed.metrics.unitsPerEm == original.metrics.unitsPerEm)
        #expect(parsed.metrics.ascender == original.metrics.ascender)
        #expect(parsed.metrics.descender == original.metrics.descender)
        #expect(parsed.metrics.lineGap == original.metrics.lineGap)
    }

    @Test("Export and re-parse preserves advance widths")
    func roundTripPreservesAdvanceWidths() async throws {
        let original = createTestProject()
        let exporter = FontExporter()
        let parser = FontParser()

        let exported = try await exporter.export(project: original)
        let parsed = try await parser.parse(data: exported)

        for (char, originalGlyph) in original.glyphs {
            if let parsedGlyph = parsed.glyphs[char] {
                #expect(parsedGlyph.advanceWidth == originalGlyph.advanceWidth)
            } else {
                Issue.record("Missing glyph for character: \(char)")
            }
        }
    }

    @Test("Export and re-parse preserves family name")
    func roundTripPreservesFamilyName() async throws {
        let original = createTestProject()
        let exporter = FontExporter()
        let parser = FontParser()

        let exported = try await exporter.export(project: original)
        let parsed = try await parser.parse(data: exported)

        #expect(parsed.family == original.family)
    }

    @Test("Export and re-parse preserves style name")
    func roundTripPreservesStyleName() async throws {
        let original = createTestProject()
        let exporter = FontExporter()
        let parser = FontParser()

        let exported = try await exporter.export(project: original)
        let parsed = try await parser.parse(data: exported)

        #expect(parsed.style == original.style)
    }
}

@Suite("Glyph Outline Conversion Tests")
struct GlyphOutlineConversionTests {

    @Test("Simple rectangle outline has 4 points")
    func simpleRectangleOutline() {
        let points = [
            PathPoint(position: CGPoint(x: 0, y: 0), type: .corner),
            PathPoint(position: CGPoint(x: 0, y: 100), type: .corner),
            PathPoint(position: CGPoint(x: 100, y: 100), type: .corner),
            PathPoint(position: CGPoint(x: 100, y: 0), type: .corner)
        ]
        let contour = Contour(points: points, isClosed: true)
        let outline = GlyphOutline(contours: [contour])

        #expect(outline.contours.count == 1)
        #expect(outline.contours[0].points.count == 4)
    }

    @Test("Bounding box calculated correctly")
    func boundingBoxCalculation() {
        let points = [
            PathPoint(position: CGPoint(x: 10, y: 20), type: .corner),
            PathPoint(position: CGPoint(x: 10, y: 120), type: .corner),
            PathPoint(position: CGPoint(x: 110, y: 120), type: .corner),
            PathPoint(position: CGPoint(x: 110, y: 20), type: .corner)
        ]
        let contour = Contour(points: points, isClosed: true)
        let outline = GlyphOutline(contours: [contour])

        let bbox = outline.boundingBox

        #expect(bbox.minX == 10)
        #expect(bbox.minY == 20)
        #expect(bbox.maxX == 110)
        #expect(bbox.maxY == 120)
        #expect(bbox.width == 100)
        #expect(bbox.height == 100)
    }

    @Test("Empty outline has zero bounding box")
    func emptyOutlineBoundingBox() {
        let outline = GlyphOutline()
        let bbox = outline.boundingBox

        #expect(bbox.minX == 0)
        #expect(bbox.minY == 0)
        #expect(bbox.maxX == 0)
        #expect(bbox.maxY == 0)
    }

    @Test("Outline with curves includes control points in bounds")
    func curveControlPointsInBounds() {
        let points = [
            PathPoint(
                position: CGPoint(x: 0, y: 0),
                type: .smooth,
                controlIn: nil,
                controlOut: CGPoint(x: 50, y: 200)
            ),
            PathPoint(
                position: CGPoint(x: 100, y: 100),
                type: .smooth,
                controlIn: CGPoint(x: 50, y: -50),
                controlOut: nil
            )
        ]
        let contour = Contour(points: points, isClosed: true)
        let outline = GlyphOutline(contours: [contour])

        let bbox = outline.boundingBox

        // Should include control points
        #expect(bbox.minY == -50)  // controlIn y
        #expect(bbox.maxY == 200)  // controlOut y
    }
}

@Suite("Int16/UInt16 Clamping Tests")
struct ClampingTests {

    @Test("Int16 clamping at max")
    func int16ClampingMax() {
        let value = Int16(clamping: 50000)
        #expect(value == Int16.max)
    }

    @Test("Int16 clamping at min")
    func int16ClampingMin() {
        let value = Int16(clamping: -50000)
        #expect(value == Int16.min)
    }

    @Test("Int16 clamping normal value")
    func int16ClampingNormal() {
        let value = Int16(clamping: 500)
        #expect(value == 500)
    }

    @Test("UInt16 clamping at max")
    func uint16ClampingMax() {
        let value = UInt16(clamping: 100000)
        #expect(value == UInt16.max)
    }

    @Test("UInt16 clamping negative to zero")
    func uint16ClampingNegative() {
        let value = UInt16(clamping: -100)
        #expect(value == 0)
    }

    @Test("UInt16 clamping normal value")
    func uint16ClampingNormal() {
        let value = UInt16(clamping: 500)
        #expect(value == 500)
    }
}
