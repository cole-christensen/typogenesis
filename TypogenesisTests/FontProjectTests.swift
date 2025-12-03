import Testing
import CoreGraphics
@testable import Typogenesis

@Suite("FontProject Tests")
struct FontProjectTests {

    @Test("Create new font project with defaults")
    func createProject() {
        let project = FontProject(name: "Test Font", family: "Test", style: "Regular")

        #expect(project.name == "Test Font")
        #expect(project.family == "Test")
        #expect(project.style == "Regular")
        #expect(project.glyphs.isEmpty)
        #expect(project.kerning.isEmpty)
    }

    @Test("Font metrics have correct defaults")
    func metricsDefaults() {
        let metrics = FontMetrics()

        #expect(metrics.unitsPerEm == 1000)
        #expect(metrics.ascender == 800)
        #expect(metrics.descender == -200)
        #expect(metrics.xHeight == 500)
        #expect(metrics.capHeight == 700)
        #expect(metrics.lineGap == 90)
        #expect(metrics.baseline == 0)
        #expect(metrics.totalHeight == 1000)
    }

    @Test("Add and retrieve glyph")
    func addGlyph() {
        var project = FontProject(name: "Test", family: "Test", style: "Regular")
        let glyph = Glyph(character: "A", advanceWidth: 600, leftSideBearing: 50)

        project.setGlyph(glyph, for: "A")

        #expect(project.glyphs.count == 1)
        #expect(project.glyph(for: "A")?.advanceWidth == 600)
        #expect(project.glyph(for: "A") == glyph)
    }

    @Test("Remove glyph")
    func removeGlyph() {
        var project = FontProject(name: "Test", family: "Test", style: "Regular")
        let glyph = Glyph(character: "A", advanceWidth: 600)

        project.setGlyph(glyph, for: "A")
        project.removeGlyph(for: "A")

        #expect(project.glyphs.isEmpty)
        #expect(project.glyph(for: "A") == nil)
    }

    @Test("Character set reflects glyphs")
    func characterSet() {
        var project = FontProject(name: "Test", family: "Test", style: "Regular")

        project.setGlyph(Glyph(character: "A"), for: "A")
        project.setGlyph(Glyph(character: "B"), for: "B")
        project.setGlyph(Glyph(character: "C"), for: "C")

        #expect(project.characterSet.contains("A" as Unicode.Scalar))
        #expect(project.characterSet.contains("B" as Unicode.Scalar))
        #expect(project.characterSet.contains("C" as Unicode.Scalar))
        #expect(!project.characterSet.contains("D" as Unicode.Scalar))
    }
}

@Suite("Glyph Tests")
struct GlyphTests {

    @Test("Create glyph with character")
    func createGlyph() {
        let glyph = Glyph(character: "A", advanceWidth: 500, leftSideBearing: 50)

        #expect(glyph.character == "A")
        #expect(glyph.advanceWidth == 500)
        #expect(glyph.leftSideBearing == 50)
        #expect(glyph.unicodeScalars == [65])
    }

    @Test("Glyph name follows naming convention")
    func glyphName() {
        let glyphA = Glyph(character: "A")
        let glyphLower = Glyph(character: "a")
        let glyphNumber = Glyph(character: "1")

        #expect(glyphA.name == "uni0041")
        #expect(glyphLower.name == "uni0061")
        #expect(glyphNumber.name == "uni0031")
    }

    @Test("Right side bearing calculation")
    func rightSideBearing() {
        let outline = GlyphOutline(contours: [
            Contour(points: [
                PathPoint(position: CGPoint(x: 50, y: 0)),
                PathPoint(position: CGPoint(x: 450, y: 0)),
                PathPoint(position: CGPoint(x: 450, y: 700)),
                PathPoint(position: CGPoint(x: 50, y: 700)),
            ])
        ])

        let glyph = Glyph(
            character: "A",
            outline: outline,
            advanceWidth: 500,
            leftSideBearing: 50
        )

        #expect(glyph.rightSideBearing == 50)
    }
}

@Suite("GlyphOutline Tests")
struct GlyphOutlineTests {

    @Test("Empty outline")
    func emptyOutline() {
        let outline = GlyphOutline()

        #expect(outline.isEmpty)
        #expect(outline.contours.isEmpty)
    }

    @Test("Bounding box calculation")
    func boundingBox() {
        let outline = GlyphOutline(contours: [
            Contour(points: [
                PathPoint(position: CGPoint(x: 100, y: 0)),
                PathPoint(position: CGPoint(x: 400, y: 0)),
                PathPoint(position: CGPoint(x: 400, y: 700)),
                PathPoint(position: CGPoint(x: 100, y: 700)),
            ])
        ])

        let bbox = outline.boundingBox

        #expect(bbox.minX == 100)
        #expect(bbox.minY == 0)
        #expect(bbox.maxX == 400)
        #expect(bbox.maxY == 700)
        #expect(bbox.width == 300)
        #expect(bbox.height == 700)
    }

    @Test("CGPath generation for simple rectangle")
    func cgPathGeneration() {
        let outline = GlyphOutline(contours: [
            Contour(points: [
                PathPoint(position: CGPoint(x: 0, y: 0)),
                PathPoint(position: CGPoint(x: 100, y: 0)),
                PathPoint(position: CGPoint(x: 100, y: 100)),
                PathPoint(position: CGPoint(x: 0, y: 100)),
            ], isClosed: true)
        ])

        let path = outline.toCGPath()

        #expect(!path.isEmpty)
        #expect(path.boundingBox.width == 100)
        #expect(path.boundingBox.height == 100)
    }

    @Test("Multiple contours")
    func multipleContours() {
        let outline = GlyphOutline(contours: [
            Contour(points: [
                PathPoint(position: CGPoint(x: 0, y: 0)),
                PathPoint(position: CGPoint(x: 100, y: 0)),
                PathPoint(position: CGPoint(x: 100, y: 100)),
            ], isClosed: true),
            Contour(points: [
                PathPoint(position: CGPoint(x: 200, y: 200)),
                PathPoint(position: CGPoint(x: 300, y: 200)),
                PathPoint(position: CGPoint(x: 300, y: 300)),
            ], isClosed: true)
        ])

        #expect(!outline.isEmpty)
        #expect(outline.contours.count == 2)
    }
}

@Suite("KerningPair Tests")
struct KerningPairTests {

    @Test("Create kerning pair")
    func createKerningPair() {
        let pair = KerningPair(left: "A", right: "V", value: -50)

        #expect(pair.left == "A")
        #expect(pair.right == "V")
        #expect(pair.value == -50)
    }

    @Test("Kerning table lookup")
    func kerningTableLookup() {
        let table = KerningTable(pairs: [
            KerningPair(left: "A", right: "V", value: -50),
            KerningPair(left: "T", right: "o", value: -30),
        ])

        #expect(table.kerning(for: "A", right: "V") == -50)
        #expect(table.kerning(for: "T", right: "o") == -30)
        #expect(table.kerning(for: "A", right: "B") == 0)
    }
}
