import Foundation
import Testing
@testable import Typogenesis

@Suite("Variable Font Export - fvar Table Tests")
struct FvarTableTests {

    func createTestConfig() -> VariableFontConfig {
        return .weightOnly()
    }

    @Test("fvar table has correct header version")
    func fvarHeaderVersion() async throws {
        let exporter = VariableFontExporter()
        let config = createTestConfig()

        let fvar = try await exporter.buildFvarTable(config: config)

        // Check version
        #expect(fvar.readUInt16(at: 0) == 1)  // majorVersion
        #expect(fvar.readUInt16(at: 2) == 0)  // minorVersion
    }

    @Test("fvar table contains correct axis count")
    func fvarAxisCount() async throws {
        let exporter = VariableFontExporter()
        let config = createTestConfig()

        let fvar = try await exporter.buildFvarTable(config: config)

        let axisCount = fvar.readUInt16(at: 8)
        #expect(axisCount == 1)  // weightOnly has 1 axis
    }

    @Test("fvar table with multiple axes")
    func fvarMultipleAxes() async throws {
        let exporter = VariableFontExporter()
        let config = VariableFontConfig.weightAndWidth()

        let fvar = try await exporter.buildFvarTable(config: config)

        let axisCount = fvar.readUInt16(at: 8)
        #expect(axisCount == 2)  // weight + width
    }

    @Test("fvar table contains instances")
    func fvarInstances() async throws {
        let exporter = VariableFontExporter()
        let config = createTestConfig()

        let fvar = try await exporter.buildFvarTable(config: config)

        let instanceCount = fvar.readUInt16(at: 12)
        #expect(instanceCount == 5)  // weightOnly has 5 instances
    }

    @Test("fvar table axis record has correct tag")
    func fvarAxisTag() async throws {
        let exporter = VariableFontExporter()
        let config = createTestConfig()

        let fvar = try await exporter.buildFvarTable(config: config)

        // Axis record starts at offset 16
        let tag = fvar.readTag(at: 16)
        #expect(tag == "wght")
    }

    @Test("fvar table throws for empty axes")
    func fvarEmptyAxesThrows() async throws {
        let exporter = VariableFontExporter()
        let config = VariableFontConfig()  // Empty config

        await #expect(throws: VariableFontExporter.VariableFontError.self) {
            try await exporter.buildFvarTable(config: config)
        }
    }

    @Test("fvar axis values are correctly encoded")
    func fvarAxisValues() async throws {
        let exporter = VariableFontExporter()
        let config = createTestConfig()

        let fvar = try await exporter.buildFvarTable(config: config)

        // Axis record: tag (4) + minValue (4) + defaultValue (4) + maxValue (4) + flags (2) + nameID (2)
        // minValue at offset 20 (16 + 4)
        let minValueFixed = fvar.readInt32(at: 20)
        let minValue = Float(minValueFixed) / 65536.0
        #expect(minValue == 100.0)  // Weight min

        // defaultValue at offset 24
        let defaultValueFixed = fvar.readInt32(at: 24)
        let defaultValue = Float(defaultValueFixed) / 65536.0
        #expect(defaultValue == 400.0)  // Weight default

        // maxValue at offset 28
        let maxValueFixed = fvar.readInt32(at: 28)
        let maxValue = Float(maxValueFixed) / 65536.0
        #expect(maxValue == 900.0)  // Weight max
    }
}

@Suite("Variable Font Export - gvar Table Tests")
struct GvarTableTests {

    func createTestProject() -> FontProject {
        var project = FontProject(
            name: "TestVarFont",
            family: "TestVarFont",
            style: "Regular"
        )

        // Add a simple glyph
        let points = [
            PathPoint(position: CGPoint(x: 50, y: 0), type: .corner),
            PathPoint(position: CGPoint(x: 50, y: 700), type: .corner),
            PathPoint(position: CGPoint(x: 450, y: 700), type: .corner),
            PathPoint(position: CGPoint(x: 450, y: 0), type: .corner)
        ]
        let contour = Contour(points: points, isClosed: true)
        let outline = GlyphOutline(contours: [contour])
        let glyph = Glyph(
            character: "A",
            outline: outline,
            advanceWidth: 500,
            leftSideBearing: 50
        )
        project.glyphs["A"] = glyph

        return project
    }

    func createConfigWithMasterGlyphs() -> VariableFontConfig {
        var config = VariableFontConfig.weightOnly()

        // Add glyphs to masters
        let lightPoints = [
            PathPoint(position: CGPoint(x: 50, y: 0), type: .corner),
            PathPoint(position: CGPoint(x: 50, y: 700), type: .corner),
            PathPoint(position: CGPoint(x: 400, y: 700), type: .corner),
            PathPoint(position: CGPoint(x: 400, y: 0), type: .corner)
        ]
        let lightContour = Contour(points: lightPoints, isClosed: true)
        let lightOutline = GlyphOutline(contours: [lightContour])
        let lightGlyph = Glyph(
            character: "A",
            outline: lightOutline,
            advanceWidth: 450,
            leftSideBearing: 50
        )

        let boldPoints = [
            PathPoint(position: CGPoint(x: 50, y: 0), type: .corner),
            PathPoint(position: CGPoint(x: 50, y: 700), type: .corner),
            PathPoint(position: CGPoint(x: 500, y: 700), type: .corner),
            PathPoint(position: CGPoint(x: 500, y: 0), type: .corner)
        ]
        let boldContour = Contour(points: boldPoints, isClosed: true)
        let boldOutline = GlyphOutline(contours: [boldContour])
        let boldGlyph = Glyph(
            character: "A",
            outline: boldOutline,
            advanceWidth: 550,
            leftSideBearing: 50
        )

        config.masters[0].glyphs["A"] = lightGlyph
        config.masters[1].glyphs["A"] = boldGlyph

        return config
    }

    @Test("gvar table has correct header version")
    func gvarHeaderVersion() async throws {
        let exporter = VariableFontExporter()
        let project = createTestProject()
        let config = createConfigWithMasterGlyphs()
        let glyphOrder: [(Character?, Glyph?)] = [(nil, nil), ("A", project.glyphs["A"])]

        let gvar = try await exporter.buildGvarTable(
            project: project,
            glyphOrder: glyphOrder,
            config: config
        )

        #expect(gvar.readUInt16(at: 0) == 1)  // majorVersion
        #expect(gvar.readUInt16(at: 2) == 0)  // minorVersion
    }

    @Test("gvar table contains correct axis count")
    func gvarAxisCount() async throws {
        let exporter = VariableFontExporter()
        let project = createTestProject()
        let config = createConfigWithMasterGlyphs()
        let glyphOrder: [(Character?, Glyph?)] = [(nil, nil), ("A", project.glyphs["A"])]

        let gvar = try await exporter.buildGvarTable(
            project: project,
            glyphOrder: glyphOrder,
            config: config
        )

        let axisCount = gvar.readUInt16(at: 4)
        #expect(axisCount == 1)
    }

    @Test("gvar table contains glyph count")
    func gvarGlyphCount() async throws {
        let exporter = VariableFontExporter()
        let project = createTestProject()
        let config = createConfigWithMasterGlyphs()
        let glyphOrder: [(Character?, Glyph?)] = [(nil, nil), ("A", project.glyphs["A"])]

        let gvar = try await exporter.buildGvarTable(
            project: project,
            glyphOrder: glyphOrder,
            config: config
        )

        let glyphCount = gvar.readUInt16(at: 12)
        #expect(glyphCount == 2)  // .notdef + A
    }

    @Test("gvar table throws for empty axes")
    func gvarEmptyAxesThrows() async throws {
        let exporter = VariableFontExporter()
        let project = createTestProject()
        let config = VariableFontConfig()  // Empty
        let glyphOrder: [(Character?, Glyph?)] = [(nil, nil), ("A", project.glyphs["A"])]

        await #expect(throws: VariableFontExporter.VariableFontError.self) {
            try await exporter.buildGvarTable(
                project: project,
                glyphOrder: glyphOrder,
                config: config
            )
        }
    }

    @Test("gvar table throws for empty masters")
    func gvarEmptyMastersThrows() async throws {
        let exporter = VariableFontExporter()
        let project = createTestProject()
        var config = VariableFontConfig.weightOnly()
        config.masters = []  // Remove masters
        let glyphOrder: [(Character?, Glyph?)] = [(nil, nil), ("A", project.glyphs["A"])]

        await #expect(throws: VariableFontExporter.VariableFontError.self) {
            try await exporter.buildGvarTable(
                project: project,
                glyphOrder: glyphOrder,
                config: config
            )
        }
    }
}

@Suite("Variable Font Export - STAT Table Tests")
struct STATTableTests {

    func createTestConfig() -> VariableFontConfig {
        return .weightOnly()
    }

    @Test("STAT table has correct header version")
    func statHeaderVersion() async throws {
        let exporter = VariableFontExporter()
        let config = createTestConfig()

        let stat = try await exporter.buildSTATTable(
            config: config,
            fontFamily: "Test",
            fontStyle: "Regular"
        )

        #expect(stat.readUInt16(at: 0) == 1)  // majorVersion
        #expect(stat.readUInt16(at: 2) == 2)  // minorVersion (1.2)
    }

    @Test("STAT table contains design axis count")
    func statDesignAxisCount() async throws {
        let exporter = VariableFontExporter()
        let config = createTestConfig()

        let stat = try await exporter.buildSTATTable(
            config: config,
            fontFamily: "Test",
            fontStyle: "Regular"
        )

        let axisCount = stat.readUInt16(at: 6)
        #expect(axisCount == 1)
    }

    @Test("STAT table with multiple axes")
    func statMultipleAxes() async throws {
        let exporter = VariableFontExporter()
        let config = VariableFontConfig.weightAndWidth()

        let stat = try await exporter.buildSTATTable(
            config: config,
            fontFamily: "Test",
            fontStyle: "Regular"
        )

        let axisCount = stat.readUInt16(at: 6)
        #expect(axisCount == 2)
    }

    @Test("STAT table throws for empty axes")
    func statEmptyAxesThrows() async throws {
        let exporter = VariableFontExporter()
        let config = VariableFontConfig()

        await #expect(throws: VariableFontExporter.VariableFontError.self) {
            try await exporter.buildSTATTable(
                config: config,
                fontFamily: "Test",
                fontStyle: "Regular"
            )
        }
    }
}

@Suite("Variable Font Export - avar Table Tests")
struct AvarTableTests {

    @Test("avar table has correct header version")
    func avarHeaderVersion() async throws {
        let exporter = VariableFontExporter()
        let config = VariableFontConfig.weightOnly()

        let avar = try await exporter.buildAvarTable(config: config)

        #expect(avar.readUInt16(at: 0) == 1)  // majorVersion
        #expect(avar.readUInt16(at: 2) == 0)  // minorVersion
    }

    @Test("avar table contains axis count")
    func avarAxisCount() async throws {
        let exporter = VariableFontExporter()
        let config = VariableFontConfig.weightOnly()

        let avar = try await exporter.buildAvarTable(config: config)

        let axisCount = avar.readUInt16(at: 6)
        #expect(axisCount == 1)
    }

    @Test("avar table has segment map for each axis")
    func avarSegmentMaps() async throws {
        let exporter = VariableFontExporter()
        let config = VariableFontConfig.weightAndWidth()

        let avar = try await exporter.buildAvarTable(config: config)

        // Header is 8 bytes, then segment maps
        // Each segment map starts with positionMapCount (2 bytes)
        let firstMapCount = avar.readUInt16(at: 8)
        #expect(firstMapCount == 3)  // Identity map has 3 points
    }

    @Test("avar table throws for empty axes")
    func avarEmptyAxesThrows() async throws {
        let exporter = VariableFontExporter()
        let config = VariableFontConfig()

        await #expect(throws: VariableFontExporter.VariableFontError.self) {
            try await exporter.buildAvarTable(config: config)
        }
    }
}

@Suite("Variable Font Export - Name Table Entries")
struct VariableFontNameEntriesTests {

    @Test("Name entries include axis names")
    func nameEntriesIncludeAxisNames() async throws {
        let exporter = VariableFontExporter()
        let config = VariableFontConfig.weightOnly()

        let entries = await exporter.getVariableFontNameEntries(config: config)

        let axisEntry = entries.first { $0.value == "Weight" }
        #expect(axisEntry != nil)
    }

    @Test("Name entries include instance names")
    func nameEntriesIncludeInstanceNames() async throws {
        let exporter = VariableFontExporter()
        let config = VariableFontConfig.weightOnly()

        let entries = await exporter.getVariableFontNameEntries(config: config)

        let boldEntry = entries.first { $0.value == "Bold" }
        #expect(boldEntry != nil)
    }

    @Test("Name entries count matches axes plus instances")
    func nameEntriesCount() async throws {
        let exporter = VariableFontExporter()
        let config = VariableFontConfig.weightOnly()

        let entries = await exporter.getVariableFontNameEntries(config: config)

        // 1 axis + 5 instances = 6 entries
        #expect(entries.count == 6)
    }
}
