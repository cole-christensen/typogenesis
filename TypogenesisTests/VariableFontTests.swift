import Foundation
import Testing
@testable import Typogenesis

@Suite("VariationAxis Tests")
struct VariationAxisTests {

    @Test("Weight axis has correct default values")
    func weightAxisDefaults() {
        let axis = VariationAxis.weight

        #expect(axis.tag == "wght")
        #expect(axis.name == "Weight")
        #expect(axis.minValue == 100)
        #expect(axis.defaultValue == 400)
        #expect(axis.maxValue == 900)
    }

    @Test("Width axis has correct default values")
    func widthAxisDefaults() {
        let axis = VariationAxis.width

        #expect(axis.tag == "wdth")
        #expect(axis.name == "Width")
        #expect(axis.minValue == 50)
        #expect(axis.defaultValue == 100)
        #expect(axis.maxValue == 200)
    }

    @Test("Slant axis has correct default values")
    func slantAxisDefaults() {
        let axis = VariationAxis.slant

        #expect(axis.tag == "slnt")
        #expect(axis.minValue == -20)
        #expect(axis.defaultValue == 0)
        #expect(axis.maxValue == 20)
    }

    @Test("Custom axis creation")
    func customAxis() {
        let axis = VariationAxis(
            tag: "GRAD",
            name: "Grade",
            minValue: -200,
            defaultValue: 0,
            maxValue: 150
        )

        #expect(axis.tag == "GRAD")
        #expect(axis.name == "Grade")
        #expect(axis.minValue == -200)
        #expect(axis.defaultValue == 0)
        #expect(axis.maxValue == 150)
    }

    @Test("Axis is Identifiable")
    func axisIdentifiable() {
        let axis1 = VariationAxis.weight
        let axis2 = VariationAxis.weight

        #expect(axis1.id != axis2.id)  // Each has unique ID
    }

    @Test("Axis is Codable")
    func axisCodable() throws {
        let original = VariationAxis.weight

        let encoder = JSONEncoder()
        let data = try encoder.encode(original)

        let decoder = JSONDecoder()
        let decoded = try decoder.decode(VariationAxis.self, from: data)

        #expect(decoded.tag == original.tag)
        #expect(decoded.name == original.name)
        #expect(decoded.minValue == original.minValue)
        #expect(decoded.defaultValue == original.defaultValue)
        #expect(decoded.maxValue == original.maxValue)
    }
}

@Suite("FontMaster Tests")
struct FontMasterTests {

    @Test("Master creation with location")
    func masterCreation() {
        let location: DesignSpaceLocation = [
            VariationAxis.weightTag: 700,
            VariationAxis.widthTag: 100
        ]

        let master = FontMaster(
            name: "Bold",
            location: location
        )

        #expect(master.name == "Bold")
        #expect(master.location[VariationAxis.weightTag] == 700)
        #expect(master.location[VariationAxis.widthTag] == 100)
        #expect(master.glyphs.isEmpty)
    }

    @Test("Master with glyphs")
    func masterWithGlyphs() {
        let glyph = Glyph(
            character: "A",
            advanceWidth: 600,
            leftSideBearing: 50
        )

        var master = FontMaster(
            name: "Regular",
            location: [VariationAxis.weightTag: 400]
        )
        master.glyphs["A"] = glyph

        #expect(master.glyphs.count == 1)
        #expect(master.glyphs["A"]?.advanceWidth == 600)
    }

    @Test("Master is Codable")
    func masterCodable() throws {
        let master = FontMaster(
            name: "Light",
            location: [VariationAxis.weightTag: 300]
        )

        let encoder = JSONEncoder()
        let data = try encoder.encode(master)

        let decoder = JSONDecoder()
        let decoded = try decoder.decode(FontMaster.self, from: data)

        #expect(decoded.name == master.name)
        #expect(decoded.location[VariationAxis.weightTag] == 300)
    }
}

@Suite("NamedInstance Tests")
struct NamedInstanceTests {

    @Test("Instance creation")
    func instanceCreation() {
        let instance = NamedInstance(
            name: "SemiBold",
            location: [VariationAxis.weightTag: 600]
        )

        #expect(instance.name == "SemiBold")
        #expect(instance.location[VariationAxis.weightTag] == 600)
    }

    @Test("Preset thin instance")
    func thinInstance() {
        let instance = NamedInstance.thin()

        #expect(instance.name == "Thin")
        #expect(instance.location[VariationAxis.weightTag] == 100)
    }

    @Test("Preset regular instance")
    func regularInstance() {
        let instance = NamedInstance.regular()

        #expect(instance.name == "Regular")
        #expect(instance.location[VariationAxis.weightTag] == 400)
    }

    @Test("Preset bold instance")
    func boldInstance() {
        let instance = NamedInstance.bold()

        #expect(instance.name == "Bold")
        #expect(instance.location[VariationAxis.weightTag] == 700)
    }

    @Test("Instance with extra axes")
    func instanceWithExtraAxes() {
        let instance = NamedInstance.bold(extraAxes: [VariationAxis.widthTag: 125])

        #expect(instance.location[VariationAxis.weightTag] == 700)
        #expect(instance.location[VariationAxis.widthTag] == 125)
    }
}

@Suite("VariableFontConfig Tests")
struct VariableFontConfigTests {

    @Test("Default config is not variable font")
    func defaultNotVariable() {
        let config = VariableFontConfig()

        #expect(!config.isVariableFont)
        #expect(config.axes.isEmpty)
        #expect(config.masters.isEmpty)
        #expect(config.instances.isEmpty)
    }

    @Test("Weight only config")
    func weightOnlyConfig() {
        let config = VariableFontConfig.weightOnly()

        #expect(config.isVariableFont)

        // Verify axis
        #expect(config.axes.count == 1)
        let weightAxis = config.axes[0]
        #expect(weightAxis.tag == VariationAxis.weightTag)
        #expect(weightAxis.name == "Weight")
        #expect(weightAxis.minValue == 100)
        #expect(weightAxis.defaultValue == 400)
        #expect(weightAxis.maxValue == 900)

        // Verify masters have correct names and design locations
        #expect(config.masters.count == 2)
        let lightMaster = config.masters[0]
        let boldMaster = config.masters[1]
        #expect(lightMaster.name == "Light Master")
        #expect(lightMaster.location[VariationAxis.weightTag] == 300)
        #expect(boldMaster.name == "Bold Master")
        #expect(boldMaster.location[VariationAxis.weightTag] == 700)

        // Verify instances have correct names and weight values
        #expect(config.instances.count == 5)
        let instanceNames = config.instances.map(\.name)
        #expect(instanceNames == ["Light", "Regular", "Medium", "Semibold", "Bold"])

        let instanceWeights = config.instances.map { $0.location[VariationAxis.weightTag] }
        #expect(instanceWeights == [300, 400, 500, 600, 700])
    }

    @Test("Weight and width config")
    func weightAndWidthConfig() {
        let config = VariableFontConfig.weightAndWidth()

        #expect(config.isVariableFont)

        // Verify axes
        #expect(config.axes.count == 2)
        let weightAxis = config.axes[0]
        #expect(weightAxis.tag == VariationAxis.weightTag)
        #expect(weightAxis.name == "Weight")
        #expect(weightAxis.minValue == 100)
        #expect(weightAxis.defaultValue == 400)
        #expect(weightAxis.maxValue == 900)

        let widthAxis = config.axes[1]
        #expect(widthAxis.tag == VariationAxis.widthTag)
        #expect(widthAxis.name == "Width")
        #expect(widthAxis.minValue == 75)
        #expect(widthAxis.defaultValue == 100)
        #expect(widthAxis.maxValue == 125)

        // Verify masters have correct names and design locations
        #expect(config.masters.count == 4)
        let masterNames = config.masters.map(\.name)
        #expect(masterNames == ["Light Condensed", "Light Expanded", "Bold Condensed", "Bold Expanded"])

        #expect(config.masters[0].location[VariationAxis.weightTag] == 300)
        #expect(config.masters[0].location[VariationAxis.widthTag] == 75)
        #expect(config.masters[1].location[VariationAxis.weightTag] == 300)
        #expect(config.masters[1].location[VariationAxis.widthTag] == 125)
        #expect(config.masters[2].location[VariationAxis.weightTag] == 700)
        #expect(config.masters[2].location[VariationAxis.widthTag] == 75)
        #expect(config.masters[3].location[VariationAxis.weightTag] == 700)
        #expect(config.masters[3].location[VariationAxis.widthTag] == 125)

        // Verify instances have correct names and locations
        #expect(config.instances.count == 9)
        let instanceNames = config.instances.map(\.name)
        #expect(instanceNames == [
            "Light Condensed", "Light", "Light Expanded",
            "Regular Condensed", "Regular", "Regular Expanded",
            "Bold Condensed", "Bold", "Bold Expanded"
        ])

        // Verify instance weight values
        let instanceWeights = config.instances.map { $0.location[VariationAxis.weightTag] }
        #expect(instanceWeights == [300, 300, 300, 400, 400, 400, 700, 700, 700])

        // Verify instance width values
        let instanceWidths = config.instances.map { $0.location[VariationAxis.widthTag] }
        #expect(instanceWidths == [75, 100, 125, 75, 100, 125, 75, 100, 125])
    }

    @Test("Config is Codable")
    func configCodable() throws {
        let original = VariableFontConfig.weightOnly()

        let encoder = JSONEncoder()
        let data = try encoder.encode(original)

        let decoder = JSONDecoder()
        let decoded = try decoder.decode(VariableFontConfig.self, from: data)

        #expect(decoded.isVariableFont == original.isVariableFont)
        #expect(decoded.axes.count == original.axes.count)
        #expect(decoded.masters.count == original.masters.count)
        #expect(decoded.instances.count == original.instances.count)

        // Verify axis values survive roundtrip
        let decodedAxis = decoded.axes[0]
        #expect(decodedAxis.tag == VariationAxis.weightTag)
        #expect(decodedAxis.minValue == 100)
        #expect(decodedAxis.defaultValue == 400)
        #expect(decodedAxis.maxValue == 900)

        // Verify master values survive roundtrip
        #expect(decoded.masters[0].name == "Light Master")
        #expect(decoded.masters[0].location[VariationAxis.weightTag] == 300)
        #expect(decoded.masters[1].name == "Bold Master")
        #expect(decoded.masters[1].location[VariationAxis.weightTag] == 700)

        // Verify instance values survive roundtrip
        let decodedInstanceNames = decoded.instances.map(\.name)
        #expect(decodedInstanceNames == ["Light", "Regular", "Medium", "Semibold", "Bold"])
        let decodedWeights = decoded.instances.map { $0.location[VariationAxis.weightTag] }
        #expect(decodedWeights == [300, 400, 500, 600, 700])
    }
}

@Suite("GlyphVariation Tests")
struct GlyphVariationTests {

    func createTestGlyph(character: Character, offset: CGFloat = 0) -> Glyph {
        let points = [
            PathPoint(position: CGPoint(x: 50 + offset, y: 0), type: .corner),
            PathPoint(position: CGPoint(x: 50 + offset, y: 700 + offset), type: .corner),
            PathPoint(position: CGPoint(x: 450 + offset, y: 700 + offset), type: .corner),
            PathPoint(position: CGPoint(x: 450 + offset, y: 0), type: .corner)
        ]
        let contour = Contour(points: points, isClosed: true)
        let outline = GlyphOutline(contours: [contour])

        return Glyph(
            character: character,
            outline: outline,
            advanceWidth: Int(500 + offset),
            leftSideBearing: Int(50 + offset / 10)
        )
    }

    @Test("Calculate variation between glyphs")
    func calculateVariation() {
        let sourceGlyph = createTestGlyph(character: "A", offset: 0)
        let targetGlyph = createTestGlyph(character: "A", offset: 50)

        let sourceMasterID = UUID()
        let targetMasterID = UUID()

        let variation = GlyphVariation.calculate(
            character: "A",
            source: sourceGlyph,
            target: targetGlyph,
            sourceMasterID: sourceMasterID,
            targetMasterID: targetMasterID
        )

        #expect(variation != nil)
        #expect(variation?.pointDeltas.count == 1)  // 1 contour
        #expect(variation?.pointDeltas.first?.count == 4)  // 4 points
        #expect(variation?.pointDeltas.first?.first?.dx == 50)  // First point moved 50 in x
    }

    @Test("Calculate variation fails with different contour counts")
    func calculateVariationDifferentContours() {
        let sourceGlyph = createTestGlyph(character: "A", offset: 0)

        // Create target with 2 contours
        let points1 = [
            PathPoint(position: CGPoint(x: 0, y: 0), type: .corner),
            PathPoint(position: CGPoint(x: 100, y: 100), type: .corner)
        ]
        let points2 = [
            PathPoint(position: CGPoint(x: 200, y: 200), type: .corner),
            PathPoint(position: CGPoint(x: 300, y: 300), type: .corner)
        ]
        let outline = GlyphOutline(contours: [
            Contour(points: points1, isClosed: true),
            Contour(points: points2, isClosed: true)
        ])
        let targetGlyph = Glyph(
            character: "A",
            outline: outline,
            advanceWidth: 500,
            leftSideBearing: 50
        )

        let variation = GlyphVariation.calculate(
            character: "A",
            source: sourceGlyph,
            target: targetGlyph,
            sourceMasterID: UUID(),
            targetMasterID: UUID()
        )

        #expect(variation == nil)  // Should fail
    }

    @Test("Apply variation to glyph")
    func applyVariation() {
        let sourceGlyph = createTestGlyph(character: "A", offset: 0)
        let targetGlyph = createTestGlyph(character: "A", offset: 100)

        let variation = GlyphVariation.calculate(
            character: "A",
            source: sourceGlyph,
            target: targetGlyph,
            sourceMasterID: UUID(),
            targetMasterID: UUID()
        )!

        // Apply at 50% interpolation
        let interpolated = variation.apply(to: sourceGlyph, factor: 0.5)

        // Check first point position (should be halfway)
        let firstPoint = interpolated.outline.contours.first?.points.first
        #expect(firstPoint?.position.x == 100)  // 50 + (100 * 0.5)
    }

    @Test("PointDelta defaults to zero")
    func pointDeltaDefaults() {
        let delta = PointDelta()

        #expect(delta.dx == 0)
        #expect(delta.dy == 0)
    }
}

@Suite("FontProject Variable Config Integration")
struct FontProjectVariableConfigTests {

    @Test("FontProject initializes with default variable config")
    func projectDefaultConfig() {
        let project = FontProject(
            name: "Test",
            family: "Test",
            style: "Regular"
        )

        #expect(!project.variableConfig.isVariableFont)
        #expect(project.variableConfig.axes.isEmpty)
    }

    @Test("FontProject with variable config")
    func projectWithVariableConfig() {
        let project = FontProject(
            name: "Variable Test",
            family: "Variable Test",
            style: "Regular",
            variableConfig: .weightOnly()
        )

        #expect(project.variableConfig.isVariableFont)
        #expect(project.variableConfig.axes.count == 1)
    }

    @Test("FontProject variable config is Codable")
    func projectVariableConfigCodable() throws {
        let original = FontProject(
            name: "Test",
            family: "Test",
            style: "Regular",
            variableConfig: .weightOnly()
        )

        let encoder = JSONEncoder()
        let data = try encoder.encode(original)

        let decoder = JSONDecoder()
        let decoded = try decoder.decode(FontProject.self, from: data)

        #expect(decoded.variableConfig.isVariableFont)
        #expect(decoded.variableConfig.axes.count == original.variableConfig.axes.count)
    }
}
