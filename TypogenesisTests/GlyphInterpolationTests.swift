import Foundation
import Testing
import CoreGraphics
@testable import Typogenesis

// MARK: - Test Helpers

/// Creates a simple square glyph with 4 corner points at known coordinates.
/// The square has its bottom-left at (leftX, bottomY) and extends to (rightX, topY).
private func createSquareGlyph(
    character: Character,
    leftX: CGFloat,
    bottomY: CGFloat,
    rightX: CGFloat,
    topY: CGFloat,
    advanceWidth: Int = 500,
    leftSideBearing: Int = 50,
    withControlPoints: Bool = false
) -> Glyph {
    var points: [PathPoint]

    if withControlPoints {
        // Create a square with bezier curves on top and bottom edges
        points = [
            PathPoint(
                position: CGPoint(x: leftX, y: bottomY),
                type: .smooth,
                controlIn: CGPoint(x: leftX - 20, y: bottomY),
                controlOut: CGPoint(x: leftX + 20, y: bottomY)
            ),
            PathPoint(
                position: CGPoint(x: rightX, y: bottomY),
                type: .smooth,
                controlIn: CGPoint(x: rightX - 20, y: bottomY),
                controlOut: CGPoint(x: rightX + 20, y: bottomY)
            ),
            PathPoint(
                position: CGPoint(x: rightX, y: topY),
                type: .smooth,
                controlIn: CGPoint(x: rightX + 20, y: topY),
                controlOut: CGPoint(x: rightX - 20, y: topY)
            ),
            PathPoint(
                position: CGPoint(x: leftX, y: topY),
                type: .smooth,
                controlIn: CGPoint(x: leftX + 20, y: topY),
                controlOut: CGPoint(x: leftX - 20, y: topY)
            )
        ]
    } else {
        // Simple corner points for a square
        points = [
            PathPoint(position: CGPoint(x: leftX, y: bottomY), type: .corner),
            PathPoint(position: CGPoint(x: rightX, y: bottomY), type: .corner),
            PathPoint(position: CGPoint(x: rightX, y: topY), type: .corner),
            PathPoint(position: CGPoint(x: leftX, y: topY), type: .corner)
        ]
    }

    let contour = Contour(points: points, isClosed: true)
    let outline = GlyphOutline(contours: [contour])

    return Glyph(
        character: character,
        outline: outline,
        advanceWidth: advanceWidth,
        leftSideBearing: leftSideBearing
    )
}

/// Creates a glyph with multiple contours (e.g., letter O with inner counter).
private func createRingGlyph(
    character: Character,
    outerRect: (left: CGFloat, bottom: CGFloat, right: CGFloat, top: CGFloat),
    innerRect: (left: CGFloat, bottom: CGFloat, right: CGFloat, top: CGFloat),
    advanceWidth: Int = 600,
    leftSideBearing: Int = 50
) -> Glyph {
    let outerPoints = [
        PathPoint(position: CGPoint(x: outerRect.left, y: outerRect.bottom), type: .corner),
        PathPoint(position: CGPoint(x: outerRect.right, y: outerRect.bottom), type: .corner),
        PathPoint(position: CGPoint(x: outerRect.right, y: outerRect.top), type: .corner),
        PathPoint(position: CGPoint(x: outerRect.left, y: outerRect.top), type: .corner)
    ]
    let outerContour = Contour(points: outerPoints, isClosed: true)

    let innerPoints = [
        PathPoint(position: CGPoint(x: innerRect.left, y: innerRect.bottom), type: .corner),
        PathPoint(position: CGPoint(x: innerRect.right, y: innerRect.bottom), type: .corner),
        PathPoint(position: CGPoint(x: innerRect.right, y: innerRect.top), type: .corner),
        PathPoint(position: CGPoint(x: innerRect.left, y: innerRect.top), type: .corner)
    ]
    let innerContour = Contour(points: innerPoints, isClosed: true)

    let outline = GlyphOutline(contours: [outerContour, innerContour])

    return Glyph(
        character: character,
        outline: outline,
        advanceWidth: advanceWidth,
        leftSideBearing: leftSideBearing
    )
}

/// Creates a FontMaster at the specified location with given glyphs.
private func createMaster(
    name: String,
    location: DesignSpaceLocation,
    glyphs: [Character: Glyph],
    metrics: FontMetrics = FontMetrics()
) -> FontMaster {
    FontMaster(
        name: name,
        location: location,
        glyphs: glyphs,
        metrics: metrics
    )
}

/// Creates a FontProject configured for variable font testing.
private func createVariableFontProject(
    axes: [VariationAxis],
    masters: [FontMaster],
    glyphs: [Character: Glyph] = [:]
) -> FontProject {
    let config = VariableFontConfig(
        isVariableFont: true,
        axes: axes,
        masters: masters,
        instances: []
    )

    return FontProject(
        name: "Test Variable Font",
        family: "Test",
        style: "Regular",
        glyphs: glyphs,
        variableConfig: config
    )
}

// MARK: - Basic Single-Axis Interpolation Tests

@Suite("Basic Single-Axis Interpolation")
struct SingleAxisInterpolationTests {

    @Test("Interpolate at midpoint produces halfway values")
    func interpolateAtMidpoint() async throws {
        // Create light master (wght=300) with square at (50, 0) - (250, 500)
        let lightGlyph = createSquareGlyph(
            character: "A",
            leftX: 50, bottomY: 0, rightX: 250, topY: 500,
            advanceWidth: 300, leftSideBearing: 50
        )
        let lightMaster = createMaster(
            name: "Light",
            location: [VariationAxis.weightTag: 300],
            glyphs: ["A": lightGlyph]
        )

        // Create bold master (wght=700) with square at (50, 0) - (450, 700)
        let boldGlyph = createSquareGlyph(
            character: "A",
            leftX: 50, bottomY: 0, rightX: 450, topY: 700,
            advanceWidth: 500, leftSideBearing: 50
        )
        let boldMaster = createMaster(
            name: "Bold",
            location: [VariationAxis.weightTag: 700],
            glyphs: ["A": boldGlyph]
        )

        let project = createVariableFontProject(
            axes: [VariationAxis.weight],
            masters: [lightMaster, boldMaster]
        )

        let interpolator = GlyphInterpolator()

        // Interpolate at wght=500 (exactly halfway)
        let result = try await interpolator.interpolate(
            character: "A",
            at: [VariationAxis.weightTag: 500],
            in: project
        )

        // Verify points are halfway between masters
        let points = result.outline.contours.first?.points ?? []
        #expect(points.count == 4)

        // First point should be at (50, 0) - same in both masters
        #expect(abs(points[0].position.x - 50) < 0.01)
        #expect(abs(points[0].position.y - 0) < 0.01)

        // Second point should be at (350, 0) - halfway between 250 and 450
        #expect(abs(points[1].position.x - 350) < 0.01)
        #expect(abs(points[1].position.y - 0) < 0.01)

        // Third point should be at (350, 600) - halfway between (250,500) and (450,700)
        #expect(abs(points[2].position.x - 350) < 0.01)
        #expect(abs(points[2].position.y - 600) < 0.01)

        // Fourth point should be at (50, 600) - halfway between (50,500) and (50,700)
        #expect(abs(points[3].position.x - 50) < 0.01)
        #expect(abs(points[3].position.y - 600) < 0.01)

        // Verify metrics are interpolated
        #expect(result.advanceWidth == 400)  // halfway between 300 and 500
        #expect(result.leftSideBearing == 50)  // same in both masters
    }

    @Test("Interpolate at 25% produces 25% blend")
    func interpolateAtQuarter() async throws {
        let lightGlyph = createSquareGlyph(
            character: "B",
            leftX: 0, bottomY: 0, rightX: 100, topY: 100,
            advanceWidth: 200, leftSideBearing: 10
        )
        let lightMaster = createMaster(
            name: "Light",
            location: [VariationAxis.weightTag: 300],
            glyphs: ["B": lightGlyph]
        )

        let boldGlyph = createSquareGlyph(
            character: "B",
            leftX: 0, bottomY: 0, rightX: 200, topY: 200,
            advanceWidth: 400, leftSideBearing: 20
        )
        let boldMaster = createMaster(
            name: "Bold",
            location: [VariationAxis.weightTag: 700],
            glyphs: ["B": boldGlyph]
        )

        let project = createVariableFontProject(
            axes: [VariationAxis.weight],
            masters: [lightMaster, boldMaster]
        )

        let interpolator = GlyphInterpolator()

        // Interpolate at wght=400 (25% of the way from 300 to 700)
        let result = try await interpolator.interpolate(
            character: "B",
            at: [VariationAxis.weightTag: 400],
            in: project
        )

        let points = result.outline.contours.first?.points ?? []

        // Point at index 1: x should be 100 + 0.25 * (200 - 100) = 125
        #expect(abs(points[1].position.x - 125) < 0.01)

        // Point at index 2: (125, 125)
        #expect(abs(points[2].position.x - 125) < 0.01)
        #expect(abs(points[2].position.y - 125) < 0.01)

        // Advance width: 200 + 0.25 * (400 - 200) = 250
        #expect(result.advanceWidth == 250)

        // Left side bearing: 10 + 0.25 * (20 - 10) = 12.5, rounded to 13
        #expect(result.leftSideBearing == 13)
    }

    @Test("Interpolate at 75% produces 75% blend")
    func interpolateAtThreeQuarters() async throws {
        let lightGlyph = createSquareGlyph(
            character: "C",
            leftX: 0, bottomY: 0, rightX: 100, topY: 100,
            advanceWidth: 200, leftSideBearing: 0
        )
        let lightMaster = createMaster(
            name: "Light",
            location: [VariationAxis.weightTag: 300],
            glyphs: ["C": lightGlyph]
        )

        let boldGlyph = createSquareGlyph(
            character: "C",
            leftX: 0, bottomY: 0, rightX: 200, topY: 200,
            advanceWidth: 400, leftSideBearing: 0
        )
        let boldMaster = createMaster(
            name: "Bold",
            location: [VariationAxis.weightTag: 700],
            glyphs: ["C": boldGlyph]
        )

        let project = createVariableFontProject(
            axes: [VariationAxis.weight],
            masters: [lightMaster, boldMaster]
        )

        let interpolator = GlyphInterpolator()

        // Interpolate at wght=600 (75% of the way from 300 to 700)
        let result = try await interpolator.interpolate(
            character: "C",
            at: [VariationAxis.weightTag: 600],
            in: project
        )

        let points = result.outline.contours.first?.points ?? []

        // Point at index 1: x should be 100 + 0.75 * (200 - 100) = 175
        #expect(abs(points[1].position.x - 175) < 0.01)

        // Advance width: 200 + 0.75 * (400 - 200) = 350
        #expect(result.advanceWidth == 350)
    }
}

// MARK: - Two-Axis Interpolation Tests

@Suite("Two-Axis Bilinear Interpolation")
struct TwoAxisInterpolationTests {

    @Test("Bilinear interpolation at center point")
    func interpolateAtCenter() async throws {
        // Create four corner masters for weight (300-700) x width (75-125)

        // Light Condensed (wght=300, wdth=75)
        let lightCondensedGlyph = createSquareGlyph(
            character: "A",
            leftX: 50, bottomY: 0, rightX: 150, topY: 500,  // narrow, short strokes
            advanceWidth: 200, leftSideBearing: 50
        )
        let lightCondensed = createMaster(
            name: "Light Condensed",
            location: [VariationAxis.weightTag: 300, VariationAxis.widthTag: 75],
            glyphs: ["A": lightCondensedGlyph]
        )

        // Light Expanded (wght=300, wdth=125)
        let lightExpandedGlyph = createSquareGlyph(
            character: "A",
            leftX: 50, bottomY: 0, rightX: 350, topY: 500,  // wide, short strokes
            advanceWidth: 400, leftSideBearing: 50
        )
        let lightExpanded = createMaster(
            name: "Light Expanded",
            location: [VariationAxis.weightTag: 300, VariationAxis.widthTag: 125],
            glyphs: ["A": lightExpandedGlyph]
        )

        // Bold Condensed (wght=700, wdth=75)
        let boldCondensedGlyph = createSquareGlyph(
            character: "A",
            leftX: 50, bottomY: 0, rightX: 250, topY: 700,  // narrow, tall strokes
            advanceWidth: 300, leftSideBearing: 50
        )
        let boldCondensed = createMaster(
            name: "Bold Condensed",
            location: [VariationAxis.weightTag: 700, VariationAxis.widthTag: 75],
            glyphs: ["A": boldCondensedGlyph]
        )

        // Bold Expanded (wght=700, wdth=125)
        let boldExpandedGlyph = createSquareGlyph(
            character: "A",
            leftX: 50, bottomY: 0, rightX: 450, topY: 700,  // wide, tall strokes
            advanceWidth: 500, leftSideBearing: 50
        )
        let boldExpanded = createMaster(
            name: "Bold Expanded",
            location: [VariationAxis.weightTag: 700, VariationAxis.widthTag: 125],
            glyphs: ["A": boldExpandedGlyph]
        )

        var widthAxis = VariationAxis.width
        widthAxis.minValue = 75
        widthAxis.maxValue = 125

        let project = createVariableFontProject(
            axes: [VariationAxis.weight, widthAxis],
            masters: [lightCondensed, lightExpanded, boldCondensed, boldExpanded]
        )

        let interpolator = GlyphInterpolator()

        // Interpolate at center (wght=500, wdth=100)
        let result = try await interpolator.interpolate(
            character: "A",
            at: [VariationAxis.weightTag: 500, VariationAxis.widthTag: 100],
            in: project
        )

        let points = result.outline.contours.first?.points ?? []
        #expect(points.count == 4)

        // At center point, all four masters contribute equally (0.25 each)
        // Right edge X values: 150, 350, 250, 450 -> average = 300
        #expect(abs(points[1].position.x - 300) < 1.0)

        // Top Y values: 500, 500, 700, 700 -> average = 600
        #expect(abs(points[2].position.y - 600) < 1.0)

        // Advance widths: 200, 400, 300, 500 -> average = 350
        #expect(result.advanceWidth == 350)
    }

    @Test("Bilinear interpolation at corner produces master glyph")
    func interpolateAtCorner() async throws {
        let cornerGlyph = createSquareGlyph(
            character: "X",
            leftX: 100, bottomY: 100, rightX: 300, topY: 400,
            advanceWidth: 400, leftSideBearing: 100
        )
        let cornerMaster = createMaster(
            name: "Corner",
            location: [VariationAxis.weightTag: 700, VariationAxis.widthTag: 125],
            glyphs: ["X": cornerGlyph]
        )

        // Create other corners with different values
        let otherGlyph = createSquareGlyph(
            character: "X",
            leftX: 0, bottomY: 0, rightX: 100, topY: 100,
            advanceWidth: 150, leftSideBearing: 0
        )

        let otherMasters = [
            createMaster(name: "M1", location: [VariationAxis.weightTag: 300, VariationAxis.widthTag: 75], glyphs: ["X": otherGlyph]),
            createMaster(name: "M2", location: [VariationAxis.weightTag: 300, VariationAxis.widthTag: 125], glyphs: ["X": otherGlyph]),
            createMaster(name: "M3", location: [VariationAxis.weightTag: 700, VariationAxis.widthTag: 75], glyphs: ["X": otherGlyph])
        ]

        var widthAxis = VariationAxis.width
        widthAxis.minValue = 75
        widthAxis.maxValue = 125

        let project = createVariableFontProject(
            axes: [VariationAxis.weight, widthAxis],
            masters: otherMasters + [cornerMaster]
        )

        let interpolator = GlyphInterpolator()

        // Interpolate exactly at the corner master's location
        let result = try await interpolator.interpolate(
            character: "X",
            at: [VariationAxis.weightTag: 700, VariationAxis.widthTag: 125],
            in: project
        )

        // Should return the exact corner master's glyph
        let points = result.outline.contours.first?.points ?? []
        #expect(abs(points[0].position.x - 100) < 0.01)
        #expect(abs(points[0].position.y - 100) < 0.01)
        #expect(abs(points[1].position.x - 300) < 0.01)
        #expect(result.advanceWidth == 400)
        #expect(result.leftSideBearing == 100)
    }
}

// MARK: - Interpolation at Master Location Tests

@Suite("Interpolation at Master Location")
struct MasterLocationInterpolationTests {

    @Test("Exact master location returns master glyph unchanged")
    func exactMasterLocation() async throws {
        let masterGlyph = createSquareGlyph(
            character: "M",
            leftX: 75, bottomY: 25, rightX: 325, topY: 575,
            advanceWidth: 400, leftSideBearing: 75
        )
        let master = createMaster(
            name: "Regular",
            location: [VariationAxis.weightTag: 400],
            glyphs: ["M": masterGlyph]
        )

        // Add another master so interpolation is possible
        let otherGlyph = createSquareGlyph(
            character: "M",
            leftX: 0, bottomY: 0, rightX: 500, topY: 700,
            advanceWidth: 600, leftSideBearing: 0
        )
        let otherMaster = createMaster(
            name: "Bold",
            location: [VariationAxis.weightTag: 700],
            glyphs: ["M": otherGlyph]
        )

        let project = createVariableFontProject(
            axes: [VariationAxis.weight],
            masters: [master, otherMaster]
        )

        let interpolator = GlyphInterpolator()

        // Request exactly at the master location
        let result = try await interpolator.interpolate(
            character: "M",
            at: [VariationAxis.weightTag: 400],
            in: project
        )

        // Should be identical to the master glyph
        let points = result.outline.contours.first?.points ?? []
        #expect(abs(points[0].position.x - 75) < 0.01)
        #expect(abs(points[0].position.y - 25) < 0.01)
        #expect(abs(points[1].position.x - 325) < 0.01)
        #expect(abs(points[2].position.y - 575) < 0.01)
        #expect(result.advanceWidth == 400)
        #expect(result.leftSideBearing == 75)
    }

    @Test("Very close to master location returns near-identical glyph")
    func nearMasterLocation() async throws {
        let masterGlyph = createSquareGlyph(
            character: "N",
            leftX: 100, bottomY: 0, rightX: 400, topY: 600,
            advanceWidth: 500, leftSideBearing: 100
        )
        let master = createMaster(
            name: "Regular",
            location: [VariationAxis.weightTag: 400],
            glyphs: ["N": masterGlyph]
        )

        let boldGlyph = createSquareGlyph(
            character: "N",
            leftX: 100, bottomY: 0, rightX: 500, topY: 700,
            advanceWidth: 600, leftSideBearing: 100
        )
        let boldMaster = createMaster(
            name: "Bold",
            location: [VariationAxis.weightTag: 700],
            glyphs: ["N": boldGlyph]
        )

        let project = createVariableFontProject(
            axes: [VariationAxis.weight],
            masters: [master, boldMaster]
        )

        let interpolator = GlyphInterpolator()

        // Request very close to master (0.1% interpolation)
        let result = try await interpolator.interpolate(
            character: "N",
            at: [VariationAxis.weightTag: 400.3],  // 0.1% toward bold
            in: project
        )

        let points = result.outline.contours.first?.points ?? []

        // Should be very close to the master values
        #expect(abs(points[1].position.x - 400) < 1.0)  // Within 1 unit of master
        #expect(abs(points[2].position.y - 600) < 1.0)
    }
}

// MARK: - Edge Cases Tests

@Suite("Edge Cases")
struct EdgeCaseTests {

    @Test("Location outside axis range should clamp")
    func locationOutsideRange() async throws {
        let lightGlyph = createSquareGlyph(
            character: "E",
            leftX: 0, bottomY: 0, rightX: 100, topY: 100,
            advanceWidth: 200, leftSideBearing: 0
        )
        let lightMaster = createMaster(
            name: "Light",
            location: [VariationAxis.weightTag: 300],
            glyphs: ["E": lightGlyph]
        )

        let boldGlyph = createSquareGlyph(
            character: "E",
            leftX: 0, bottomY: 0, rightX: 200, topY: 200,
            advanceWidth: 400, leftSideBearing: 0
        )
        let boldMaster = createMaster(
            name: "Bold",
            location: [VariationAxis.weightTag: 700],
            glyphs: ["E": boldGlyph]
        )

        // Weight axis with range 100-900
        let weightAxis = VariationAxis.weight

        let project = createVariableFontProject(
            axes: [weightAxis],
            masters: [lightMaster, boldMaster]
        )

        let interpolator = GlyphInterpolator()

        // Request below minimum (wght=50, should clamp to 100 which is below our masters)
        // The interpolator should throw an error for out-of-range values
        await #expect(throws: GlyphInterpolator.InterpolationError.self) {
            _ = try await interpolator.interpolate(
                character: "E",
                at: [VariationAxis.weightTag: 50],  // Below axis minimum of 100
                in: project
            )
        }

        // Request above maximum (wght=1000, should clamp to 900 which is above our masters)
        await #expect(throws: GlyphInterpolator.InterpolationError.self) {
            _ = try await interpolator.interpolate(
                character: "E",
                at: [VariationAxis.weightTag: 1000],  // Above axis maximum of 900
                in: project
            )
        }
    }

    @Test("Glyph missing from one master throws appropriate error")
    func glyphMissingFromMaster() async throws {
        let glyphA = createSquareGlyph(
            character: "A",
            leftX: 0, bottomY: 0, rightX: 100, topY: 100,
            advanceWidth: 200, leftSideBearing: 0
        )

        let lightMaster = createMaster(
            name: "Light",
            location: [VariationAxis.weightTag: 300],
            glyphs: ["A": glyphA]  // Has glyph A
        )

        // Bold master doesn't have glyph A
        let boldMaster = createMaster(
            name: "Bold",
            location: [VariationAxis.weightTag: 700],
            glyphs: [:]  // Empty - no glyph A
        )

        let project = createVariableFontProject(
            axes: [VariationAxis.weight],
            masters: [lightMaster, boldMaster]
        )

        let interpolator = GlyphInterpolator()

        // Should throw because glyph A is missing from bold master
        // and we need at least 2 masters for interpolation
        await #expect(throws: GlyphInterpolator.InterpolationError.self) {
            _ = try await interpolator.interpolate(
                character: "A",
                at: [VariationAxis.weightTag: 500],
                in: project
            )
        }
    }

    @Test("Glyph not found in any master throws error")
    func glyphNotFoundInAnyMaster() async throws {
        let glyphA = createSquareGlyph(
            character: "A",
            leftX: 0, bottomY: 0, rightX: 100, topY: 100,
            advanceWidth: 200, leftSideBearing: 0
        )

        let masters = [
            createMaster(name: "Light", location: [VariationAxis.weightTag: 300], glyphs: ["A": glyphA]),
            createMaster(name: "Bold", location: [VariationAxis.weightTag: 700], glyphs: ["A": glyphA])
        ]

        let project = createVariableFontProject(
            axes: [VariationAxis.weight],
            masters: masters
        )

        let interpolator = GlyphInterpolator()

        // Request a glyph that doesn't exist in any master
        await #expect(throws: GlyphInterpolator.InterpolationError.self) {
            _ = try await interpolator.interpolate(
                character: "Z",  // Not in any master
                at: [VariationAxis.weightTag: 500],
                in: project
            )
        }
    }

    @Test("Incompatible glyph structure across masters throws error")
    func incompatibleGlyphStructure() async throws {
        // Light master has simple square (1 contour, 4 points)
        let simpleGlyph = createSquareGlyph(
            character: "O",
            leftX: 0, bottomY: 0, rightX: 100, topY: 100,
            advanceWidth: 200, leftSideBearing: 0
        )
        let lightMaster = createMaster(
            name: "Light",
            location: [VariationAxis.weightTag: 300],
            glyphs: ["O": simpleGlyph]
        )

        // Bold master has ring (2 contours)
        let ringGlyph = createRingGlyph(
            character: "O",
            outerRect: (0, 0, 200, 200),
            innerRect: (50, 50, 150, 150),
            advanceWidth: 300, leftSideBearing: 0
        )
        let boldMaster = createMaster(
            name: "Bold",
            location: [VariationAxis.weightTag: 700],
            glyphs: ["O": ringGlyph]
        )

        let project = createVariableFontProject(
            axes: [VariationAxis.weight],
            masters: [lightMaster, boldMaster]
        )

        let interpolator = GlyphInterpolator()

        // Should throw because contour counts don't match
        await #expect(throws: GlyphInterpolator.InterpolationError.self) {
            _ = try await interpolator.interpolate(
                character: "O",
                at: [VariationAxis.weightTag: 500],
                in: project
            )
        }
    }

    @Test("Different point counts per contour throws error")
    func differentPointCounts() async throws {
        // Light master has 4 points
        let fourPointGlyph = createSquareGlyph(
            character: "P",
            leftX: 0, bottomY: 0, rightX: 100, topY: 100,
            advanceWidth: 200, leftSideBearing: 0
        )
        let lightMaster = createMaster(
            name: "Light",
            location: [VariationAxis.weightTag: 300],
            glyphs: ["P": fourPointGlyph]
        )

        // Bold master has 5 points in the contour
        let fivePointContour = Contour(
            points: [
                PathPoint(position: CGPoint(x: 0, y: 0), type: .corner),
                PathPoint(position: CGPoint(x: 100, y: 0), type: .corner),
                PathPoint(position: CGPoint(x: 150, y: 50), type: .corner),  // Extra point
                PathPoint(position: CGPoint(x: 100, y: 100), type: .corner),
                PathPoint(position: CGPoint(x: 0, y: 100), type: .corner)
            ],
            isClosed: true
        )
        let fivePointGlyph = Glyph(
            character: "P",
            outline: GlyphOutline(contours: [fivePointContour]),
            advanceWidth: 250,
            leftSideBearing: 0
        )
        let boldMaster = createMaster(
            name: "Bold",
            location: [VariationAxis.weightTag: 700],
            glyphs: ["P": fivePointGlyph]
        )

        let project = createVariableFontProject(
            axes: [VariationAxis.weight],
            masters: [lightMaster, boldMaster]
        )

        let interpolator = GlyphInterpolator()

        // Should throw because point counts don't match
        await #expect(throws: GlyphInterpolator.InterpolationError.self) {
            _ = try await interpolator.interpolate(
                character: "P",
                at: [VariationAxis.weightTag: 500],
                in: project
            )
        }
    }

    @Test("No masters defined throws error")
    func noMastersDefined() async throws {
        let config = VariableFontConfig(
            isVariableFont: true,
            axes: [VariationAxis.weight],
            masters: [],  // No masters
            instances: []
        )

        let project = FontProject(
            name: "Empty",
            family: "Empty",
            style: "Regular",
            variableConfig: config
        )

        let interpolator = GlyphInterpolator()

        await #expect(throws: GlyphInterpolator.InterpolationError.self) {
            _ = try await interpolator.interpolate(
                character: "A",
                at: [VariationAxis.weightTag: 500],
                in: project
            )
        }
    }

    @Test("No axes defined throws error")
    func noAxesDefined() async throws {
        let glyph = createSquareGlyph(character: "A", leftX: 0, bottomY: 0, rightX: 100, topY: 100)
        let master = createMaster(name: "Only", location: [:], glyphs: ["A": glyph])

        let config = VariableFontConfig(
            isVariableFont: true,
            axes: [],  // No axes
            masters: [master],
            instances: []
        )

        let project = FontProject(
            name: "NoAxes",
            family: "NoAxes",
            style: "Regular",
            variableConfig: config
        )

        let interpolator = GlyphInterpolator()

        await #expect(throws: GlyphInterpolator.InterpolationError.self) {
            _ = try await interpolator.interpolate(
                character: "A",
                at: [VariationAxis.weightTag: 500],
                in: project
            )
        }
    }
}

// MARK: - Metrics Interpolation Tests

@Suite("Metrics Interpolation")
struct MetricsInterpolationTests {

    @Test("AdvanceWidth interpolates correctly")
    func advanceWidthInterpolation() async throws {
        let lightGlyph = createSquareGlyph(
            character: "W",
            leftX: 0, bottomY: 0, rightX: 100, topY: 100,
            advanceWidth: 200,  // Light: 200
            leftSideBearing: 20
        )
        let lightMaster = createMaster(
            name: "Light",
            location: [VariationAxis.weightTag: 300],
            glyphs: ["W": lightGlyph]
        )

        let boldGlyph = createSquareGlyph(
            character: "W",
            leftX: 0, bottomY: 0, rightX: 200, topY: 200,
            advanceWidth: 600,  // Bold: 600
            leftSideBearing: 40
        )
        let boldMaster = createMaster(
            name: "Bold",
            location: [VariationAxis.weightTag: 700],
            glyphs: ["W": boldGlyph]
        )

        let project = createVariableFontProject(
            axes: [VariationAxis.weight],
            masters: [lightMaster, boldMaster]
        )

        let interpolator = GlyphInterpolator()

        // At midpoint, advance width should be 400
        let midResult = try await interpolator.interpolate(
            character: "W",
            at: [VariationAxis.weightTag: 500],
            in: project
        )
        #expect(midResult.advanceWidth == 400)

        // At 25%, advance width should be 300
        let quarterResult = try await interpolator.interpolate(
            character: "W",
            at: [VariationAxis.weightTag: 400],
            in: project
        )
        #expect(quarterResult.advanceWidth == 300)

        // At 75%, advance width should be 500
        let threeQuarterResult = try await interpolator.interpolate(
            character: "W",
            at: [VariationAxis.weightTag: 600],
            in: project
        )
        #expect(threeQuarterResult.advanceWidth == 500)
    }

    @Test("LeftSideBearing interpolates correctly")
    func leftSideBearingInterpolation() async throws {
        let lightGlyph = createSquareGlyph(
            character: "L",
            leftX: 20, bottomY: 0, rightX: 100, topY: 100,
            advanceWidth: 150,
            leftSideBearing: 20  // Light: 20
        )
        let lightMaster = createMaster(
            name: "Light",
            location: [VariationAxis.weightTag: 300],
            glyphs: ["L": lightGlyph]
        )

        let boldGlyph = createSquareGlyph(
            character: "L",
            leftX: 60, bottomY: 0, rightX: 200, topY: 200,
            advanceWidth: 280,
            leftSideBearing: 60  // Bold: 60
        )
        let boldMaster = createMaster(
            name: "Bold",
            location: [VariationAxis.weightTag: 700],
            glyphs: ["L": boldGlyph]
        )

        let project = createVariableFontProject(
            axes: [VariationAxis.weight],
            masters: [lightMaster, boldMaster]
        )

        let interpolator = GlyphInterpolator()

        // At midpoint, LSB should be 40
        let result = try await interpolator.interpolate(
            character: "L",
            at: [VariationAxis.weightTag: 500],
            in: project
        )
        #expect(result.leftSideBearing == 40)
    }

    @Test("FontMetrics interpolate correctly")
    func fontMetricsInterpolation() async throws {
        let lightMetrics = FontMetrics(
            unitsPerEm: 1000,
            ascender: 700,
            descender: -200,
            xHeight: 450,
            capHeight: 650,
            lineGap: 80
        )

        let boldMetrics = FontMetrics(
            unitsPerEm: 1000,
            ascender: 800,
            descender: -250,
            xHeight: 500,
            capHeight: 750,
            lineGap: 100
        )

        let lightGlyph = createSquareGlyph(character: "T", leftX: 0, bottomY: 0, rightX: 100, topY: 100)
        let boldGlyph = createSquareGlyph(character: "T", leftX: 0, bottomY: 0, rightX: 200, topY: 200)

        let lightMaster = createMaster(
            name: "Light",
            location: [VariationAxis.weightTag: 300],
            glyphs: ["T": lightGlyph],
            metrics: lightMetrics
        )

        let boldMaster = createMaster(
            name: "Bold",
            location: [VariationAxis.weightTag: 700],
            glyphs: ["T": boldGlyph],
            metrics: boldMetrics
        )

        let project = createVariableFontProject(
            axes: [VariationAxis.weight],
            masters: [lightMaster, boldMaster]
        )

        let interpolator = GlyphInterpolator()

        // At midpoint
        let result = try await interpolator.interpolateMetrics(
            at: [VariationAxis.weightTag: 500],
            in: project
        )

        // unitsPerEm should remain 1000 (same in both)
        #expect(result.unitsPerEm == 1000)

        // ascender: (700 + 800) / 2 = 750
        #expect(result.ascender == 750)

        // descender: (-200 + -250) / 2 = -225
        #expect(result.descender == -225)

        // xHeight: (450 + 500) / 2 = 475
        #expect(result.xHeight == 475)

        // capHeight: (650 + 750) / 2 = 700
        #expect(result.capHeight == 700)

        // lineGap: (80 + 100) / 2 = 90
        #expect(result.lineGap == 90)
    }
}

// MARK: - Control Handle Interpolation Tests

@Suite("Control Handle Interpolation")
struct ControlHandleInterpolationTests {

    @Test("Bezier control points interpolate correctly")
    func controlPointInterpolation() async throws {
        // Light master with control points
        let lightGlyph = createSquareGlyph(
            character: "S",
            leftX: 50, bottomY: 0, rightX: 150, topY: 200,
            advanceWidth: 200, leftSideBearing: 50,
            withControlPoints: true
        )
        let lightMaster = createMaster(
            name: "Light",
            location: [VariationAxis.weightTag: 300],
            glyphs: ["S": lightGlyph]
        )

        // Bold master with control points (scaled up)
        let boldGlyph = createSquareGlyph(
            character: "S",
            leftX: 50, bottomY: 0, rightX: 250, topY: 400,
            advanceWidth: 300, leftSideBearing: 50,
            withControlPoints: true
        )
        let boldMaster = createMaster(
            name: "Bold",
            location: [VariationAxis.weightTag: 700],
            glyphs: ["S": boldGlyph]
        )

        let project = createVariableFontProject(
            axes: [VariationAxis.weight],
            masters: [lightMaster, boldMaster]
        )

        let interpolator = GlyphInterpolator()

        let result = try await interpolator.interpolate(
            character: "S",
            at: [VariationAxis.weightTag: 500],
            in: project
        )

        let points = result.outline.contours.first?.points ?? []
        #expect(points.count == 4)

        // All points should have control handles
        for point in points {
            #expect(point.controlIn != nil)
            #expect(point.controlOut != nil)
        }

        // First point controlOut should be interpolated
        // Light: (50 + 20, 0) = (70, 0)
        // Bold: (50 + 20, 0) = (70, 0) - same x offset but different base
        // At midpoint, the control point should be at interpolated position
        let firstPoint = points[0]
        #expect(firstPoint.controlOut != nil)

        // The positions should be reasonably between the masters
        let expectedX = (150.0 + 250.0) / 2  // 200
        #expect(abs(points[1].position.x - expectedX) < 1.0)
    }

    @Test("Mixed control points handled correctly")
    func mixedControlPoints() async throws {
        // Light master has control points
        let lightGlyph = createSquareGlyph(
            character: "Q",
            leftX: 0, bottomY: 0, rightX: 100, topY: 100,
            advanceWidth: 150, leftSideBearing: 0,
            withControlPoints: true
        )
        let lightMaster = createMaster(
            name: "Light",
            location: [VariationAxis.weightTag: 300],
            glyphs: ["Q": lightGlyph]
        )

        // Bold master has NO control points (corners)
        let boldGlyph = createSquareGlyph(
            character: "Q",
            leftX: 0, bottomY: 0, rightX: 200, topY: 200,
            advanceWidth: 250, leftSideBearing: 0,
            withControlPoints: false
        )
        let boldMaster = createMaster(
            name: "Bold",
            location: [VariationAxis.weightTag: 700],
            glyphs: ["Q": boldGlyph]
        )

        let project = createVariableFontProject(
            axes: [VariationAxis.weight],
            masters: [lightMaster, boldMaster]
        )

        let interpolator = GlyphInterpolator()

        // This should still work - the interpolator handles mixed control point presence
        let result = try await interpolator.interpolate(
            character: "Q",
            at: [VariationAxis.weightTag: 500],
            in: project
        )

        let points = result.outline.contours.first?.points ?? []
        #expect(points.count == 4)

        // Positions should still be interpolated correctly
        let expectedX = (100.0 + 200.0) / 2  // 150
        #expect(abs(points[1].position.x - expectedX) < 1.0)
    }

    @Test("Control point deltas interpolate proportionally")
    func controlPointDeltas() async throws {
        // Create glyphs where control points move differently than main points
        let lightPoints = [
            PathPoint(
                position: CGPoint(x: 0, y: 0),
                type: .smooth,
                controlIn: CGPoint(x: -10, y: 0),
                controlOut: CGPoint(x: 10, y: 0)
            ),
            PathPoint(
                position: CGPoint(x: 100, y: 0),
                type: .smooth,
                controlIn: CGPoint(x: 90, y: 0),
                controlOut: CGPoint(x: 110, y: 0)
            ),
            PathPoint(
                position: CGPoint(x: 100, y: 100),
                type: .smooth,
                controlIn: CGPoint(x: 110, y: 100),
                controlOut: CGPoint(x: 90, y: 100)
            ),
            PathPoint(
                position: CGPoint(x: 0, y: 100),
                type: .smooth,
                controlIn: CGPoint(x: 10, y: 100),
                controlOut: CGPoint(x: -10, y: 100)
            )
        ]
        let lightContour = Contour(points: lightPoints, isClosed: true)
        let lightGlyph = Glyph(
            character: "R",
            outline: GlyphOutline(contours: [lightContour]),
            advanceWidth: 150,
            leftSideBearing: 0
        )
        let lightMaster = createMaster(
            name: "Light",
            location: [VariationAxis.weightTag: 300],
            glyphs: ["R": lightGlyph]
        )

        // Bold has larger control point offsets
        let boldPoints = [
            PathPoint(
                position: CGPoint(x: 0, y: 0),
                type: .smooth,
                controlIn: CGPoint(x: -30, y: 0),
                controlOut: CGPoint(x: 30, y: 0)
            ),
            PathPoint(
                position: CGPoint(x: 200, y: 0),
                type: .smooth,
                controlIn: CGPoint(x: 170, y: 0),
                controlOut: CGPoint(x: 230, y: 0)
            ),
            PathPoint(
                position: CGPoint(x: 200, y: 200),
                type: .smooth,
                controlIn: CGPoint(x: 230, y: 200),
                controlOut: CGPoint(x: 170, y: 200)
            ),
            PathPoint(
                position: CGPoint(x: 0, y: 200),
                type: .smooth,
                controlIn: CGPoint(x: 30, y: 200),
                controlOut: CGPoint(x: -30, y: 200)
            )
        ]
        let boldContour = Contour(points: boldPoints, isClosed: true)
        let boldGlyph = Glyph(
            character: "R",
            outline: GlyphOutline(contours: [boldContour]),
            advanceWidth: 250,
            leftSideBearing: 0
        )
        let boldMaster = createMaster(
            name: "Bold",
            location: [VariationAxis.weightTag: 700],
            glyphs: ["R": boldGlyph]
        )

        let project = createVariableFontProject(
            axes: [VariationAxis.weight],
            masters: [lightMaster, boldMaster]
        )

        let interpolator = GlyphInterpolator()

        let result = try await interpolator.interpolate(
            character: "R",
            at: [VariationAxis.weightTag: 500],
            in: project
        )

        let points = result.outline.contours.first?.points ?? []

        // First point controlOut: interpolate between 10 and 30 = 20
        let firstControlOut = points[0].controlOut
        #expect(firstControlOut != nil)
        #expect(abs((firstControlOut?.x ?? 0) - 20) < 1.0)

        // Second point position: interpolate between 100 and 200 = 150
        #expect(abs(points[1].position.x - 150) < 1.0)

        // Second point controlIn: interpolate between 90 and 170 = 130
        let secondControlIn = points[1].controlIn
        #expect(secondControlIn != nil)
        #expect(abs((secondControlIn?.x ?? 0) - 130) < 1.0)
    }
}

// MARK: - Multi-Contour Tests

@Suite("Multi-Contour Interpolation")
struct MultiContourInterpolationTests {

    @Test("Glyph with multiple contours interpolates all contours")
    func multiContourInterpolation() async throws {
        // Light master: ring with smaller dimensions
        let lightGlyph = createRingGlyph(
            character: "O",
            outerRect: (50, 0, 150, 200),
            innerRect: (75, 25, 125, 175),
            advanceWidth: 200,
            leftSideBearing: 50
        )
        let lightMaster = createMaster(
            name: "Light",
            location: [VariationAxis.weightTag: 300],
            glyphs: ["O": lightGlyph]
        )

        // Bold master: ring with larger dimensions
        let boldGlyph = createRingGlyph(
            character: "O",
            outerRect: (50, 0, 250, 400),
            innerRect: (100, 50, 200, 350),
            advanceWidth: 300,
            leftSideBearing: 50
        )
        let boldMaster = createMaster(
            name: "Bold",
            location: [VariationAxis.weightTag: 700],
            glyphs: ["O": boldGlyph]
        )

        let project = createVariableFontProject(
            axes: [VariationAxis.weight],
            masters: [lightMaster, boldMaster]
        )

        let interpolator = GlyphInterpolator()

        let result = try await interpolator.interpolate(
            character: "O",
            at: [VariationAxis.weightTag: 500],
            in: project
        )

        // Should have 2 contours
        #expect(result.outline.contours.count == 2)

        // Outer contour
        let outerContour = result.outline.contours[0]
        #expect(outerContour.points.count == 4)

        // Outer right edge: (150 + 250) / 2 = 200
        #expect(abs(outerContour.points[1].position.x - 200) < 1.0)

        // Outer top: (200 + 400) / 2 = 300
        #expect(abs(outerContour.points[2].position.y - 300) < 1.0)

        // Inner contour
        let innerContour = result.outline.contours[1]
        #expect(innerContour.points.count == 4)

        // Inner right edge: (125 + 200) / 2 = 162.5
        #expect(abs(innerContour.points[1].position.x - 162.5) < 1.0)
    }
}

// MARK: - Interpolation Consistency Tests

@Suite("Interpolation Consistency")
struct InterpolationConsistencyTests {

    @Test("Interpolation is deterministic")
    func deterministicInterpolation() async throws {
        let lightGlyph = createSquareGlyph(
            character: "D",
            leftX: 0, bottomY: 0, rightX: 100, topY: 100,
            advanceWidth: 200, leftSideBearing: 0
        )
        let boldGlyph = createSquareGlyph(
            character: "D",
            leftX: 0, bottomY: 0, rightX: 200, topY: 200,
            advanceWidth: 400, leftSideBearing: 0
        )

        let masters = [
            createMaster(name: "Light", location: [VariationAxis.weightTag: 300], glyphs: ["D": lightGlyph]),
            createMaster(name: "Bold", location: [VariationAxis.weightTag: 700], glyphs: ["D": boldGlyph])
        ]

        let project = createVariableFontProject(
            axes: [VariationAxis.weight],
            masters: masters
        )

        let interpolator = GlyphInterpolator()
        let location: DesignSpaceLocation = [VariationAxis.weightTag: 500]

        // Run interpolation multiple times
        let result1 = try await interpolator.interpolate(character: "D", at: location, in: project)
        let result2 = try await interpolator.interpolate(character: "D", at: location, in: project)
        let result3 = try await interpolator.interpolate(character: "D", at: location, in: project)

        // All results should be identical
        #expect(result1.advanceWidth == result2.advanceWidth)
        #expect(result2.advanceWidth == result3.advanceWidth)

        let points1 = result1.outline.contours.first?.points ?? []
        let points2 = result2.outline.contours.first?.points ?? []
        let points3 = result3.outline.contours.first?.points ?? []

        for i in 0..<points1.count {
            #expect(abs(points1[i].position.x - points2[i].position.x) < 0.001)
            #expect(abs(points2[i].position.x - points3[i].position.x) < 0.001)
        }
    }

    @Test("Interpolation preserves contour closure")
    func preserveContourClosure() async throws {
        let closedContour = Contour(
            points: [
                PathPoint(position: CGPoint(x: 0, y: 0), type: .corner),
                PathPoint(position: CGPoint(x: 100, y: 0), type: .corner),
                PathPoint(position: CGPoint(x: 100, y: 100), type: .corner),
                PathPoint(position: CGPoint(x: 0, y: 100), type: .corner)
            ],
            isClosed: true
        )
        let closedGlyph = Glyph(
            character: "C",
            outline: GlyphOutline(contours: [closedContour]),
            advanceWidth: 150, leftSideBearing: 0
        )

        let openContour = Contour(
            points: [
                PathPoint(position: CGPoint(x: 0, y: 0), type: .corner),
                PathPoint(position: CGPoint(x: 200, y: 0), type: .corner),
                PathPoint(position: CGPoint(x: 200, y: 200), type: .corner),
                PathPoint(position: CGPoint(x: 0, y: 200), type: .corner)
            ],
            isClosed: false  // Open contour
        )
        let openGlyph = Glyph(
            character: "C",
            outline: GlyphOutline(contours: [openContour]),
            advanceWidth: 250, leftSideBearing: 0
        )

        let masters = [
            createMaster(name: "Light", location: [VariationAxis.weightTag: 300], glyphs: ["C": closedGlyph]),
            createMaster(name: "Bold", location: [VariationAxis.weightTag: 700], glyphs: ["C": openGlyph])
        ]

        let project = createVariableFontProject(
            axes: [VariationAxis.weight],
            masters: masters
        )

        let interpolator = GlyphInterpolator()

        let result = try await interpolator.interpolate(
            character: "C",
            at: [VariationAxis.weightTag: 500],
            in: project
        )

        // Result should have isClosed = true (if either master is closed, result is closed)
        #expect(result.outline.contours.first?.isClosed == true)
    }

    @Test("Interpolation preserves point types")
    func preservePointTypes() async throws {
        let mixedPoints = [
            PathPoint(position: CGPoint(x: 0, y: 0), type: .corner),
            PathPoint(position: CGPoint(x: 50, y: 0), type: .smooth, controlIn: CGPoint(x: 40, y: 0), controlOut: CGPoint(x: 60, y: 0)),
            PathPoint(position: CGPoint(x: 100, y: 50), type: .symmetric, controlIn: CGPoint(x: 100, y: 40), controlOut: CGPoint(x: 100, y: 60)),
            PathPoint(position: CGPoint(x: 50, y: 100), type: .corner)
        ]
        let lightContour = Contour(points: mixedPoints, isClosed: true)
        let lightGlyph = Glyph(
            character: "Y",
            outline: GlyphOutline(contours: [lightContour]),
            advanceWidth: 150, leftSideBearing: 0
        )

        let boldPoints = mixedPoints.map { point in
            PathPoint(
                position: CGPoint(x: point.position.x * 2, y: point.position.y * 2),
                type: point.type,  // Same type
                controlIn: point.controlIn.map { CGPoint(x: $0.x * 2, y: $0.y * 2) },
                controlOut: point.controlOut.map { CGPoint(x: $0.x * 2, y: $0.y * 2) }
            )
        }
        let boldContour = Contour(points: boldPoints, isClosed: true)
        let boldGlyph = Glyph(
            character: "Y",
            outline: GlyphOutline(contours: [boldContour]),
            advanceWidth: 300, leftSideBearing: 0
        )

        let masters = [
            createMaster(name: "Light", location: [VariationAxis.weightTag: 300], glyphs: ["Y": lightGlyph]),
            createMaster(name: "Bold", location: [VariationAxis.weightTag: 700], glyphs: ["Y": boldGlyph])
        ]

        let project = createVariableFontProject(
            axes: [VariationAxis.weight],
            masters: masters
        )

        let interpolator = GlyphInterpolator()

        let result = try await interpolator.interpolate(
            character: "Y",
            at: [VariationAxis.weightTag: 500],
            in: project
        )

        let resultPoints = result.outline.contours.first?.points ?? []

        // Point types should be preserved
        #expect(resultPoints[0].type == .corner)
        #expect(resultPoints[1].type == .smooth)
        #expect(resultPoints[2].type == .symmetric)
        #expect(resultPoints[3].type == .corner)
    }
}
