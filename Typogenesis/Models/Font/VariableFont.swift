import Foundation

/// Represents a variation axis in a variable font
struct VariationAxis: Identifiable, Codable, Sendable, Equatable {
    let id: UUID
    var tag: String           // 4-character tag (e.g., "wght", "wdth")
    var name: String          // Human-readable name
    var minValue: CGFloat     // Minimum axis value
    var defaultValue: CGFloat // Default axis value
    var maxValue: CGFloat     // Maximum axis value

    // Standard axis tags
    static let weightTag = "wght"
    static let widthTag = "wdth"
    static let slantTag = "slnt"
    static let italicTag = "ital"
    static let opticalSizeTag = "opsz"

    init(
        id: UUID = UUID(),
        tag: String,
        name: String,
        minValue: CGFloat,
        defaultValue: CGFloat,
        maxValue: CGFloat
    ) {
        self.id = id
        self.tag = tag
        self.name = name
        self.minValue = minValue
        self.defaultValue = defaultValue
        self.maxValue = maxValue
    }

    /// Standard weight axis (100-900)
    static var weight: VariationAxis {
        VariationAxis(
            tag: weightTag,
            name: "Weight",
            minValue: 100,
            defaultValue: 400,
            maxValue: 900
        )
    }

    /// Standard width axis (50-200)
    static var width: VariationAxis {
        VariationAxis(
            tag: widthTag,
            name: "Width",
            minValue: 50,
            defaultValue: 100,
            maxValue: 200
        )
    }

    /// Slant axis (-20 to 20 degrees)
    static var slant: VariationAxis {
        VariationAxis(
            tag: slantTag,
            name: "Slant",
            minValue: -20,
            defaultValue: 0,
            maxValue: 20
        )
    }

    /// Italic axis (0 or 1)
    static var italic: VariationAxis {
        VariationAxis(
            tag: italicTag,
            name: "Italic",
            minValue: 0,
            defaultValue: 0,
            maxValue: 1
        )
    }

    /// Optical size axis
    static var opticalSize: VariationAxis {
        VariationAxis(
            tag: opticalSizeTag,
            name: "Optical Size",
            minValue: 8,
            defaultValue: 14,
            maxValue: 144
        )
    }
}

/// A location in the design space, specified as axis tag to value
typealias DesignSpaceLocation = [String: CGFloat]

/// Represents a master (source design) at a specific location in design space
struct FontMaster: Identifiable, Codable, Sendable, Equatable {
    let id: UUID
    var name: String
    var location: DesignSpaceLocation
    var glyphs: [Character: Glyph]
    var metrics: FontMetrics

    init(
        id: UUID = UUID(),
        name: String,
        location: DesignSpaceLocation,
        glyphs: [Character: Glyph] = [:],
        metrics: FontMetrics = FontMetrics()
    ) {
        self.id = id
        self.name = name
        self.location = location
        self.glyphs = glyphs
        self.metrics = metrics
    }
}

/// A named instance in the design space (e.g., "Bold", "Light")
struct NamedInstance: Identifiable, Codable, Sendable, Equatable {
    let id: UUID
    var name: String
    var location: DesignSpaceLocation

    init(
        id: UUID = UUID(),
        name: String,
        location: DesignSpaceLocation
    ) {
        self.id = id
        self.name = name
        self.location = location
    }

    // Common weight instances
    static func thin(extraAxes: DesignSpaceLocation = [:]) -> NamedInstance {
        var location = extraAxes
        location[VariationAxis.weightTag] = 100
        return NamedInstance(name: "Thin", location: location)
    }

    static func light(extraAxes: DesignSpaceLocation = [:]) -> NamedInstance {
        var location = extraAxes
        location[VariationAxis.weightTag] = 300
        return NamedInstance(name: "Light", location: location)
    }

    static func regular(extraAxes: DesignSpaceLocation = [:]) -> NamedInstance {
        var location = extraAxes
        location[VariationAxis.weightTag] = 400
        return NamedInstance(name: "Regular", location: location)
    }

    static func medium(extraAxes: DesignSpaceLocation = [:]) -> NamedInstance {
        var location = extraAxes
        location[VariationAxis.weightTag] = 500
        return NamedInstance(name: "Medium", location: location)
    }

    static func semibold(extraAxes: DesignSpaceLocation = [:]) -> NamedInstance {
        var location = extraAxes
        location[VariationAxis.weightTag] = 600
        return NamedInstance(name: "Semibold", location: location)
    }

    static func bold(extraAxes: DesignSpaceLocation = [:]) -> NamedInstance {
        var location = extraAxes
        location[VariationAxis.weightTag] = 700
        return NamedInstance(name: "Bold", location: location)
    }

    static func extraBold(extraAxes: DesignSpaceLocation = [:]) -> NamedInstance {
        var location = extraAxes
        location[VariationAxis.weightTag] = 800
        return NamedInstance(name: "ExtraBold", location: location)
    }

    static func black(extraAxes: DesignSpaceLocation = [:]) -> NamedInstance {
        var location = extraAxes
        location[VariationAxis.weightTag] = 900
        return NamedInstance(name: "Black", location: location)
    }
}

/// Configuration for a variable font project
struct VariableFontConfig: Codable, Sendable, Equatable {
    var isVariableFont: Bool
    var axes: [VariationAxis]
    var masters: [FontMaster]
    var instances: [NamedInstance]

    init(
        isVariableFont: Bool = false,
        axes: [VariationAxis] = [],
        masters: [FontMaster] = [],
        instances: [NamedInstance] = []
    ) {
        self.isVariableFont = isVariableFont
        self.axes = axes
        self.masters = masters
        self.instances = instances
    }

    /// Create a simple weight-only variable font configuration
    static func weightOnly() -> VariableFontConfig {
        let lightMaster = FontMaster(
            name: "Light Master",
            location: [VariationAxis.weightTag: 300]
        )
        let boldMaster = FontMaster(
            name: "Bold Master",
            location: [VariationAxis.weightTag: 700]
        )

        return VariableFontConfig(
            isVariableFont: true,
            axes: [.weight],
            masters: [lightMaster, boldMaster],
            instances: [
                .light(),
                .regular(),
                .medium(),
                .semibold(),
                .bold()
            ]
        )
    }

    /// Create a weight + width variable font configuration
    static func weightAndWidth() -> VariableFontConfig {
        let lightCondensed = FontMaster(
            name: "Light Condensed",
            location: [VariationAxis.weightTag: 300, VariationAxis.widthTag: 75]
        )
        let lightExpanded = FontMaster(
            name: "Light Expanded",
            location: [VariationAxis.weightTag: 300, VariationAxis.widthTag: 125]
        )
        let boldCondensed = FontMaster(
            name: "Bold Condensed",
            location: [VariationAxis.weightTag: 700, VariationAxis.widthTag: 75]
        )
        let boldExpanded = FontMaster(
            name: "Bold Expanded",
            location: [VariationAxis.weightTag: 700, VariationAxis.widthTag: 125]
        )

        var widthAxis = VariationAxis.width
        widthAxis.minValue = 75
        widthAxis.maxValue = 125

        return VariableFontConfig(
            isVariableFont: true,
            axes: [.weight, widthAxis],
            masters: [lightCondensed, lightExpanded, boldCondensed, boldExpanded],
            instances: [
                NamedInstance(name: "Light Condensed", location: [VariationAxis.weightTag: 300, VariationAxis.widthTag: 75]),
                NamedInstance(name: "Light", location: [VariationAxis.weightTag: 300, VariationAxis.widthTag: 100]),
                NamedInstance(name: "Light Expanded", location: [VariationAxis.weightTag: 300, VariationAxis.widthTag: 125]),
                NamedInstance(name: "Regular Condensed", location: [VariationAxis.weightTag: 400, VariationAxis.widthTag: 75]),
                NamedInstance(name: "Regular", location: [VariationAxis.weightTag: 400, VariationAxis.widthTag: 100]),
                NamedInstance(name: "Regular Expanded", location: [VariationAxis.weightTag: 400, VariationAxis.widthTag: 125]),
                NamedInstance(name: "Bold Condensed", location: [VariationAxis.weightTag: 700, VariationAxis.widthTag: 75]),
                NamedInstance(name: "Bold", location: [VariationAxis.weightTag: 700, VariationAxis.widthTag: 100]),
                NamedInstance(name: "Bold Expanded", location: [VariationAxis.weightTag: 700, VariationAxis.widthTag: 125])
            ]
        )
    }
}

/// Point delta for glyph variation
struct PointDelta: Codable, Sendable {
    var dx: CGFloat
    var dy: CGFloat

    init(dx: CGFloat = 0, dy: CGFloat = 0) {
        self.dx = dx
        self.dy = dy
    }
}

/// Glyph variation data between two masters
struct GlyphVariation: Codable, Sendable {
    var character: Character
    var sourceMasterID: UUID
    var targetMasterID: UUID
    var pointDeltas: [[PointDelta]]  // Per contour, per point

    init(
        character: Character,
        sourceMasterID: UUID,
        targetMasterID: UUID,
        pointDeltas: [[PointDelta]]
    ) {
        self.character = character
        self.sourceMasterID = sourceMasterID
        self.targetMasterID = targetMasterID
        self.pointDeltas = pointDeltas
    }
}

/// Extension to calculate glyph variations
extension GlyphVariation {
    /// Calculate deltas between two glyphs
    static func calculate(
        character: Character,
        source: Glyph,
        target: Glyph,
        sourceMasterID: UUID,
        targetMasterID: UUID
    ) -> GlyphVariation? {
        // Ensure same structure
        guard source.outline.contours.count == target.outline.contours.count else {
            return nil
        }

        var pointDeltas: [[PointDelta]] = []

        for (sourceContour, targetContour) in zip(source.outline.contours, target.outline.contours) {
            guard sourceContour.points.count == targetContour.points.count else {
                return nil
            }

            var contourDeltas: [PointDelta] = []
            for (sourcePoint, targetPoint) in zip(sourceContour.points, targetContour.points) {
                let delta = PointDelta(
                    dx: targetPoint.position.x - sourcePoint.position.x,
                    dy: targetPoint.position.y - sourcePoint.position.y
                )
                contourDeltas.append(delta)
            }
            pointDeltas.append(contourDeltas)
        }

        return GlyphVariation(
            character: character,
            sourceMasterID: sourceMasterID,
            targetMasterID: targetMasterID,
            pointDeltas: pointDeltas
        )
    }

    /// Apply variation deltas to a glyph at a given interpolation factor
    func apply(to glyph: Glyph, factor: CGFloat) -> Glyph {
        var result = glyph

        for (contourIndex, contour) in glyph.outline.contours.enumerated() {
            guard contourIndex < pointDeltas.count else { continue }
            let contourDeltas = pointDeltas[contourIndex]

            for (pointIndex, _) in contour.points.enumerated() {
                guard pointIndex < contourDeltas.count else { continue }
                let delta = contourDeltas[pointIndex]

                let currentPoint = result.outline.contours[contourIndex].points[pointIndex]
                let newPosition = CGPoint(
                    x: currentPoint.position.x + delta.dx * factor,
                    y: currentPoint.position.y + delta.dy * factor
                )

                result.outline.contours[contourIndex].points[pointIndex].position = newPosition

                // Also interpolate control points if present
                if var controlOut = currentPoint.controlOut {
                    controlOut.x += delta.dx * factor
                    controlOut.y += delta.dy * factor
                    result.outline.contours[contourIndex].points[pointIndex].controlOut = controlOut
                }
                if var controlIn = currentPoint.controlIn {
                    controlIn.x += delta.dx * factor
                    controlIn.y += delta.dy * factor
                    result.outline.contours[contourIndex].points[pointIndex].controlIn = controlIn
                }
            }
        }

        // Interpolate metrics
        result.advanceWidth = Int(CGFloat(glyph.advanceWidth) + CGFloat(glyph.advanceWidth) * 0.1 * factor)
        result.leftSideBearing = Int(CGFloat(glyph.leftSideBearing) + CGFloat(glyph.leftSideBearing) * 0.1 * factor)

        return result
    }
}
