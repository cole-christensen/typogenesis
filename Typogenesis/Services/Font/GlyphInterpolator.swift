import Foundation
import CoreGraphics

/// Service for interpolating glyphs at arbitrary locations in a variable font's design space.
///
/// The `GlyphInterpolator` takes a `FontProject` with `VariableFontConfig` containing multiple
/// masters at different design space locations and produces interpolated glyphs at any target
/// location within the defined axes.
///
/// ## Interpolation Strategies
///
/// - **Single-axis**: Linear interpolation between two bracketing masters
/// - **Two-axis**: Bilinear interpolation using four corner masters, or triangular with three
/// - **N-axis**: Weighted average based on inverse distance in normalized design space
///
/// ## Usage
///
/// ```swift
/// let interpolator = GlyphInterpolator()
/// let location: DesignSpaceLocation = ["wght": 500, "wdth": 90]
/// let glyph = try await interpolator.interpolate(
///     character: "A",
///     at: location,
///     in: project
/// )
/// ```
actor GlyphInterpolator {

    // MARK: - Error Types

    /// Errors that can occur during glyph interpolation
    enum InterpolationError: Error, LocalizedError {
        /// The character is not present in any master
        case glyphNotFound(Character)

        /// The character is missing from one or more masters needed for interpolation
        case glyphMissingFromMasters(Character, missingMasterNames: [String])

        /// Glyphs across masters have incompatible structure (different contour/point counts)
        case incompatibleGlyphStructure(Character, details: String)

        /// The target location is outside the valid range for one or more axes
        case locationOutsideAxisRange(axisTag: String, value: CGFloat, validRange: ClosedRange<CGFloat>)

        /// No masters are defined in the variable font configuration
        case noMastersDefined

        /// No axes are defined in the variable font configuration
        case noAxesDefined

        /// Could not find suitable masters to bracket the target location
        case noSuitableMastersFound(DesignSpaceLocation)

        var errorDescription: String? {
            switch self {
            case .glyphNotFound(let char):
                return "Glyph '\(char)' not found in any master"
            case .glyphMissingFromMasters(let char, let names):
                return "Glyph '\(char)' is missing from masters: \(names.joined(separator: ", "))"
            case .incompatibleGlyphStructure(let char, let details):
                return "Glyph '\(char)' has incompatible structure across masters: \(details)"
            case .locationOutsideAxisRange(let tag, let value, let range):
                return "Axis '\(tag)' value \(value) is outside valid range \(range.lowerBound)...\(range.upperBound)"
            case .noMastersDefined:
                return "No masters are defined in the variable font configuration"
            case .noAxesDefined:
                return "No axes are defined in the variable font configuration"
            case .noSuitableMastersFound(let location):
                return "Could not find suitable masters to interpolate at location: \(location)"
            }
        }
    }

    // MARK: - Internal Types

    /// A master with its calculated interpolation weight
    private struct WeightedMaster {
        let master: FontMaster
        let weight: CGFloat
        let normalizedLocation: [String: CGFloat]
    }

    /// Compatibility information for a glyph across masters
    private struct GlyphCompatibility {
        let contourCounts: [Int]
        let pointCountsPerContour: [[Int]]
        let isCompatible: Bool
        let incompatibilityReason: String?
    }

    // MARK: - Public Interface

    /// Interpolates a glyph at the specified design space location.
    ///
    /// - Parameters:
    ///   - character: The character to interpolate
    ///   - location: The target location in design space (e.g., `["wght": 500, "wdth": 90]`)
    ///   - project: The font project containing variable font configuration
    /// - Returns: An interpolated `Glyph` at the target location
    /// - Throws: `InterpolationError` if interpolation fails
    func interpolate(
        character: Character,
        at location: DesignSpaceLocation,
        in project: FontProject
    ) throws -> Glyph {
        let config = project.variableConfig

        // Validate configuration
        guard !config.axes.isEmpty else {
            throw InterpolationError.noAxesDefined
        }
        guard !config.masters.isEmpty else {
            throw InterpolationError.noMastersDefined
        }

        // Validate location is within axis ranges
        try validateLocation(location, axes: config.axes)

        // Find masters that have this glyph
        let mastersWithGlyph = config.masters.filter { $0.glyphs[character] != nil }

        guard !mastersWithGlyph.isEmpty else {
            throw InterpolationError.glyphNotFound(character)
        }

        // Check glyph compatibility across masters
        let compatibility = checkGlyphCompatibility(
            character: character,
            masters: mastersWithGlyph
        )

        guard compatibility.isCompatible else {
            throw InterpolationError.incompatibleGlyphStructure(
                character,
                details: compatibility.incompatibilityReason ?? "Unknown incompatibility"
            )
        }

        // Check if all required masters have the glyph
        let mastersWithoutGlyph = config.masters.filter { $0.glyphs[character] == nil }
        if !mastersWithoutGlyph.isEmpty && mastersWithGlyph.count < 2 {
            throw InterpolationError.glyphMissingFromMasters(
                character,
                missingMasterNames: mastersWithoutGlyph.map(\.name)
            )
        }

        // Calculate weighted masters based on location
        let weightedMasters = try calculateInterpolationWeights(
            targetLocation: location,
            masters: mastersWithGlyph,
            axes: config.axes
        )

        // Perform the interpolation
        return interpolateGlyph(
            character: character,
            weightedMasters: weightedMasters
        )
    }

    /// Interpolates all glyphs in the font at the specified location.
    ///
    /// - Parameters:
    ///   - location: The target location in design space
    ///   - project: The font project containing variable font configuration
    /// - Returns: A dictionary of interpolated glyphs keyed by character.
    ///            Glyphs that fail to interpolate are logged and skipped.
    func interpolateAllGlyphs(
        at location: DesignSpaceLocation,
        in project: FontProject
    ) -> [Character: Glyph] {
        let config = project.variableConfig

        // Collect all unique characters across all masters
        var allCharacters = Set<Character>()
        for master in config.masters {
            allCharacters.formUnion(master.glyphs.keys)
        }

        var interpolatedGlyphs: [Character: Glyph] = [:]
        var failedCharacters: [(Character, Error)] = []

        for character in allCharacters {
            do {
                let glyph = try interpolate(character: character, at: location, in: project)
                interpolatedGlyphs[character] = glyph
            } catch {
                // Log failures instead of silently swallowing
                failedCharacters.append((character, error))
            }
        }

        // Report failures for debugging - don't silently swallow errors
        if !failedCharacters.isEmpty {
            print("[GlyphInterpolator] WARNING: Failed to interpolate \(failedCharacters.count) glyphs:")
            for (char, error) in failedCharacters.prefix(10) {
                print("  - '\(char)': \(error.localizedDescription)")
            }
            if failedCharacters.count > 10 {
                print("  ... and \(failedCharacters.count - 10) more")
            }
        }

        return interpolatedGlyphs
    }

    /// Interpolates font metrics at the specified location.
    ///
    /// - Parameters:
    ///   - location: The target location in design space
    ///   - project: The font project containing variable font configuration
    /// - Returns: Interpolated `FontMetrics`
    /// - Throws: `InterpolationError` if metrics cannot be interpolated
    func interpolateMetrics(
        at location: DesignSpaceLocation,
        in project: FontProject
    ) throws -> FontMetrics {
        let config = project.variableConfig

        guard !config.axes.isEmpty else {
            throw InterpolationError.noAxesDefined
        }
        guard !config.masters.isEmpty else {
            throw InterpolationError.noMastersDefined
        }

        try validateLocation(location, axes: config.axes)

        let weightedMasters = try calculateInterpolationWeights(
            targetLocation: location,
            masters: config.masters,
            axes: config.axes
        )

        return interpolateMetrics(weightedMasters: weightedMasters)
    }

    // MARK: - Location Validation

    /// Validates that the target location is within all axis ranges.
    private func validateLocation(_ location: DesignSpaceLocation, axes: [VariationAxis]) throws {
        for axis in axes {
            if let value = location[axis.tag] {
                let validRange = axis.minValue...axis.maxValue
                if !validRange.contains(value) {
                    throw InterpolationError.locationOutsideAxisRange(
                        axisTag: axis.tag,
                        value: value,
                        validRange: validRange
                    )
                }
            }
        }
    }

    // MARK: - Glyph Compatibility

    /// Checks if a glyph has compatible structure across all provided masters.
    private func checkGlyphCompatibility(
        character: Character,
        masters: [FontMaster]
    ) -> GlyphCompatibility {
        let glyphs = masters.compactMap { $0.glyphs[character] }

        guard let firstGlyph = glyphs.first else {
            return GlyphCompatibility(
                contourCounts: [],
                pointCountsPerContour: [],
                isCompatible: false,
                incompatibilityReason: "No glyphs found"
            )
        }

        let referenceContourCount = firstGlyph.outline.contours.count
        let referencePointCounts = firstGlyph.outline.contours.map(\.points.count)

        var contourCounts: [Int] = [referenceContourCount]
        var pointCountsPerContour: [[Int]] = [referencePointCounts]

        for glyph in glyphs.dropFirst() {
            let contourCount = glyph.outline.contours.count
            let pointCounts = glyph.outline.contours.map(\.points.count)

            contourCounts.append(contourCount)
            pointCountsPerContour.append(pointCounts)

            // Check contour count
            if contourCount != referenceContourCount {
                return GlyphCompatibility(
                    contourCounts: contourCounts,
                    pointCountsPerContour: pointCountsPerContour,
                    isCompatible: false,
                    incompatibilityReason: "Contour count mismatch: expected \(referenceContourCount), found \(contourCount)"
                )
            }

            // Check point counts per contour
            for (index, (refCount, actualCount)) in zip(referencePointCounts, pointCounts).enumerated() {
                if refCount != actualCount {
                    return GlyphCompatibility(
                        contourCounts: contourCounts,
                        pointCountsPerContour: pointCountsPerContour,
                        isCompatible: false,
                        incompatibilityReason: "Point count mismatch in contour \(index): expected \(refCount), found \(actualCount)"
                    )
                }
            }
        }

        return GlyphCompatibility(
            contourCounts: contourCounts,
            pointCountsPerContour: pointCountsPerContour,
            isCompatible: true,
            incompatibilityReason: nil
        )
    }

    // MARK: - Weight Calculation

    /// Calculates interpolation weights for each master based on the target location.
    ///
    /// The algorithm varies based on the number of axes:
    /// - Single axis: Linear interpolation between two bracketing masters
    /// - Two axes: Bilinear interpolation between up to four corner masters
    /// - N axes: Weighted average using inverse distance weighting in normalized space
    private func calculateInterpolationWeights(
        targetLocation: DesignSpaceLocation,
        masters: [FontMaster],
        axes: [VariationAxis]
    ) throws -> [WeightedMaster] {
        // Normalize the target location
        let normalizedTarget = normalizeLocation(targetLocation, axes: axes)

        // Normalize all master locations
        let normalizedMasters = masters.map { master in
            (master: master, normalized: normalizeLocation(master.location, axes: axes))
        }

        let axisCount = axes.count

        switch axisCount {
        case 1:
            return try calculateSingleAxisWeights(
                target: normalizedTarget,
                masters: normalizedMasters,
                axis: axes[0]
            )
        case 2:
            return try calculateBilinearWeights(
                target: normalizedTarget,
                masters: normalizedMasters,
                axes: axes
            )
        default:
            return calculateMultiAxisWeights(
                target: normalizedTarget,
                masters: normalizedMasters,
                axes: axes
            )
        }
    }

    /// Normalizes a design space location to 0-1 range for each axis.
    private func normalizeLocation(
        _ location: DesignSpaceLocation,
        axes: [VariationAxis]
    ) -> [String: CGFloat] {
        var normalized: [String: CGFloat] = [:]

        for axis in axes {
            let value = location[axis.tag] ?? axis.defaultValue
            let range = axis.maxValue - axis.minValue

            if range > 0 {
                normalized[axis.tag] = (value - axis.minValue) / range
            } else {
                normalized[axis.tag] = 0.5 // Degenerate axis
            }
        }

        return normalized
    }

    // MARK: - Single-Axis Interpolation

    /// Calculates weights for single-axis linear interpolation.
    private func calculateSingleAxisWeights(
        target: [String: CGFloat],
        masters: [(master: FontMaster, normalized: [String: CGFloat])],
        axis: VariationAxis
    ) throws -> [WeightedMaster] {
        let tag = axis.tag
        let targetValue = target[tag] ?? 0.5

        // Sort masters by their position on this axis
        let sortedMasters = masters.sorted { m1, m2 in
            (m1.normalized[tag] ?? 0) < (m2.normalized[tag] ?? 0)
        }

        // Find the two bracketing masters
        var lowerMaster: (master: FontMaster, normalized: [String: CGFloat])?
        var upperMaster: (master: FontMaster, normalized: [String: CGFloat])?

        for master in sortedMasters {
            let masterValue = master.normalized[tag] ?? 0

            if masterValue <= targetValue {
                lowerMaster = master
            }
            if masterValue >= targetValue && upperMaster == nil {
                upperMaster = master
            }
        }

        // If we're exactly at a master location, return 100% weight for that master
        if let lower = lowerMaster, let upper = upperMaster {
            let lowerValue = lower.normalized[tag] ?? 0
            let upperValue = upper.normalized[tag] ?? 0

            if abs(lowerValue - targetValue) < 0.0001 {
                return [WeightedMaster(master: lower.master, weight: 1.0, normalizedLocation: lower.normalized)]
            }
            if abs(upperValue - targetValue) < 0.0001 {
                return [WeightedMaster(master: upper.master, weight: 1.0, normalizedLocation: upper.normalized)]
            }

            // Linear interpolation
            let range = upperValue - lowerValue
            if range > 0 {
                let t = (targetValue - lowerValue) / range
                return [
                    WeightedMaster(master: lower.master, weight: 1.0 - t, normalizedLocation: lower.normalized),
                    WeightedMaster(master: upper.master, weight: t, normalizedLocation: upper.normalized)
                ]
            }
        }

        // Fallback: use closest master
        if let lower = lowerMaster {
            return [WeightedMaster(master: lower.master, weight: 1.0, normalizedLocation: lower.normalized)]
        }
        if let upper = upperMaster {
            return [WeightedMaster(master: upper.master, weight: 1.0, normalizedLocation: upper.normalized)]
        }

        throw InterpolationError.noSuitableMastersFound(target)
    }

    // MARK: - Bilinear Interpolation (Two Axes)

    /// Calculates weights for two-axis bilinear interpolation.
    private func calculateBilinearWeights(
        target: [String: CGFloat],
        masters: [(master: FontMaster, normalized: [String: CGFloat])],
        axes: [VariationAxis]
    ) throws -> [WeightedMaster] {
        let tag0 = axes[0].tag
        let tag1 = axes[1].tag

        let tx = target[tag0] ?? 0.5
        let ty = target[tag1] ?? 0.5

        // Find the four corner masters (or as many as we have)
        // We need masters at approximately (low, low), (high, low), (low, high), (high, high)

        var corners: [(master: FontMaster, normalized: [String: CGFloat], dx: CGFloat, dy: CGFloat)] = []

        for master in masters {
            let mx = master.normalized[tag0] ?? 0.5
            let my = master.normalized[tag1] ?? 0.5
            corners.append((master.master, master.normalized, mx - tx, my - ty))
        }

        // If we have exactly 4 masters arranged in a grid, use true bilinear interpolation
        if masters.count == 4 {
            // Find the bounding box of masters
            let xValues = masters.map { $0.normalized[tag0] ?? 0.5 }
            let yValues = masters.map { $0.normalized[tag1] ?? 0.5 }

            guard let minX = xValues.min(), let maxX = xValues.max(),
                  let minY = yValues.min(), let maxY = yValues.max(),
                  maxX > minX, maxY > minY else {
                return calculateMultiAxisWeights(target: target, masters: masters, axes: axes)
            }

            // Normalize target within the master bounding box
            let s = (tx - minX) / (maxX - minX)
            let t = (ty - minY) / (maxY - minY)

            // Find masters at each corner
            var cornerMasters: [String: (FontMaster, [String: CGFloat])] = [:]
            for master in masters {
                let mx = master.normalized[tag0] ?? 0.5
                let my = master.normalized[tag1] ?? 0.5

                let isLowX = abs(mx - minX) < abs(mx - maxX)
                let isLowY = abs(my - minY) < abs(my - maxY)

                let key = "\(isLowX ? "L" : "H")\(isLowY ? "L" : "H")"
                cornerMasters[key] = (master.master, master.normalized)
            }

            // If we have all 4 corners, compute bilinear weights
            if let ll = cornerMasters["LL"],
               let hl = cornerMasters["HL"],
               let lh = cornerMasters["LH"],
               let hh = cornerMasters["HH"] {

                let w00 = (1 - s) * (1 - t)  // LL weight
                let w10 = s * (1 - t)        // HL weight
                let w01 = (1 - s) * t        // LH weight
                let w11 = s * t              // HH weight

                return [
                    WeightedMaster(master: ll.0, weight: w00, normalizedLocation: ll.1),
                    WeightedMaster(master: hl.0, weight: w10, normalizedLocation: hl.1),
                    WeightedMaster(master: lh.0, weight: w01, normalizedLocation: lh.1),
                    WeightedMaster(master: hh.0, weight: w11, normalizedLocation: hh.1)
                ].filter { $0.weight > 0.0001 }
            }
        }

        // If we have 3 masters, use triangular (barycentric) interpolation
        if masters.count == 3 {
            return try calculateBarycentricWeights(
                target: target,
                masters: masters,
                axes: axes
            )
        }

        // Fallback to distance-based weighting
        return calculateMultiAxisWeights(target: target, masters: masters, axes: axes)
    }

    /// Calculates barycentric weights for three masters forming a triangle.
    private func calculateBarycentricWeights(
        target: [String: CGFloat],
        masters: [(master: FontMaster, normalized: [String: CGFloat])],
        axes: [VariationAxis]
    ) throws -> [WeightedMaster] {
        guard masters.count == 3 else {
            return calculateMultiAxisWeights(target: target, masters: masters, axes: axes)
        }

        let tag0 = axes[0].tag
        let tag1 = axes[1].tag

        // Get triangle vertices
        let p0 = CGPoint(
            x: masters[0].normalized[tag0] ?? 0.5,
            y: masters[0].normalized[tag1] ?? 0.5
        )
        let p1 = CGPoint(
            x: masters[1].normalized[tag0] ?? 0.5,
            y: masters[1].normalized[tag1] ?? 0.5
        )
        let p2 = CGPoint(
            x: masters[2].normalized[tag0] ?? 0.5,
            y: masters[2].normalized[tag1] ?? 0.5
        )

        let targetPoint = CGPoint(x: target[tag0] ?? 0.5, y: target[tag1] ?? 0.5)

        // Calculate barycentric coordinates
        let v0 = CGPoint(x: p1.x - p0.x, y: p1.y - p0.y)
        let v1 = CGPoint(x: p2.x - p0.x, y: p2.y - p0.y)
        let v2 = CGPoint(x: targetPoint.x - p0.x, y: targetPoint.y - p0.y)

        let dot00 = v0.x * v0.x + v0.y * v0.y
        let dot01 = v0.x * v1.x + v0.y * v1.y
        let dot02 = v0.x * v2.x + v0.y * v2.y
        let dot11 = v1.x * v1.x + v1.y * v1.y
        let dot12 = v1.x * v2.x + v1.y * v2.y

        let denom = dot00 * dot11 - dot01 * dot01

        guard abs(denom) > 0.0001 else {
            // Degenerate triangle, fall back to distance weighting
            return calculateMultiAxisWeights(target: target, masters: masters, axes: axes)
        }

        let u = (dot11 * dot02 - dot01 * dot12) / denom
        let v = (dot00 * dot12 - dot01 * dot02) / denom
        let w = 1 - u - v

        // Clamp to valid range (point might be outside triangle)
        let clampedU = max(0, min(1, u))
        let clampedV = max(0, min(1, v))
        let clampedW = max(0, min(1, w))

        // Renormalize
        let sum = clampedU + clampedV + clampedW
        guard sum > 0 else {
            return calculateMultiAxisWeights(target: target, masters: masters, axes: axes)
        }

        return [
            WeightedMaster(
                master: masters[1].master,
                weight: clampedU / sum,
                normalizedLocation: masters[1].normalized
            ),
            WeightedMaster(
                master: masters[2].master,
                weight: clampedV / sum,
                normalizedLocation: masters[2].normalized
            ),
            WeightedMaster(
                master: masters[0].master,
                weight: clampedW / sum,
                normalizedLocation: masters[0].normalized
            )
        ].filter { $0.weight > 0.0001 }
    }

    // MARK: - Multi-Axis Interpolation (N-dimensional)

    /// Calculates weights for N-axis interpolation using inverse distance weighting.
    private func calculateMultiAxisWeights(
        target: [String: CGFloat],
        masters: [(master: FontMaster, normalized: [String: CGFloat])],
        axes: [VariationAxis]
    ) -> [WeightedMaster] {
        // Use inverse distance weighting (Shepard's method)
        // Weight = 1 / distance^p where p typically = 2
        let p: CGFloat = 2.0
        let epsilon: CGFloat = 0.0001

        var weights: [(master: FontMaster, normalized: [String: CGFloat], weight: CGFloat)] = []
        var totalWeight: CGFloat = 0

        for master in masters {
            // Calculate Euclidean distance in normalized space
            var distanceSquared: CGFloat = 0

            for axis in axes {
                let targetValue = target[axis.tag] ?? 0.5
                let masterValue = master.normalized[axis.tag] ?? 0.5
                let diff = targetValue - masterValue
                distanceSquared += diff * diff
            }

            let distance = sqrt(distanceSquared)

            // If we're very close to a master, return that master with full weight
            if distance < epsilon {
                return [WeightedMaster(master: master.master, weight: 1.0, normalizedLocation: master.normalized)]
            }

            let weight = 1.0 / pow(distance, p)
            weights.append((master.master, master.normalized, weight))
            totalWeight += weight
        }

        // Normalize weights
        guard totalWeight > 0 else {
            // Fallback: equal weights
            let equalWeight = 1.0 / CGFloat(masters.count)
            return masters.map {
                WeightedMaster(master: $0.master, weight: equalWeight, normalizedLocation: $0.normalized)
            }
        }

        return weights.map {
            WeightedMaster(master: $0.master, weight: $0.weight / totalWeight, normalizedLocation: $0.normalized)
        }.filter { $0.weight > 0.0001 }
    }

    // MARK: - Glyph Interpolation

    /// Performs the actual glyph interpolation using pre-calculated weights.
    private func interpolateGlyph(
        character: Character,
        weightedMasters: [WeightedMaster]
    ) -> Glyph {
        guard let firstMaster = weightedMasters.first,
              let baseGlyph = firstMaster.master.glyphs[character] else {
            // This shouldn't happen if we validated properly, but return an empty glyph as fallback
            return Glyph(character: character)
        }

        // If there's only one master, return its glyph directly
        if weightedMasters.count == 1 {
            return baseGlyph
        }

        // Collect all glyphs with their weights
        var glyphsWithWeights: [(glyph: Glyph, weight: CGFloat)] = []
        for wm in weightedMasters {
            if let glyph = wm.master.glyphs[character] {
                glyphsWithWeights.append((glyph, wm.weight))
            }
        }

        // Interpolate outline
        let interpolatedOutline = interpolateOutline(
            glyphsWithWeights: glyphsWithWeights
        )

        // Interpolate metrics
        var interpolatedAdvanceWidth: CGFloat = 0
        var interpolatedLSB: CGFloat = 0

        for (glyph, weight) in glyphsWithWeights {
            interpolatedAdvanceWidth += CGFloat(glyph.advanceWidth) * weight
            interpolatedLSB += CGFloat(glyph.leftSideBearing) * weight
        }

        return Glyph(
            id: UUID(),
            character: character,
            unicodeScalars: baseGlyph.unicodeScalars,
            outline: interpolatedOutline,
            advanceWidth: Int(interpolatedAdvanceWidth.rounded()),
            leftSideBearing: Int(interpolatedLSB.rounded()),
            generatedBy: .placeholder,
            styleConfidence: nil
        )
    }

    /// Interpolates glyph outlines using weighted average.
    private func interpolateOutline(
        glyphsWithWeights: [(glyph: Glyph, weight: CGFloat)]
    ) -> GlyphOutline {
        guard let firstGlyph = glyphsWithWeights.first?.glyph else {
            return GlyphOutline()
        }

        var interpolatedContours: [Contour] = []

        for (contourIndex, contour) in firstGlyph.outline.contours.enumerated() {
            var interpolatedPoints: [PathPoint] = []

            for (pointIndex, point) in contour.points.enumerated() {
                var interpolatedPosition = CGPoint.zero
                var interpolatedControlIn: CGPoint?
                var interpolatedControlOut: CGPoint?

                // Check if any glyph has control points at this position
                var hasControlIn = false
                var hasControlOut = false

                for (glyph, _) in glyphsWithWeights {
                    guard contourIndex < glyph.outline.contours.count,
                          pointIndex < glyph.outline.contours[contourIndex].points.count else {
                        continue
                    }
                    let p = glyph.outline.contours[contourIndex].points[pointIndex]
                    if p.controlIn != nil { hasControlIn = true }
                    if p.controlOut != nil { hasControlOut = true }
                }

                if hasControlIn {
                    interpolatedControlIn = .zero
                }
                if hasControlOut {
                    interpolatedControlOut = .zero
                }

                // Interpolate positions
                for (glyph, weight) in glyphsWithWeights {
                    guard contourIndex < glyph.outline.contours.count,
                          pointIndex < glyph.outline.contours[contourIndex].points.count else {
                        continue
                    }

                    let p = glyph.outline.contours[contourIndex].points[pointIndex]

                    interpolatedPosition.x += p.position.x * weight
                    interpolatedPosition.y += p.position.y * weight

                    if let ctrlIn = p.controlIn {
                        interpolatedControlIn?.x += ctrlIn.x * weight
                        interpolatedControlIn?.y += ctrlIn.y * weight
                    } else if hasControlIn {
                        // Use the point position as fallback for control point
                        interpolatedControlIn?.x += p.position.x * weight
                        interpolatedControlIn?.y += p.position.y * weight
                    }

                    if let ctrlOut = p.controlOut {
                        interpolatedControlOut?.x += ctrlOut.x * weight
                        interpolatedControlOut?.y += ctrlOut.y * weight
                    } else if hasControlOut {
                        // Use the point position as fallback for control point
                        interpolatedControlOut?.x += p.position.x * weight
                        interpolatedControlOut?.y += p.position.y * weight
                    }
                }

                let interpolatedPoint = PathPoint(
                    id: UUID(),
                    position: interpolatedPosition,
                    type: point.type,
                    controlIn: interpolatedControlIn,
                    controlOut: interpolatedControlOut
                )

                interpolatedPoints.append(interpolatedPoint)
            }

            let interpolatedContour = Contour(
                id: UUID(),
                points: interpolatedPoints,
                isClosed: contour.isClosed
            )

            interpolatedContours.append(interpolatedContour)
        }

        return GlyphOutline(contours: interpolatedContours)
    }

    // MARK: - Metrics Interpolation

    /// Interpolates font metrics using weighted average.
    private func interpolateMetrics(weightedMasters: [WeightedMaster]) -> FontMetrics {
        var unitsPerEm: CGFloat = 0
        var ascender: CGFloat = 0
        var descender: CGFloat = 0
        var xHeight: CGFloat = 0
        var capHeight: CGFloat = 0
        var lineGap: CGFloat = 0

        for wm in weightedMasters {
            let metrics = wm.master.metrics
            let weight = wm.weight

            unitsPerEm += CGFloat(metrics.unitsPerEm) * weight
            ascender += CGFloat(metrics.ascender) * weight
            descender += CGFloat(metrics.descender) * weight
            xHeight += CGFloat(metrics.xHeight) * weight
            capHeight += CGFloat(metrics.capHeight) * weight
            lineGap += CGFloat(metrics.lineGap) * weight
        }

        return FontMetrics(
            unitsPerEm: Int(unitsPerEm.rounded()),
            ascender: Int(ascender.rounded()),
            descender: Int(descender.rounded()),
            xHeight: Int(xHeight.rounded()),
            capHeight: Int(capHeight.rounded()),
            lineGap: Int(lineGap.rounded())
        )
    }
}
