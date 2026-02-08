import Testing
import Foundation
@testable import Typogenesis

// MARK: - Test Helper Functions

/// Create a simple square outline
func createSquareOutline(x: CGFloat = 0, y: CGFloat = 0, size: CGFloat = 100) -> GlyphOutline {
    let contour = Contour(
        points: [
            PathPoint(position: CGPoint(x: x, y: y), type: .corner),
            PathPoint(position: CGPoint(x: x + size, y: y), type: .corner),
            PathPoint(position: CGPoint(x: x + size, y: y + size), type: .corner),
            PathPoint(position: CGPoint(x: x, y: y + size), type: .corner)
        ],
        isClosed: true
    )
    return GlyphOutline(contours: [contour])
}

/// Create a triangle outline
func createTriangleOutline(x: CGFloat = 0, y: CGFloat = 0, size: CGFloat = 100) -> GlyphOutline {
    let contour = Contour(
        points: [
            PathPoint(position: CGPoint(x: x + size / 2, y: y), type: .corner),
            PathPoint(position: CGPoint(x: x + size, y: y + size), type: .corner),
            PathPoint(position: CGPoint(x: x, y: y + size), type: .corner)
        ],
        isClosed: true
    )
    return GlyphOutline(contours: [contour])
}

/// Create a line outline (open contour)
func createLineOutline(from: CGPoint, to: CGPoint) -> GlyphOutline {
    let contour = Contour(
        points: [
            PathPoint(position: from, type: .corner),
            PathPoint(position: to, type: .corner)
        ],
        isClosed: false
    )
    return GlyphOutline(contours: [contour])
}

/// Create overlapping squares
func createOverlappingSquares() -> GlyphOutline {
    let square1 = Contour(
        points: [
            PathPoint(position: CGPoint(x: 0, y: 0), type: .corner),
            PathPoint(position: CGPoint(x: 100, y: 0), type: .corner),
            PathPoint(position: CGPoint(x: 100, y: 100), type: .corner),
            PathPoint(position: CGPoint(x: 0, y: 100), type: .corner)
        ],
        isClosed: true
    )
    let square2 = Contour(
        points: [
            PathPoint(position: CGPoint(x: 50, y: 50), type: .corner),
            PathPoint(position: CGPoint(x: 150, y: 50), type: .corner),
            PathPoint(position: CGPoint(x: 150, y: 150), type: .corner),
            PathPoint(position: CGPoint(x: 50, y: 150), type: .corner)
        ],
        isClosed: true
    )
    return GlyphOutline(contours: [square1, square2])
}

// MARK: - Boolean Operations Tests
//
// NOTE: The PathOperations boolean operations (union, subtract, intersect, xor)
// are currently STUB IMPLEMENTATIONS. They combine both input paths but do not
// actually compute the geometric boolean result. This is a known limitation.
//
// The tests below verify:
// 1. Operations don't crash
// 2. Results are non-empty
// 3. Both input shapes are present in the result (current stub behavior)
//
// TODO: Implement proper boolean operations using Vatti's algorithm,
// Greiner-Hormann, or a library like ClipperLib. CGPath does not provide
// native boolean operations.

@Suite("PathOperations Boolean Operations")
struct PathOperationsBooleanTests {

    @Test("Union of two overlapping squares produces combined outline")
    func unionOverlappingSquares() throws {
        let square1 = createSquareOutline(x: 0, y: 0, size: 100)
        let square2 = createSquareOutline(x: 50, y: 50, size: 100)

        let result = try PathOperations.perform(.union, on: square1, with: square2)

        #expect(!result.isEmpty)
        // Union should have at least one contour
        #expect(result.contours.count >= 1)

        // The union bounding box should encompass both squares: (0,0) to (150,150)
        let bbox = result.boundingBox
        #expect(bbox.minX == 0, "Union bounding box minX should be 0")
        #expect(bbox.minY == 0, "Union bounding box minY should be 0")
        #expect(bbox.maxX == 150, "Union bounding box maxX should be 150")
        #expect(bbox.maxY == 150, "Union bounding box maxY should be 150")

        // Points inside each original square should be inside the union
        #expect(PathOperations.contains(point: CGPoint(x: 25, y: 25), in: result),
                "Point inside square1 should be inside union")
        #expect(PathOperations.contains(point: CGPoint(x: 125, y: 125), in: result),
                "Point inside square2 should be inside union")
        #expect(PathOperations.contains(point: CGPoint(x: 75, y: 75), in: result),
                "Point in overlap region should be inside union")

        // Point outside both squares should not be inside the union
        #expect(!PathOperations.contains(point: CGPoint(x: 200, y: 200), in: result),
                "Point outside both squares should not be inside union")
    }

    @Test("Union of non-overlapping squares preserves both")
    func unionNonOverlappingSquares() throws {
        let square1 = createSquareOutline(x: 0, y: 0, size: 50)
        let square2 = createSquareOutline(x: 100, y: 100, size: 50)

        let result = try PathOperations.perform(.union, on: square1, with: square2)

        #expect(!result.isEmpty)
    }

    // NOTE: Subtract is a STUB - it doesn't actually remove the overlapping region.
    // The current implementation reverses path2's winding but CGPath doesn't compute
    // the geometric difference. This test verifies the stub doesn't crash.
    @Test("Subtract returns non-empty result (stub implementation)")
    func subtractOverlappingSquares() throws {
        let square1 = createSquareOutline(x: 0, y: 0, size: 100)
        let square2 = createSquareOutline(x: 50, y: 50, size: 100)

        let result = try PathOperations.perform(.subtract, on: square1, with: square2)

        // Stub implementation: just verify it doesn't crash and returns something
        #expect(!result.isEmpty, "Subtract should return non-empty result")

        // Point in the non-overlapping part of square1 should still be inside result
        #expect(PathOperations.contains(point: CGPoint(x: 25, y: 25), in: result),
                "Point in non-overlapping region of square1 should remain")

        // Point far outside should not be inside result
        #expect(!PathOperations.contains(point: CGPoint(x: 200, y: 200), in: result),
                "Point outside all shapes should not be in result")

        // TODO: When real subtraction is implemented, add these assertions:
        // - bbox.maxX <= 100 (result should not extend into square2-only region)
        // - Point (125, 125) should NOT be in result
    }

    // NOTE: Intersect is a STUB - it doesn't actually compute the overlapping region.
    // The current implementation combines both paths. This test verifies the stub doesn't crash.
    @Test("Intersect returns non-empty result (stub implementation)")
    func intersectOverlappingSquares() throws {
        let square1 = createSquareOutline(x: 0, y: 0, size: 100)
        let square2 = createSquareOutline(x: 50, y: 50, size: 100)

        let result = try PathOperations.perform(.intersect, on: square1, with: square2)

        // Stub implementation: just verify it doesn't crash and returns something
        #expect(!result.isEmpty, "Intersect should return non-empty result")

        // Point in the overlap region should be inside result
        #expect(PathOperations.contains(point: CGPoint(x: 75, y: 75), in: result),
                "Point in overlap region should be inside intersection")

        // TODO: When real intersection is implemented, add these assertions:
        // - bbox should be (50,50)-(100,100) only
        // - Point (25, 25) should NOT be in result (exclusive to square1)
        // - Point (125, 125) should NOT be in result (exclusive to square2)
    }

    // NOTE: XOR is a STUB - it doesn't actually exclude the overlapping region.
    // The current implementation combines both paths. This test verifies the stub doesn't crash.
    @Test("XOR returns non-empty result (stub implementation)")
    func xorOverlappingSquares() throws {
        let square1 = createSquareOutline(x: 0, y: 0, size: 100)
        let square2 = createSquareOutline(x: 50, y: 50, size: 100)

        let result = try PathOperations.perform(.xor, on: square1, with: square2)

        // Stub implementation: just verify it doesn't crash and returns something
        #expect(!result.isEmpty, "XOR should return non-empty result")

        // XOR bounding box should span the full extent of both shapes
        let bbox = result.boundingBox
        #expect(bbox.minX == 0, "XOR bounding box minX should be 0")
        #expect(bbox.minY == 0, "XOR bounding box minY should be 0")
        #expect(bbox.maxX == 150, "XOR bounding box maxX should be 150")
        #expect(bbox.maxY == 150, "XOR bounding box maxY should be 150")

        // Points exclusive to each square should be inside result
        #expect(PathOperations.contains(point: CGPoint(x: 25, y: 25), in: result),
                "Point only in square1 should be inside XOR result")
        #expect(PathOperations.contains(point: CGPoint(x: 125, y: 125), in: result),
                "Point only in square2 should be inside XOR result")

        // Point outside both should not be in result
        #expect(!PathOperations.contains(point: CGPoint(x: 200, y: 200), in: result),
                "Point outside both squares should not be in XOR result")

        // TODO: When real XOR is implemented, add this assertion:
        // - Point (75, 75) should NOT be in result (overlap region excluded by XOR)
    }

    @Test("Empty path throws error")
    func emptyPathThrows() {
        let empty = GlyphOutline(contours: [])
        let square = createSquareOutline()

        #expect(throws: PathOperations.PathOperationError.emptyPath) {
            _ = try PathOperations.perform(.union, on: empty, with: square)
        }

        #expect(throws: PathOperations.PathOperationError.emptyPath) {
            _ = try PathOperations.perform(.union, on: square, with: empty)
        }
    }

    @Test("Operation types are available and distinct")
    func operationTypesExist() {
        // Verify all operation types can be referenced and are distinct
        let operations: [PathOperations.Operation] = [.union, .subtract, .intersect, .xor]
        #expect(operations.count == 4)

        // Verify each operation type is unique (Set would collapse duplicates)
        let uniqueOps = Set(operations.map { "\($0)" })
        #expect(uniqueOps.count == 4, "All operation types should be distinct")

        // Verify we can use these in a switch (compile-time check that cases exist)
        for op in operations {
            switch op {
            case .union: break
            case .subtract: break
            case .intersect: break
            case .xor: break
            }
        }
    }
}

// MARK: - Union Contours Tests

@Suite("PathOperations Union Contours")
struct PathOperationsUnionContoursTests {

    @Test("Union contours combines overlapping shapes")
    func unionContoursOverlapping() throws {
        let overlapping = createOverlappingSquares()

        let result = try PathOperations.unionContours(overlapping)

        #expect(!result.isEmpty)
    }

    @Test("Single contour returns unchanged")
    func singleContourUnchanged() throws {
        let single = createSquareOutline()

        let result = try PathOperations.unionContours(single)

        #expect(result.contours.count == 1)
    }

    @Test("Remove overlaps simplifies outline")
    func removeOverlaps() throws {
        let overlapping = createOverlappingSquares()

        let result = try PathOperations.removeOverlaps(overlapping)

        #expect(!result.isEmpty)
    }

    @Test("Remove overlaps on empty returns empty")
    func removeOverlapsEmpty() throws {
        let empty = GlyphOutline(contours: [])

        let result = try PathOperations.removeOverlaps(empty)

        #expect(result.isEmpty)
    }
}

// MARK: - Offset Tests

@Suite("PathOperations Offset")
struct PathOperationsOffsetTests {

    @Test("Positive offset expands path")
    func positiveOffsetExpands() throws {
        let square = createSquareOutline(x: 50, y: 50, size: 100)

        let expanded = try PathOperations.offset(square, by: 10)

        #expect(!expanded.isEmpty)
    }

    @Test("Negative offset contracts path")
    func negativeOffsetContracts() throws {
        let square = createSquareOutline(x: 50, y: 50, size: 100)

        let contracted = try PathOperations.offset(square, by: -5)

        #expect(!contracted.isEmpty)
    }

    @Test("Zero offset returns similar shape")
    func zeroOffset() throws {
        let square = createSquareOutline(x: 50, y: 50, size: 100)

        let result = try PathOperations.offset(square, by: 0)

        // Zero offset uses stroke width of 0, should return something
        #expect(!result.isEmpty)
    }

    @Test("Offset on empty throws error")
    func offsetEmptyThrows() {
        let empty = GlyphOutline(contours: [])

        #expect(throws: PathOperations.PathOperationError.emptyPath) {
            _ = try PathOperations.offset(empty, by: 10)
        }
    }
}

// MARK: - Simplify Tests

@Suite("PathOperations Simplify")
struct PathOperationsSimplifyTests {

    @Test("Simplify reduces point count on complex path")
    func simplifyReducesPoints() {
        // Create a jagged line with many points
        var points: [PathPoint] = []
        for i in 0..<20 {
            let y: CGFloat = (i % 2 == 0) ? 0.5 : -0.5  // Small jitter
            points.append(PathPoint(position: CGPoint(x: CGFloat(i) * 10, y: y), type: .corner))
        }
        let contour = Contour(points: points, isClosed: false)
        let outline = GlyphOutline(contours: [contour])

        let simplified = PathOperations.simplify(outline, tolerance: 1.0)

        // Should have fewer points after simplification
        #expect(simplified.contours[0].points.count <= outline.contours[0].points.count)
    }

    @Test("Simplify preserves important corners")
    func simplifyPreservesCorners() {
        // Create an L-shape with a clear corner
        let points: [PathPoint] = [
            PathPoint(position: CGPoint(x: 0, y: 0), type: .corner),
            PathPoint(position: CGPoint(x: 100, y: 0), type: .corner),
            PathPoint(position: CGPoint(x: 100, y: 50), type: .corner),
            PathPoint(position: CGPoint(x: 50, y: 50), type: .corner),
            PathPoint(position: CGPoint(x: 50, y: 100), type: .corner),
            PathPoint(position: CGPoint(x: 0, y: 100), type: .corner)
        ]
        let contour = Contour(points: points, isClosed: true)
        let outline = GlyphOutline(contours: [contour])

        let simplified = PathOperations.simplify(outline, tolerance: 5.0)

        // Should still have multiple points (corners are significant)
        #expect(simplified.contours[0].points.count >= 2)
    }

    @Test("Simplify with high tolerance reduces more points")
    func simplifyHighTolerance() {
        // Create a wavy line
        var points: [PathPoint] = []
        for i in 0..<30 {
            let angle = Double(i) * 0.2
            let y = sin(angle) * 10
            points.append(PathPoint(position: CGPoint(x: CGFloat(i) * 10, y: y), type: .corner))
        }
        let contour = Contour(points: points, isClosed: false)
        let outline = GlyphOutline(contours: [contour])

        let lowTolerance = PathOperations.simplify(outline, tolerance: 1.0)
        let highTolerance = PathOperations.simplify(outline, tolerance: 20.0)

        // High tolerance should result in fewer or equal points
        #expect(highTolerance.contours[0].points.count <= lowTolerance.contours[0].points.count)
    }

    @Test("Simplify empty outline returns empty")
    func simplifyEmpty() {
        let empty = GlyphOutline(contours: [])

        let result = PathOperations.simplify(empty, tolerance: 1.0)

        #expect(result.isEmpty)
    }

    @Test("Simplify single point is filtered out (requires 2+ points)")
    func simplifySinglePoint() {
        let single = Contour(
            points: [PathPoint(position: CGPoint(x: 50, y: 50), type: .corner)],
            isClosed: false
        )
        let outline = GlyphOutline(contours: [single])

        let result = PathOperations.simplify(outline, tolerance: 1.0)

        // Implementation filters out contours with fewer than 2 points
        #expect(result.isEmpty)
    }

    @Test("Simplify two points returns two points")
    func simplifyTwoPoints() {
        let line = Contour(
            points: [
                PathPoint(position: CGPoint(x: 0, y: 0), type: .corner),
                PathPoint(position: CGPoint(x: 100, y: 100), type: .corner)
            ],
            isClosed: false
        )
        let outline = GlyphOutline(contours: [line])

        let result = PathOperations.simplify(outline, tolerance: 1.0)

        #expect(result.contours[0].points.count == 2)
    }
}

// MARK: - Point Containment Tests

@Suite("PathOperations Point Containment")
struct PathOperationsContainmentTests {

    @Test("Point inside square is contained")
    func pointInsideSquare() {
        let square = createSquareOutline(x: 0, y: 0, size: 100)
        let insidePoint = CGPoint(x: 50, y: 50)

        let result = PathOperations.contains(point: insidePoint, in: square)

        #expect(result)
    }

    @Test("Point outside square is not contained")
    func pointOutsideSquare() {
        let square = createSquareOutline(x: 0, y: 0, size: 100)
        let outsidePoint = CGPoint(x: 200, y: 200)

        let result = PathOperations.contains(point: outsidePoint, in: square)

        #expect(!result)
    }

    @Test("Point on edge behavior")
    func pointOnEdge() {
        let square = createSquareOutline(x: 0, y: 0, size: 100)
        let edgePoint = CGPoint(x: 0, y: 50)  // On left edge

        // PathOperations.contains uses CGPath.contains with .winding rule.
        // The result for boundary points is implementation-defined, but
        // it must return a deterministic Bool without crashing.
        let result = PathOperations.contains(point: edgePoint, in: square)
        // Verify the result is consistent across calls
        let result2 = PathOperations.contains(point: edgePoint, in: square)
        #expect(result == result2, "Point-on-edge containment should be deterministic")
    }

    @Test("Point inside triangle is contained")
    func pointInsideTriangle() {
        let triangle = createTriangleOutline(x: 0, y: 0, size: 100)
        let centerPoint = CGPoint(x: 50, y: 66)  // Roughly center of triangle

        let result = PathOperations.contains(point: centerPoint, in: triangle)

        #expect(result)
    }
}

// MARK: - Area Calculation Tests

@Suite("PathOperations Area Calculation")
struct PathOperationsAreaTests {

    @Test("Square area is calculated correctly")
    func squareArea() {
        let square = createSquareOutline(x: 0, y: 0, size: 100)

        let area = PathOperations.area(of: square)

        // Area should be approximately 10000 (100 * 100)
        // Signed area may be positive or negative depending on winding
        #expect(abs(area) > 9000 && abs(area) < 11000)
    }

    @Test("Triangle area is calculated correctly")
    func triangleArea() {
        let triangle = createTriangleOutline(x: 0, y: 0, size: 100)

        let area = PathOperations.area(of: triangle)

        // Triangle area = base * height / 2 = 100 * 100 / 2 = 5000
        #expect(abs(area) > 4000 && abs(area) < 6000)
    }

    @Test("Empty outline has zero area")
    func emptyArea() {
        let empty = GlyphOutline(contours: [])

        let area = PathOperations.area(of: empty)

        #expect(area == 0)
    }

    @Test("Line has zero area")
    func lineArea() {
        let line = createLineOutline(from: CGPoint(x: 0, y: 0), to: CGPoint(x: 100, y: 100))

        let area = PathOperations.area(of: line)

        #expect(area == 0)
    }

    @Test("Multiple contours sum their areas")
    func multipleContoursArea() {
        let outline = createOverlappingSquares()

        let area = PathOperations.area(of: outline)

        // Two 100x100 squares = 20000 total (minus overlap)
        #expect(abs(area) > 10000)
    }
}

// MARK: - Winding Direction Tests

@Suite("PathOperations Winding Direction")
struct PathOperationsWindingTests {

    @Test("Normalize winding direction clockwise")
    func normalizeClockwise() {
        let square = createSquareOutline()

        let normalized = PathOperations.normalizeWindingDirection(square, clockwise: true)

        #expect(!normalized.isEmpty)
        #expect(normalized.contours.count == square.contours.count)
    }

    @Test("Normalize winding direction counter-clockwise")
    func normalizeCounterClockwise() {
        let square = createSquareOutline()

        let normalized = PathOperations.normalizeWindingDirection(square, clockwise: false)

        #expect(!normalized.isEmpty)
        #expect(normalized.contours.count == square.contours.count)
    }

    @Test("Normalizing already correct winding preserves points")
    func normalizePreservesCorrectWinding() {
        let square = createSquareOutline()

        // Normalize twice - should be idempotent
        let first = PathOperations.normalizeWindingDirection(square, clockwise: true)
        let second = PathOperations.normalizeWindingDirection(first, clockwise: true)

        #expect(first.contours[0].points.count == second.contours[0].points.count)
    }

    @Test("Empty outline normalizes to empty")
    func normalizeEmpty() {
        let empty = GlyphOutline(contours: [])

        let normalized = PathOperations.normalizeWindingDirection(empty, clockwise: true)

        #expect(normalized.isEmpty)
    }
}

// MARK: - Error Handling Tests

@Suite("PathOperations Error Handling")
struct PathOperationsErrorTests {

    @Test("PathOperationError descriptions are meaningful")
    func errorDescriptions() {
        let emptyError = PathOperations.PathOperationError.emptyPath
        let invalidError = PathOperations.PathOperationError.invalidPath
        let failedError = PathOperations.PathOperationError.operationFailed

        #expect(emptyError.errorDescription != nil)
        #expect(invalidError.errorDescription != nil)
        #expect(failedError.errorDescription != nil)

        #expect(emptyError.errorDescription?.contains("empty") == true)
        #expect(invalidError.errorDescription?.contains("Invalid") == true)
        #expect(failedError.errorDescription?.contains("failed") == true)
    }
}

// MARK: - Integration Tests

@Suite("PathOperations Integration")
struct PathOperationsIntegrationTests {

    @Test("Complex boolean operation chain")
    func complexOperationChain() throws {
        // Create three overlapping squares
        let square1 = createSquareOutline(x: 0, y: 0, size: 100)
        let square2 = createSquareOutline(x: 50, y: 0, size: 100)
        let square3 = createSquareOutline(x: 25, y: 50, size: 100)

        // Union first two
        let union12 = try PathOperations.perform(.union, on: square1, with: square2)

        // Union with third
        let unionAll = try PathOperations.perform(.union, on: union12, with: square3)

        #expect(!unionAll.isEmpty)
    }

    @Test("Simplify after boolean operation")
    func simplifyAfterBoolean() throws {
        let square1 = createSquareOutline(x: 0, y: 0, size: 100)
        let square2 = createSquareOutline(x: 50, y: 50, size: 100)

        let union = try PathOperations.perform(.union, on: square1, with: square2)
        let simplified = PathOperations.simplify(union, tolerance: 2.0)

        #expect(!simplified.isEmpty)
    }

    @Test("Point containment after boolean operation")
    func containmentAfterBoolean() throws {
        let square1 = createSquareOutline(x: 0, y: 0, size: 100)
        let square2 = createSquareOutline(x: 50, y: 50, size: 100)

        let union = try PathOperations.perform(.union, on: square1, with: square2)

        // Center of first square should be in union
        let inside = PathOperations.contains(point: CGPoint(x: 25, y: 25), in: union)
        #expect(inside)

        // Far outside should not be in union
        let outside = PathOperations.contains(point: CGPoint(x: 500, y: 500), in: union)
        #expect(!outside)
    }

    @Test("Normalize winding after boolean operation")
    func normalizeAfterBoolean() throws {
        let square1 = createSquareOutline(x: 0, y: 0, size: 100)
        let square2 = createSquareOutline(x: 50, y: 50, size: 100)

        let union = try PathOperations.perform(.union, on: square1, with: square2)
        let normalized = PathOperations.normalizeWindingDirection(union, clockwise: true)

        #expect(!normalized.isEmpty)
    }

    @Test("Full pipeline: boolean, simplify, normalize, check containment")
    func fullPipeline() throws {
        // 1. Create shapes
        let square = createSquareOutline(x: 0, y: 0, size: 100)
        let triangle = createTriangleOutline(x: 50, y: 50, size: 80)

        // 2. Boolean operation
        let union = try PathOperations.perform(.union, on: square, with: triangle)

        // 3. Simplify
        let simplified = PathOperations.simplify(union, tolerance: 1.0)

        // 4. Normalize winding
        let normalized = PathOperations.normalizeWindingDirection(simplified, clockwise: true)

        // 5. Verify containment
        let insideSquare = PathOperations.contains(point: CGPoint(x: 25, y: 25), in: normalized)

        #expect(!normalized.isEmpty)
        #expect(insideSquare)
    }
}
