import Testing
import Foundation
import CoreGraphics
import AppKit
@testable import Typogenesis

@Suite("Image Processing Tests")
struct ImageProcessingTests {

    // MARK: - ImageProcessor Tests

    @Test("PixelData toBinary threshold conversion")
    func testToBinaryThreshold() throws {
        // Create simple pixel data - need to provide data array
        var data = [UInt8](repeating: 200, count: 16)  // 4x4 with light values

        // Set first row as dark (foreground)
        for x in 0..<4 {
            data[x] = 50  // Dark = foreground
        }

        let pixelData = ImageProcessor.PixelData(data: data, width: 4, height: 4)
        let binary = pixelData.toBinary(threshold: 128)

        // First row should be true (dark = foreground)
        #expect(binary[0][0])
        #expect(binary[0][1])
        #expect(binary[0][2])
        #expect(binary[0][3])

        // Second row should be false (light = background)
        #expect(!binary[1][0])
        #expect(!binary[1][1])
    }

    @Test("PixelData pixel access")
    func testPixelAccess() throws {
        var data = [UInt8](repeating: 128, count: 9)  // 3x3
        data[4] = 50  // Center pixel is dark

        let pixelData = ImageProcessor.PixelData(data: data, width: 3, height: 3)

        #expect(pixelData.pixel(x: 1, y: 1) == 50)  // Center
        #expect(pixelData.pixel(x: 0, y: 0) == 128)  // Corner
        #expect(pixelData.isBlack(x: 1, y: 1, threshold: 128) == true)
        #expect(pixelData.isBlack(x: 0, y: 0, threshold: 128) == false)
    }

    @Test("BoundingBox detection from pixel data")
    func testBoundingBoxDetection() throws {
        // Create 10x10 image with a square in the middle
        var data = [UInt8](repeating: 255, count: 100)  // All white

        // Create a 4x4 dark square starting at (3,3)
        for y in 3..<7 {
            for x in 3..<7 {
                data[y * 10 + x] = 50  // Dark pixel
            }
        }

        let pixelData = ImageProcessor.PixelData(data: data, width: 10, height: 10)
        let bounds = ImageProcessor.detectCharacterBounds(
            in: pixelData,
            minSize: 2,
            padding: 0
        )

        #expect(bounds.count == 1)
        guard let first = bounds.first else {
            Issue.record("Expected at least one bounding box but got none")
            return
        }
        #expect(first.origin.x >= 3)
        #expect(first.origin.y >= 3)
        #expect(first.size.width >= 4)
        #expect(first.size.height >= 4)
    }

    // MARK: - EdgeDetector Tests

    @Test("Edge detection finds edges")
    func testEdgeDetection() throws {
        // Create a simple 5x5 binary image with a filled square
        var binary = [[Bool]](repeating: [Bool](repeating: false, count: 5), count: 5)

        // Fill a 3x3 square in the middle
        for y in 1..<4 {
            for x in 1..<4 {
                binary[y][x] = true
            }
        }

        let chains = EdgeDetector.detectEdges(binary: binary, width: 5, height: 5)

        // Should find at least one edge chain
        #expect(chains.count >= 1)

        // First chain should have points
        guard let first = chains.first else {
            Issue.record("Expected at least one edge chain but got none")
            return
        }
        #expect(first.points.count >= 4)  // At least 4 points for a square
    }

    @Test("Douglas-Peucker simplification reduces points")
    func testSimplification() throws {
        // Create a chain with many points on a roughly straight line
        var points: [EdgeDetector.EdgePoint] = []
        for i in 0..<20 {
            points.append(EdgeDetector.EdgePoint(
                x: i,
                y: i + (i % 2 == 0 ? 0 : 1),  // Slight zigzag
                direction: .right
            ))
        }

        let chain = EdgeDetector.EdgeChain(points: points, isClosed: false)
        let simplified = EdgeDetector.simplify(chain: chain, tolerance: 2.0)

        // Simplified should have fewer points
        #expect(simplified.count < points.count)
        #expect(simplified.count >= 2)  // At least start and end
    }

    @Test("Corner detection identifies sharp angles")
    func testCornerDetection() throws {
        // Create an L-shape with a clear corner at index 1
        // The corner detection works by looking at angle change between segments
        let points: [CGPoint] = [
            CGPoint(x: 0, y: 0),
            CGPoint(x: 100, y: 0),   // End of horizontal segment, start of vertical
            CGPoint(x: 100, y: 100)  // End of vertical segment
        ]

        let corners = EdgeDetector.detectCorners(in: points, angleThreshold: 45)

        // Should detect at least one corner
        #expect(corners.count >= 1)

        // The middle point (index 1) should be detected as a corner
        // since there's a 90-degree angle change there
        #expect(corners.contains(1))
    }

    // MARK: - ContourTracer Tests

    @Test("Trace contours from binary image")
    func testContourTracing() throws {
        // Create a simple filled shape
        var binary = [[Bool]](repeating: [Bool](repeating: false, count: 10), count: 10)

        // Fill a 6x6 square
        for y in 2..<8 {
            for x in 2..<8 {
                binary[y][x] = true
            }
        }

        let contours = try ContourTracer.trace(
            binary: binary,
            width: 10,
            height: 10,
            settings: ContourTracer.TracingSettings(
                simplificationTolerance: 1.0,
                minContourLength: 4
            )
        )

        #expect(contours.count >= 1)

        guard let first = contours.first else {
            Issue.record("Expected at least one traced contour but got none")
            return
        }
        #expect(first.points.count >= 4)
        #expect(first.isClosed)
    }

    @Test("toGlyphContour converts traced contour")
    func testToGlyphContour() throws {
        let tracedContour = ContourTracer.TracedContour(
            points: [
                CGPoint(x: 0, y: 0),
                CGPoint(x: 100, y: 0),
                CGPoint(x: 100, y: 100),
                CGPoint(x: 0, y: 100)
            ],
            cornerIndices: [0, 1, 2, 3],
            isClosed: true
        )

        let glyphContour = tracedContour.toGlyphContour()

        #expect(glyphContour.points.count == 4)
        #expect(glyphContour.isClosed)

        // All points should be corners
        for point in glyphContour.points {
            #expect(point.type == .corner)
        }
    }

    // MARK: - BezierFitter Tests

    @Test("Fit bezier to straight line")
    func testBezierFitStraightLine() throws {
        let points: [CGPoint] = [
            CGPoint(x: 0, y: 0),
            CGPoint(x: 50, y: 50),
            CGPoint(x: 100, y: 100)
        ]

        let segments = BezierFitter.fitCurves(
            to: points,
            isClosed: false,
            settings: BezierFitter.FittingSettings(errorThreshold: 5.0)
        )

        // Should produce at least one segment
        #expect(segments.count >= 1)

        guard let first = segments.first else {
            Issue.record("Expected at least one bezier segment but got none")
            return
        }
        // Start should be at origin
        #expect(abs(first.start.x - 0) < 1)
        #expect(abs(first.start.y - 0) < 1)
    }

    @Test("Fit bezier to curve")
    func testBezierFitCurve() throws {
        // Create a quarter circle
        var points: [CGPoint] = []
        for i in 0...20 {
            let angle = Double(i) / 20.0 * .pi / 2
            points.append(CGPoint(
                x: cos(angle) * 100,
                y: sin(angle) * 100
            ))
        }

        let segments = BezierFitter.fitCurves(
            to: points,
            isClosed: false,
            settings: BezierFitter.FittingSettings(errorThreshold: 5.0)
        )

        // Should produce segments to approximate the curve
        #expect(segments.count >= 1)
    }

    @Test("Convert bezier segments to PathPoints")
    func testToPathPoints() throws {
        let segments = [
            BezierFitter.BezierSegment(
                start: CGPoint(x: 0, y: 0),
                control1: CGPoint(x: 33, y: 0),
                control2: CGPoint(x: 66, y: 100),
                end: CGPoint(x: 100, y: 100)
            )
        ]

        let pathPoints = BezierFitter.toPathPoints(segments: segments)

        #expect(pathPoints.count >= 2)  // At least start and end

        // First point should be at segment start
        guard let first = pathPoints.first else {
            Issue.record("Expected at least one path point but got none")
            return
        }
        #expect(abs(first.position.x - 0) < 1)
        #expect(abs(first.position.y - 0) < 1)
    }

    @Test("BezierSegment evaluation")
    func testBezierEvaluation() throws {
        let segment = BezierFitter.BezierSegment(
            start: CGPoint(x: 0, y: 0),
            control1: CGPoint(x: 0, y: 50),
            control2: CGPoint(x: 100, y: 50),
            end: CGPoint(x: 100, y: 100)
        )

        // At t=0, should be at start
        let start = segment.evaluate(at: 0)
        #expect(abs(start.x - 0) < 0.001)
        #expect(abs(start.y - 0) < 0.001)

        // At t=1, should be at end
        let end = segment.evaluate(at: 1)
        #expect(abs(end.x - 100) < 0.001)
        #expect(abs(end.y - 100) < 0.001)

        // At t=0.5, should be somewhere in the middle
        let mid = segment.evaluate(at: 0.5)
        #expect(mid.x > 0 && mid.x < 100)
        #expect(mid.y > 0 && mid.y < 100)
    }

    // MARK: - Vectorizer Tests

    @Test("Vectorizer creates glyph outline from image")
    func testVectorizerCreatesOutline() async throws {
        // Create a simple test image with a square
        let size = 64
        guard let context = CGContext(
            data: nil,
            width: size,
            height: size,
            bitsPerComponent: 8,
            bytesPerRow: size * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            throw VectorizerTestError.failedToCreateImage
        }

        // White background
        context.setFillColor(CGColor.white)
        context.fill(CGRect(x: 0, y: 0, width: size, height: size))

        // Black square
        context.setFillColor(CGColor.black)
        context.fill(CGRect(x: 16, y: 16, width: 32, height: 32))

        guard let cgImage = context.makeImage() else {
            throw VectorizerTestError.failedToCreateImage
        }

        let metrics = FontMetrics(
            unitsPerEm: 1000,
            ascender: 800,
            descender: -200,
            xHeight: 500,
            capHeight: 700,
            lineGap: 90
        )

        let outline = try Vectorizer.vectorizeCharacter(
            image: cgImage,
            metrics: metrics,
            settings: .default
        )

        // Should produce at least one contour
        #expect(outline.contours.count >= 1)

        // Contour should have points
        guard let first = outline.contours.first else {
            Issue.record("Expected at least one contour from vectorization but got none")
            return
        }
        #expect(first.points.count >= 4)
    }

    enum VectorizerTestError: Error {
        case failedToCreateImage
    }
}

@Suite("Coordinate Transformation Tests")
struct CoordinateTransformationTests {

    @Test("toGlyphOutline scales to cap height")
    func testScaleToCapHeight() throws {
        let tracedContours = [
            ContourTracer.TracedContour(
                points: [
                    CGPoint(x: 0, y: 0),
                    CGPoint(x: 100, y: 0),
                    CGPoint(x: 100, y: 100),
                    CGPoint(x: 0, y: 100)
                ],
                cornerIndices: [0, 1, 2, 3],
                isClosed: true
            )
        ]

        let metrics = FontMetrics(
            unitsPerEm: 1000,
            ascender: 800,
            descender: -200,
            xHeight: 500,
            capHeight: 700,
            lineGap: 90
        )

        let outline = ContourTracer.toGlyphOutline(
            contours: tracedContours,
            metrics: metrics,
            fitToCapHeight: true
        )

        #expect(outline.contours.count == 1)

        // Height should be scaled to cap height
        let bounds = outline.boundingBox
        // Allow some tolerance for scaling
        #expect(bounds.height >= 600)  // Should be close to cap height
        #expect(bounds.height <= 800)
    }

    @Test("Empty contours produce empty outline")
    func testEmptyContours() throws {
        let outline = ContourTracer.toGlyphOutline(
            contours: [],
            metrics: FontMetrics()
        )

        #expect(outline.isEmpty)
        #expect(outline.contours.count == 0)
    }
}
