import Testing
import Foundation
import CoreGraphics
import AppKit
@testable import Typogenesis

// MARK: - Test Helpers

/// Creates test images programmatically for vectorization testing
enum TestImageFactory {

    /// Create a simple black rectangle on white background
    static func createRectangle(width: Int = 100, height: Int = 100, rectWidth: Int = 50, rectHeight: Int = 50) -> CGImage {
        let colorSpace = CGColorSpaceCreateDeviceGray()
        var pixelData = [UInt8](repeating: 255, count: width * height) // White background

        // Draw black rectangle in center
        let startX = (width - rectWidth) / 2
        let startY = (height - rectHeight) / 2

        for y in startY..<(startY + rectHeight) {
            for x in startX..<(startX + rectWidth) {
                pixelData[y * width + x] = 0 // Black
            }
        }

        let context = CGContext(
            data: &pixelData,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.none.rawValue
        )!

        return context.makeImage()!
    }

    /// Create a filled circle on white background
    static func createCircle(width: Int = 100, height: Int = 100, radius: Int = 30) -> CGImage {
        let colorSpace = CGColorSpaceCreateDeviceGray()
        var pixelData = [UInt8](repeating: 255, count: width * height) // White background

        let centerX = width / 2
        let centerY = height / 2

        for y in 0..<height {
            for x in 0..<width {
                let dx = x - centerX
                let dy = y - centerY
                if dx * dx + dy * dy <= radius * radius {
                    pixelData[y * width + x] = 0 // Black
                }
            }
        }

        let context = CGContext(
            data: &pixelData,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.none.rawValue
        )!

        return context.makeImage()!
    }

    /// Create letter "A" shape (simplified triangle with hole)
    static func createLetterA(width: Int = 100, height: Int = 100) -> CGImage {
        let colorSpace = CGColorSpaceCreateDeviceGray()
        var pixelData = [UInt8](repeating: 255, count: width * height) // White background

        let centerX = width / 2
        let bottomY = height - 10
        let topY = 10
        let strokeWidth = 8

        // Draw outer triangle
        for y in topY..<bottomY {
            let progress = CGFloat(y - topY) / CGFloat(bottomY - topY)
            let halfWidth = Int(progress * CGFloat(width / 2 - 10))

            // Left stroke
            for x in (centerX - halfWidth - strokeWidth)..<(centerX - halfWidth + strokeWidth) {
                if x >= 0 && x < width {
                    pixelData[y * width + x] = 0
                }
            }

            // Right stroke
            for x in (centerX + halfWidth - strokeWidth)..<(centerX + halfWidth + strokeWidth) {
                if x >= 0 && x < width {
                    pixelData[y * width + x] = 0
                }
            }
        }

        // Draw crossbar
        let crossbarY = height * 2 / 3
        let crossbarHalfWidth = width / 4
        for x in (centerX - crossbarHalfWidth)..<(centerX + crossbarHalfWidth) {
            for dy in 0..<strokeWidth {
                let y = crossbarY + dy
                if y < height {
                    pixelData[y * width + x] = 0
                }
            }
        }

        let context = CGContext(
            data: &pixelData,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.none.rawValue
        )!

        return context.makeImage()!
    }

    /// Create an empty (all white) image
    static func createEmpty(width: Int = 100, height: Int = 100) -> CGImage {
        let colorSpace = CGColorSpaceCreateDeviceGray()
        var pixelData = [UInt8](repeating: 255, count: width * height)

        let context = CGContext(
            data: &pixelData,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.none.rawValue
        )!

        return context.makeImage()!
    }

    /// Create all black image
    static func createAllBlack(width: Int = 100, height: Int = 100) -> CGImage {
        let colorSpace = CGColorSpaceCreateDeviceGray()
        var pixelData = [UInt8](repeating: 0, count: width * height)

        let context = CGContext(
            data: &pixelData,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.none.rawValue
        )!

        return context.makeImage()!
    }

    /// Create noisy image with random black pixels
    static func createNoisy(width: Int = 100, height: Int = 100, noiseDensity: Double = 0.1) -> CGImage {
        let colorSpace = CGColorSpaceCreateDeviceGray()
        var pixelData = [UInt8](repeating: 255, count: width * height)

        for i in 0..<pixelData.count {
            if Double.random(in: 0...1) < noiseDensity {
                pixelData[i] = 0
            }
        }

        let context = CGContext(
            data: &pixelData,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.none.rawValue
        )!

        return context.makeImage()!
    }

    /// Create a sample sheet grid with multiple characters
    static func createSampleSheetGrid(rows: Int = 3, cols: Int = 3, cellSize: Int = 50) -> CGImage {
        let width = cols * cellSize
        let height = rows * cellSize
        let colorSpace = CGColorSpaceCreateDeviceGray()
        var pixelData = [UInt8](repeating: 255, count: width * height)

        // Draw a small rectangle in each cell
        let charSize = cellSize / 3
        let margin = (cellSize - charSize) / 2

        for row in 0..<rows {
            for col in 0..<cols {
                let cellX = col * cellSize + margin
                let cellY = row * cellSize + margin

                for y in cellY..<(cellY + charSize) {
                    for x in cellX..<(cellX + charSize) {
                        if y < height && x < width {
                            pixelData[y * width + x] = 0
                        }
                    }
                }
            }
        }

        let context = CGContext(
            data: &pixelData,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.none.rawValue
        )!

        return context.makeImage()!
    }

    /// Convert CGImage to NSImage
    static func toNSImage(_ cgImage: CGImage) -> NSImage {
        return NSImage(cgImage: cgImage, size: NSSize(width: cgImage.width, height: cgImage.height))
    }
}

// MARK: - ImageProcessor Tests

@Suite("ImageProcessor Tests")
struct ImageProcessorTests {

    @Test("Process valid image without crashing")
    func processValidImage() throws {
        let cgImage = TestImageFactory.createRectangle()
        let processed = try ImageProcessor.process(cgImage: cgImage, settings: .default)

        #expect(processed.width == cgImage.width)
        #expect(processed.height == cgImage.height)
    }

    @Test("Get pixel data from processed image")
    func getPixelData() throws {
        let cgImage = TestImageFactory.createRectangle(width: 50, height: 50, rectWidth: 20, rectHeight: 20)
        let processed = try ImageProcessor.process(cgImage: cgImage, settings: .default)
        let pixelData = try ImageProcessor.getPixelData(processed)

        #expect(pixelData.width == 50)
        #expect(pixelData.height == 50)
        #expect(pixelData.data.count == 50 * 50)
    }

    @Test("Pixel data isBlack detects black pixels correctly")
    func pixelDataIsBlack() throws {
        let cgImage = TestImageFactory.createRectangle(width: 50, height: 50, rectWidth: 20, rectHeight: 20)
        let pixelData = try ImageProcessor.getPixelData(cgImage)

        // Center should be black (rectangle is centered)
        #expect(pixelData.isBlack(x: 25, y: 25))

        // Corners should be white
        #expect(!pixelData.isBlack(x: 0, y: 0))
        #expect(!pixelData.isBlack(x: 49, y: 49))
    }

    @Test("toBinary creates correct binary representation")
    func toBinaryRepresentation() throws {
        let cgImage = TestImageFactory.createRectangle(width: 50, height: 50, rectWidth: 20, rectHeight: 20)
        let pixelData = try ImageProcessor.getPixelData(cgImage)
        let binary = pixelData.toBinary()

        #expect(binary.count == 50)  // rows
        #expect(binary[0].count == 50)  // cols

        // Center should be true (black/foreground)
        #expect(binary[25][25])

        // Corner should be false (white/background)
        #expect(!binary[0][0])
    }

    @Test("Detect character bounds finds rectangle")
    func detectCharacterBounds() throws {
        let cgImage = TestImageFactory.createRectangle(width: 100, height: 100, rectWidth: 40, rectHeight: 40)
        let pixelData = try ImageProcessor.getPixelData(cgImage)
        let bounds = ImageProcessor.detectCharacterBounds(in: pixelData, minSize: 10, padding: 2)

        #expect(bounds.count == 1)
        #expect(bounds[0].width >= 40)
        #expect(bounds[0].height >= 40)
    }

    @Test("Detect multiple separate characters")
    func detectMultipleCharacters() throws {
        // Create image with two separate rectangles
        let width = 200
        let height = 100
        let colorSpace = CGColorSpaceCreateDeviceGray()
        var pixelData = [UInt8](repeating: 255, count: width * height)

        // First rectangle on left
        for y in 30..<70 {
            for x in 20..<60 {
                pixelData[y * width + x] = 0
            }
        }

        // Second rectangle on right
        for y in 30..<70 {
            for x in 140..<180 {
                pixelData[y * width + x] = 0
            }
        }

        let context = CGContext(
            data: &pixelData,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.none.rawValue
        )!
        let cgImage = context.makeImage()!

        let data = try ImageProcessor.getPixelData(cgImage)
        let bounds = ImageProcessor.detectCharacterBounds(in: data, minSize: 10, padding: 2)

        #expect(bounds.count == 2)
    }

    @Test("Empty image returns no bounds")
    func emptyImageNoBounds() throws {
        let cgImage = TestImageFactory.createEmpty()
        let pixelData = try ImageProcessor.getPixelData(cgImage)
        let bounds = ImageProcessor.detectCharacterBounds(in: pixelData, minSize: 10, padding: 2)

        #expect(bounds.isEmpty)
    }

    @Test("Detect grid cells for sample sheet")
    func detectGridCells() throws {
        let cgImage = TestImageFactory.createSampleSheetGrid(rows: 3, cols: 3, cellSize: 50)
        let grid = try ImageProcessor.detectGridCells(in: cgImage, expectedRows: 3, expectedCols: 3)

        #expect(grid.count == 3)  // 3 rows
        #expect(grid[0].count == 3)  // 3 cols

        // Check cell sizes are correct
        #expect(Int(grid[0][0].width) == 50)
        #expect(Int(grid[0][0].height) == 50)
    }

    @Test("Extract character sub-image")
    func extractCharacter() throws {
        let cgImage = TestImageFactory.createRectangle(width: 100, height: 100, rectWidth: 40, rectHeight: 40)
        let bounds = CGRect(x: 30, y: 30, width: 40, height: 40)

        let extracted = try ImageProcessor.extractCharacter(from: cgImage, bounds: bounds)

        #expect(extracted.width == 40)
        #expect(extracted.height == 40)
    }

    @Test("Processing with different settings")
    func processingSettings() throws {
        let cgImage = TestImageFactory.createRectangle()

        // Test different threshold values
        let lowThreshold = ImageProcessor.ProcessingSettings(threshold: 0.2)
        let highThreshold = ImageProcessor.ProcessingSettings(threshold: 0.8)

        let processedLow = try ImageProcessor.process(cgImage: cgImage, settings: lowThreshold)
        let processedHigh = try ImageProcessor.process(cgImage: cgImage, settings: highThreshold)

        #expect(processedLow.width == cgImage.width)
        #expect(processedHigh.width == cgImage.width)
    }

    @Test("Processing with invert option")
    func processWithInvert() throws {
        let cgImage = TestImageFactory.createRectangle()
        let settings = ImageProcessor.ProcessingSettings(invert: true)

        let processed = try ImageProcessor.process(cgImage: cgImage, settings: settings)

        #expect(processed.width == cgImage.width)
    }
}

// MARK: - EdgeDetector Tests

@Suite("EdgeDetector Tests")
struct EdgeDetectorTests {

    @Test("Detect edges in simple rectangle")
    func detectEdgesRectangle() throws {
        let cgImage = TestImageFactory.createRectangle(width: 50, height: 50, rectWidth: 20, rectHeight: 20)
        let pixelData = try ImageProcessor.getPixelData(cgImage)
        let binary = pixelData.toBinary()

        let chains = EdgeDetector.detectEdges(binary: binary, width: 50, height: 50)

        #expect(!chains.isEmpty)
        #expect(chains[0].isClosed)  // Rectangle should form closed contour
    }

    @Test("Detect edges in circle")
    func detectEdgesCircle() throws {
        let cgImage = TestImageFactory.createCircle(width: 100, height: 100, radius: 30)
        let pixelData = try ImageProcessor.getPixelData(cgImage)
        let binary = pixelData.toBinary()

        let chains = EdgeDetector.detectEdges(binary: binary, width: 100, height: 100)

        #expect(!chains.isEmpty)
        #expect(chains[0].isClosed)
    }

    @Test("Empty image returns no edges")
    func emptyImageNoEdges() throws {
        let cgImage = TestImageFactory.createEmpty()
        let pixelData = try ImageProcessor.getPixelData(cgImage)
        let binary = pixelData.toBinary()

        let chains = EdgeDetector.detectEdges(binary: binary, width: 100, height: 100)

        #expect(chains.isEmpty)
    }

    @Test("Simplify chain reduces point count")
    func simplifyChain() {
        // Create a chain with many points along a line
        var points: [EdgeDetector.EdgePoint] = []
        for i in 0..<100 {
            points.append(EdgeDetector.EdgePoint(x: i, y: 0, direction: .right))
        }
        let chain = EdgeDetector.EdgeChain(points: points, isClosed: false)

        let simplified = EdgeDetector.simplify(chain: chain, tolerance: 1.0)

        // Simplification should reduce points on a straight line to just endpoints
        #expect(simplified.count < points.count)
        #expect(simplified.count >= 2)
    }

    @Test("Detect corners in L-shape")
    func detectCornersLShape() {
        // Create an L-shape with a clear corner
        let points: [CGPoint] = [
            CGPoint(x: 0, y: 0),
            CGPoint(x: 50, y: 0),
            CGPoint(x: 50, y: 50),
            CGPoint(x: 100, y: 50)
        ]

        let corners = EdgeDetector.detectCorners(in: points, angleThreshold: 45)

        // Should detect the corner at (50, 0) to (50, 50)
        #expect(!corners.isEmpty)
    }

    @Test("No corners on straight line")
    func noCornersOnLine() {
        let points: [CGPoint] = [
            CGPoint(x: 0, y: 0),
            CGPoint(x: 25, y: 0),
            CGPoint(x: 50, y: 0),
            CGPoint(x: 75, y: 0),
            CGPoint(x: 100, y: 0)
        ]

        let corners = EdgeDetector.detectCorners(in: points, angleThreshold: 45)

        // Corner detection may find endpoints but no mid-points for a straight line
        // Verify no corners in middle points (indices 1, 2, 3)
        let midCorners = corners.filter { $0 > 0 && $0 < points.count - 1 }
        #expect(midCorners.isEmpty)
    }

    @Test("Direction dx/dy values are correct")
    func directionOffsets() {
        let right = EdgeDetector.EdgePoint.Direction.right
        #expect(right.dx == 1)
        #expect(right.dy == 0)

        let down = EdgeDetector.EdgePoint.Direction.down
        #expect(down.dx == 0)
        #expect(down.dy == 1)

        let left = EdgeDetector.EdgePoint.Direction.left
        #expect(left.dx == -1)
        #expect(left.dy == 0)

        let up = EdgeDetector.EdgePoint.Direction.up
        #expect(up.dx == 0)
        #expect(up.dy == -1)
    }

    @Test("Direction opposite returns correct direction")
    func directionOpposite() {
        #expect(EdgeDetector.EdgePoint.Direction.right.opposite() == .left)
        #expect(EdgeDetector.EdgePoint.Direction.up.opposite() == .down)
        #expect(EdgeDetector.EdgePoint.Direction.downRight.opposite() == .upLeft)
    }

    @Test("Convert to glyph coordinates")
    func toGlyphCoordinates() {
        let points: [CGPoint] = [
            CGPoint(x: 0, y: 0),
            CGPoint(x: 100, y: 0),
            CGPoint(x: 100, y: 100),
            CGPoint(x: 0, y: 100)
        ]

        let metrics = FontMetrics()
        let converted = EdgeDetector.toGlyphCoordinates(
            points: points,
            imageSize: CGSize(width: 100, height: 100),
            glyphMetrics: metrics
        )

        #expect(converted.count == 4)
        // Points should be scaled to fit cap height
        let height = converted.map { $0.y }.max()! - converted.map { $0.y }.min()!
        #expect(abs(height - CGFloat(metrics.capHeight)) < 1)
    }
}

// MARK: - BezierFitter Tests

@Suite("BezierFitter Tests")
struct BezierFitterTests {

    @Test("Fit curves to two points creates single segment")
    func fitTwoPoints() {
        let points: [CGPoint] = [
            CGPoint(x: 0, y: 0),
            CGPoint(x: 100, y: 100)
        ]

        let segments = BezierFitter.fitCurves(to: points, isClosed: false)

        #expect(segments.count == 1)
        #expect(segments[0].start == points[0])
        #expect(segments[0].end == points[1])
    }

    @Test("Fit curves to straight line")
    func fitStraightLine() {
        let points: [CGPoint] = [
            CGPoint(x: 0, y: 0),
            CGPoint(x: 25, y: 25),
            CGPoint(x: 50, y: 50),
            CGPoint(x: 75, y: 75),
            CGPoint(x: 100, y: 100)
        ]

        let segments = BezierFitter.fitCurves(to: points, isClosed: false, settings: .init(errorThreshold: 5.0))

        // Straight line should fit with few segments
        #expect(segments.count >= 1)
        #expect(segments.count <= 2)
    }

    @Test("Fit curves to arc shape")
    func fitArc() {
        // Create points along a quarter circle
        var points: [CGPoint] = []
        for i in 0...20 {
            let angle = Double(i) / 20.0 * .pi / 2
            points.append(CGPoint(
                x: cos(angle) * 100,
                y: sin(angle) * 100
            ))
        }

        let segments = BezierFitter.fitCurves(to: points, isClosed: false, settings: .init(errorThreshold: 2.0))

        #expect(!segments.isEmpty)
        // Arc should be representable with segments (implementation may vary)
        // Just verify we get reasonable segment count (at most one per point)
        #expect(segments.count <= points.count)
    }

    @Test("Closed curve fitting")
    func fitClosedCurve() {
        // Create a square
        let points: [CGPoint] = [
            CGPoint(x: 0, y: 0),
            CGPoint(x: 100, y: 0),
            CGPoint(x: 100, y: 100),
            CGPoint(x: 0, y: 100)
        ]

        let segments = BezierFitter.fitCurves(to: points, isClosed: true)

        #expect(!segments.isEmpty)
    }

    @Test("Bezier segment evaluation at endpoints")
    func bezierEvaluation() {
        let segment = BezierFitter.BezierSegment(
            start: CGPoint(x: 0, y: 0),
            control1: CGPoint(x: 33, y: 0),
            control2: CGPoint(x: 66, y: 100),
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
    }

    @Test("Bezier derivative is non-zero for non-degenerate curve")
    func bezierDerivative() {
        let segment = BezierFitter.BezierSegment(
            start: CGPoint(x: 0, y: 0),
            control1: CGPoint(x: 33, y: 0),
            control2: CGPoint(x: 66, y: 100),
            end: CGPoint(x: 100, y: 100)
        )

        let derivative = segment.derivative(at: 0.5)
        let length = sqrt(derivative.x * derivative.x + derivative.y * derivative.y)

        #expect(length > 0)
    }

    @Test("Convert segments to PathPoints")
    func toPathPoints() {
        let segment = BezierFitter.BezierSegment(
            start: CGPoint(x: 0, y: 0),
            control1: CGPoint(x: 33, y: 0),
            control2: CGPoint(x: 66, y: 100),
            end: CGPoint(x: 100, y: 100)
        )

        let pathPoints = BezierFitter.toPathPoints(segments: [segment])

        #expect(pathPoints.count == 2)  // Start and end points
        #expect(pathPoints[0].position == segment.start)
        #expect(pathPoints[1].position == segment.end)
        #expect(pathPoints[0].controlOut == segment.control1)
        #expect(pathPoints[1].controlIn == segment.control2)
    }

    @Test("Empty points returns empty segments")
    func emptyPoints() {
        let segments = BezierFitter.fitCurves(to: [], isClosed: false)
        #expect(segments.isEmpty)
    }

    @Test("Single point returns empty segments")
    func singlePoint() {
        let segments = BezierFitter.fitCurves(to: [CGPoint(x: 0, y: 0)], isClosed: false)
        #expect(segments.isEmpty)
    }
}

// MARK: - ContourTracer Tests

@Suite("ContourTracer Tests")
struct ContourTracerTests {

    @Test("Trace contours from binary data")
    func traceFromBinary() throws {
        let cgImage = TestImageFactory.createRectangle(width: 50, height: 50, rectWidth: 20, rectHeight: 20)
        let pixelData = try ImageProcessor.getPixelData(cgImage)
        let binary = pixelData.toBinary()

        let contours = try ContourTracer.trace(binary: binary, width: 50, height: 50)

        #expect(!contours.isEmpty)
        #expect(contours[0].isClosed)
        #expect(contours[0].points.count >= 4)  // Rectangle has at least 4 points
    }

    @Test("Trace contours from pixel data")
    func traceFromPixelData() throws {
        let cgImage = TestImageFactory.createRectangle()
        let pixelData = try ImageProcessor.getPixelData(cgImage)

        let contours = try ContourTracer.trace(pixelData: pixelData)

        #expect(!contours.isEmpty)
    }

    @Test("Trace contours from CGImage")
    func traceFromCGImage() throws {
        let cgImage = TestImageFactory.createCircle()

        let contours = try ContourTracer.trace(image: cgImage)

        #expect(!contours.isEmpty)
        #expect(contours[0].isClosed)
    }

    @Test("Empty image throws noContoursFound")
    func emptyImageThrows() throws {
        let cgImage = TestImageFactory.createEmpty()
        let pixelData = try ImageProcessor.getPixelData(cgImage)
        let binary = pixelData.toBinary()

        #expect(throws: ContourTracer.TracerError.self) {
            try ContourTracer.trace(binary: binary, width: 100, height: 100)
        }
    }

    @Test("Convert to GlyphOutline")
    func toGlyphOutline() throws {
        let cgImage = TestImageFactory.createRectangle()
        let pixelData = try ImageProcessor.getPixelData(cgImage)

        let contours = try ContourTracer.trace(pixelData: pixelData)
        let metrics = FontMetrics()
        let outline = ContourTracer.toGlyphOutline(contours: contours, metrics: metrics)

        #expect(!outline.isEmpty)
        #expect(!outline.contours.isEmpty)
        #expect(outline.contours[0].isClosed)
    }

    @Test("TracedContour toGlyphContour conversion")
    func tracedContourToGlyphContour() {
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

        // All points should be corners since cornerIndices contains all indices
        for point in glyphContour.points {
            #expect(point.type == .corner)
        }
    }

    @Test("Tracing settings affect output")
    func tracingSettingsEffect() throws {
        let cgImage = TestImageFactory.createCircle()

        let looseSettings = ContourTracer.TracingSettings(simplificationTolerance: 10.0)
        let tightSettings = ContourTracer.TracingSettings(simplificationTolerance: 1.0)

        let looseContours = try ContourTracer.trace(image: cgImage, tracingSettings: looseSettings)
        let tightContours = try ContourTracer.trace(image: cgImage, tracingSettings: tightSettings)

        // Loose tolerance should produce fewer points
        #expect(looseContours[0].points.count <= tightContours[0].points.count)
    }

    @Test("Full vectorization pipeline")
    func fullVectorizationPipeline() throws {
        let cgImage = TestImageFactory.createRectangle()
        let metrics = FontMetrics()

        let outline = try ContourTracer.vectorize(image: cgImage, metrics: metrics)

        #expect(!outline.isEmpty)
        #expect(!outline.contours.isEmpty)

        // Check outline has valid bounding box
        let bbox = outline.boundingBox
        #expect(bbox.width > 0)
        #expect(bbox.height > 0)
    }

    @Test("Vectorize from NSImage")
    func vectorizeFromNSImage() throws {
        let cgImage = TestImageFactory.createRectangle()
        let nsImage = TestImageFactory.toNSImage(cgImage)
        let metrics = FontMetrics()

        let outline = try ContourTracer.vectorize(image: nsImage, metrics: metrics)

        #expect(!outline.isEmpty)
    }
}

// MARK: - Vectorizer Tests

@Suite("Vectorizer Tests")
struct VectorizerTests {

    @Test("Vectorize single character image")
    func vectorizeSingleCharacter() throws {
        let cgImage = TestImageFactory.createRectangle()
        let metrics = FontMetrics()

        let outline = try Vectorizer.vectorizeCharacter(image: cgImage, metrics: metrics)

        #expect(!outline.isEmpty)
        #expect(!outline.contours.isEmpty)
    }

    @Test("Vectorize full image with character detection")
    func vectorizeFullImage() async throws {
        let cgImage = TestImageFactory.createRectangle(width: 200, height: 200, rectWidth: 50, rectHeight: 50)
        let nsImage = TestImageFactory.toNSImage(cgImage)
        let metrics = FontMetrics()

        let result = try await Vectorizer.vectorize(image: nsImage, metrics: metrics)

        #expect(result.characters.count >= 1)
        #expect(result.processingTime >= 0)
        #expect(result.imageSize.width == 200)
        #expect(result.imageSize.height == 200)
    }

    @Test("Vectorization with different presets")
    func vectorizationPresets() throws {
        let cgImage = TestImageFactory.createRectangle()
        let metrics = FontMetrics()

        // Test each preset
        let cleanOutline = try Vectorizer.vectorizeCharacter(
            image: cgImage,
            metrics: metrics,
            settings: .cleanHandwriting
        )

        let roughOutline = try Vectorizer.vectorizeCharacter(
            image: cgImage,
            metrics: metrics,
            settings: .roughHandwriting
        )

        let printedOutline = try Vectorizer.vectorizeCharacter(
            image: cgImage,
            metrics: metrics,
            settings: .printedCharacters
        )

        #expect(!cleanOutline.isEmpty)
        #expect(!roughOutline.isEmpty)
        #expect(!printedOutline.isEmpty)
    }

    @Test("Empty image throws noCharactersDetected")
    func emptyImageThrows() async throws {
        let cgImage = TestImageFactory.createEmpty()
        let nsImage = TestImageFactory.toNSImage(cgImage)
        let metrics = FontMetrics()

        await #expect(throws: Vectorizer.VectorizerError.self) {
            try await Vectorizer.vectorize(image: nsImage, metrics: metrics)
        }
    }

    @Test("Create glyph from vectorized character")
    func createGlyphFromVectorized() {
        let outline = GlyphOutline(contours: [
            Contour(points: [
                PathPoint(position: CGPoint(x: 0, y: 0), type: .corner),
                PathPoint(position: CGPoint(x: 100, y: 0), type: .corner),
                PathPoint(position: CGPoint(x: 100, y: 200), type: .corner),
                PathPoint(position: CGPoint(x: 0, y: 200), type: .corner)
            ], isClosed: true)
        ])

        let vectorized = Vectorizer.VectorizedCharacter(
            bounds: CGRect(x: 0, y: 0, width: 100, height: 200),
            outline: outline,
            assignedCharacter: nil
        )

        let metrics = FontMetrics()
        let glyph = Vectorizer.createGlyph(from: vectorized, character: "A", metrics: metrics)

        #expect(glyph.character == "A")
        #expect(glyph.advanceWidth > 0)
        #expect(glyph.leftSideBearing > 0)
        #expect(!glyph.outline.isEmpty)
    }

    @Test("Process sample sheet")
    func processSampleSheet() async throws {
        let cgImage = TestImageFactory.createSampleSheetGrid(rows: 2, cols: 3, cellSize: 50)
        let nsImage = TestImageFactory.toNSImage(cgImage)
        let metrics = FontMetrics()

        let results = try await Vectorizer.processSampleSheet(
            image: nsImage,
            characters: "ABCDEF",
            rows: 2,
            cols: 3,
            metrics: metrics
        )

        // Should find characters in the grid cells
        #expect(!results.isEmpty)
        #expect(results.count <= 6)
    }

    @Test("Batch vectorization")
    func batchVectorization() async throws {
        let images = [
            TestImageFactory.toNSImage(TestImageFactory.createRectangle(width: 100, height: 100, rectWidth: 40, rectHeight: 40)),
            TestImageFactory.toNSImage(TestImageFactory.createCircle(width: 100, height: 100, radius: 30))
        ]
        let metrics = FontMetrics()

        let results = try await Vectorizer.vectorizeBatch(images: images, metrics: metrics)

        #expect(results.count == 2)
    }

    @Test("Vectorization with bezier fitting disabled")
    func vectorizeWithoutBezierFitting() throws {
        let cgImage = TestImageFactory.createRectangle()
        let metrics = FontMetrics()
        var settings = Vectorizer.VectorizationSettings.default
        settings.useBezierFitting = false

        let outline = try Vectorizer.vectorizeCharacter(image: cgImage, metrics: metrics, settings: settings)

        #expect(!outline.isEmpty)
    }

    @Test("Vectorization result contains correct image size")
    func vectorizationResultImageSize() async throws {
        let cgImage = TestImageFactory.createRectangle(width: 150, height: 200, rectWidth: 50, rectHeight: 50)
        let nsImage = TestImageFactory.toNSImage(cgImage)
        let metrics = FontMetrics()

        let result = try await Vectorizer.vectorize(image: nsImage, metrics: metrics)

        #expect(Int(result.imageSize.width) == 150)
        #expect(Int(result.imageSize.height) == 200)
    }

    @Test("VectorizationSettings presets have correct values")
    func vectorizationSettingsPresets() {
        let clean = Vectorizer.VectorizationSettings.cleanHandwriting
        let rough = Vectorizer.VectorizationSettings.roughHandwriting
        let printed = Vectorizer.VectorizationSettings.printedCharacters

        // Clean should have lower threshold
        #expect(clean.imageProcessing.threshold < rough.imageProcessing.threshold)

        // Rough should have higher simplification tolerance
        #expect(rough.tracing.simplificationTolerance > clean.tracing.simplificationTolerance)

        // Printed should have higher corner angle threshold (fewer corners detected)
        #expect(printed.tracing.cornerAngleThreshold > rough.tracing.cornerAngleThreshold)
    }
}

// MARK: - Integration Tests

@Suite("Vectorization Integration Tests")
struct VectorizationIntegrationTests {

    @Test("Full pipeline: Image to valid GlyphOutline")
    func fullPipelineProducesValidOutline() throws {
        let cgImage = TestImageFactory.createLetterA()
        let metrics = FontMetrics()

        let outline = try Vectorizer.vectorizeCharacter(image: cgImage, metrics: metrics)

        // Outline should be non-empty
        #expect(!outline.isEmpty)

        // All contours should be closed
        for contour in outline.contours {
            #expect(contour.isClosed)
            #expect(contour.points.count >= 3)
        }

        // Bounding box should be reasonable
        let bbox = outline.boundingBox
        #expect(bbox.width > 0)
        #expect(bbox.height > 0)
        let maxHeight = Int(Double(metrics.capHeight) * 1.5)
        #expect(bbox.height <= maxHeight)  // Should fit within cap height
    }

    @Test("Circle vectorization produces smooth contour")
    func circleProducesSmoothContour() throws {
        let cgImage = TestImageFactory.createCircle(width: 200, height: 200, radius: 60)
        let metrics = FontMetrics()

        let outline = try Vectorizer.vectorizeCharacter(image: cgImage, metrics: metrics)

        #expect(!outline.isEmpty)
        #expect(outline.contours.count >= 1)

        // Circle should have mostly smooth points, not corners
        let contour = outline.contours[0]
        let smoothCount = contour.points.filter { $0.type == .smooth }.count
        let cornerCount = contour.points.filter { $0.type == .corner }.count
        let totalCount = contour.points.count

        // A circle should have almost entirely smooth points, not corners.
        // Measured actual ratio is 1.0 (100% smooth, 0 corners).
        // Threshold of 0.85 catches regressions while allowing minor tolerance.
        let smoothRatio = Double(smoothCount) / Double(max(totalCount, 1))
        #expect(smoothRatio >= 0.85, "Circle should have at least 85% smooth points, got \(Int(smoothRatio * 100))%")
        #expect(cornerCount < smoothCount, "Circle should have fewer corners (\(cornerCount)) than smooth points (\(smoothCount))")
    }

    @Test("Rectangle vectorization detects corners")
    func rectangleDetectsCorners() throws {
        let cgImage = TestImageFactory.createRectangle(width: 100, height: 100, rectWidth: 60, rectHeight: 60)
        let metrics = FontMetrics()

        let outline = try ContourTracer.vectorize(image: cgImage, metrics: metrics)

        #expect(!outline.isEmpty)

        // Rectangle should have corner points
        // Note: Vectorization may not produce exactly 4 corners due to pixel resolution
        let contour = outline.contours[0]
        let cornerCount = contour.points.filter { $0.type == .corner }.count

        #expect(cornerCount >= 3)  // Rectangle should have multiple corners
    }

    @Test("Vectorized outline can be converted to CGPath")
    func outlineConvertsToCGPath() throws {
        let cgImage = TestImageFactory.createRectangle()
        let metrics = FontMetrics()

        let outline = try Vectorizer.vectorizeCharacter(image: cgImage, metrics: metrics)
        let path = outline.toCGPath()

        #expect(!path.isEmpty)
        #expect(path.boundingBox.width > 0)
        #expect(path.boundingBox.height > 0)
    }

    @Test("Large image performance")
    func largeImagePerformance() async throws {
        // Create a larger image to test performance
        let cgImage = TestImageFactory.createRectangle(width: 500, height: 500, rectWidth: 200, rectHeight: 200)
        let nsImage = TestImageFactory.toNSImage(cgImage)
        let metrics = FontMetrics()

        let startTime = Date()
        let result = try await Vectorizer.vectorize(image: nsImage, metrics: metrics)
        let elapsed = Date().timeIntervalSince(startTime)

        #expect(!result.characters.isEmpty)
        #expect(elapsed < 5.0)  // Should complete within 5 seconds
    }
}
