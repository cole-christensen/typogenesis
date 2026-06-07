import SwiftUI

/// Shared drawing routines used by both GlyphCanvas (read-only preview)
/// and InteractiveGlyphCanvas (editable canvas).
///
/// All methods are static and pure -- they only write into the provided
/// GraphicsContext and never mutate external state.
enum GlyphCanvasRenderer {

    // MARK: - Coordinate Transform

    /// Builds the affine transform that maps glyph units to screen points.
    ///
    /// The transform centres the glyph in the view, scales it to fit at 70 %
    /// of the smaller dimension, and flips the Y axis so that positive-Y
    /// points upward (matching font coordinate conventions).
    static func makeTransform(
        size: CGSize,
        metrics: FontMetrics,
        scale: CGFloat,
        offset: CGSize
    ) -> CGAffineTransform {
        let safeUnitsPerEm = max(CGFloat(metrics.unitsPerEm), 1)
        let minDimension = max(min(size.width, size.height), 1)
        let baseScale = minDimension / safeUnitsPerEm * 0.7
        let finalScale = baseScale * scale

        let centerX = size.width / 2 + offset.width
        let centerY = size.height / 2 + offset.height

        return CGAffineTransform(translationX: centerX, y: centerY)
            .scaledBy(x: finalScale, y: -finalScale)
    }

    // MARK: - Grid

    /// Draws a unit-space grid that covers the visible portion of the canvas.
    static func drawGrid(
        context: GraphicsContext,
        size: CGSize,
        transform: CGAffineTransform,
        color: Color = Color.gray.opacity(0.2),
        gridSize: CGFloat = 50
    ) {
        let inverseTransform = transform.inverted()

        let topLeft = CGPoint(x: 0, y: 0).applying(inverseTransform)
        let bottomRight = CGPoint(x: size.width, y: size.height).applying(inverseTransform)

        let minX = (topLeft.x / gridSize).rounded(.down) * gridSize
        let maxX = (bottomRight.x / gridSize).rounded(.up) * gridSize
        let minY = (bottomRight.y / gridSize).rounded(.down) * gridSize
        let maxY = (topLeft.y / gridSize).rounded(.up) * gridSize

        var path = Path()

        var x = minX
        while x <= maxX {
            let start = CGPoint(x: x, y: minY).applying(transform)
            let end = CGPoint(x: x, y: maxY).applying(transform)
            path.move(to: start)
            path.addLine(to: end)
            x += gridSize
        }

        var y = minY
        while y <= maxY {
            let start = CGPoint(x: minX, y: y).applying(transform)
            let end = CGPoint(x: maxX, y: y).applying(transform)
            path.move(to: start)
            path.addLine(to: end)
            y += gridSize
        }

        context.stroke(path, with: .color(color), lineWidth: 0.5)
    }

    // MARK: - Metrics Lines

    /// Draws horizontal metric lines (baseline, x-height, cap-height, ascender,
    /// descender) and vertical dashed lines for left-side-bearing and advance width.
    static func drawMetrics(
        context: GraphicsContext,
        size: CGSize,
        transform: CGAffineTransform,
        metrics: FontMetrics,
        leftSideBearing: Int,
        advanceWidth: Int
    ) {
        let inverseTransform = transform.inverted()
        let left = CGPoint(x: 0, y: 0).applying(inverseTransform).x
        let right = CGPoint(x: size.width, y: 0).applying(inverseTransform).x

        let metricsLines: [(Int, Color)] = [
            (metrics.baseline, .blue),
            (metrics.xHeight, .green),
            (metrics.capHeight, .orange),
            (metrics.ascender, .red),
            (metrics.descender, .purple)
        ]

        for (yValue, color) in metricsLines {
            let start = CGPoint(x: left, y: CGFloat(yValue)).applying(transform)
            let end = CGPoint(x: right, y: CGFloat(yValue)).applying(transform)

            var linePath = Path()
            linePath.move(to: start)
            linePath.addLine(to: end)
            context.stroke(linePath, with: .color(color.opacity(0.5)), lineWidth: 1)
        }

        let lsb = CGPoint(x: CGFloat(leftSideBearing), y: 0).applying(transform)
        let advance = CGPoint(x: CGFloat(advanceWidth), y: 0).applying(transform)

        var verticalPath = Path()
        verticalPath.move(to: CGPoint(x: lsb.x, y: 0))
        verticalPath.addLine(to: CGPoint(x: lsb.x, y: size.height))
        verticalPath.move(to: CGPoint(x: advance.x, y: 0))
        verticalPath.addLine(to: CGPoint(x: advance.x, y: size.height))

        context.stroke(
            verticalPath,
            with: .color(Color.gray.opacity(0.5)),
            style: StrokeStyle(lineWidth: 1, dash: [4, 4])
        )
    }
}
