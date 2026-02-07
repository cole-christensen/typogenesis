import SwiftUI

struct GlyphCanvas: View {
    let glyph: Glyph
    let metrics: FontMetrics

    @State private var scale: CGFloat = 1.0
    @State private var baseScale: CGFloat = 1.0
    @State private var offset: CGSize = .zero
    @State private var lastCommittedOffset: CGSize = .zero
    @State private var showGrid = true
    @State private var showMetrics = true

    private let gridColor = Color.gray.opacity(0.2)
    private let metricsColor = Color.blue.opacity(0.5)
    private let outlineColor = Color.primary

    var body: some View {
        GeometryReader { geometry in
            ZStack {
                Color(nsColor: .controlBackgroundColor)

                Canvas { context, size in
                    let transform = makeTransform(size: size)

                    if showGrid {
                        drawGrid(context: context, size: size, transform: transform)
                    }

                    if showMetrics {
                        drawMetrics(context: context, size: size, transform: transform)
                    }

                    drawGlyph(context: context, transform: transform)
                }
            }
            .gesture(magnificationGesture)
            .gesture(dragGesture)
            .overlay(alignment: .topTrailing) {
                canvasControls
            }
        }
    }

    private func makeTransform(size: CGSize) -> CGAffineTransform {
        // Guard against division by zero
        let safeUnitsPerEm = max(CGFloat(metrics.unitsPerEm), 1)
        let minDimension = max(min(size.width, size.height), 1)
        let baseScale = minDimension / safeUnitsPerEm * 0.7
        let finalScale = baseScale * scale

        let centerX = size.width / 2 + offset.width
        let centerY = size.height / 2 + offset.height

        return CGAffineTransform(translationX: centerX, y: centerY)
            .scaledBy(x: finalScale, y: -finalScale)
    }

    private func drawGrid(context: GraphicsContext, size: CGSize, transform: CGAffineTransform) {
        let gridSize: CGFloat = 50
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

        context.stroke(path, with: .color(gridColor), lineWidth: 0.5)
    }

    private func drawMetrics(context: GraphicsContext, size: CGSize, transform: CGAffineTransform) {
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

        for (y, color) in metricsLines {
            let start = CGPoint(x: left, y: CGFloat(y)).applying(transform)
            let end = CGPoint(x: right, y: CGFloat(y)).applying(transform)

            var linePath = Path()
            linePath.move(to: start)
            linePath.addLine(to: end)
            context.stroke(linePath, with: .color(color.opacity(0.5)), lineWidth: 1)
        }

        let lsb = CGPoint(x: CGFloat(glyph.leftSideBearing), y: 0).applying(transform)
        let advance = CGPoint(x: CGFloat(glyph.advanceWidth), y: 0).applying(transform)

        var verticalPath = Path()
        verticalPath.move(to: CGPoint(x: lsb.x, y: 0))
        verticalPath.addLine(to: CGPoint(x: lsb.x, y: size.height))
        verticalPath.move(to: CGPoint(x: advance.x, y: 0))
        verticalPath.addLine(to: CGPoint(x: advance.x, y: size.height))

        context.stroke(verticalPath, with: .color(Color.gray.opacity(0.5)), style: StrokeStyle(lineWidth: 1, dash: [4, 4]))
    }

    private func drawGlyph(context: GraphicsContext, transform: CGAffineTransform) {
        guard !glyph.outline.isEmpty else { return }

        let cgPath = glyph.outline.toCGPath()
        let transformedPath = Path(cgPath).applying(transform)

        context.fill(transformedPath, with: .color(Color.primary.opacity(0.8)))
        context.stroke(transformedPath, with: .color(outlineColor), lineWidth: 1)
    }

    private var magnificationGesture: some Gesture {
        MagnificationGesture()
            .onChanged { value in
                scale = max(0.1, min(10, baseScale * value))
            }
            .onEnded { value in
                baseScale = max(0.1, min(10, baseScale * value))
                scale = baseScale
            }
    }

    private var dragGesture: some Gesture {
        DragGesture()
            .onChanged { value in
                offset = CGSize(
                    width: lastCommittedOffset.width + value.translation.width,
                    height: lastCommittedOffset.height + value.translation.height
                )
            }
            .onEnded { value in
                lastCommittedOffset = CGSize(
                    width: lastCommittedOffset.width + value.translation.width,
                    height: lastCommittedOffset.height + value.translation.height
                )
                offset = lastCommittedOffset
            }
    }

    private var canvasControls: some View {
        VStack(spacing: 8) {
            Toggle(isOn: $showGrid) {
                Image(systemName: "grid")
            }
            .toggleStyle(.button)

            Toggle(isOn: $showMetrics) {
                Image(systemName: "ruler")
            }
            .toggleStyle(.button)

            Button {
                scale = 1.0
                baseScale = 1.0
                offset = .zero
                lastCommittedOffset = .zero
            } label: {
                Image(systemName: "arrow.counterclockwise")
            }
        }
        .padding(8)
    }
}

#Preview {
    let outline = GlyphOutline(contours: [
        Contour(points: [
            PathPoint(position: CGPoint(x: 100, y: 0), type: .corner),
            PathPoint(position: CGPoint(x: 100, y: 500), type: .corner),
            PathPoint(position: CGPoint(x: 150, y: 500), type: .corner),
            PathPoint(position: CGPoint(x: 150, y: 50), type: .corner),
            PathPoint(position: CGPoint(x: 350, y: 50), type: .corner),
            PathPoint(position: CGPoint(x: 350, y: 500), type: .corner),
            PathPoint(position: CGPoint(x: 400, y: 500), type: .corner),
            PathPoint(position: CGPoint(x: 400, y: 0), type: .corner),
        ], isClosed: true)
    ])

    let glyph = Glyph(
        character: "H",
        outline: outline,
        advanceWidth: 500,
        leftSideBearing: 50
    )

    return GlyphCanvas(glyph: glyph, metrics: FontMetrics())
        .frame(width: 600, height: 500)
}
