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
        GlyphCanvasRenderer.makeTransform(
            size: size, metrics: metrics, scale: scale, offset: offset
        )
    }

    private func drawGrid(context: GraphicsContext, size: CGSize, transform: CGAffineTransform) {
        GlyphCanvasRenderer.drawGrid(
            context: context, size: size, transform: transform, color: gridColor
        )
    }

    private func drawMetrics(context: GraphicsContext, size: CGSize, transform: CGAffineTransform) {
        GlyphCanvasRenderer.drawMetrics(
            context: context, size: size, transform: transform,
            metrics: metrics,
            leftSideBearing: glyph.leftSideBearing,
            advanceWidth: glyph.advanceWidth
        )
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
