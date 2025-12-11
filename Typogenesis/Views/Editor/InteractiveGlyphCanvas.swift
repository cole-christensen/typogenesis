import SwiftUI

struct InteractiveGlyphCanvas: View {
    @ObservedObject var viewModel: GlyphEditorViewModel
    let metrics: FontMetrics

    @State private var scale: CGFloat = 1.0
    @State private var offset: CGSize = .zero
    @State private var showGrid = true
    @State private var showMetrics = true
    @State private var showControlPoints = true
    @State private var currentDragHit: GlyphEditorViewModel.HitTestResult?
    @State private var lastDragPosition: CGPoint?
    @State private var canvasSize: CGSize = .zero

    private let gridColor = Color.gray.opacity(0.2)
    private let pointColor = Color.blue
    private let selectedPointColor = Color.orange
    private let controlPointColor = Color.purple
    private let controlLineColor = Color.purple.opacity(0.5)
    private let hitTolerance: CGFloat = 12

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

                    if showControlPoints {
                        drawControlPoints(context: context, transform: transform)
                    }
                }
                .gesture(editingGesture(in: geometry.size))
                .simultaneousGesture(magnificationGesture)
            }
            .onAppear {
                canvasSize = geometry.size
            }
            .onChange(of: geometry.size) { _, newSize in
                canvasSize = newSize
            }
            .overlay(alignment: .topTrailing) {
                canvasControls
            }
            .overlay(alignment: .topLeading) {
                toolPicker
            }
            .overlay(alignment: .bottom) {
                statusBar
            }
        }
        .onKeyPress(keys: [.delete, .deleteForward]) { _ in
            viewModel.deleteSelectedPoints()
            return .handled
        }
        .onKeyPress(.escape) {
            if viewModel.isDrawingPath {
                viewModel.finishPenPath()
                return .handled
            }
            viewModel.clearSelection()
            return .handled
        }
        .onKeyPress(.return) {
            if viewModel.isDrawingPath {
                viewModel.closePenPath()
                return .handled
            }
            return .ignored
        }
        .onKeyPress(characters: .alphanumerics) { press in
            // Handle keyboard shortcuts
            if press.key == KeyEquivalent("t") && press.modifiers.isEmpty {
                viewModel.togglePointType()
                return .handled
            }
            if press.key == KeyEquivalent("s") && press.modifiers.isEmpty && !viewModel.selectedPointIDs.isEmpty {
                viewModel.smoothSelectedPoints()
                return .handled
            }
            if press.key == KeyEquivalent("c") && press.modifiers.isEmpty && !viewModel.selectedPointIDs.isEmpty {
                viewModel.cornerSelectedPoints()
                return .handled
            }
            return .ignored
        }
        .focusable()
        .focusEffectDisabled()
    }

    // MARK: - Transform

    private func makeTransform(size: CGSize) -> CGAffineTransform {
        let baseScale = min(size.width, size.height) / CGFloat(metrics.unitsPerEm) * 0.7
        let finalScale = baseScale * scale

        let centerX = size.width / 2 + offset.width
        let centerY = size.height / 2 + offset.height

        return CGAffineTransform(translationX: centerX, y: centerY)
            .scaledBy(x: finalScale, y: -finalScale)
    }

    private func screenToGlyph(_ point: CGPoint, size: CGSize) -> CGPoint {
        let transform = makeTransform(size: size).inverted()
        return point.applying(transform)
    }

    private func glyphToScreen(_ point: CGPoint, size: CGSize) -> CGPoint {
        let transform = makeTransform(size: size)
        return point.applying(transform)
    }

    // MARK: - Gestures

    private func editingGesture(in size: CGSize) -> some Gesture {
        DragGesture(minimumDistance: 0)
            .onChanged { value in
                let glyphPoint = screenToGlyph(value.location, size: size)
                let screenTolerance = hitTolerance / (scale * min(size.width, size.height) / CGFloat(metrics.unitsPerEm) * 0.7)

                switch viewModel.currentTool {
                case .select:
                    handleSelectDrag(value: value, glyphPoint: glyphPoint, screenTolerance: screenTolerance, size: size)
                case .pen:
                    handlePenDrag(value: value, glyphPoint: glyphPoint, size: size)
                case .addPoint:
                    // Add point handled on tap end
                    break
                case .deletePoint:
                    // Delete handled on tap end
                    break
                }
            }
            .onEnded { value in
                let glyphPoint = screenToGlyph(value.location, size: size)
                let screenTolerance = hitTolerance / (scale * min(size.width, size.height) / CGFloat(metrics.unitsPerEm) * 0.7)

                switch viewModel.currentTool {
                case .select:
                    handleSelectEnd()
                case .pen:
                    handlePenEnd(value: value, glyphPoint: glyphPoint, size: size)
                case .addPoint:
                    if value.translation.width.magnitude < 3 && value.translation.height.magnitude < 3 {
                        viewModel.addPoint(at: glyphPoint)
                    }
                case .deletePoint:
                    if let hit = viewModel.hitTest(point: glyphPoint, tolerance: screenTolerance) {
                        viewModel.selectPoint(id: hit.pointID)
                        viewModel.deleteSelectedPoints()
                    }
                }
            }
    }

    private func handleSelectDrag(value: DragGesture.Value, glyphPoint: CGPoint, screenTolerance: CGFloat, size: CGSize) {
        if currentDragHit == nil {
            // Starting a new drag - check if we hit something
            if let hit = viewModel.hitTest(point: glyphPoint, tolerance: screenTolerance) {
                currentDragHit = hit
                viewModel.beginDrag()

                // If clicking on an unselected point without shift, select just that point
                let shiftPressed = NSEvent.modifierFlags.contains(.shift)
                if case .point(_, _, let id) = hit, !viewModel.selectedPointIDs.contains(id) {
                    viewModel.selectPoint(id: id, addToSelection: shiftPressed)
                }
            } else {
                // Clicked on empty space - clear selection unless shift is held
                let shiftPressed = NSEvent.modifierFlags.contains(.shift)
                if !shiftPressed {
                    viewModel.clearSelection()
                }
                // Start panning
                offset = CGSize(
                    width: offset.width + value.translation.width - (lastDragPosition?.x ?? 0),
                    height: offset.height + value.translation.height - (lastDragPosition?.y ?? 0)
                )
            }
            lastDragPosition = CGPoint(x: value.translation.width, y: value.translation.height)
        } else {
            // Continue dragging
            let deltaScreen = CGSize(
                width: value.translation.width - (lastDragPosition?.x ?? 0),
                height: value.translation.height - (lastDragPosition?.y ?? 0)
            )
            lastDragPosition = CGPoint(x: value.translation.width, y: value.translation.height)

            // Convert screen delta to glyph delta
            let scaleFactor = scale * min(size.width, size.height) / CGFloat(metrics.unitsPerEm) * 0.7
            let glyphDelta = CGSize(
                width: deltaScreen.width / scaleFactor,
                height: -deltaScreen.height / scaleFactor  // Y is inverted
            )

            if let hit = currentDragHit {
                switch hit {
                case .point:
                    viewModel.moveSelectedPoints(by: glyphDelta)
                case .controlIn:
                    viewModel.moveControlIn(at: hit, to: glyphPoint)
                case .controlOut:
                    viewModel.moveControlOut(at: hit, to: glyphPoint)
                }
            }
        }
    }

    private func handleSelectEnd() {
        currentDragHit = nil
        lastDragPosition = nil
        viewModel.endDrag()
    }

    // MARK: - Pen Tool Handling

    @State private var penDragStartPoint: CGPoint?

    private func handlePenDrag(value: DragGesture.Value, glyphPoint: CGPoint, size: CGSize) {
        if penDragStartPoint == nil {
            penDragStartPoint = glyphPoint
        }

        // If we're dragging significantly, show control handles preview
        if value.translation.width.magnitude > 5 || value.translation.height.magnitude > 5 {
            if let startPoint = penDragStartPoint {
                viewModel.penToolDrag(from: startPoint, to: glyphPoint)
            }
        }
    }

    private func handlePenEnd(value: DragGesture.Value, glyphPoint: CGPoint, size: CGSize) {
        let wasDragging = value.translation.width.magnitude > 5 || value.translation.height.magnitude > 5

        if wasDragging {
            // Drag ended - control handles already set via penToolDrag
            penDragStartPoint = nil
        } else {
            // Simple click - add a corner point
            viewModel.penToolClick(at: glyphPoint)
        }

        penDragStartPoint = nil
    }

    private var magnificationGesture: some Gesture {
        MagnificationGesture()
            .onChanged { value in
                scale = max(0.1, min(10, value))
            }
    }

    // MARK: - Drawing

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

        let lsb = CGPoint(x: CGFloat(viewModel.glyph.leftSideBearing), y: 0).applying(transform)
        let advance = CGPoint(x: CGFloat(viewModel.glyph.advanceWidth), y: 0).applying(transform)

        var verticalPath = Path()
        verticalPath.move(to: CGPoint(x: lsb.x, y: 0))
        verticalPath.addLine(to: CGPoint(x: lsb.x, y: size.height))
        verticalPath.move(to: CGPoint(x: advance.x, y: 0))
        verticalPath.addLine(to: CGPoint(x: advance.x, y: size.height))

        context.stroke(verticalPath, with: .color(Color.gray.opacity(0.5)), style: StrokeStyle(lineWidth: 1, dash: [4, 4]))
    }

    private func drawGlyph(context: GraphicsContext, transform: CGAffineTransform) {
        guard !viewModel.glyph.outline.isEmpty else { return }

        let cgPath = viewModel.glyph.outline.toCGPath()
        let transformedPath = Path(cgPath).applying(transform)

        context.fill(transformedPath, with: .color(Color.primary.opacity(0.3)))
        context.stroke(transformedPath, with: .color(Color.primary), lineWidth: 1)
    }

    private func drawControlPoints(context: GraphicsContext, transform: CGAffineTransform) {
        for contour in viewModel.glyph.outline.contours {
            for point in contour.points {
                let screenPos = point.position.applying(transform)
                let isSelected = viewModel.selectedPointIDs.contains(point.id)

                // Draw control handles
                if let controlIn = point.controlIn {
                    let controlScreenPos = controlIn.applying(transform)
                    var handleLine = Path()
                    handleLine.move(to: screenPos)
                    handleLine.addLine(to: controlScreenPos)
                    context.stroke(handleLine, with: .color(controlLineColor), lineWidth: 1)

                    let handleRect = CGRect(x: controlScreenPos.x - 4, y: controlScreenPos.y - 4, width: 8, height: 8)
                    context.fill(Path(ellipseIn: handleRect), with: .color(controlPointColor))
                }

                if let controlOut = point.controlOut {
                    let controlScreenPos = controlOut.applying(transform)
                    var handleLine = Path()
                    handleLine.move(to: screenPos)
                    handleLine.addLine(to: controlScreenPos)
                    context.stroke(handleLine, with: .color(controlLineColor), lineWidth: 1)

                    let handleRect = CGRect(x: controlScreenPos.x - 4, y: controlScreenPos.y - 4, width: 8, height: 8)
                    context.fill(Path(ellipseIn: handleRect), with: .color(controlPointColor))
                }

                // Draw main point
                let pointSize: CGFloat = isSelected ? 10 : 8
                let pointRect = CGRect(
                    x: screenPos.x - pointSize / 2,
                    y: screenPos.y - pointSize / 2,
                    width: pointSize,
                    height: pointSize
                )

                let pointPath: Path
                switch point.type {
                case .corner:
                    pointPath = Path(roundedRect: pointRect, cornerRadius: 1)
                case .smooth:
                    pointPath = Path(ellipseIn: pointRect)
                case .symmetric:
                    // Diamond shape
                    var diamond = Path()
                    diamond.move(to: CGPoint(x: screenPos.x, y: screenPos.y - pointSize / 2))
                    diamond.addLine(to: CGPoint(x: screenPos.x + pointSize / 2, y: screenPos.y))
                    diamond.addLine(to: CGPoint(x: screenPos.x, y: screenPos.y + pointSize / 2))
                    diamond.addLine(to: CGPoint(x: screenPos.x - pointSize / 2, y: screenPos.y))
                    diamond.closeSubpath()
                    pointPath = diamond
                }

                context.fill(pointPath, with: .color(isSelected ? selectedPointColor : pointColor))
                context.stroke(pointPath, with: .color(.white), lineWidth: 1)
            }
        }
    }

    // MARK: - Controls

    private var canvasControls: some View {
        VStack(spacing: 8) {
            Toggle(isOn: $showGrid) {
                Image(systemName: "grid")
            }
            .toggleStyle(.button)
            .help("Toggle Grid")

            Toggle(isOn: $showMetrics) {
                Image(systemName: "ruler")
            }
            .toggleStyle(.button)
            .help("Toggle Metrics")

            Toggle(isOn: $showControlPoints) {
                Image(systemName: "point.topleft.down.to.point.bottomright.curvepath")
            }
            .toggleStyle(.button)
            .help("Toggle Control Points")

            Divider()
                .frame(width: 20)

            Button {
                scale = 1.0
                offset = .zero
            } label: {
                Image(systemName: "arrow.counterclockwise")
            }
            .help("Reset View")

            Divider()
                .frame(width: 20)

            Button {
                viewModel.undo()
            } label: {
                Image(systemName: "arrow.uturn.backward")
            }
            .disabled(!viewModel.canUndo)
            .help("Undo (⌘Z)")

            Button {
                viewModel.redo()
            } label: {
                Image(systemName: "arrow.uturn.forward")
            }
            .disabled(!viewModel.canRedo)
            .help("Redo (⇧⌘Z)")

            Divider()
                .frame(width: 20)

            pathOperationsMenu
        }
        .padding(8)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 8))
        .padding(8)
    }

    private var pathOperationsMenu: some View {
        Menu {
            Section("Boolean Operations") {
                Button("Union") {
                    viewModel.unionSelectedContours()
                }
                .disabled(viewModel.selectedContourIndices.count < 2)

                Button("Subtract") {
                    viewModel.subtractSelectedContours()
                }
                .disabled(viewModel.selectedContourIndices.count != 2)

                Button("Intersect") {
                    viewModel.intersectSelectedContours()
                }
                .disabled(viewModel.selectedContourIndices.count < 2)

                Button("Exclude (XOR)") {
                    viewModel.xorSelectedContours()
                }
                .disabled(viewModel.selectedContourIndices.count < 2)
            }

            Divider()

            Section("Path Cleanup") {
                Button("Remove Overlaps") {
                    viewModel.removeOverlaps()
                }
                .disabled(viewModel.glyph.outline.contours.isEmpty)

                Button("Simplify Path") {
                    viewModel.simplifyPath()
                }
                .disabled(viewModel.glyph.outline.contours.isEmpty)

                Button("Correct Direction") {
                    viewModel.normalizeWindingDirection()
                }
                .disabled(viewModel.glyph.outline.contours.isEmpty)
            }

            Divider()

            Section("Offset") {
                Button("Expand (+10)") {
                    viewModel.offsetOutline(by: 10)
                }
                .disabled(viewModel.glyph.outline.contours.isEmpty)

                Button("Contract (-10)") {
                    viewModel.offsetOutline(by: -10)
                }
                .disabled(viewModel.glyph.outline.contours.isEmpty)
            }
        } label: {
            Image(systemName: "square.on.square.intersection.dashed")
        }
        .menuStyle(.borderlessButton)
        .help("Path Operations")
    }

    private var toolPicker: some View {
        HStack(spacing: 4) {
            ForEach(GlyphEditorViewModel.EditorTool.allCases, id: \.self) { tool in
                Button {
                    viewModel.currentTool = tool
                } label: {
                    Image(systemName: toolIcon(for: tool))
                        .frame(width: 24, height: 24)
                }
                .buttonStyle(.bordered)
                .tint(viewModel.currentTool == tool ? .accentColor : .secondary)
                .help(tool.rawValue)
            }
        }
        .padding(8)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 8))
        .padding(8)
    }

    private func toolIcon(for tool: GlyphEditorViewModel.EditorTool) -> String {
        switch tool {
        case .select: return "arrow.up.left.and.arrow.down.right"
        case .pen: return "pencil.tip"
        case .addPoint: return "plus.circle"
        case .deletePoint: return "minus.circle"
        }
    }

    @ViewBuilder
    private var statusBar: some View {
        HStack {
            // Tool info
            Text(viewModel.currentTool.rawValue)
                .fontWeight(.medium)

            Divider()
                .frame(height: 16)

            // Context-specific info
            if viewModel.isDrawingPath {
                HStack(spacing: 8) {
                    Image(systemName: "pencil.tip.crop.circle")
                        .foregroundColor(.orange)
                    Text("Drawing path")
                    Text("•")
                        .foregroundColor(.secondary)
                    Text("Return: close")
                        .foregroundColor(.secondary)
                    Text("Esc: finish")
                        .foregroundColor(.secondary)
                }
            } else if !viewModel.selectedPointIDs.isEmpty {
                HStack(spacing: 8) {
                    Text("\(viewModel.selectedPointIDs.count) point\(viewModel.selectedPointIDs.count == 1 ? "" : "s") selected")
                    Text("•")
                        .foregroundColor(.secondary)
                    Text("T: toggle type")
                        .foregroundColor(.secondary)
                    Text("S: smooth")
                        .foregroundColor(.secondary)
                    Text("C: corner")
                        .foregroundColor(.secondary)
                }
            } else {
                Text("\(viewModel.glyph.outline.contours.count) contour\(viewModel.glyph.outline.contours.count == 1 ? "" : "s")")
                    .foregroundColor(.secondary)
            }

            Spacer()

            // Scale indicator
            Text("\(Int(scale * 100))%")
                .foregroundColor(.secondary)
                .monospacedDigit()
        }
        .font(.caption)
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .background(.regularMaterial)
    }
}

#Preview {
    let outline = GlyphOutline(contours: [
        Contour(points: [
            PathPoint(position: CGPoint(x: 100, y: 0), type: .corner),
            PathPoint(position: CGPoint(x: 100, y: 500), type: .smooth,
                     controlIn: CGPoint(x: 100, y: 400),
                     controlOut: CGPoint(x: 100, y: 600)),
            PathPoint(position: CGPoint(x: 250, y: 700), type: .symmetric,
                     controlIn: CGPoint(x: 150, y: 700),
                     controlOut: CGPoint(x: 350, y: 700)),
            PathPoint(position: CGPoint(x: 400, y: 500), type: .smooth,
                     controlIn: CGPoint(x: 400, y: 600),
                     controlOut: CGPoint(x: 400, y: 400)),
            PathPoint(position: CGPoint(x: 400, y: 0), type: .corner),
        ], isClosed: true)
    ])

    let glyph = Glyph(
        character: "A",
        outline: outline,
        advanceWidth: 500,
        leftSideBearing: 50
    )

    let viewModel = GlyphEditorViewModel(glyph: glyph)

    return InteractiveGlyphCanvas(viewModel: viewModel, metrics: FontMetrics())
        .frame(width: 600, height: 500)
}
