import SwiftUI
import Combine

@MainActor
final class GlyphEditorViewModel: ObservableObject {
    @Published var glyph: Glyph
    @Published var selectedPointIDs: Set<UUID> = []
    @Published var hoveredPointID: UUID?
    @Published var currentTool: EditorTool = .select
    @Published var isDragging = false

    // Pen tool state
    @Published var isDrawingPath = false
    @Published var currentContourIndex: Int?
    @Published var pendingPoint: CGPoint?

    // Error handling for path operations
    @Published var operationError: String?
    @Published var showOperationError = false

    private var undoStack: [Glyph] = []
    private var redoStack: [Glyph] = []
    private let maxUndoLevels = 50

    enum EditorTool: String, CaseIterable {
        case select = "Select"
        case pen = "Pen"
        case addPoint = "Add Point"
        case deletePoint = "Delete Point"
    }

    init(glyph: Glyph) {
        self.glyph = glyph
    }

    // MARK: - Point Selection

    func selectPoint(id: UUID, addToSelection: Bool = false) {
        if addToSelection {
            if selectedPointIDs.contains(id) {
                selectedPointIDs.remove(id)
            } else {
                selectedPointIDs.insert(id)
            }
        } else {
            selectedPointIDs = [id]
        }
    }

    func clearSelection() {
        selectedPointIDs.removeAll()
    }

    func selectAllPoints() {
        selectedPointIDs = Set(glyph.outline.contours.flatMap { $0.points.map { $0.id } })
    }

    // MARK: - Hit Testing

    func hitTest(point: CGPoint, tolerance: CGFloat = 8) -> HitTestResult? {
        for (contourIndex, contour) in glyph.outline.contours.enumerated() {
            for (pointIndex, pathPoint) in contour.points.enumerated() {
                // Check control handles first (smaller hit area)
                if let controlIn = pathPoint.controlIn {
                    let distance = hypot(point.x - controlIn.x, point.y - controlIn.y)
                    if distance <= tolerance * 0.7 {
                        return .controlIn(contourIndex: contourIndex, pointIndex: pointIndex, pointID: pathPoint.id)
                    }
                }

                if let controlOut = pathPoint.controlOut {
                    let distance = hypot(point.x - controlOut.x, point.y - controlOut.y)
                    if distance <= tolerance * 0.7 {
                        return .controlOut(contourIndex: contourIndex, pointIndex: pointIndex, pointID: pathPoint.id)
                    }
                }

                // Check main point
                let distance = hypot(point.x - pathPoint.position.x, point.y - pathPoint.position.y)
                if distance <= tolerance {
                    return .point(contourIndex: contourIndex, pointIndex: pointIndex, pointID: pathPoint.id)
                }
            }
        }
        return nil
    }

    enum HitTestResult: Equatable {
        case point(contourIndex: Int, pointIndex: Int, pointID: UUID)
        case controlIn(contourIndex: Int, pointIndex: Int, pointID: UUID)
        case controlOut(contourIndex: Int, pointIndex: Int, pointID: UUID)

        var pointID: UUID {
            switch self {
            case .point(_, _, let id), .controlIn(_, _, let id), .controlOut(_, _, let id):
                return id
            }
        }
    }

    // MARK: - Point Manipulation

    func moveSelectedPoints(by delta: CGSize) {
        guard !selectedPointIDs.isEmpty else { return }

        for contourIndex in glyph.outline.contours.indices {
            for pointIndex in glyph.outline.contours[contourIndex].points.indices {
                let point = glyph.outline.contours[contourIndex].points[pointIndex]
                if selectedPointIDs.contains(point.id) {
                    glyph.outline.contours[contourIndex].points[pointIndex].position.x += delta.width
                    glyph.outline.contours[contourIndex].points[pointIndex].position.y += delta.height

                    // Move control points with the main point
                    if var controlIn = point.controlIn {
                        controlIn.x += delta.width
                        controlIn.y += delta.height
                        glyph.outline.contours[contourIndex].points[pointIndex].controlIn = controlIn
                    }
                    if var controlOut = point.controlOut {
                        controlOut.x += delta.width
                        controlOut.y += delta.height
                        glyph.outline.contours[contourIndex].points[pointIndex].controlOut = controlOut
                    }
                }
            }
        }
    }

    func movePoint(at hitResult: HitTestResult, to newPosition: CGPoint) {
        guard case let .point(contourIndex, pointIndex, _) = hitResult else { return }

        let oldPosition = glyph.outline.contours[contourIndex].points[pointIndex].position
        let delta = CGSize(width: newPosition.x - oldPosition.x, height: newPosition.y - oldPosition.y)

        glyph.outline.contours[contourIndex].points[pointIndex].position = newPosition

        // Move control points with the main point
        if var controlIn = glyph.outline.contours[contourIndex].points[pointIndex].controlIn {
            controlIn.x += delta.width
            controlIn.y += delta.height
            glyph.outline.contours[contourIndex].points[pointIndex].controlIn = controlIn
        }
        if var controlOut = glyph.outline.contours[contourIndex].points[pointIndex].controlOut {
            controlOut.x += delta.width
            controlOut.y += delta.height
            glyph.outline.contours[contourIndex].points[pointIndex].controlOut = controlOut
        }
    }

    func moveControlIn(at hitResult: HitTestResult, to newPosition: CGPoint) {
        guard case let .controlIn(contourIndex, pointIndex, _) = hitResult else { return }

        glyph.outline.contours[contourIndex].points[pointIndex].controlIn = newPosition
        mirrorSymmetricControlHandle(
            at: (contourIndex, pointIndex),
            movedHandle: newPosition,
            mirrorTarget: \.controlOut
        )
    }

    func moveControlOut(at hitResult: HitTestResult, to newPosition: CGPoint) {
        guard case let .controlOut(contourIndex, pointIndex, _) = hitResult else { return }

        glyph.outline.contours[contourIndex].points[pointIndex].controlOut = newPosition
        mirrorSymmetricControlHandle(
            at: (contourIndex, pointIndex),
            movedHandle: newPosition,
            mirrorTarget: \.controlIn
        )
    }

    /// Mirrors a control handle for symmetric points.
    /// When one control handle moves, the opposite handle mirrors around the anchor point.
    private func mirrorSymmetricControlHandle(
        at location: (contourIndex: Int, pointIndex: Int),
        movedHandle newPosition: CGPoint,
        mirrorTarget: WritableKeyPath<PathPoint, CGPoint?>
    ) {
        let point = glyph.outline.contours[location.contourIndex].points[location.pointIndex]
        guard point.type == .symmetric, point[keyPath: mirrorTarget] != nil else { return }

        let dx = point.position.x - newPosition.x
        let dy = point.position.y - newPosition.y
        glyph.outline.contours[location.contourIndex].points[location.pointIndex][keyPath: mirrorTarget] = CGPoint(
            x: point.position.x + dx,
            y: point.position.y + dy
        )
    }

    // MARK: - Point Operations

    func deleteSelectedPoints() {
        guard !selectedPointIDs.isEmpty else { return }
        saveStateForUndo()

        for contourIndex in glyph.outline.contours.indices.reversed() {
            glyph.outline.contours[contourIndex].points.removeAll { selectedPointIDs.contains($0.id) }
        }

        // Remove empty contours
        glyph.outline.contours.removeAll { $0.points.isEmpty }
        selectedPointIDs.removeAll()
    }

    func addPoint(at position: CGPoint, toContourIndex: Int? = nil) {
        saveStateForUndo()

        let newPoint = PathPoint(position: position, type: .corner)

        if let contourIndex = toContourIndex, contourIndex < glyph.outline.contours.count {
            glyph.outline.contours[contourIndex].points.append(newPoint)
        } else if glyph.outline.contours.isEmpty {
            // Create new contour
            glyph.outline.contours.append(Contour(points: [newPoint], isClosed: false))
        } else {
            // Add to last contour
            glyph.outline.contours[glyph.outline.contours.count - 1].points.append(newPoint)
        }

        selectedPointIDs = [newPoint.id]
    }

    func togglePointType() {
        guard selectedPointIDs.count == 1, let pointID = selectedPointIDs.first else { return }
        saveStateForUndo()

        for contourIndex in glyph.outline.contours.indices {
            for pointIndex in glyph.outline.contours[contourIndex].points.indices {
                if glyph.outline.contours[contourIndex].points[pointIndex].id == pointID {
                    let currentType = glyph.outline.contours[contourIndex].points[pointIndex].type
                    let newType: PathPoint.PointType
                    switch currentType {
                    case .corner: newType = .smooth
                    case .smooth: newType = .symmetric
                    case .symmetric: newType = .corner
                    }
                    glyph.outline.contours[contourIndex].points[pointIndex].type = newType
                    return
                }
            }
        }
    }

    // MARK: - Undo/Redo

    func saveStateForUndo() {
        undoStack.append(glyph)
        if undoStack.count > maxUndoLevels {
            undoStack.removeFirst()
        }
        redoStack.removeAll()
    }

    func undo() {
        guard let previousState = undoStack.popLast() else { return }
        redoStack.append(glyph)
        glyph = previousState
        selectedPointIDs.removeAll()
    }

    func redo() {
        guard let nextState = redoStack.popLast() else { return }
        undoStack.append(glyph)
        glyph = nextState
        selectedPointIDs.removeAll()
    }

    var canUndo: Bool { !undoStack.isEmpty }
    var canRedo: Bool { !redoStack.isEmpty }

    // MARK: - Drag Operations

    func beginDrag() {
        if !isDragging {
            saveStateForUndo()
            isDragging = true
        }
    }

    func endDrag() {
        isDragging = false
    }

    // MARK: - Pen Tool Operations

    /// Start drawing a new contour or continue an existing one
    func penToolClick(at position: CGPoint) {
        // Check if clicking near the first point of current contour (to close it)
        if isDrawingPath,
           let contourIndex = currentContourIndex,
           contourIndex < glyph.outline.contours.count {
            let contour = glyph.outline.contours[contourIndex]
            if let firstPoint = contour.points.first {
                let distance = hypot(position.x - firstPoint.position.x, position.y - firstPoint.position.y)
                if distance < 15 && contour.points.count >= 3 {
                    // Close the contour
                    closePenPath()
                    return
                }
            }
        }

        // Check if clicking on an existing point to start extending from there
        if !isDrawingPath {
            if let hitResult = hitTest(point: position, tolerance: 12) {
                if case let .point(contourIndex, pointIndex, _) = hitResult {
                    let contour = glyph.outline.contours[contourIndex]
                    // Can only extend from endpoints
                    if pointIndex == 0 || pointIndex == contour.points.count - 1 {
                        saveStateForUndo()
                        isDrawingPath = true
                        currentContourIndex = contourIndex
                        selectedPointIDs = [contour.points[pointIndex].id]
                        return
                    }
                }
            }
        }

        // Add a new point
        saveStateForUndo()

        let newPoint = PathPoint(position: position, type: .corner)

        if isDrawingPath, let contourIndex = currentContourIndex, contourIndex < glyph.outline.contours.count {
            // Add to current contour
            glyph.outline.contours[contourIndex].points.append(newPoint)
        } else {
            // Start a new contour
            let newContour = Contour(points: [newPoint], isClosed: false)
            glyph.outline.contours.append(newContour)
            currentContourIndex = glyph.outline.contours.count - 1
            isDrawingPath = true
        }

        selectedPointIDs = [newPoint.id]
    }

    /// Add a curve point with control handles while drawing
    func penToolDrag(from startPosition: CGPoint, to currentPosition: CGPoint) {
        guard isDrawingPath,
              let contourIndex = currentContourIndex,
              contourIndex < glyph.outline.contours.count,
              !glyph.outline.contours[contourIndex].points.isEmpty else { return }

        let lastPointIndex = glyph.outline.contours[contourIndex].points.count - 1

        // Calculate control handles based on drag
        let dx = currentPosition.x - startPosition.x
        let dy = currentPosition.y - startPosition.y

        // Set the control out handle
        glyph.outline.contours[contourIndex].points[lastPointIndex].controlOut = currentPosition
        glyph.outline.contours[contourIndex].points[lastPointIndex].type = .smooth

        // Set opposite control in handle (symmetric)
        glyph.outline.contours[contourIndex].points[lastPointIndex].controlIn = CGPoint(
            x: startPosition.x - dx,
            y: startPosition.y - dy
        )
    }

    /// Close the current path being drawn
    func closePenPath() {
        guard isDrawingPath,
              let contourIndex = currentContourIndex,
              contourIndex < glyph.outline.contours.count else { return }

        glyph.outline.contours[contourIndex].isClosed = true
        finishPenPath()
    }

    /// Finish drawing without closing (open path)
    func finishPenPath() {
        isDrawingPath = false
        currentContourIndex = nil
        pendingPoint = nil
        selectedPointIDs.removeAll()
    }

    /// Cancel the current drawing operation
    func cancelPenPath() {
        guard isDrawingPath,
              let contourIndex = currentContourIndex else {
            finishPenPath()
            return
        }

        // Remove the contour if it only has one point
        if contourIndex < glyph.outline.contours.count {
            if glyph.outline.contours[contourIndex].points.count <= 1 {
                glyph.outline.contours.remove(at: contourIndex)
            }
        }

        finishPenPath()
        undo()  // Undo the entire drawing operation
    }

    // MARK: - Contour Operations

    /// Create a new contour from the selected points
    func createContourFromSelection() {
        guard selectedPointIDs.count >= 2 else { return }
        saveStateForUndo()

        var newPoints: [PathPoint] = []

        // Collect selected points in order
        for contour in glyph.outline.contours {
            for point in contour.points {
                if selectedPointIDs.contains(point.id) {
                    newPoints.append(PathPoint(
                        position: point.position,
                        type: point.type,
                        controlIn: point.controlIn,
                        controlOut: point.controlOut
                    ))
                }
            }
        }

        if newPoints.count >= 2 {
            let newContour = Contour(points: newPoints, isClosed: false)
            glyph.outline.contours.append(newContour)
        }
    }

    /// Reverse the direction of a contour
    func reverseContour(at contourIndex: Int) {
        guard contourIndex < glyph.outline.contours.count else { return }
        saveStateForUndo()

        let reversedPoints = glyph.outline.contours[contourIndex].points.reversed().map { point in
            // Swap control in and control out
            PathPoint(
                id: point.id,
                position: point.position,
                type: point.type,
                controlIn: point.controlOut,
                controlOut: point.controlIn
            )
        }

        glyph.outline.contours[contourIndex].points = Array(reversedPoints)
    }

    /// Delete an entire contour
    func deleteContour(at contourIndex: Int) {
        guard contourIndex < glyph.outline.contours.count else { return }
        saveStateForUndo()

        // Clear selection of points in this contour
        let pointIDs = Set(glyph.outline.contours[contourIndex].points.map { $0.id })
        selectedPointIDs.subtract(pointIDs)

        glyph.outline.contours.remove(at: contourIndex)
    }

    /// Toggle whether a contour is closed
    func toggleContourClosed(at contourIndex: Int) {
        guard contourIndex < glyph.outline.contours.count else { return }
        saveStateForUndo()

        glyph.outline.contours[contourIndex].isClosed.toggle()
    }

    // MARK: - Point Smoothing

    /// Convert selected corner points to smooth curves
    func smoothSelectedPoints() {
        guard !selectedPointIDs.isEmpty else { return }
        saveStateForUndo()

        for contourIndex in glyph.outline.contours.indices {
            let contour = glyph.outline.contours[contourIndex]
            for pointIndex in contour.points.indices {
                let point = contour.points[pointIndex]
                guard selectedPointIDs.contains(point.id) else { continue }

                // Get neighboring points
                let prevIndex = pointIndex > 0 ? pointIndex - 1 : (contour.isClosed ? contour.points.count - 1 : nil)
                let nextIndex = pointIndex < contour.points.count - 1 ? pointIndex + 1 : (contour.isClosed ? 0 : nil)

                guard let prev = prevIndex, let next = nextIndex else { continue }

                let prevPoint = contour.points[prev]
                let nextPoint = contour.points[next]

                // Calculate smooth control handles
                let dx = nextPoint.position.x - prevPoint.position.x
                let dy = nextPoint.position.y - prevPoint.position.y
                let handleLength = hypot(dx, dy) * 0.25

                let angle = atan2(dy, dx)

                glyph.outline.contours[contourIndex].points[pointIndex].type = .smooth
                glyph.outline.contours[contourIndex].points[pointIndex].controlIn = CGPoint(
                    x: point.position.x - cos(angle) * handleLength,
                    y: point.position.y - sin(angle) * handleLength
                )
                glyph.outline.contours[contourIndex].points[pointIndex].controlOut = CGPoint(
                    x: point.position.x + cos(angle) * handleLength,
                    y: point.position.y + sin(angle) * handleLength
                )
            }
        }
    }

    /// Convert selected points to corner points (remove handles)
    func cornerSelectedPoints() {
        guard !selectedPointIDs.isEmpty else { return }
        saveStateForUndo()

        for contourIndex in glyph.outline.contours.indices {
            for pointIndex in glyph.outline.contours[contourIndex].points.indices {
                let point = glyph.outline.contours[contourIndex].points[pointIndex]
                if selectedPointIDs.contains(point.id) {
                    glyph.outline.contours[contourIndex].points[pointIndex].type = .corner
                    glyph.outline.contours[contourIndex].points[pointIndex].controlIn = nil
                    glyph.outline.contours[contourIndex].points[pointIndex].controlOut = nil
                }
            }
        }
    }

    // MARK: - Path Boolean Operations

    /// Get indices of selected contours
    var selectedContourIndices: Set<Int> {
        var indices: Set<Int> = []
        for (contourIndex, contour) in glyph.outline.contours.enumerated() {
            for point in contour.points {
                if selectedPointIDs.contains(point.id) {
                    indices.insert(contourIndex)
                    break
                }
            }
        }
        return indices
    }

    // MARK: - Boolean Operations (Unified Implementation)

    /// Performs a boolean path operation on selected contours
    /// - Parameters:
    ///   - operation: The type of boolean operation to perform
    ///   - requireExactlyTwo: If true, requires exactly 2 selected contours (for subtract)
    private func performBooleanOperation(
        _ operation: PathOperations.Operation,
        requireExactlyTwo: Bool = false
    ) {
        let indices = Array(selectedContourIndices).sorted()
        let minCount = 2
        let maxCount = requireExactlyTwo ? 2 : Int.max

        guard indices.count >= minCount && indices.count <= maxCount else { return }
        saveStateForUndo()

        do {
            let outline1 = GlyphOutline(contours: [glyph.outline.contours[indices[0]]])
            var resultOutline = outline1

            for i in 1..<indices.count {
                let outline2 = GlyphOutline(contours: [glyph.outline.contours[indices[i]]])
                resultOutline = try PathOperations.perform(operation, on: resultOutline, with: outline2)
            }

            // Remove original contours (in reverse order to maintain indices)
            for index in indices.reversed() {
                glyph.outline.contours.remove(at: index)
            }

            // Add result contours
            glyph.outline.contours.append(contentsOf: resultOutline.contours)
            selectedPointIDs.removeAll()

        } catch {
            operationError = "\(operation.displayName) operation failed: \(error.localizedDescription)"
            showOperationError = true
            undo()
        }
    }

    /// Union selected contours
    func unionSelectedContours() {
        performBooleanOperation(.union)
    }

    /// Subtract second selected contour from first
    func subtractSelectedContours() {
        performBooleanOperation(.subtract, requireExactlyTwo: true)
    }

    /// Intersect selected contours
    func intersectSelectedContours() {
        performBooleanOperation(.intersect)
    }

    /// XOR selected contours (exclude overlapping areas)
    func xorSelectedContours() {
        performBooleanOperation(.xor)
    }

    /// Remove overlapping regions within all contours
    func removeOverlaps() {
        guard glyph.outline.contours.count > 0 else { return }
        saveStateForUndo()

        do {
            let resultOutline = try PathOperations.removeOverlaps(glyph.outline)
            glyph.outline = resultOutline
            selectedPointIDs.removeAll()
        } catch {
            operationError = "Remove overlaps failed: \(error.localizedDescription)"
            showOperationError = true
            undo()
        }
    }

    /// Simplify path by removing redundant points
    func simplifyPath(tolerance: CGFloat = 2.0) {
        saveStateForUndo()
        glyph.outline = PathOperations.simplify(glyph.outline, tolerance: tolerance)
        selectedPointIDs.removeAll()
    }

    /// Normalize winding direction of all contours
    func normalizeWindingDirection(clockwise: Bool = true) {
        saveStateForUndo()
        glyph.outline = PathOperations.normalizeWindingDirection(glyph.outline, clockwise: clockwise)
    }

    /// Offset (expand/contract) the outline
    func offsetOutline(by amount: CGFloat) {
        saveStateForUndo()

        do {
            glyph.outline = try PathOperations.offset(glyph.outline, by: amount)
            selectedPointIDs.removeAll()
        } catch {
            operationError = "Offset operation failed: \(error.localizedDescription)"
            showOperationError = true
            undo()
        }
    }
}
