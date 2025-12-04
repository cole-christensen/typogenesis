import SwiftUI
import Combine

@MainActor
final class GlyphEditorViewModel: ObservableObject {
    @Published var glyph: Glyph
    @Published var selectedPointIDs: Set<UUID> = []
    @Published var hoveredPointID: UUID?
    @Published var currentTool: EditorTool = .select
    @Published var isDragging = false

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

        // For symmetric points, mirror the control handle
        let point = glyph.outline.contours[contourIndex].points[pointIndex]
        if point.type == .symmetric, let _ = point.controlOut {
            let dx = point.position.x - newPosition.x
            let dy = point.position.y - newPosition.y
            glyph.outline.contours[contourIndex].points[pointIndex].controlOut = CGPoint(
                x: point.position.x + dx,
                y: point.position.y + dy
            )
        }
    }

    func moveControlOut(at hitResult: HitTestResult, to newPosition: CGPoint) {
        guard case let .controlOut(contourIndex, pointIndex, _) = hitResult else { return }

        glyph.outline.contours[contourIndex].points[pointIndex].controlOut = newPosition

        // For symmetric points, mirror the control handle
        let point = glyph.outline.contours[contourIndex].points[pointIndex]
        if point.type == .symmetric, let _ = point.controlIn {
            let dx = point.position.x - newPosition.x
            let dy = point.position.y - newPosition.y
            glyph.outline.contours[contourIndex].points[pointIndex].controlIn = CGPoint(
                x: point.position.x + dx,
                y: point.position.y + dy
            )
        }
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
}
