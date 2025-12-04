import Testing
import CoreGraphics
@testable import Typogenesis

@Suite("GlyphEditorViewModel Tests")
struct GlyphEditorViewModelTests {

    @MainActor
    func createTestGlyph() -> Glyph {
        let outline = GlyphOutline(contours: [
            Contour(points: [
                PathPoint(position: CGPoint(x: 100, y: 0), type: .corner),
                PathPoint(position: CGPoint(x: 100, y: 500), type: .corner),
                PathPoint(position: CGPoint(x: 400, y: 500), type: .corner),
                PathPoint(position: CGPoint(x: 400, y: 0), type: .corner),
            ], isClosed: true)
        ])
        return Glyph(character: "H", outline: outline, advanceWidth: 500, leftSideBearing: 50)
    }

    @Test("Initial state has no selection")
    @MainActor
    func initialState() {
        let glyph = createTestGlyph()
        let viewModel = GlyphEditorViewModel(glyph: glyph)

        #expect(viewModel.selectedPointIDs.isEmpty)
        #expect(viewModel.currentTool == .select)
        #expect(!viewModel.canUndo)
        #expect(!viewModel.canRedo)
    }

    @Test("Select single point")
    @MainActor
    func selectSinglePoint() {
        let glyph = createTestGlyph()
        let viewModel = GlyphEditorViewModel(glyph: glyph)
        let pointID = glyph.outline.contours[0].points[0].id

        viewModel.selectPoint(id: pointID)

        #expect(viewModel.selectedPointIDs.count == 1)
        #expect(viewModel.selectedPointIDs.contains(pointID))
    }

    @Test("Add to selection with shift")
    @MainActor
    func addToSelection() {
        let glyph = createTestGlyph()
        let viewModel = GlyphEditorViewModel(glyph: glyph)
        let point1ID = glyph.outline.contours[0].points[0].id
        let point2ID = glyph.outline.contours[0].points[1].id

        viewModel.selectPoint(id: point1ID)
        viewModel.selectPoint(id: point2ID, addToSelection: true)

        #expect(viewModel.selectedPointIDs.count == 2)
        #expect(viewModel.selectedPointIDs.contains(point1ID))
        #expect(viewModel.selectedPointIDs.contains(point2ID))
    }

    @Test("Toggle selection removes point")
    @MainActor
    func toggleSelection() {
        let glyph = createTestGlyph()
        let viewModel = GlyphEditorViewModel(glyph: glyph)
        let pointID = glyph.outline.contours[0].points[0].id

        viewModel.selectPoint(id: pointID)
        viewModel.selectPoint(id: pointID, addToSelection: true)

        #expect(viewModel.selectedPointIDs.isEmpty)
    }

    @Test("Clear selection")
    @MainActor
    func clearSelection() {
        let glyph = createTestGlyph()
        let viewModel = GlyphEditorViewModel(glyph: glyph)
        let pointID = glyph.outline.contours[0].points[0].id

        viewModel.selectPoint(id: pointID)
        viewModel.clearSelection()

        #expect(viewModel.selectedPointIDs.isEmpty)
    }

    @Test("Select all points")
    @MainActor
    func selectAllPoints() {
        let glyph = createTestGlyph()
        let viewModel = GlyphEditorViewModel(glyph: glyph)

        viewModel.selectAllPoints()

        #expect(viewModel.selectedPointIDs.count == 4)
    }

    @Test("Hit test finds point")
    @MainActor
    func hitTestFindsPoint() {
        let glyph = createTestGlyph()
        let viewModel = GlyphEditorViewModel(glyph: glyph)

        let result = viewModel.hitTest(point: CGPoint(x: 100, y: 0), tolerance: 10)

        #expect(result != nil)
        if case .point(let contourIndex, let pointIndex, _) = result {
            #expect(contourIndex == 0)
            #expect(pointIndex == 0)
        } else {
            Issue.record("Expected point hit result")
        }
    }

    @Test("Hit test returns nil for miss")
    @MainActor
    func hitTestMiss() {
        let glyph = createTestGlyph()
        let viewModel = GlyphEditorViewModel(glyph: glyph)

        let result = viewModel.hitTest(point: CGPoint(x: 1000, y: 1000), tolerance: 10)

        #expect(result == nil)
    }

    @Test("Move selected points")
    @MainActor
    func moveSelectedPoints() {
        let glyph = createTestGlyph()
        let viewModel = GlyphEditorViewModel(glyph: glyph)
        let pointID = glyph.outline.contours[0].points[0].id
        let originalPosition = glyph.outline.contours[0].points[0].position

        viewModel.selectPoint(id: pointID)
        viewModel.moveSelectedPoints(by: CGSize(width: 50, height: 50))

        let newPosition = viewModel.glyph.outline.contours[0].points[0].position
        #expect(newPosition.x == originalPosition.x + 50)
        #expect(newPosition.y == originalPosition.y + 50)
    }

    @Test("Delete selected points")
    @MainActor
    func deleteSelectedPoints() {
        let glyph = createTestGlyph()
        let viewModel = GlyphEditorViewModel(glyph: glyph)
        let pointID = glyph.outline.contours[0].points[0].id

        viewModel.selectPoint(id: pointID)
        viewModel.deleteSelectedPoints()

        #expect(viewModel.glyph.outline.contours[0].points.count == 3)
        #expect(viewModel.selectedPointIDs.isEmpty)
    }

    @Test("Add point creates new point")
    @MainActor
    func addPoint() {
        let glyph = createTestGlyph()
        let viewModel = GlyphEditorViewModel(glyph: glyph)
        let initialCount = viewModel.glyph.outline.contours[0].points.count

        viewModel.addPoint(at: CGPoint(x: 250, y: 250), toContourIndex: 0)

        #expect(viewModel.glyph.outline.contours[0].points.count == initialCount + 1)
        #expect(viewModel.selectedPointIDs.count == 1)
    }

    @Test("Undo restores previous state")
    @MainActor
    func undoRestoresState() {
        let glyph = createTestGlyph()
        let viewModel = GlyphEditorViewModel(glyph: glyph)
        let pointID = glyph.outline.contours[0].points[0].id
        let originalCount = viewModel.glyph.outline.contours[0].points.count

        viewModel.selectPoint(id: pointID)
        viewModel.deleteSelectedPoints()

        #expect(viewModel.glyph.outline.contours[0].points.count == originalCount - 1)
        #expect(viewModel.canUndo)

        viewModel.undo()

        #expect(viewModel.glyph.outline.contours[0].points.count == originalCount)
    }

    @Test("Redo restores undone state")
    @MainActor
    func redoRestoresState() {
        let glyph = createTestGlyph()
        let viewModel = GlyphEditorViewModel(glyph: glyph)
        let pointID = glyph.outline.contours[0].points[0].id
        let originalCount = viewModel.glyph.outline.contours[0].points.count

        viewModel.selectPoint(id: pointID)
        viewModel.deleteSelectedPoints()
        viewModel.undo()

        #expect(viewModel.canRedo)

        viewModel.redo()

        #expect(viewModel.glyph.outline.contours[0].points.count == originalCount - 1)
    }

    @Test("Toggle point type cycles through types")
    @MainActor
    func togglePointType() {
        let glyph = createTestGlyph()
        let viewModel = GlyphEditorViewModel(glyph: glyph)
        let pointID = glyph.outline.contours[0].points[0].id

        viewModel.selectPoint(id: pointID)

        #expect(viewModel.glyph.outline.contours[0].points[0].type == .corner)

        viewModel.togglePointType()
        #expect(viewModel.glyph.outline.contours[0].points[0].type == .smooth)

        viewModel.togglePointType()
        #expect(viewModel.glyph.outline.contours[0].points[0].type == .symmetric)

        viewModel.togglePointType()
        #expect(viewModel.glyph.outline.contours[0].points[0].type == .corner)
    }
}
