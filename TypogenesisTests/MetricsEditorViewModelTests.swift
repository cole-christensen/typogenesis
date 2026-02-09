import Testing
@testable import Typogenesis

@Suite("MetricsEditorViewModel Tests")
struct MetricsEditorViewModelTests {

    private func makeProject(metrics: FontMetrics = FontMetrics()) -> FontProject {
        FontProject(name: "Test", family: "Test", style: "Regular", metrics: metrics)
    }

    @Test("Initial state: editedMetrics is nil and showPreview is true")
    @MainActor
    func initialState() {
        let vm = MetricsEditorViewModel()

        #expect(vm.editedMetrics == nil)
        #expect(vm.showPreview == true)
    }

    @Test("loadMetrics sets editedMetrics from project")
    @MainActor
    func loadMetricsSetsFromProject() {
        let vm = MetricsEditorViewModel()
        let metrics = FontMetrics(unitsPerEm: 2048, ascender: 900, descender: -300, xHeight: 550, capHeight: 750, lineGap: 100)
        let project = makeProject(metrics: metrics)

        vm.loadMetrics(from: project)

        #expect(vm.editedMetrics == metrics)
    }

    @Test("loadMetrics does not overwrite if editedMetrics already set")
    @MainActor
    func loadMetricsDoesNotOverwrite() {
        let vm = MetricsEditorViewModel()
        let originalMetrics = FontMetrics(unitsPerEm: 2048)
        let newMetrics = FontMetrics(unitsPerEm: 500)

        vm.loadMetrics(from: makeProject(metrics: originalMetrics))
        vm.loadMetrics(from: makeProject(metrics: newMetrics))

        #expect(vm.editedMetrics == originalMetrics)
    }

    @Test("applyChanges returns editedMetrics when changed")
    @MainActor
    func applyChangesReturnsEditedMetrics() {
        let vm = MetricsEditorViewModel()
        let metrics = FontMetrics()
        vm.loadMetrics(from: makeProject(metrics: metrics))

        vm.editedMetrics?.ascender = 900

        let result = vm.applyChanges()
        #expect(result != nil)
        #expect(result?.ascender == 900)
    }

    @Test("applyChanges returns nil when no changes made")
    @MainActor
    func applyChangesReturnsNilWhenUnchanged() {
        let vm = MetricsEditorViewModel()
        vm.loadMetrics(from: makeProject())

        let result = vm.applyChanges()
        #expect(result == nil)
    }

    @Test("hasChanges returns false when metrics match")
    @MainActor
    func hasChangesReturnsFalseWhenMatch() {
        let vm = MetricsEditorViewModel()
        let metrics = FontMetrics()
        vm.loadMetrics(from: makeProject(metrics: metrics))

        #expect(vm.hasChanges(vs: metrics) == false)
    }

    @Test("hasChanges returns true when metrics differ")
    @MainActor
    func hasChangesReturnsTrueWhenDifferent() {
        let vm = MetricsEditorViewModel()
        let metrics = FontMetrics()
        vm.loadMetrics(from: makeProject(metrics: metrics))

        vm.editedMetrics?.ascender = 900

        #expect(vm.hasChanges(vs: metrics) == true)
    }

    @Test("hasChanges returns false when either is nil")
    @MainActor
    func hasChangesReturnsFalseWhenNil() {
        let vm = MetricsEditorViewModel()

        #expect(vm.hasChanges(vs: nil) == false)
        #expect(vm.hasChanges(vs: FontMetrics()) == false)
    }

    @Test("resetToDefaults sets editedMetrics to default FontMetrics")
    @MainActor
    func resetToDefaultsSetsDefaultMetrics() {
        let vm = MetricsEditorViewModel()
        let custom = FontMetrics(unitsPerEm: 2048, ascender: 900, descender: -300, xHeight: 550, capHeight: 750, lineGap: 100)
        vm.loadMetrics(from: makeProject(metrics: custom))

        vm.resetToDefaults()

        #expect(vm.editedMetrics == FontMetrics())
        #expect(vm.editedMetrics?.unitsPerEm == 1000)
        #expect(vm.editedMetrics?.ascender == 800)
        #expect(vm.editedMetrics?.descender == -200)
        #expect(vm.editedMetrics?.xHeight == 500)
        #expect(vm.editedMetrics?.capHeight == 700)
        #expect(vm.editedMetrics?.lineGap == 90)
    }
}
