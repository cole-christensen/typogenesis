import Testing
@testable import Typogenesis

@Suite("VariableFontEditorViewModel Tests")
struct VariableFontEditorViewModelTests {

    // MARK: - Initial State

    @Test("Initial state has correct defaults")
    @MainActor
    func initialState() {
        let vm = VariableFontEditorViewModel()

        #expect(vm.variableConfig.isVariableFont == false)
        #expect(vm.variableConfig.axes.isEmpty)
        #expect(vm.variableConfig.masters.isEmpty)
        #expect(vm.variableConfig.instances.isEmpty)
        #expect(vm.selectedMasterID == nil)
        #expect(vm.showAddAxisSheet == false)
        #expect(vm.showAddMasterSheet == false)
        #expect(vm.showAddInstanceSheet == false)
    }

    // MARK: - loadConfig

    @Test("loadConfig copies variableConfig from project")
    @MainActor
    func loadConfigCopiesFromProject() {
        let vm = VariableFontEditorViewModel()
        let project = FontProject(
            name: "Test",
            family: "Test",
            style: "Regular",
            variableConfig: .weightOnly()
        )

        vm.loadConfig(from: project)

        #expect(vm.variableConfig.isVariableFont == true)
        #expect(vm.variableConfig.axes.count == 1)
        #expect(vm.variableConfig.axes[0].tag == VariationAxis.weightTag)
        #expect(vm.variableConfig.masters.count == 2)
        #expect(vm.variableConfig.instances.count == 5)
    }

    // MARK: - toggleVariableFont

    @Test("toggleVariableFont enables with default weight axis when axes empty")
    @MainActor
    func toggleEnableAddsDefaultAxis() {
        let vm = VariableFontEditorViewModel()
        #expect(vm.variableConfig.axes.isEmpty)

        vm.toggleVariableFont(enabled: true)

        #expect(vm.variableConfig.isVariableFont == true)
        #expect(vm.variableConfig.axes.count == 1)
        #expect(vm.variableConfig.axes[0].tag == VariationAxis.weightTag)
        #expect(vm.variableConfig.axes[0].name == "Weight")
        #expect(vm.variableConfig.axes[0].minValue == 100)
        #expect(vm.variableConfig.axes[0].defaultValue == 400)
        #expect(vm.variableConfig.axes[0].maxValue == 900)
    }

    @Test("toggleVariableFont enables without adding axis when axes already exist")
    @MainActor
    func toggleEnableKeepsExistingAxes() {
        let vm = VariableFontEditorViewModel()
        let widthAxis = VariationAxis.width
        vm.variableConfig.axes = [widthAxis]

        vm.toggleVariableFont(enabled: true)

        #expect(vm.variableConfig.isVariableFont == true)
        #expect(vm.variableConfig.axes.count == 1)
        #expect(vm.variableConfig.axes[0].tag == VariationAxis.widthTag)
    }

    @Test("toggleVariableFont disables")
    @MainActor
    func toggleDisable() {
        let vm = VariableFontEditorViewModel()
        vm.variableConfig.isVariableFont = true
        vm.variableConfig.axes = [.weight]

        vm.toggleVariableFont(enabled: false)

        #expect(vm.variableConfig.isVariableFont == false)
        // Axes remain; only the flag changes
        #expect(vm.variableConfig.axes.count == 1)
    }

    // MARK: - Axis Management

    @Test("addAxis appends to axes array")
    @MainActor
    func addAxisAppendsToArray() {
        let vm = VariableFontEditorViewModel()
        let axis = VariationAxis.width

        vm.addAxis(axis)

        #expect(vm.variableConfig.axes.count == 1)
        #expect(vm.variableConfig.axes[0].tag == VariationAxis.widthTag)
        #expect(vm.variableConfig.axes[0].name == "Width")
    }

    @Test("removeAxis removes correct axis by ID")
    @MainActor
    func removeAxisByID() {
        let vm = VariableFontEditorViewModel()
        let axis1 = VariationAxis.weight
        let axis2 = VariationAxis.width
        vm.variableConfig.axes = [axis1, axis2]

        vm.removeAxis(id: axis1.id)

        #expect(vm.variableConfig.axes.count == 1)
        #expect(vm.variableConfig.axes[0].id == axis2.id)
    }

    // MARK: - Master Management

    @Test("addMaster appends to masters array")
    @MainActor
    func addMasterAppendsToArray() {
        let vm = VariableFontEditorViewModel()
        let master = FontMaster(
            name: "Bold",
            location: [VariationAxis.weightTag: 700]
        )

        vm.addMaster(master)

        #expect(vm.variableConfig.masters.count == 1)
        #expect(vm.variableConfig.masters[0].name == "Bold")
        #expect(vm.variableConfig.masters[0].location[VariationAxis.weightTag] == 700)
    }

    @Test("removeMaster removes correct master by ID")
    @MainActor
    func removeMasterByID() {
        let vm = VariableFontEditorViewModel()
        let master1 = FontMaster(
            name: "Light",
            location: [VariationAxis.weightTag: 300]
        )
        let master2 = FontMaster(
            name: "Bold",
            location: [VariationAxis.weightTag: 700]
        )
        vm.variableConfig.masters = [master1, master2]

        vm.removeMaster(id: master1.id)

        #expect(vm.variableConfig.masters.count == 1)
        #expect(vm.variableConfig.masters[0].id == master2.id)
        #expect(vm.variableConfig.masters[0].name == "Bold")
    }

    // MARK: - Instance Management

    @Test("addInstance appends to instances array")
    @MainActor
    func addInstanceAppendsToArray() {
        let vm = VariableFontEditorViewModel()
        let instance = NamedInstance(
            name: "SemiBold",
            location: [VariationAxis.weightTag: 600]
        )

        vm.addInstance(instance)

        #expect(vm.variableConfig.instances.count == 1)
        #expect(vm.variableConfig.instances[0].name == "SemiBold")
        #expect(vm.variableConfig.instances[0].location[VariationAxis.weightTag] == 600)
    }

    @Test("removeInstance removes correct instance by ID")
    @MainActor
    func removeInstanceByID() {
        let vm = VariableFontEditorViewModel()
        let instance1 = NamedInstance(
            name: "Regular",
            location: [VariationAxis.weightTag: 400]
        )
        let instance2 = NamedInstance(
            name: "Bold",
            location: [VariationAxis.weightTag: 700]
        )
        vm.variableConfig.instances = [instance1, instance2]

        vm.removeInstance(id: instance1.id)

        #expect(vm.variableConfig.instances.count == 1)
        #expect(vm.variableConfig.instances[0].id == instance2.id)
        #expect(vm.variableConfig.instances[0].name == "Bold")
    }
}
