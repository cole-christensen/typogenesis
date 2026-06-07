import Testing
@testable import Typogenesis

@Suite("KerningEditorViewModel Tests")
struct KerningEditorViewModelTests {

    @MainActor
    private func makeProject(kerning: [KerningPair] = []) -> FontProject {
        FontProject(
            name: "Test Font",
            family: "Test",
            style: "Regular",
            kerning: kerning
        )
    }

    @MainActor
    private func makeViewModel() -> KerningEditorViewModel {
        KerningEditorViewModel()
    }

    // MARK: - Initial State

    @Test("Initial state has empty pairs and no selection")
    @MainActor
    func initialState() {
        let vm = makeViewModel()

        #expect(vm.kerningPairs.isEmpty)
        #expect(vm.selectedPairIndex == nil)
        #expect(vm.leftChar == "")
        #expect(vm.rightChar == "")
        #expect(vm.kerningValue == 0)
        #expect(vm.previewText == "AVAST Wavy Type")
        #expect(vm.selectedPair == nil)
    }

    // MARK: - loadPairs

    @Test("loadPairs populates kerningPairs from project")
    @MainActor
    func loadPairsFromProject() {
        let vm = makeViewModel()
        let pairs = [
            KerningPair(left: "A", right: "V", value: -80),
            KerningPair(left: "T", right: "o", value: -40),
        ]
        let project = makeProject(kerning: pairs)

        vm.loadPairs(from: project)

        #expect(vm.kerningPairs.count == 2)
        #expect(vm.kerningPairs[0].left == Character("A"))
        #expect(vm.kerningPairs[0].right == Character("V"))
        #expect(vm.kerningPairs[0].value == -80)
        #expect(vm.kerningPairs[1].left == Character("T"))
        #expect(vm.kerningPairs[1].right == Character("o"))
        #expect(vm.kerningPairs[1].value == -40)
    }

    // MARK: - addPair

    @Test("addPair adds new pair to kerningPairs")
    @MainActor
    func addPairAddsNew() {
        let vm = makeViewModel()

        vm.addPair(left: "A", right: "V", value: -80)

        #expect(vm.kerningPairs.count == 1)
        #expect(vm.kerningPairs[0].left == Character("A"))
        #expect(vm.kerningPairs[0].right == Character("V"))
        #expect(vm.kerningPairs[0].value == -80)
        #expect(vm.selectedPairIndex == 0)
    }

    @Test("addPair updates existing pair if same left/right exists")
    @MainActor
    func addPairUpdatesExisting() {
        let vm = makeViewModel()

        vm.addPair(left: "A", right: "V", value: -80)
        vm.addPair(left: "A", right: "V", value: -120)

        #expect(vm.kerningPairs.count == 1)
        #expect(vm.kerningPairs[0].value == -120)
        #expect(vm.selectedPairIndex == 0)
    }

    // MARK: - addQuickPair

    @Test("addQuickPair validates input and adds pair")
    @MainActor
    func addQuickPairValidatesAndAdds() {
        let vm = makeViewModel()
        vm.leftChar = "A"
        vm.rightChar = "V"
        vm.kerningValue = -50

        vm.addQuickPair()

        #expect(vm.kerningPairs.count == 1)
        #expect(vm.kerningPairs[0].left == Character("A"))
        #expect(vm.kerningPairs[0].right == Character("V"))
        #expect(vm.kerningPairs[0].value == -50)
    }

    @Test("addQuickPair does nothing with empty input")
    @MainActor
    func addQuickPairRejectsEmptyInput() {
        let vm = makeViewModel()
        vm.leftChar = ""
        vm.rightChar = "V"

        vm.addQuickPair()

        #expect(vm.kerningPairs.isEmpty)
    }

    @Test("addQuickPair clears input fields after adding")
    @MainActor
    func addQuickPairClearsFields() {
        let vm = makeViewModel()
        vm.leftChar = "A"
        vm.rightChar = "V"
        vm.kerningValue = -50

        vm.addQuickPair()

        #expect(vm.leftChar == "")
        #expect(vm.rightChar == "")
        #expect(vm.kerningValue == 0)
    }

    // MARK: - updatePairValue

    @Test("updatePairValue changes value at index")
    @MainActor
    func updatePairValueChangesValue() {
        let vm = makeViewModel()
        vm.addPair(left: "A", right: "V", value: -80)

        vm.updatePairValue(at: 0, value: -120)

        #expect(vm.kerningPairs[0].value == -120)
        #expect(vm.kerningPairs[0].left == Character("A"))
        #expect(vm.kerningPairs[0].right == Character("V"))
    }

    @Test("updatePairValue does nothing for out-of-bounds index")
    @MainActor
    func updatePairValueOutOfBounds() {
        let vm = makeViewModel()
        vm.addPair(left: "A", right: "V", value: -80)

        vm.updatePairValue(at: 5, value: -120)

        #expect(vm.kerningPairs[0].value == -80)
    }

    // MARK: - deletePair

    @Test("deletePair removes pair and adjusts selection")
    @MainActor
    func deletePairRemovesAndAdjusts() {
        let vm = makeViewModel()
        vm.addPair(left: "A", right: "V", value: -80)
        vm.addPair(left: "T", right: "o", value: -40)
        vm.addPair(left: "W", right: "a", value: -60)
        // selectedPairIndex is now 2 (last added)
        vm.selectedPairIndex = 2

        vm.deletePair(at: 0)

        #expect(vm.kerningPairs.count == 2)
        #expect(vm.kerningPairs[0].left == Character("T"))
        // Selection should shift down by 1 since deleted index < selected
        #expect(vm.selectedPairIndex == 1)
    }

    @Test("deletePair sets selectedPairIndex to nil when deleting selected")
    @MainActor
    func deletePairClearsSelectionWhenDeletingSelected() {
        let vm = makeViewModel()
        vm.addPair(left: "A", right: "V", value: -80)
        vm.addPair(left: "T", right: "o", value: -40)
        vm.selectedPairIndex = 0

        vm.deletePair(at: 0)

        #expect(vm.kerningPairs.count == 1)
        #expect(vm.selectedPairIndex == nil)
    }

    // MARK: - applyAutoKerning

    @Test("applyAutoKerning with replaceExisting=true replaces all pairs")
    @MainActor
    func applyAutoKerningReplacesAll() {
        let vm = makeViewModel()
        vm.addPair(left: "A", right: "V", value: -80)
        vm.addPair(left: "T", right: "o", value: -40)

        let newPairs = [
            KerningPair(left: "W", right: "a", value: -60),
            KerningPair(left: "L", right: "T", value: -30),
        ]

        vm.applyAutoKerning(pairs: newPairs, replaceExisting: true)

        #expect(vm.kerningPairs.count == 2)
        // Should be sorted: L-T before W-a
        #expect(vm.kerningPairs[0].left == Character("L"))
        #expect(vm.kerningPairs[0].right == Character("T"))
        #expect(vm.kerningPairs[1].left == Character("W"))
        #expect(vm.kerningPairs[1].right == Character("a"))
        #expect(vm.selectedPairIndex == nil)
    }

    @Test("applyAutoKerning with replaceExisting=false merges pairs")
    @MainActor
    func applyAutoKerningMergesPairs() {
        let vm = makeViewModel()
        vm.addPair(left: "A", right: "V", value: -80)
        vm.addPair(left: "T", right: "o", value: -40)

        let newPairs = [
            KerningPair(left: "A", right: "V", value: -100),  // update existing
            KerningPair(left: "W", right: "a", value: -60),    // new pair
        ]

        vm.applyAutoKerning(pairs: newPairs, replaceExisting: false)

        #expect(vm.kerningPairs.count == 3)
        // A-V should be updated
        let avPair = vm.kerningPairs.first(where: { $0.left == Character("A") && $0.right == Character("V") })
        #expect(avPair?.value == -100)
        // W-a should be added
        let waPair = vm.kerningPairs.first(where: { $0.left == Character("W") && $0.right == Character("a") })
        #expect(waPair != nil)
        #expect(waPair?.value == -60)
    }

    @Test("applyAutoKerning sorts pairs by left then right character")
    @MainActor
    func applyAutoKerningSortsPairs() {
        let vm = makeViewModel()

        let pairs = [
            KerningPair(left: "W", right: "a", value: -60),
            KerningPair(left: "A", right: "V", value: -80),
            KerningPair(left: "A", right: "T", value: -30),
            KerningPair(left: "T", right: "o", value: -40),
        ]

        vm.applyAutoKerning(pairs: pairs, replaceExisting: true)

        #expect(vm.kerningPairs.count == 4)
        #expect(vm.kerningPairs[0].left == Character("A"))
        #expect(vm.kerningPairs[0].right == Character("T"))
        #expect(vm.kerningPairs[1].left == Character("A"))
        #expect(vm.kerningPairs[1].right == Character("V"))
        #expect(vm.kerningPairs[2].left == Character("T"))
        #expect(vm.kerningPairs[2].right == Character("o"))
        #expect(vm.kerningPairs[3].left == Character("W"))
        #expect(vm.kerningPairs[3].right == Character("a"))
    }

    // MARK: - selectedPair

    @Test("selectedPair returns correct pair")
    @MainActor
    func selectedPairReturnsCorrect() {
        let vm = makeViewModel()
        vm.addPair(left: "A", right: "V", value: -80)
        vm.addPair(left: "T", right: "o", value: -40)
        vm.selectedPairIndex = 1

        let pair = vm.selectedPair
        #expect(pair != nil)
        #expect(pair?.left == Character("T"))
        #expect(pair?.right == Character("o"))
        #expect(pair?.value == -40)
    }

    @Test("selectedPair returns nil for out-of-bounds index")
    @MainActor
    func selectedPairReturnsNilOutOfBounds() {
        let vm = makeViewModel()
        vm.addPair(left: "A", right: "V", value: -80)
        vm.selectedPairIndex = 5

        #expect(vm.selectedPair == nil)
    }

    @Test("selectedPair returns nil when no selection")
    @MainActor
    func selectedPairReturnsNilNoSelection() {
        let vm = makeViewModel()
        vm.addPair(left: "A", right: "V", value: -80)
        vm.selectedPairIndex = nil

        #expect(vm.selectedPair == nil)
    }
}
