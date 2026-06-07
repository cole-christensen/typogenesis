import Testing
import UniformTypeIdentifiers
@testable import Typogenesis

// MARK: - Tests

@Suite("ImportFontViewModel Tests")
struct ImportFontViewModelTests {

    @Test("Initial state has no URL, not analyzing, no imported project")
    @MainActor
    func initialState() {
        let mock = MockFileDialogService()
        let vm = ImportFontViewModel(fileDialogService: mock)

        #expect(vm.selectedURL == nil)
        #expect(vm.isAnalyzing == false)
        #expect(vm.importedProject == nil)
        #expect(vm.extractedStyle == nil)
        #expect(vm.analysisError == nil)
        #expect(vm.showError == false)
    }

    @Test("reset clears importedProject and extractedStyle")
    @MainActor
    func resetClearsState() {
        let mock = MockFileDialogService()
        let vm = ImportFontViewModel(fileDialogService: mock)

        // Manually set state to simulate a completed analysis
        vm.importedProject = FontProject(name: "Test", family: "Test", style: "Regular")
        vm.extractedStyle = .default

        vm.reset()

        #expect(vm.importedProject == nil)
        #expect(vm.extractedStyle == nil)
    }

    // MARK: - Description Helper Tests

    @Test("weightDescription returns correct labels for ranges")
    @MainActor
    func weightDescriptionLabels() {
        let mock = MockFileDialogService()
        let vm = ImportFontViewModel(fileDialogService: mock)

        #expect(vm.weightDescription(0.0) == "Light")
        #expect(vm.weightDescription(0.15) == "Light")
        #expect(vm.weightDescription(0.29) == "Light")
        #expect(vm.weightDescription(0.3) == "Regular")
        #expect(vm.weightDescription(0.4) == "Regular")
        #expect(vm.weightDescription(0.5) == "Medium")
        #expect(vm.weightDescription(0.6) == "Medium")
        #expect(vm.weightDescription(0.7) == "Bold")
        #expect(vm.weightDescription(0.8) == "Bold")
        #expect(vm.weightDescription(0.85) == "Heavy")
        #expect(vm.weightDescription(1.0) == "Heavy")
    }

    @Test("contrastDescription returns correct labels for ranges")
    @MainActor
    func contrastDescriptionLabels() {
        let mock = MockFileDialogService()
        let vm = ImportFontViewModel(fileDialogService: mock)

        #expect(vm.contrastDescription(0.0) == "Monolinear")
        #expect(vm.contrastDescription(0.1) == "Monolinear")
        #expect(vm.contrastDescription(0.2) == "Low contrast")
        #expect(vm.contrastDescription(0.3) == "Low contrast")
        #expect(vm.contrastDescription(0.4) == "Moderate")
        #expect(vm.contrastDescription(0.5) == "Moderate")
        #expect(vm.contrastDescription(0.6) == "High contrast")
        #expect(vm.contrastDescription(0.7) == "High contrast")
        #expect(vm.contrastDescription(0.8) == "Very high")
        #expect(vm.contrastDescription(1.0) == "Very high")
    }

    @Test("roundnessDescription returns correct labels for ranges")
    @MainActor
    func roundnessDescriptionLabels() {
        let mock = MockFileDialogService()
        let vm = ImportFontViewModel(fileDialogService: mock)

        #expect(vm.roundnessDescription(0.0) == "Geometric")
        #expect(vm.roundnessDescription(0.2) == "Geometric")
        #expect(vm.roundnessDescription(0.3) == "Mixed")
        #expect(vm.roundnessDescription(0.4) == "Mixed")
        #expect(vm.roundnessDescription(0.5) == "Organic")
        #expect(vm.roundnessDescription(0.6) == "Organic")
        #expect(vm.roundnessDescription(0.7) == "Fluid")
        #expect(vm.roundnessDescription(1.0) == "Fluid")
    }

    @Test("regularityDescription returns correct labels for ranges")
    @MainActor
    func regularityDescriptionLabels() {
        let mock = MockFileDialogService()
        let vm = ImportFontViewModel(fileDialogService: mock)

        #expect(vm.regularityDescription(0.0) == "Irregular")
        #expect(vm.regularityDescription(0.3) == "Irregular")
        #expect(vm.regularityDescription(0.4) == "Moderate")
        #expect(vm.regularityDescription(0.5) == "Moderate")
        #expect(vm.regularityDescription(0.6) == "Consistent")
        #expect(vm.regularityDescription(0.7) == "Consistent")
        #expect(vm.regularityDescription(0.8) == "Very uniform")
        #expect(vm.regularityDescription(1.0) == "Very uniform")
    }

    // MARK: - selectFile Tests

    @Test("selectFile with mock returning nil keeps selectedURL as nil")
    @MainActor
    func selectFileReturningNil() async {
        let mock = MockFileDialogService()
        mock.urlToReturn = nil
        let vm = ImportFontViewModel(fileDialogService: mock)

        await vm.selectFile()

        #expect(vm.selectedURL == nil)
        #expect(mock.selectFileCalled == true)
    }

    @Test("selectFile with mock returning URL sets selectedURL")
    @MainActor
    func selectFileReturningURL() async {
        let mock = MockFileDialogService()
        let testURL = URL(fileURLWithPath: "/tmp/TestFont.ttf")
        mock.urlToReturn = testURL
        let vm = ImportFontViewModel(fileDialogService: mock)

        await vm.selectFile()

        #expect(vm.selectedURL == testURL)
        #expect(mock.selectFileCalled == true)
    }
}
