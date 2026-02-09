import Testing
@testable import Typogenesis

// MARK: - Test Helpers

private func makeProject(
    name: String = "TestFont",
    family: String = "TestFamily",
    variableConfig: VariableFontConfig = VariableFontConfig()
) -> FontProject {
    FontProject(
        name: name,
        family: family,
        style: "Regular",
        variableConfig: variableConfig
    )
}

private func makeVariableProject() -> FontProject {
    makeProject(variableConfig: .weightOnly())
}

// MARK: - Tests

@Suite("ExportViewModel Tests")
struct ExportViewModelTests {

    @Test("Initial state has correct defaults")
    @MainActor
    func initialState() {
        let vm = ExportViewModel(fileDialogService: MockFileDialogService())

        #expect(vm.selectedFormat == .ttf)
        #expect(vm.includeKerning == true)
        #expect(vm.isExporting == false)
        #expect(vm.exportError == nil)
        #expect(vm.showingError == false)
    }

    @Test("availableFormats excludes designspace when project is not variable font")
    @MainActor
    func availableFormatsNonVariable() {
        let vm = ExportViewModel(fileDialogService: MockFileDialogService())
        let project = makeProject()

        let formats = vm.availableFormats(for: project)

        #expect(!formats.contains(.designspace))
        #expect(formats.contains(.ttf))
        #expect(formats.contains(.otf))
        #expect(formats.contains(.woff))
        #expect(formats.contains(.woff2))
        #expect(formats.contains(.ufo))
    }

    @Test("availableFormats includes designspace when project is variable font")
    @MainActor
    func availableFormatsVariable() {
        let vm = ExportViewModel(fileDialogService: MockFileDialogService())
        let project = makeVariableProject()

        let formats = vm.availableFormats(for: project)

        #expect(formats.contains(.designspace))
        #expect(formats.contains(.ttf))
    }

    @Test("isFormatAvailable returns false for designspace with fewer than 2 masters")
    @MainActor
    func designspaceRequiresTwoMasters() {
        let vm = ExportViewModel(fileDialogService: MockFileDialogService())
        let config = VariableFontConfig(
            isVariableFont: true,
            axes: [.weight],
            masters: [FontMaster(name: "Regular", location: ["wght": 400])],
            instances: []
        )
        let project = makeProject(variableConfig: config)

        #expect(vm.isFormatAvailable(.designspace, for: project) == false)
    }

    @Test("isFormatAvailable returns true for ttf with any project")
    @MainActor
    func ttfAlwaysAvailable() {
        let vm = ExportViewModel(fileDialogService: MockFileDialogService())
        let project = makeProject()

        #expect(vm.isFormatAvailable(.ttf, for: project) == true)
    }

    @Test("isFormatAvailable returns false for nil project")
    @MainActor
    func nilProjectFormatsUnavailable() {
        let vm = ExportViewModel(fileDialogService: MockFileDialogService())

        #expect(vm.isFormatAvailable(.ttf, for: nil) == false)
        #expect(vm.isFormatAvailable(.designspace, for: nil) == false)
    }

    @Test("fileExtension returns correct extension for each format")
    @MainActor
    func fileExtensions() {
        let vm = ExportViewModel(fileDialogService: MockFileDialogService())

        vm.selectedFormat = .ttf
        #expect(vm.fileExtension == "ttf")

        vm.selectedFormat = .otf
        #expect(vm.fileExtension == "otf")

        vm.selectedFormat = .woff
        #expect(vm.fileExtension == "woff")

        vm.selectedFormat = .woff2
        #expect(vm.fileExtension == "woff2")

        vm.selectedFormat = .ufo
        #expect(vm.fileExtension == "ufo")

        vm.selectedFormat = .designspace
        #expect(vm.fileExtension == "designspace")
    }

    @Test("ExportFormat.isDirectory is true for ufo and designspace only")
    @MainActor
    func isDirectoryFormats() {
        #expect(ExportViewModel.ExportFormat.ufo.isDirectory == true)
        #expect(ExportViewModel.ExportFormat.designspace.isDirectory == true)
        #expect(ExportViewModel.ExportFormat.ttf.isDirectory == false)
        #expect(ExportViewModel.ExportFormat.otf.isDirectory == false)
        #expect(ExportViewModel.ExportFormat.woff.isDirectory == false)
        #expect(ExportViewModel.ExportFormat.woff2.isDirectory == false)
    }
}
