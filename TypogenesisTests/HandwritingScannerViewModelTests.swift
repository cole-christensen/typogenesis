import Testing
import AppKit
@testable import Typogenesis

// MARK: - Tests

@Suite("HandwritingScannerViewModel Tests")
struct HandwritingScannerViewModelTests {

    // MARK: - Helpers

    @MainActor
    private func makeVM(fileDialog: MockFileDialogService = MockFileDialogService()) -> HandwritingScannerViewModel {
        HandwritingScannerViewModel(fileDialogService: fileDialog)
    }

    @MainActor
    private func makeDetectedCharacters(count: Int) -> [HandwritingScannerViewModel.DetectedCharacter] {
        (0..<count).map { i in
            HandwritingScannerViewModel.DetectedCharacter(
                boundingBox: CGRect(x: i * 50, y: 0, width: 40, height: 40),
                outline: GlyphOutline(contours: [
                    Contour(points: [
                        PathPoint(position: CGPoint(x: 0, y: 0), type: .corner),
                        PathPoint(position: CGPoint(x: 40, y: 0), type: .corner),
                        PathPoint(position: CGPoint(x: 40, y: 40), type: .corner),
                        PathPoint(position: CGPoint(x: 0, y: 40), type: .corner),
                    ], isClosed: true)
                ]),
                assignedCharacter: nil
            )
        }
    }

    @MainActor
    private func makeProject() -> FontProject {
        FontProject(name: "Test", family: "Test", style: "Regular")
    }

    // MARK: - Tests

    @Test("Initial state is upload step with no image and no detected characters")
    @MainActor func initialState() {
        let vm = makeVM()

        #expect(vm.currentStep == .upload)
        #expect(vm.uploadedImage == nil)
        #expect(vm.detectedCharacters.isEmpty)
        #expect(vm.selectedCharacterIndex == nil)
        #expect(vm.isProcessing == false)
        #expect(vm.threshold == 0.5)
        #expect(vm.simplification == 2.0)
        #expect(vm.showSampleSheetInfo == false)
        #expect(vm.processingError == nil)
        #expect(vm.replaceExisting == false)
        #expect(vm.autoFitMetrics == true)
        #expect(vm.generateKerning == false)
        #expect(vm.isGeneratingKerning == false)
    }

    @Test("canProceed returns false at upload step with no image")
    @MainActor func canProceedFalseAtUploadWithNoImage() {
        let vm = makeVM()

        #expect(vm.currentStep == .upload)
        #expect(vm.uploadedImage == nil)
        #expect(vm.canProceed == false)
    }

    @Test("canProceed returns true at upload step with an image")
    @MainActor func canProceedTrueAtUploadWithImage() {
        let vm = makeVM()
        // Create a minimal 1x1 NSImage
        vm.uploadedImage = NSImage(size: NSSize(width: 1, height: 1))

        #expect(vm.currentStep == .upload)
        #expect(vm.canProceed == true)
    }

    @Test("canProceed returns false at process step with no detected characters")
    @MainActor func canProceedFalseAtProcessWithNoCharacters() {
        let vm = makeVM()
        vm.currentStep = .process

        #expect(vm.canProceed == false)
    }

    @Test("canProceed returns true at process step with detected characters")
    @MainActor func canProceedTrueAtProcessWithCharacters() {
        let vm = makeVM()
        vm.currentStep = .process
        vm.detectedCharacters = makeDetectedCharacters(count: 3)

        #expect(vm.canProceed == true)
    }

    @Test("canProceed returns false at assign step when no characters are assigned")
    @MainActor func canProceedFalseAtAssignWithNoAssigned() {
        let vm = makeVM()
        vm.currentStep = .assign
        vm.detectedCharacters = makeDetectedCharacters(count: 3)

        #expect(vm.canProceed == false)
    }

    @Test("canProceed returns true at assign step when at least one character is assigned")
    @MainActor func canProceedTrueAtAssignWithAssigned() {
        let vm = makeVM()
        vm.currentStep = .assign
        vm.detectedCharacters = makeDetectedCharacters(count: 3)
        vm.detectedCharacters[0].assignedCharacter = "A"

        #expect(vm.canProceed == true)
    }

    @Test("autoAssignFromTemplate assigns A-Z to first 26 characters")
    @MainActor func autoAssignFromTemplate26() {
        let vm = makeVM()
        vm.detectedCharacters = makeDetectedCharacters(count: 30)

        vm.autoAssignFromTemplate()

        let letters = Array("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        for (i, letter) in letters.enumerated() {
            #expect(vm.detectedCharacters[i].assignedCharacter == letter)
        }
        // Characters beyond 26 should remain unassigned
        for i in 26..<30 {
            #expect(vm.detectedCharacters[i].assignedCharacter == nil)
        }
    }

    @Test("autoAssignFromTemplate handles fewer than 26 detected characters")
    @MainActor func autoAssignFromTemplateFewerThan26() {
        let vm = makeVM()
        vm.detectedCharacters = makeDetectedCharacters(count: 5)

        vm.autoAssignFromTemplate()

        let letters = Array("ABCDE")
        for (i, letter) in letters.enumerated() {
            #expect(vm.detectedCharacters[i].assignedCharacter == letter)
        }
    }

    @Test("importGlyphs creates new glyphs in project for assigned characters")
    @MainActor func importGlyphsCreatesNewGlyphs() {
        let vm = makeVM()
        vm.detectedCharacters = makeDetectedCharacters(count: 3)
        vm.detectedCharacters[0].assignedCharacter = "A"
        vm.detectedCharacters[1].assignedCharacter = "B"
        // Index 2 left unassigned

        var project = makeProject()
        #expect(project.glyphs.isEmpty)

        vm.importGlyphs(into: &project)

        #expect(project.glyphs.count == 2)
        #expect(project.glyphs["A"] != nil)
        #expect(project.glyphs["B"] != nil)
        #expect(project.glyphs["A"]?.character == "A")
        #expect(project.glyphs["B"]?.character == "B")
    }

    @Test("importGlyphs skips existing glyphs when replaceExisting is false")
    @MainActor func importGlyphsRespectsReplaceExistingFalse() {
        let vm = makeVM()
        vm.replaceExisting = false
        vm.detectedCharacters = makeDetectedCharacters(count: 1)
        vm.detectedCharacters[0].assignedCharacter = "A"

        var project = makeProject()
        let existingGlyph = Glyph(character: "A", advanceWidth: 999, leftSideBearing: 50)
        project.glyphs["A"] = existingGlyph

        vm.importGlyphs(into: &project)

        // Original glyph should be preserved
        #expect(project.glyphs["A"]?.advanceWidth == 999)
    }

    @Test("importGlyphs replaces existing glyphs when replaceExisting is true")
    @MainActor func importGlyphsRespectsReplaceExistingTrue() {
        let vm = makeVM()
        vm.replaceExisting = true
        vm.detectedCharacters = makeDetectedCharacters(count: 1)
        vm.detectedCharacters[0].assignedCharacter = "A"

        var project = makeProject()
        let existingGlyph = Glyph(character: "A", advanceWidth: 999, leftSideBearing: 50)
        project.glyphs["A"] = existingGlyph

        vm.importGlyphs(into: &project)

        // Glyph should have been replaced (advance width recalculated)
        #expect(project.glyphs["A"]?.advanceWidth != 999)
        #expect(project.glyphs["A"]?.outline.contours.isEmpty == false)
    }

    @Test("resetScanner returns to initial state")
    @MainActor func resetScannerResetsState() {
        let vm = makeVM()

        // Mutate state
        vm.currentStep = .assign
        vm.uploadedImage = NSImage(size: NSSize(width: 1, height: 1))
        vm.detectedCharacters = makeDetectedCharacters(count: 5)
        vm.selectedCharacterIndex = 2
        vm.isProcessing = true
        vm.threshold = 0.8
        vm.simplification = 4.0
        vm.showSampleSheetInfo = true
        vm.processingError = "some error"
        vm.replaceExisting = true
        vm.autoFitMetrics = false
        vm.generateKerning = true
        vm.isGeneratingKerning = true

        vm.resetScanner()

        #expect(vm.currentStep == .upload)
        #expect(vm.uploadedImage == nil)
        #expect(vm.detectedCharacters.isEmpty)
        #expect(vm.selectedCharacterIndex == nil)
        #expect(vm.isProcessing == false)
        #expect(vm.threshold == 0.5)
        #expect(vm.simplification == 2.0)
        #expect(vm.showSampleSheetInfo == false)
        #expect(vm.processingError == nil)
        #expect(vm.replaceExisting == false)
        #expect(vm.autoFitMetrics == true)
        #expect(vm.generateKerning == false)
        #expect(vm.isGeneratingKerning == false)
    }

    @Test("generateSampleSheetPDFData returns non-empty data")
    @MainActor func generateSampleSheetPDFDataNonEmpty() {
        let vm = makeVM()

        let data = vm.generateSampleSheetPDFData()

        #expect(data.count > 0)
        // PDF files start with %PDF
        let prefix = String(data: data.prefix(4), encoding: .ascii)
        #expect(prefix == "%PDF")
    }

    @Test("scaleRect scales correctly for matching view and image sizes")
    @MainActor func scaleRectIdentity() {
        let vm = makeVM()
        let rect = CGRect(x: 10, y: 20, width: 30, height: 40)
        let size = CGSize(width: 100, height: 100)

        let result = vm.scaleRect(rect, to: size, imageSize: size)

        #expect(abs(result.origin.x - 10) < 0.001)
        #expect(abs(result.origin.y - 20) < 0.001)
        #expect(abs(result.width - 30) < 0.001)
        #expect(abs(result.height - 40) < 0.001)
    }

    @Test("scaleRect scales down when view is smaller than image")
    @MainActor func scaleRectScalesDown() {
        let vm = makeVM()
        let rect = CGRect(x: 100, y: 100, width: 200, height: 200)
        let viewSize = CGSize(width: 500, height: 500)
        let imageSize = CGSize(width: 1000, height: 1000)

        let result = vm.scaleRect(rect, to: viewSize, imageSize: imageSize)

        // Scale factor is 0.5 (500/1000), offset is 0 (centered, same aspect ratio)
        #expect(abs(result.origin.x - 50) < 0.001)
        #expect(abs(result.origin.y - 50) < 0.001)
        #expect(abs(result.width - 100) < 0.001)
        #expect(abs(result.height - 100) < 0.001)
    }

    @Test("scaleRect handles zero image size gracefully")
    @MainActor func scaleRectZeroImageSize() {
        let vm = makeVM()
        let rect = CGRect(x: 10, y: 20, width: 30, height: 40)
        let viewSize = CGSize(width: 500, height: 500)
        let imageSize = CGSize(width: 0, height: 0)

        // Should not crash -- uses max(1) guard
        let result = vm.scaleRect(rect, to: viewSize, imageSize: imageSize)
        #expect(result.width.isFinite)
        #expect(result.height.isFinite)
    }

    @Test("selectImage calls file dialog service")
    @MainActor func selectImageCallsFileDialog() async {
        let mockDialog = MockFileDialogService()
        mockDialog.selectFileResult = nil
        let vm = makeVM(fileDialog: mockDialog)

        await vm.selectImage()

        #expect(mockDialog.selectFileCalled == true)
        // No URL returned, so image should remain nil
        #expect(vm.uploadedImage == nil)
        #expect(vm.currentStep == .upload)
    }

    @Test("ScannerStep titles are correct")
    @MainActor func scannerStepTitles() {
        #expect(HandwritingScannerViewModel.ScannerStep.upload.title == "Upload")
        #expect(HandwritingScannerViewModel.ScannerStep.process.title == "Process")
        #expect(HandwritingScannerViewModel.ScannerStep.assign.title == "Assign")
        #expect(HandwritingScannerViewModel.ScannerStep.import_.title == "Import")
    }

    @Test("ScannerStep icons are correct")
    @MainActor func scannerStepIcons() {
        #expect(HandwritingScannerViewModel.ScannerStep.upload.icon == "photo")
        #expect(HandwritingScannerViewModel.ScannerStep.process.icon == "wand.and.rays")
        #expect(HandwritingScannerViewModel.ScannerStep.assign.icon == "character.textbox")
        #expect(HandwritingScannerViewModel.ScannerStep.import_.icon == "square.and.arrow.down")
    }
}
