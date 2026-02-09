import Testing
import CoreGraphics
@testable import Typogenesis

// MARK: - Tests

@Suite("GenerationViewModel Tests")
struct GenerationViewModelTests {

    @Test("Initial state has correct defaults")
    @MainActor
    func initialState() {
        let vm = GenerationViewModel(fileDialogService: MockFileDialogService())

        #expect(vm.selectedMode == .completeFont)
        #expect(vm.selectedCharacterSet == .basicLatin)
        #expect(vm.stylePrompt == "")
        #expect(vm.referenceImage == nil)
        #expect(vm.referenceFontStyle == nil)
        #expect(vm.isGenerating == false)
        #expect(vm.progress == 0)
        #expect(vm.generatedCount == 0)
        #expect(vm.generatedGlyphs.isEmpty)
        #expect(vm.errorMessage == nil)
        #expect(vm.showingError == false)
    }

    @Test("modeDescription returns correct string for completeFont")
    @MainActor
    func modeDescriptionCompleteFont() {
        let vm = GenerationViewModel(fileDialogService: MockFileDialogService())
        vm.selectedMode = .completeFont
        #expect(vm.modeDescription == "Generate a complete set of glyphs from scratch based on style description")
    }

    @Test("modeDescription returns correct string for missingGlyphs")
    @MainActor
    func modeDescriptionMissingGlyphs() {
        let vm = GenerationViewModel(fileDialogService: MockFileDialogService())
        vm.selectedMode = .missingGlyphs
        #expect(vm.modeDescription == "Generate only the glyphs missing from your current font")
    }

    @Test("modeDescription returns correct string for styleTransfer")
    @MainActor
    func modeDescriptionStyleTransfer() {
        let vm = GenerationViewModel(fileDialogService: MockFileDialogService())
        vm.selectedMode = .styleTransfer
        #expect(vm.modeDescription == "Clone the style from an existing font or image")
    }

    @Test("modeDescription returns correct string for variation")
    @MainActor
    func modeDescriptionVariation() {
        let vm = GenerationViewModel(fileDialogService: MockFileDialogService())
        vm.selectedMode = .variation
        #expect(vm.modeDescription == "Create a variation (bold, italic, etc.) of existing glyphs")
    }

    @Test("canGenerate is true when character set is not empty")
    @MainActor
    func canGenerateWithBasicLatin() {
        let vm = GenerationViewModel(fileDialogService: MockFileDialogService())
        vm.selectedCharacterSet = .basicLatin
        #expect(vm.canGenerate == true)
    }

    @Test("canGenerate is false for custom with no characters")
    @MainActor
    func canGenerateCustomEmpty() {
        let vm = GenerationViewModel(fileDialogService: MockFileDialogService())
        vm.selectedCharacterSet = .custom
        #expect(vm.canGenerate == false)
    }

    @Test("charactersToGenerate returns full set for basicLatin")
    @MainActor
    func charactersToGenerateBasicLatin() {
        let vm = GenerationViewModel(fileDialogService: MockFileDialogService())
        vm.selectedCharacterSet = .basicLatin

        let chars = vm.charactersToGenerate(existingGlyphs: nil)
        let expected = Array("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")
        #expect(chars == expected)
    }

    @Test("charactersToGenerate filters out existing glyphs in missingGlyphs mode")
    @MainActor
    func charactersToGenerateFiltersMissing() {
        let vm = GenerationViewModel(fileDialogService: MockFileDialogService())
        vm.selectedMode = .missingGlyphs
        vm.selectedCharacterSet = .basicLatin

        // Simulate some existing glyphs
        let existingGlyphs: [Character: Glyph] = [
            "A": Glyph(character: "A"),
            "B": Glyph(character: "B"),
            "C": Glyph(character: "C"),
        ]

        let chars = vm.charactersToGenerate(existingGlyphs: existingGlyphs)

        #expect(!chars.contains("A"))
        #expect(!chars.contains("B"))
        #expect(!chars.contains("C"))
        #expect(chars.contains("D"))
        #expect(chars.contains("a"))
        #expect(chars.contains("0"))

        let fullSet = Array("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")
        #expect(chars.count == fullSet.count - 3)
    }

    @Test("totalGlyphsToGenerate matches characters count")
    @MainActor
    func totalGlyphsToGenerateMatchesCount() {
        let vm = GenerationViewModel(fileDialogService: MockFileDialogService())
        vm.selectedCharacterSet = .basicLatin

        let chars = vm.charactersToGenerate(existingGlyphs: nil)
        let total = vm.totalGlyphsToGenerate(existingGlyphs: nil)
        #expect(total == chars.count)
        #expect(total == 62) // 26 + 26 + 10
    }

    @Test("applyToProject merges glyphs and clears generatedGlyphs")
    @MainActor
    func applyToProjectMergesAndClears() {
        let vm = GenerationViewModel(fileDialogService: MockFileDialogService())

        // Simulate generated glyphs
        let glyphA = Glyph(character: "A", advanceWidth: 600)
        let glyphB = Glyph(character: "B", advanceWidth: 550)
        vm.generatedGlyphs = ["A": glyphA, "B": glyphB]

        var project = FontProject(name: "Test", family: "Test", style: "Regular")
        // Pre-existing glyph that should remain
        let glyphC = Glyph(character: "C", advanceWidth: 500)
        project.glyphs["C"] = glyphC

        vm.applyToProject(&project)

        // Generated glyphs should be merged into project
        #expect(project.glyphs["A"] != nil)
        #expect(project.glyphs["B"] != nil)
        // Pre-existing glyph should still be there
        #expect(project.glyphs["C"] != nil)
        // generatedGlyphs should be cleared
        #expect(vm.generatedGlyphs.isEmpty)
    }
}
