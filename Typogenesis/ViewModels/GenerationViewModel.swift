import SwiftUI
import UniformTypeIdentifiers

@MainActor
final class GenerationViewModel: ObservableObject {

    // MARK: - Enums

    enum GenerationMode: String, CaseIterable {
        case completeFont = "Complete Font"
        case missingGlyphs = "Missing Glyphs"
        case styleTransfer = "Style Transfer"
        case variation = "Create Variation"
    }

    enum CharacterSetOption: String, CaseIterable {
        case basicLatin = "Basic Latin (A-Z, a-z, 0-9)"
        case extendedLatin = "Extended Latin"
        case punctuation = "Punctuation & Symbols"
        case cyrillic = "Cyrillic"
        case greek = "Greek"
        case custom = "Custom Selection"

        var characters: String {
            switch self {
            case .basicLatin:
                return "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
            case .extendedLatin:
                return "ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ"
            case .punctuation:
                return "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
            case .cyrillic:
                return "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя"
            case .greek:
                return "ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩαβγδεζηθικλμνξοπρστυφχψω"
            case .custom:
                return ""
            }
        }
    }

    // MARK: - Published Properties

    @Published var selectedMode: GenerationMode = .completeFont
    @Published var selectedCharacterSet: CharacterSetOption = .basicLatin
    @Published var stylePrompt: String = ""
    @Published var referenceImage: NSImage?
    @Published var referenceFontStyle: StyleEncoder.FontStyle?
    @Published var isGenerating = false
    @Published var progress: Double = 0
    @Published var generatedCount = 0
    @Published var generatedGlyphs: [Character: Glyph] = [:]
    @Published var errorMessage: String?
    @Published var showingError = false

    // MARK: - Dependencies

    private let fileDialogService: FileDialogService

    // MARK: - Init

    init(fileDialogService: FileDialogService = NSPanelFileDialogService()) {
        self.fileDialogService = fileDialogService
    }

    // MARK: - Computed Properties

    var modeDescription: String {
        switch selectedMode {
        case .completeFont:
            return "Generate a complete set of glyphs from scratch based on style description"
        case .missingGlyphs:
            return "Generate only the glyphs missing from your current font"
        case .styleTransfer:
            return "Clone the style from an existing font or image"
        case .variation:
            return "Create a variation (bold, italic, etc.) of existing glyphs"
        }
    }

    var canGenerate: Bool {
        !charactersToGenerate(existingGlyphs: nil).isEmpty
    }

    func charactersToGenerate(existingGlyphs: [Character: Glyph]?) -> [Character] {
        if selectedMode == .missingGlyphs, let existing = existingGlyphs {
            return Array(selectedCharacterSet.characters).filter { existing[$0] == nil }
        }
        return Array(selectedCharacterSet.characters)
    }

    func totalGlyphsToGenerate(existingGlyphs: [Character: Glyph]?) -> Int {
        charactersToGenerate(existingGlyphs: existingGlyphs).count
    }

    // MARK: - Actions

    func startGeneration(metrics: FontMetrics) {
        isGenerating = true
        progress = 0
        generatedCount = 0
        generatedGlyphs = [:]

        let characters = charactersToGenerate(existingGlyphs: nil)
        let style = referenceFontStyle ?? StyleEncoder.FontStyle.default

        Task {
            let generator = GlyphGenerator()

            do {
                let results = try await generator.generateBatch(
                    characters: characters,
                    mode: .fromScratch(style: style),
                    metrics: metrics,
                    settings: .fast
                ) { completed, total in
                    Task { @MainActor in
                        self.generatedCount = completed
                        self.progress = Double(completed) / Double(total)
                    }
                }

                for (index, result) in results.enumerated() {
                    self.generatedGlyphs[characters[index]] = result.glyph
                }
                self.isGenerating = false
            } catch {
                self.errorMessage = "Generation failed: \(error.localizedDescription)"
                self.showingError = true
                self.isGenerating = false
            }
        }
    }

    func selectReferenceFont() {
        Task {
            guard let url = await fileDialogService.selectFile(types: [.font], message: nil) else {
                return
            }

            do {
                let parser = FontParser()
                let project = try await parser.parse(url: url)

                let encoder = StyleEncoder()
                let style = try await encoder.extractStyle(from: project)

                self.referenceFontStyle = style
            } catch {
                self.errorMessage = "Failed to analyze font: \(error.localizedDescription)"
                self.showingError = true
            }
        }
    }

    func selectReferenceImage() {
        Task {
            guard let url = await fileDialogService.selectFile(types: [.image], message: nil) else {
                return
            }

            self.referenceImage = NSImage(contentsOf: url)
        }
    }

    func applyToProject(_ project: inout FontProject) {
        for (char, glyph) in generatedGlyphs {
            project.glyphs[char] = glyph
        }
        generatedGlyphs = [:]
    }
}
