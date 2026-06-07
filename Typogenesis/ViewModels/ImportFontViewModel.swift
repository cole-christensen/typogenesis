import SwiftUI
import UniformTypeIdentifiers

@MainActor
final class ImportFontViewModel: ObservableObject {
    // MARK: - Published Properties

    @Published var selectedURL: URL?
    @Published var isAnalyzing = false
    @Published var importedProject: FontProject?
    @Published var extractedStyle: StyleEncoder.FontStyle?
    @Published var analysisError: String?
    @Published var showError = false

    // MARK: - Dependencies

    private let fileDialogService: FileDialogService

    // MARK: - Init

    init(fileDialogService: FileDialogService = NSPanelFileDialogService()) {
        self.fileDialogService = fileDialogService
    }

    // MARK: - Actions

    func selectFile() async {
        let url = await fileDialogService.selectFile(
            types: [
                UTType(filenameExtension: "ttf") ?? .data,
                UTType(filenameExtension: "otf") ?? .data,
            ],
            message: "Select a font file to import"
        )
        selectedURL = url
    }

    func analyzeFont() async {
        guard let url = selectedURL else { return }

        isAnalyzing = true

        do {
            let parser = FontParser()
            let project = try await parser.parse(url: url)

            let encoder = StyleEncoder()
            let style = try await encoder.extractStyle(from: project)

            importedProject = project
            extractedStyle = style
            isAnalyzing = false
        } catch {
            isAnalyzing = false
            analysisError = error.localizedDescription
            showError = true
        }
    }

    func reset() {
        importedProject = nil
        extractedStyle = nil
    }

    // MARK: - Description Helpers

    func weightDescription(_ value: Float) -> String {
        switch value {
        case 0..<0.3: return "Light"
        case 0.3..<0.5: return "Regular"
        case 0.5..<0.7: return "Medium"
        case 0.7..<0.85: return "Bold"
        default: return "Heavy"
        }
    }

    func contrastDescription(_ value: Float) -> String {
        switch value {
        case 0..<0.2: return "Monolinear"
        case 0.2..<0.4: return "Low contrast"
        case 0.4..<0.6: return "Moderate"
        case 0.6..<0.8: return "High contrast"
        default: return "Very high"
        }
    }

    func roundnessDescription(_ value: Float) -> String {
        switch value {
        case 0..<0.3: return "Geometric"
        case 0.3..<0.5: return "Mixed"
        case 0.5..<0.7: return "Organic"
        default: return "Fluid"
        }
    }

    func regularityDescription(_ value: Float) -> String {
        switch value {
        case 0..<0.4: return "Irregular"
        case 0.4..<0.6: return "Moderate"
        case 0.6..<0.8: return "Consistent"
        default: return "Very uniform"
        }
    }
}
