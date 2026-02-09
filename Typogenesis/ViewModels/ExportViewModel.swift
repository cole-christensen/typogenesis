import SwiftUI
import UniformTypeIdentifiers

@MainActor
final class ExportViewModel: ObservableObject {

    // MARK: - Types

    enum ExportFormat: String, CaseIterable, Sendable {
        case ttf = "TrueType (.ttf)"
        case otf = "OpenType (.otf)"
        case woff = "Web Open Font (.woff)"
        case woff2 = "Web Open Font 2 (.woff2)"
        case ufo = "Unified Font Object (.ufo)"
        case designspace = "DesignSpace (.designspace)"

        var isDirectory: Bool {
            self == .ufo || self == .designspace
        }

        var requiresVariableFont: Bool {
            self == .designspace
        }
    }

    enum ExportError: Error, LocalizedError {
        case unsupportedFormat

        var errorDescription: String? {
            switch self {
            case .unsupportedFormat:
                return "This export format is not yet supported"
            }
        }
    }

    // MARK: - Published Properties

    @Published var selectedFormat: ExportFormat = .ttf
    @Published var includeKerning = true
    @Published var isExporting = false
    @Published var exportError: String?
    @Published var showingError = false

    // MARK: - Callbacks

    var onExportComplete: (() -> Void)?

    // MARK: - Dependencies

    private let fileDialogService: FileDialogService

    // MARK: - Init

    init(fileDialogService: FileDialogService = NSPanelFileDialogService()) {
        self.fileDialogService = fileDialogService
    }

    // MARK: - Computed Properties

    func availableFormats(for project: FontProject?) -> [ExportFormat] {
        guard let project else {
            return ExportFormat.allCases.filter { !$0.requiresVariableFont }
        }

        if project.variableConfig.isVariableFont {
            return ExportFormat.allCases
        } else {
            return ExportFormat.allCases.filter { !$0.requiresVariableFont }
        }
    }

    func isFormatAvailable(_ format: ExportFormat, for project: FontProject?) -> Bool {
        guard let project else { return false }

        if format == .designspace {
            return project.variableConfig.isVariableFont &&
                   project.variableConfig.masters.count >= 2 &&
                   !project.variableConfig.axes.isEmpty
        }

        return true
    }

    var fileExtension: String {
        switch selectedFormat {
        case .ttf: return "ttf"
        case .otf: return "otf"
        case .woff: return "woff"
        case .woff2: return "woff2"
        case .ufo: return "ufo"
        case .designspace: return "designspace"
        }
    }

    // MARK: - Export Actions

    func exportFont(project: FontProject) async {
        if selectedFormat.isDirectory {
            let message: String
            if selectedFormat == .designspace {
                message = "Choose a location to save the DesignSpace package"
            } else {
                message = "Choose a location to save the UFO package"
            }

            guard let baseURL = await fileDialogService.selectDirectory(message: message) else { return }

            if selectedFormat == .designspace {
                let dsURL = baseURL.appendingPathComponent("\(project.family).designspace")
                await exportDesignSpace(project: project, to: dsURL)
            } else {
                let ufoURL = baseURL.appendingPathComponent("\(project.name).ufo")
                await exportUFO(project: project, to: ufoURL)
            }
        } else {
            let defaultName = "\(project.name).\(fileExtension)"
            var types: [UTType] = []
            if let contentType = UTType(filenameExtension: fileExtension) {
                types = [contentType]
            }

            guard let url = await fileDialogService.selectSaveLocation(
                defaultName: defaultName,
                types: types,
                message: nil
            ) else { return }

            await exportBinaryFormat(project: project, to: url)
        }
    }

    func exportBinaryFormat(project: FontProject, to url: URL) async {
        isExporting = true

        do {
            let exporter = FontExporter()
            let format: FontExporter.ExportFormat

            switch selectedFormat {
            case .ttf:
                format = .ttf
            case .otf:
                format = .otf
            case .woff:
                format = .woff
            case .woff2:
                format = .woff2
            case .ufo, .designspace:
                throw ExportError.unsupportedFormat
            }

            let options = FontExporter.ExportOptions(
                format: format,
                includeKerning: includeKerning
            )
            try await exporter.export(project: project, to: url, options: options)

            isExporting = false
            onExportComplete?()
        } catch {
            isExporting = false
            exportError = error.localizedDescription
            showingError = true
        }
    }

    func exportUFO(project: FontProject, to url: URL) async {
        isExporting = true

        do {
            let ufoExporter = UFOExporter()
            let options = UFOExporter.ExportOptions(
                includeKerning: includeKerning
            )
            try await ufoExporter.export(project: project, to: url, options: options)

            isExporting = false
            onExportComplete?()
        } catch {
            isExporting = false
            exportError = error.localizedDescription
            showingError = true
        }
    }

    func exportDesignSpace(project: FontProject, to url: URL) async {
        isExporting = true

        do {
            let designSpaceExporter = DesignSpaceExporter()
            let options = DesignSpaceExporter.ExportOptions(
                includeKerning: includeKerning
            )
            try await designSpaceExporter.export(project: project, to: url, options: options)

            isExporting = false
            onExportComplete?()
        } catch {
            isExporting = false
            exportError = error.localizedDescription
            showingError = true
        }
    }
}
