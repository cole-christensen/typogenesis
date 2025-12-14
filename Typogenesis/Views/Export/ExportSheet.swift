import SwiftUI
import UniformTypeIdentifiers

struct ExportSheet: View {
    @EnvironmentObject var appState: AppState
    @Environment(\.dismiss) var dismiss

    @State private var selectedFormat: ExportFormat = .ttf
    @State private var includeKerning = true
    @State private var isExporting = false
    @State private var exportError: String?
    @State private var showingError = false

    enum ExportFormat: String, CaseIterable {
        case ttf = "TrueType (.ttf)"
        case otf = "OpenType (.otf)"
        case woff = "Web Open Font (.woff)"
        case woff2 = "Web Open Font 2 (.woff2)"
        case ufo = "Unified Font Object (.ufo)"

        var isSupported: Bool {
            switch self {
            case .ttf, .otf, .woff, .ufo: return true
            case .woff2: return false
            }
        }

        var statusText: String? {
            switch self {
            case .ttf, .otf, .woff, .ufo: return nil
            case .woff2: return "Requires Brotli"
            }
        }

        var isDirectory: Bool {
            self == .ufo
        }
    }

    var body: some View {
        VStack(spacing: 20) {
            Text("Export Font")
                .font(.title2)
                .fontWeight(.semibold)

            if let project = appState.currentProject {
                VStack(alignment: .leading, spacing: 12) {
                    Text("Font: \(project.name)")
                        .font(.headline)

                    HStack {
                        Text("Glyphs: \(project.glyphs.count)")
                        if !project.kerning.isEmpty {
                            Text("Kerning pairs: \(project.kerning.count)")
                        }
                    }
                    .foregroundColor(.secondary)

                    if project.glyphs.isEmpty {
                        Label("No glyphs to export", systemImage: "exclamationmark.triangle")
                            .foregroundColor(.orange)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding()
                .background(Color(nsColor: .controlBackgroundColor))
                .cornerRadius(8)
            }

            VStack(alignment: .leading, spacing: 8) {
                Text("Format")
                    .font(.headline)

                ForEach(ExportFormat.allCases, id: \.self) { format in
                    HStack {
                        Button(action: {
                            if format.isSupported {
                                selectedFormat = format
                            }
                        }) {
                            HStack {
                                Image(systemName: selectedFormat == format ? "largecircle.fill.circle" : "circle")
                                    .foregroundColor(format.isSupported ? .accentColor : .secondary)
                                Text(format.rawValue)
                                    .foregroundColor(format.isSupported ? .primary : .secondary)
                            }
                        }
                        .buttonStyle(.plain)
                        .disabled(!format.isSupported)

                        if let status = format.statusText {
                            Text(status)
                                .font(.caption)
                                .foregroundColor(.secondary)
                                .padding(.horizontal, 6)
                                .padding(.vertical, 2)
                                .background(Color.secondary.opacity(0.2))
                                .cornerRadius(4)
                        }
                    }
                    .padding(.vertical, 2)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)

            if selectedFormat == .ttf || selectedFormat == .otf || selectedFormat == .woff || selectedFormat == .ufo {
                Toggle("Include kerning data", isOn: $includeKerning)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .accessibilityIdentifier(AccessibilityID.Export.includeKerningToggle)
            }

            if selectedFormat == .ufo {
                Text("UFO is a directory-based format used by font editors like Glyphs and RoboFont.")
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }

            Divider()

            HStack {
                Button("Cancel") {
                    dismiss()
                }
                .keyboardShortcut(.cancelAction)
                .accessibilityIdentifier(AccessibilityID.Export.cancelButton)

                Spacer()

                Button(action: exportFont) {
                    if isExporting {
                        ProgressView()
                            .scaleEffect(0.7)
                            .frame(width: 16, height: 16)
                    } else {
                        Text("Export...")
                    }
                }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.defaultAction)
                .disabled(isExporting || appState.currentProject?.glyphs.isEmpty == true)
                .accessibilityIdentifier(AccessibilityID.Export.exportButton)
            }
        }
        .padding(24)
        .frame(width: 400)
        .accessibilityIdentifier(AccessibilityID.Export.sheet)
        .alert("Export Error", isPresented: $showingError) {
            Button("OK") {}
        } message: {
            Text(exportError ?? "Unknown error")
        }
    }

    private func exportFont() {
        guard let project = appState.currentProject else { return }

        // UFO is a directory, so we need different handling
        if selectedFormat.isDirectory {
            let panel = NSOpenPanel()
            panel.canChooseDirectories = true
            panel.canChooseFiles = false
            panel.canCreateDirectories = true
            panel.message = "Choose a location to save the UFO package"
            panel.prompt = "Save UFO"

            guard panel.runModal() == .OK, let baseURL = panel.url else { return }

            let ufoURL = baseURL.appendingPathComponent("\(project.name).ufo")
            exportUFO(project: project, to: ufoURL)
        } else {
            let panel = NSSavePanel()
            panel.nameFieldStringValue = "\(project.name).\(fileExtension)"

            if let contentType = UTType(filenameExtension: fileExtension) {
                panel.allowedContentTypes = [contentType]
            }

            guard panel.runModal() == .OK, let url = panel.url else { return }
            exportBinaryFormat(project: project, to: url)
        }
    }

    private func exportBinaryFormat(project: FontProject, to url: URL) {
        isExporting = true

        Task {
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
                case .woff2, .ufo:
                    throw ExportError.unsupportedFormat
                }

                let options = FontExporter.ExportOptions(
                    format: format,
                    includeKerning: includeKerning
                )
                try await exporter.export(project: project, to: url, options: options)

                await MainActor.run {
                    isExporting = false
                    dismiss()
                }
            } catch {
                await MainActor.run {
                    isExporting = false
                    exportError = error.localizedDescription
                    showingError = true
                }
            }
        }
    }

    private func exportUFO(project: FontProject, to url: URL) {
        isExporting = true

        Task {
            do {
                let ufoExporter = UFOExporter()
                let options = UFOExporter.ExportOptions(
                    includeKerning: includeKerning
                )
                try await ufoExporter.export(project: project, to: url, options: options)

                await MainActor.run {
                    isExporting = false
                    dismiss()
                }
            } catch {
                await MainActor.run {
                    isExporting = false
                    exportError = error.localizedDescription
                    showingError = true
                }
            }
        }
    }

    private var fileExtension: String {
        switch selectedFormat {
        case .ttf: return "ttf"
        case .otf: return "otf"
        case .woff: return "woff"
        case .woff2: return "woff2"
        case .ufo: return "ufo"
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
}

#Preview {
    ExportSheet()
        .environmentObject(AppState())
}
