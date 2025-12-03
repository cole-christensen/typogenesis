import SwiftUI

struct ExportSheet: View {
    @EnvironmentObject var appState: AppState
    @Environment(\.dismiss) var dismiss

    @State private var selectedFormat: ExportFormat = .otf
    @State private var isExporting = false

    enum ExportFormat: String, CaseIterable {
        case ttf = "TrueType (.ttf)"
        case otf = "OpenType (.otf)"
        case woff = "Web Open Font (.woff)"
        case woff2 = "Web Open Font 2 (.woff2)"
        case ufo = "Unified Font Object (.ufo)"
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

                    Text("Glyphs: \(project.glyphs.count)")
                        .foregroundColor(.secondary)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding()
                .background(Color(nsColor: .controlBackgroundColor))
                .cornerRadius(8)
            }

            Picker("Format", selection: $selectedFormat) {
                ForEach(ExportFormat.allCases, id: \.self) { format in
                    Text(format.rawValue).tag(format)
                }
            }
            .pickerStyle(.radioGroup)

            HStack {
                Button("Cancel") {
                    dismiss()
                }
                .keyboardShortcut(.cancelAction)

                Spacer()

                Button("Export...") {
                    exportFont()
                }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.defaultAction)
                .disabled(isExporting)
            }
        }
        .padding(24)
        .frame(width: 400)
    }

    private func exportFont() {
        guard let project = appState.currentProject else { return }

        let panel = NSSavePanel()
        panel.nameFieldStringValue = "\(project.name).\(fileExtension)"
        panel.allowedContentTypes = [.init(filenameExtension: fileExtension)!]

        if panel.runModal() == .OK, let url = panel.url {
            isExporting = true
            // Export logic will be implemented in Phase 2
            isExporting = false
            dismiss()
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
}

#Preview {
    ExportSheet()
        .environmentObject(AppState())
}
