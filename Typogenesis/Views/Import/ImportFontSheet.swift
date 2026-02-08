import SwiftUI
import UniformTypeIdentifiers

struct ImportFontSheet: View {
    @EnvironmentObject var appState: AppState
    @Environment(\.dismiss) var dismiss

    @State private var selectedURL: URL?
    @State private var isAnalyzing = false
    @State private var importedProject: FontProject?
    @State private var extractedStyle: StyleEncoder.FontStyle?
    @State private var analysisError: String?
    @State private var showError = false

    var body: some View {
        VStack(spacing: 20) {
            // Header
            HStack {
                Image(systemName: "doc.badge.plus")
                    .font(.title2)
                    .foregroundColor(.accentColor)
                Text("Import Font")
                    .font(.title2)
                    .fontWeight(.semibold)
            }

            if let project = importedProject {
                // Show analysis results
                analysisResultsView(project: project)
            } else {
                // Show file picker
                filePickerView
            }
        }
        .padding(24)
        .frame(width: 500)
        .alert("Import Error", isPresented: $showError) {
            Button("OK", role: .cancel) { }
        } message: {
            Text(analysisError ?? "Unknown error")
        }
    }

    @ViewBuilder
    var filePickerView: some View {
        VStack(spacing: 16) {
            // Instructions
            Text("Select a TrueType (.ttf) or OpenType (.otf) font file to import and analyze.")
                .font(.body)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)

            // File selection
            if let url = selectedURL {
                HStack {
                    Image(systemName: "doc.fill")
                        .foregroundColor(.accentColor)
                    Text(url.lastPathComponent)
                        .lineLimit(1)
                    Spacer()
                    Button("Change") {
                        selectFile()
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                }
                .padding()
                .background(Color(nsColor: .controlBackgroundColor))
                .cornerRadius(8)
            } else {
                Button(action: selectFile) {
                    VStack(spacing: 12) {
                        Image(systemName: "plus.circle.dashed")
                            .font(.system(size: 40))
                            .foregroundColor(.secondary)
                        Text("Select Font File")
                            .font(.headline)
                        Text("TTF or OTF format")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    .frame(maxWidth: .infinity)
                    .padding(40)
                    .background(
                        RoundedRectangle(cornerRadius: 12)
                            .strokeBorder(style: StrokeStyle(lineWidth: 2, dash: [8]))
                            .foregroundColor(.secondary.opacity(0.5))
                    )
                }
                .buttonStyle(.plain)
            }

            Divider()

            // Actions
            HStack {
                Button("Cancel") {
                    dismiss()
                }
                .keyboardShortcut(.cancelAction)
                .accessibilityIdentifier(AccessibilityID.Import.cancelButton)

                Spacer()

                if isAnalyzing {
                    ProgressView()
                        .scaleEffect(0.7)
                        .frame(width: 20, height: 20)
                        .accessibilityIdentifier(AccessibilityID.Import.analyzingIndicator)
                    Text("Analyzing...")
                        .foregroundColor(.secondary)
                } else {
                    Button("Analyze & Import") {
                        Task {
                            await analyzeFont()
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .keyboardShortcut(.defaultAction)
                    .disabled(selectedURL == nil)
                    .accessibilityIdentifier(AccessibilityID.Import.analyzeButton)
                }
            }
        }
        .accessibilityIdentifier(AccessibilityID.Import.sheet)
    }

    @ViewBuilder
    func analysisResultsView(project: FontProject) -> some View {
        VStack(spacing: 16) {
            // Font info
            VStack(alignment: .leading, spacing: 8) {
                Text("Font Information")
                    .font(.headline)

                Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 4) {
                    GridRow {
                        Text("Family:").foregroundColor(.secondary)
                        Text(project.family).fontWeight(.medium)
                    }
                    GridRow {
                        Text("Style:").foregroundColor(.secondary)
                        Text(project.style)
                    }
                    GridRow {
                        Text("Glyphs:").foregroundColor(.secondary)
                        Text("\(project.glyphs.count)")
                    }
                    GridRow {
                        Text("Kerning pairs:").foregroundColor(.secondary)
                        Text("\(project.kerning.count)")
                    }
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding()
            .background(Color(nsColor: .controlBackgroundColor))
            .cornerRadius(8)

            // Style analysis
            if let style = extractedStyle {
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Text("Style Analysis")
                            .font(.headline)
                        Spacer()
                        Text(style.serifStyle.rawValue)
                            .font(.caption)
                            .padding(.horizontal, 8)
                            .padding(.vertical, 2)
                            .background(Color.accentColor.opacity(0.2))
                            .cornerRadius(4)
                    }

                    // Style metrics
                    VStack(spacing: 8) {
                        styleMetricRow(label: "Stroke Weight", value: style.strokeWeight, description: weightDescription(style.strokeWeight))
                        styleMetricRow(label: "Stroke Contrast", value: style.strokeContrast, description: contrastDescription(style.strokeContrast))
                        styleMetricRow(label: "Roundness", value: style.roundness, description: roundnessDescription(style.roundness))
                        styleMetricRow(label: "Regularity", value: style.regularity, description: regularityDescription(style.regularity))
                    }

                    Divider()

                    // Proportions
                    HStack {
                        VStack(alignment: .leading) {
                            Text("x-Height Ratio")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            Text(String(format: "%.0f%%", style.xHeightRatio * 100))
                                .font(.headline)
                        }
                        Spacer()
                        VStack(alignment: .leading) {
                            Text("Width Ratio")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            Text(String(format: "%.0f%%", style.widthRatio * 100))
                                .font(.headline)
                        }
                        Spacer()
                        VStack(alignment: .leading) {
                            Text("Slant")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            Text(String(format: "%.1f", style.slant))
                                .font(.headline)
                        }
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding()
                .background(Color(nsColor: .controlBackgroundColor))
                .cornerRadius(8)
            }

            Divider()

            // Actions
            HStack {
                Button("Back") {
                    importedProject = nil
                    extractedStyle = nil
                }
                .accessibilityIdentifier(AccessibilityID.Import.backButton)

                Spacer()

                Button("Cancel") {
                    dismiss()
                }
                .accessibilityIdentifier(AccessibilityID.Import.cancelButton)

                Button("Import Font") {
                    applyImport()
                }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.defaultAction)
                .accessibilityIdentifier(AccessibilityID.Import.importButton)
            }
        }
        .accessibilityIdentifier(AccessibilityID.Import.styleAnalysis)
    }

    @ViewBuilder
    func styleMetricRow(label: String, value: Float, description: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack {
                Text(label)
                    .font(.subheadline)
                Spacer()
                Text(description)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            GeometryReader { geometry in
                ZStack(alignment: .leading) {
                    Rectangle()
                        .fill(Color.secondary.opacity(0.2))
                        .frame(height: 6)
                        .cornerRadius(3)
                    Rectangle()
                        .fill(Color.accentColor)
                        .frame(width: geometry.size.width * CGFloat(value), height: 6)
                        .cornerRadius(3)
                }
            }
            .frame(height: 6)
        }
    }

    // MARK: - Private Methods

    private func selectFile() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [
            UTType(filenameExtension: "ttf") ?? .data,
            UTType(filenameExtension: "otf") ?? .data
        ]
        panel.allowsMultipleSelection = false
        panel.message = "Select a font file to import"

        if panel.runModal() == .OK {
            selectedURL = panel.url
        }
    }

    private func analyzeFont() async {
        guard let url = selectedURL else { return }

        isAnalyzing = true

        do {
            // Parse font
            let parser = FontParser()
            let project = try await parser.parse(url: url)

            // Extract style
            let encoder = StyleEncoder()
            let style = try await encoder.extractStyle(from: project)

            await MainActor.run {
                importedProject = project
                extractedStyle = style
                isAnalyzing = false
            }
        } catch {
            await MainActor.run {
                isAnalyzing = false
                analysisError = error.localizedDescription
                showError = true
            }
        }
    }

    private func applyImport() {
        guard let project = importedProject else { return }
        appState.currentProject = project
        appState.selectedGlyph = nil
        if let url = selectedURL {
            appState.addToRecentProjects(url)
        }
        dismiss()
    }

    // MARK: - Description Helpers

    private func weightDescription(_ value: Float) -> String {
        switch value {
        case 0..<0.3: return "Light"
        case 0.3..<0.5: return "Regular"
        case 0.5..<0.7: return "Medium"
        case 0.7..<0.85: return "Bold"
        default: return "Heavy"
        }
    }

    private func contrastDescription(_ value: Float) -> String {
        switch value {
        case 0..<0.2: return "Monolinear"
        case 0.2..<0.4: return "Low contrast"
        case 0.4..<0.6: return "Moderate"
        case 0.6..<0.8: return "High contrast"
        default: return "Very high"
        }
    }

    private func roundnessDescription(_ value: Float) -> String {
        switch value {
        case 0..<0.3: return "Geometric"
        case 0.3..<0.5: return "Mixed"
        case 0.5..<0.7: return "Organic"
        default: return "Fluid"
        }
    }

    private func regularityDescription(_ value: Float) -> String {
        switch value {
        case 0..<0.4: return "Irregular"
        case 0.4..<0.6: return "Moderate"
        case 0.6..<0.8: return "Consistent"
        default: return "Very uniform"
        }
    }
}

#Preview {
    ImportFontSheet()
        .environmentObject(AppState())
}
