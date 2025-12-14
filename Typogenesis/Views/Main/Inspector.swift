import SwiftUI

struct Inspector: View {
    @EnvironmentObject var appState: AppState
    @State private var showPreview = true

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                if let project = appState.currentProject {
                    projectInfoSection(project)

                    if let character = appState.selectedGlyph,
                       let glyph = project.glyph(for: character) {
                        glyphInfoSection(glyph)
                    }

                    metricsSection(project)

                    Divider()

                    quickPreviewSection(project)
                } else {
                    Text("No project selected")
                        .foregroundColor(.secondary)
                }
            }
            .padding()
        }
        .background(Color(nsColor: .controlBackgroundColor))
    }

    func projectInfoSection(_ project: FontProject) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Project")
                .font(.headline)

            LabeledContent("Name", value: project.name)
            LabeledContent("Family", value: project.family)
            LabeledContent("Style", value: project.style)
            LabeledContent("Glyphs", value: "\(project.glyphs.count)")
        }
    }

    func glyphInfoSection(_ glyph: Glyph) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Selected Glyph")
                .font(.headline)

            LabeledContent("Character", value: String(glyph.character))
            LabeledContent("Name", value: glyph.name)
            LabeledContent("Advance Width", value: "\(glyph.advanceWidth)")
            LabeledContent("LSB", value: "\(glyph.leftSideBearing)")
            LabeledContent("RSB", value: "\(glyph.rightSideBearing)")
            LabeledContent("Contours", value: "\(glyph.outline.contours.count)")

            if let source = glyph.generatedBy {
                LabeledContent("Source", value: source.rawValue)
            }
        }
    }

    func metricsSection(_ project: FontProject) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Metrics")
                .font(.headline)

            LabeledContent("Units/Em", value: "\(project.metrics.unitsPerEm)")
            LabeledContent("Ascender", value: "\(project.metrics.ascender)")
            LabeledContent("Descender", value: "\(project.metrics.descender)")
            LabeledContent("x-Height", value: "\(project.metrics.xHeight)")
            LabeledContent("Cap Height", value: "\(project.metrics.capHeight)")
        }
    }

    @ViewBuilder
    func quickPreviewSection(_ project: FontProject) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Quick Preview")
                    .font(.headline)

                Spacer()

                Button(action: { showPreview.toggle() }) {
                    Image(systemName: showPreview ? "chevron.down" : "chevron.right")
                }
                .buttonStyle(.borderless)
            }

            if showPreview {
                QuickFontPreview(project: project)
                    .frame(height: 120)
                    .background(Color(nsColor: .textBackgroundColor))
                    .cornerRadius(8)
            }
        }
    }
}

/// Compact font preview for the Inspector sidebar
struct QuickFontPreview: View {
    let project: FontProject
    @State private var sampleText = "Aa Bb Cc"

    var body: some View {
        VStack(spacing: 8) {
            // Rendered text
            FontTextRenderer(
                text: sampleText,
                project: project,
                fontSize: 32
            )
            .frame(maxWidth: .infinity, alignment: .center)
            .padding(.top, 8)

            Spacer()

            // Editable sample text
            TextField("Sample", text: $sampleText)
                .textFieldStyle(.roundedBorder)
                .font(.caption)
                .padding(.horizontal, 8)
                .padding(.bottom, 8)
        }
    }
}

#Preview {
    Inspector()
        .environmentObject(AppState())
        .frame(width: 280)
}
