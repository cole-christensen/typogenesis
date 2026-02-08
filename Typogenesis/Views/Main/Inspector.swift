import SwiftUI

struct Inspector: View {
    @EnvironmentObject var appState: AppState
    @State private var showPreview = true

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                if let project = appState.currentProject {
                    projectInfoSection(project)

                    if let character = appState.selectedGlyph,
                       let glyph = project.glyph(for: character) {
                        glyphInfoSection(glyph)
                    }

                    metricsSection(project)

                    quickPreviewSection(project)
                } else {
                    Text("No project selected")
                        .foregroundColor(.secondary)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                }
            }
            .padding()
        }
        .background(Color(nsColor: .controlBackgroundColor))
    }

    @ViewBuilder
    func projectInfoSection(_ project: FontProject) -> some View {
        InspectorSection(title: "Project") {
            InspectorField(label: "Name", value: project.name)
            InspectorField(label: "Family", value: project.family)
            InspectorField(label: "Style", value: project.style)
            InspectorField(label: "Glyphs", value: "\(project.glyphs.count)")
        }
    }

    @ViewBuilder
    func glyphInfoSection(_ glyph: Glyph) -> some View {
        InspectorSection(title: "Selected Glyph") {
            InspectorField(label: "Character", value: String(glyph.character))
            InspectorField(label: "Name", value: glyph.name)
            InspectorField(label: "Width", value: "\(glyph.advanceWidth)")
            InspectorField(label: "LSB", value: "\(glyph.leftSideBearing)")
            InspectorField(label: "RSB", value: "\(glyph.rightSideBearing)")
            InspectorField(label: "Contours", value: "\(glyph.outline.contours.count)")

            if let source = glyph.generatedBy {
                InspectorField(label: "Source", value: source.rawValue)
            }
        }
    }

    @ViewBuilder
    func metricsSection(_ project: FontProject) -> some View {
        InspectorSection(title: "Metrics") {
            InspectorField(label: "Units/Em", value: "\(project.metrics.unitsPerEm)")
            InspectorField(label: "Ascender", value: "\(project.metrics.ascender)")
            InspectorField(label: "Descender", value: "\(project.metrics.descender)")
            InspectorField(label: "x-Height", value: "\(project.metrics.xHeight)")
            InspectorField(label: "Cap Height", value: "\(project.metrics.capHeight)")
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

// MARK: - Inspector Components

/// A section with a header and content
struct InspectorSection<Content: View>: View {
    let title: String
    @ViewBuilder let content: () -> Content

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title)
                .font(.headline)
                .foregroundColor(.primary)

            VStack(alignment: .leading, spacing: 6) {
                content()
            }
            .padding(12)
            .background(Color(nsColor: .textBackgroundColor).opacity(0.5))
            .cornerRadius(8)
        }
    }
}

/// A label-value field for the inspector
struct InspectorField: View {
    let label: String
    let value: String

    var body: some View {
        HStack(alignment: .top, spacing: 8) {
            Text(label)
                .font(.callout)
                .foregroundColor(.secondary)

            Spacer()

            Text(value.isEmpty ? "—" : value)
                .font(.callout)
                .fontWeight(.medium)
                .foregroundColor(.primary)
                .help(value)
        }
    }
}

/// Compact font preview for the Inspector sidebar
struct QuickFontPreview: View {
    let project: FontProject
    @State private var sampleText = "Aa Bb Cc"

    var body: some View {
        VStack(spacing: 8) {
            if project.glyphs.isEmpty {
                // Empty state when no glyphs defined
                VStack(spacing: 4) {
                    Image(systemName: "character.textbox")
                        .font(.title2)
                        .foregroundColor(.secondary)
                    Text("No glyphs defined")
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .lineLimit(1)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
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
}

#Preview {
    Inspector()
        .environmentObject(AppState())
}
