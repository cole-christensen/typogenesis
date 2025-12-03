import SwiftUI

struct Inspector: View {
    @EnvironmentObject var appState: AppState

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
}

#Preview {
    Inspector()
        .environmentObject(AppState())
        .frame(width: 280)
}
