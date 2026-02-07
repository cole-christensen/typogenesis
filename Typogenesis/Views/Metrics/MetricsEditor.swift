import SwiftUI

struct MetricsEditor: View {
    @EnvironmentObject var appState: AppState
    @State private var editedMetrics: FontMetrics?
    @State private var showPreview = true

    var body: some View {
        HSplitView {
            metricsForm
                .layoutPriority(0)

            if showPreview {
                metricsPreview
                    .layoutPriority(1)
            }
        }
        .onAppear {
            editedMetrics = appState.currentProject?.metrics
        }
        .onChange(of: appState.currentProject?.metrics) { _, newMetrics in
            if editedMetrics == nil {
                editedMetrics = newMetrics
            }
        }
    }

    @ViewBuilder
    var metricsForm: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                headerSection

                if editedMetrics != nil {
                    unitsSection
                    verticalMetricsSection
                    metadataSection
                    actionsSection
                }
            }
            .padding()
        }
    }

    @ViewBuilder
    var headerSection: some View {
        HStack {
            Text("Font Metrics")
                .font(.title2)
                .fontWeight(.semibold)

            Spacer()

            Toggle("Preview", isOn: $showPreview)
                .toggleStyle(.switch)
                .controlSize(.small)
        }
    }

    @ViewBuilder
    var unitsSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Units")
                .font(.headline)

            MetricField(
                label: "Units Per Em",
                value: Binding(
                    get: { editedMetrics?.unitsPerEm ?? 1000 },
                    set: { editedMetrics?.unitsPerEm = $0 }
                ),
                help: "The size of the em square. Standard is 1000 or 2048."
            )
        }
        .padding()
        .background(Color(nsColor: .controlBackgroundColor))
        .cornerRadius(8)
    }

    @ViewBuilder
    var verticalMetricsSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Vertical Metrics")
                .font(.headline)

            MetricField(
                label: "Ascender",
                value: Binding(
                    get: { editedMetrics?.ascender ?? 800 },
                    set: { editedMetrics?.ascender = $0 }
                ),
                help: "Height of tallest glyphs above baseline"
            )

            MetricField(
                label: "Cap Height",
                value: Binding(
                    get: { editedMetrics?.capHeight ?? 700 },
                    set: { editedMetrics?.capHeight = $0 }
                ),
                help: "Height of capital letters"
            )

            MetricField(
                label: "x-Height",
                value: Binding(
                    get: { editedMetrics?.xHeight ?? 500 },
                    set: { editedMetrics?.xHeight = $0 }
                ),
                help: "Height of lowercase letters (without ascenders)"
            )

            MetricField(
                label: "Descender",
                value: Binding(
                    get: { editedMetrics?.descender ?? -200 },
                    set: { editedMetrics?.descender = $0 }
                ),
                help: "Depth below baseline (typically negative)"
            )

            MetricField(
                label: "Line Gap",
                value: Binding(
                    get: { editedMetrics?.lineGap ?? 90 },
                    set: { editedMetrics?.lineGap = $0 }
                ),
                help: "Additional space between lines of text"
            )
        }
        .padding()
        .background(Color(nsColor: .controlBackgroundColor))
        .cornerRadius(8)
    }

    @ViewBuilder
    var metadataSection: some View {
        if let project = appState.currentProject {
            VStack(alignment: .leading, spacing: 12) {
                Text("Font Info")
                    .font(.headline)

                LabeledContent("Family") {
                    Text(project.family)
                        .foregroundColor(.secondary)
                }

                LabeledContent("Style") {
                    Text(project.style)
                        .foregroundColor(.secondary)
                }

                LabeledContent("Glyphs") {
                    Text("\(project.glyphs.count)")
                        .foregroundColor(.secondary)
                }

                LabeledContent("Kerning Pairs") {
                    Text("\(project.kerning.count)")
                        .foregroundColor(.secondary)
                }
            }
            .padding()
            .background(Color(nsColor: .controlBackgroundColor))
            .cornerRadius(8)
        }
    }

    @ViewBuilder
    var actionsSection: some View {
        HStack {
            Button("Reset to Defaults") {
                editedMetrics = FontMetrics()
            }
            .buttonStyle(.bordered)

            Spacer()

            Button("Apply Changes") {
                applyChanges()
            }
            .buttonStyle(.borderedProminent)
            .disabled(!hasChanges)
        }
    }

    @ViewBuilder
    var metricsPreview: some View {
        VStack(spacing: 0) {
            Text("Preview")
                .font(.headline)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding()
                .background(Color(nsColor: .controlBackgroundColor))

            MetricsPreviewCanvas(metrics: editedMetrics ?? FontMetrics())
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }

    private var hasChanges: Bool {
        guard let current = appState.currentProject?.metrics,
              let edited = editedMetrics else { return false }
        return current != edited
    }

    private func applyChanges() {
        guard var project = appState.currentProject,
              let metrics = editedMetrics else { return }
        project.metrics = metrics
        appState.currentProject = project
    }
}

struct MetricField: View {
    let label: String
    @Binding var value: Int
    let help: String

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(label)
                    .frame(width: 100, alignment: .leading)

                TextField("", value: $value, format: .number)
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 100)

                Stepper("", value: $value, step: 10)
                    .labelsHidden()
            }

            Text(help)
                .font(.caption)
                .foregroundColor(.secondary)
        }
    }
}

struct MetricsPreviewCanvas: View {
    let metrics: FontMetrics
    @State private var sampleText = "Hamburgefonstiv"

    private let scale: CGFloat = 0.3

    var body: some View {
        VStack {
            TextField("Sample Text", text: $sampleText)
                .textFieldStyle(.roundedBorder)
                .padding()

            GeometryReader { geometry in
                Canvas { context, size in
                    // Guard against zero-size canvas
                    guard size.width > 0, size.height > 0 else { return }

                    let centerX = size.width / 2
                    let baselineY = size.height * 0.7

                    // Draw metrics lines - clamp y values to visible range
                    let clampY: (CGFloat) -> CGFloat = { y in
                        max(0, min(y, size.height))
                    }

                    drawMetricLine(context: context, y: clampY(baselineY), width: size.width, color: .blue, label: "Baseline")
                    drawMetricLine(context: context, y: clampY(baselineY - CGFloat(metrics.xHeight) * scale), width: size.width, color: .green, label: "x-Height")
                    drawMetricLine(context: context, y: clampY(baselineY - CGFloat(metrics.capHeight) * scale), width: size.width, color: .orange, label: "Cap Height")
                    drawMetricLine(context: context, y: clampY(baselineY - CGFloat(metrics.ascender) * scale), width: size.width, color: .red, label: "Ascender")
                    drawMetricLine(context: context, y: clampY(baselineY - CGFloat(metrics.descender) * scale), width: size.width, color: .purple, label: "Descender")

                    // Draw em square indicator - clamp to visible area
                    let safeEmSize = max(CGFloat(metrics.unitsPerEm), 1) * scale
                    let emRectY = baselineY - CGFloat(metrics.ascender) * scale
                    let emRect = CGRect(
                        x: max(0, centerX - safeEmSize / 2),
                        y: max(0, emRectY),
                        width: min(safeEmSize, size.width),
                        height: min(safeEmSize, size.height)
                    )
                    context.stroke(Path(emRect), with: .color(.gray.opacity(0.5)), lineWidth: 1)

                    // Draw sample text using system font scaled to match metrics
                    // Guard against zero/negative capHeight
                    let safeFontSize = max(CGFloat(metrics.capHeight) * scale * 0.8, 8)
                    let text = Text(sampleText)
                        .font(.system(size: safeFontSize))
                    context.draw(text, at: CGPoint(x: centerX, y: clampY(baselineY - CGFloat(metrics.xHeight) * scale / 2)), anchor: .center)
                }
            }
        }
        .background(Color(nsColor: .textBackgroundColor))
    }

    private func drawMetricLine(context: GraphicsContext, y: CGFloat, width: CGFloat, color: Color, label: String) {
        var path = Path()
        path.move(to: CGPoint(x: 20, y: y))
        path.addLine(to: CGPoint(x: width - 20, y: y))
        context.stroke(path, with: .color(color.opacity(0.7)), lineWidth: 1)

        // Draw label
        let text = Text(label)
            .font(.caption2)
            .foregroundColor(color)
        context.draw(text, at: CGPoint(x: 25, y: y - 10), anchor: .leading)
    }
}

#Preview {
    MetricsEditor()
        .environmentObject({
            let state = AppState()
            state.createNewProject()
            return state
        }())
        .frame(width: 800, height: 600)
}
