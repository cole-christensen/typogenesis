import SwiftUI

/// A panel that displays sample text rendered with the current font project's glyphs
struct FontPreviewPanel: View {
    let project: FontProject

    @State private var sampleText: String = "The quick brown fox jumps over the lazy dog"
    @State private var fontSize: CGFloat = 48
    @State private var showSettings = false
    @State private var previewMode: PreviewMode = .paragraph

    enum PreviewMode: String, CaseIterable {
        case paragraph = "Paragraph"
        case waterfall = "Waterfall"
        case glyphProof = "Glyph Proof"
        case kerning = "Kerning"
    }

    var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack {
                Text("Preview")
                    .font(.headline)

                Spacer()

                Picker("", selection: $previewMode) {
                    ForEach(PreviewMode.allCases, id: \.self) { mode in
                        Text(mode.rawValue).tag(mode)
                    }
                }
                .pickerStyle(.segmented)
                .frame(width: 300)

                Button(action: { showSettings.toggle() }) {
                    Image(systemName: "slider.horizontal.3")
                }
                .buttonStyle(.borderless)
            }
            .padding()
            .background(Color(nsColor: .controlBackgroundColor))

            Divider()

            // Preview content
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    switch previewMode {
                    case .paragraph:
                        paragraphPreview
                    case .waterfall:
                        waterfallPreview
                    case .glyphProof:
                        glyphProofPreview
                    case .kerning:
                        kerningPreview
                    }
                }
                .padding()
            }
            .background(Color(nsColor: .textBackgroundColor))

            // Settings bar (collapsible)
            if showSettings {
                Divider()
                settingsBar
            }
        }
    }

    // MARK: - Preview Modes

    @ViewBuilder
    var paragraphPreview: some View {
        VStack(alignment: .leading, spacing: 16) {
            // Editable sample text
            TextField("Sample text", text: $sampleText, axis: .vertical)
                .textFieldStyle(.plain)
                .font(.system(size: 14))
                .padding(8)
                .background(Color(nsColor: .controlBackgroundColor))
                .cornerRadius(6)

            // Rendered preview
            FontTextRenderer(
                text: sampleText,
                project: project,
                fontSize: fontSize
            )
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    @ViewBuilder
    var waterfallPreview: some View {
        let sizes: [CGFloat] = [12, 14, 18, 24, 36, 48, 72, 96]

        VStack(alignment: .leading, spacing: 12) {
            ForEach(sizes, id: \.self) { size in
                VStack(alignment: .leading, spacing: 4) {
                    Text("\(Int(size))pt")
                        .font(.caption)
                        .foregroundColor(.secondary)

                    FontTextRenderer(
                        text: sampleText.isEmpty ? "AaBbCcDdEeFfGgHhIiJjKk" : sampleText,
                        project: project,
                        fontSize: size
                    )
                }
            }
        }
    }

    @ViewBuilder
    var glyphProofPreview: some View {
        let uppercaseRow = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        let lowercaseRow = "abcdefghijklmnopqrstuvwxyz"
        let numbersRow = "0123456789"
        let punctuationRow = "!@#$%^&*()_+-=[]{}|;':\",./<>?"

        VStack(alignment: .leading, spacing: 24) {
            proofRow("Uppercase", text: uppercaseRow)
            proofRow("Lowercase", text: lowercaseRow)
            proofRow("Numbers", text: numbersRow)
            proofRow("Punctuation", text: punctuationRow)
        }
    }

    @ViewBuilder
    func proofRow(_ label: String, text: String) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(label)
                .font(.caption)
                .foregroundColor(.secondary)

            FontTextRenderer(
                text: text,
                project: project,
                fontSize: fontSize
            )
        }
    }

    @ViewBuilder
    var kerningPreview: some View {
        let kerningPairs = [
            "AV AW AT AY",
            "VA Vo We Ya",
            "To Ta Te Tr Ty",
            "LT LV LY",
            "ff fi fl ffi ffl"
        ]

        VStack(alignment: .leading, spacing: 16) {
            Text("Common kerning pairs:")
                .font(.caption)
                .foregroundColor(.secondary)

            ForEach(kerningPairs, id: \.self) { pair in
                FontTextRenderer(
                    text: pair,
                    project: project,
                    fontSize: fontSize
                )
            }

            Divider()

            Text("Kerning test string:")
                .font(.caption)
                .foregroundColor(.secondary)

            FontTextRenderer(
                text: "AVAST WATER YOUTH",
                project: project,
                fontSize: fontSize * 1.5
            )
        }
    }

    // MARK: - Settings Bar

    @ViewBuilder
    var settingsBar: some View {
        HStack(spacing: 20) {
            HStack {
                Text("Size:")
                    .foregroundColor(.secondary)
                Slider(value: $fontSize, in: 12...144)
                    .frame(width: 120)
                Text("\(Int(fontSize))pt")
                    .frame(width: 40, alignment: .trailing)
            }

            Divider()
                .frame(height: 20)

            Button("Reset") {
                sampleText = "The quick brown fox jumps over the lazy dog"
                fontSize = 48
            }
            .buttonStyle(.bordered)
            .controlSize(.small)

            Spacer()
        }
        .padding()
        .background(Color(nsColor: .controlBackgroundColor))
    }
}

// MARK: - Font Text Renderer

/// Renders text using glyphs from a FontProject
struct FontTextRenderer: View {
    let text: String
    let project: FontProject
    let fontSize: CGFloat

    var body: some View {
        Canvas { context, size in
            var xOffset: CGFloat = 0
            let baseline = fontSize * 0.8  // Approximate baseline

            // Guard against division by zero
            let safeUnitsPerEm = max(CGFloat(project.metrics.unitsPerEm), 1)
            let scale = fontSize / safeUnitsPerEm

            var currentIndex = text.startIndex
            for char in text {
                if let glyph = project.glyphs[char] {
                    // Draw glyph
                    let path = glyph.outline.cgPath

                    var transform = CGAffineTransform.identity
                    transform = transform.translatedBy(x: xOffset, y: baseline)
                    transform = transform.scaledBy(x: scale, y: -scale)  // Flip Y

                    if let transformedPath = path.copy(using: &transform) {
                        context.fill(Path(transformedPath), with: .color(.primary))
                    }

                    // Apply kerning using tracked index (not firstIndex which breaks for repeated chars)
                    let nextIndex = text.index(after: currentIndex)
                    if nextIndex < text.endIndex {
                        let nextChar = text[nextIndex]
                        if let kerningPair = project.kerning.first(where: { $0.left == char && $0.right == nextChar }) {
                            xOffset += CGFloat(kerningPair.value) * scale
                        }
                    }

                    xOffset += CGFloat(glyph.advanceWidth) * scale
                } else if char == " " {
                    // Space - use default width
                    xOffset += fontSize * 0.3
                } else {
                    // Missing glyph - draw placeholder
                    let placeholderSize = fontSize * 0.6
                    let rect = CGRect(
                        x: xOffset + fontSize * 0.05,
                        y: baseline - placeholderSize,
                        width: placeholderSize,
                        height: placeholderSize
                    )
                    context.stroke(
                        Path(rect),
                        with: .color(.secondary.opacity(0.5)),
                        lineWidth: 1
                    )
                    xOffset += fontSize * 0.7
                }
                currentIndex = text.index(after: currentIndex)
            }
        }
        .frame(height: fontSize * 1.2)
    }
}

// MARK: - Preview

#Preview {
    var project = FontProject(name: "Test", family: "Test", style: "Regular")

    // Add some sample glyphs
    let characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    for char in characters {
        let outline = GlyphOutline(contours: [
            Contour(
                points: [
                    PathPoint(position: CGPoint(x: 50, y: 0), type: .corner),
                    PathPoint(position: CGPoint(x: 450, y: 0), type: .corner),
                    PathPoint(position: CGPoint(x: 450, y: 700), type: .corner),
                    PathPoint(position: CGPoint(x: 50, y: 700), type: .corner)
                ],
                isClosed: true
            )
        ])
        project.glyphs[char] = Glyph(
            character: char,
            outline: outline,
            advanceWidth: 500,
            leftSideBearing: 50
        )
    }

    return FontPreviewPanel(project: project)
        .frame(width: 800, height: 600)
}
