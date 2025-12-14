import SwiftUI

struct GenerateView: View {
    @EnvironmentObject var appState: AppState
    @StateObject private var modelManager = ModelManager.shared
    @State private var selectedMode: GenerationMode = .completeFont
    @State private var selectedCharacterSet: CharacterSetOption = .basicLatin
    @State private var stylePrompt: String = ""
    @State private var referenceImage: NSImage?
    @State private var referenceFontStyle: StyleEncoder.FontStyle?
    @State private var isGenerating = false
    @State private var progress: Double = 0
    @State private var generatedCount = 0
    @State private var generatedGlyphs: [Character: Glyph] = [:]
    @State private var errorMessage: String?
    @State private var showingError = false

    enum GenerationMode: String, CaseIterable {
        case completeFont = "Complete Font"
        case missingGlyphs = "Missing Glyphs"
        case styleTransfer = "Style Transfer"
        case variation = "Create Variation"
    }

    enum CharacterSetOption: String, CaseIterable {
        case basicLatin = "Basic Latin (A-Z, a-z, 0-9)"
        case extendedLatin = "Extended Latin"
        case punctuation = "Punctuation & Symbols"
        case cyrillic = "Cyrillic"
        case greek = "Greek"
        case custom = "Custom Selection"

        var characters: String {
            switch self {
            case .basicLatin:
                return "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
            case .extendedLatin:
                return "ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ"
            case .punctuation:
                return "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
            case .cyrillic:
                return "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя"
            case .greek:
                return "ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩαβγδεζηθικλμνξοπρστυφχψω"
            case .custom:
                return ""
            }
        }
    }

    var body: some View {
        HSplitView {
            settingsPanel
                .frame(minWidth: 300, maxWidth: 400)

            previewPanel
        }
        .alert("Generation Error", isPresented: $showingError) {
            Button("OK") {}
        } message: {
            Text(errorMessage ?? "An unknown error occurred")
        }
    }

    @ViewBuilder
    var settingsPanel: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                // Header
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Image(systemName: "wand.and.stars")
                            .font(.title)
                            .foregroundColor(.accentColor)
                        Text("AI Generation")
                            .font(.title2)
                            .fontWeight(.semibold)
                    }
                    Text("Generate glyphs using AI models")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }

                Divider()

                // Generation Mode
                VStack(alignment: .leading, spacing: 8) {
                    Text("Generation Mode")
                        .font(.headline)

                    Picker("Mode", selection: $selectedMode) {
                        ForEach(GenerationMode.allCases, id: \.self) { mode in
                            Text(mode.rawValue).tag(mode)
                        }
                    }
                    .pickerStyle(.segmented)
                    .labelsHidden()

                    Text(modeDescription)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }

                // Character Set Selection
                VStack(alignment: .leading, spacing: 8) {
                    Text("Character Set")
                        .font(.headline)

                    Picker("Characters", selection: $selectedCharacterSet) {
                        ForEach(CharacterSetOption.allCases, id: \.self) { charset in
                            Text(charset.rawValue).tag(charset)
                        }
                    }
                    .labelsHidden()

                    if selectedCharacterSet != .custom {
                        Text("\(charactersToGenerate.count) characters")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }

                // Style Prompt
                if selectedMode == .completeFont || selectedMode == .variation {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Style Description")
                            .font(.headline)

                        TextField("e.g., Modern geometric sans-serif with rounded corners", text: $stylePrompt, axis: .vertical)
                            .textFieldStyle(.roundedBorder)
                            .lineLimit(3...6)

                        Text("Describe the visual style you want for the generated glyphs")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }

                // Reference Font (for style transfer)
                if selectedMode == .styleTransfer {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Reference Font")
                            .font(.headline)

                        HStack {
                            if let image = referenceImage {
                                Image(nsImage: image)
                                    .resizable()
                                    .aspectRatio(contentMode: .fit)
                                    .frame(height: 60)
                                    .background(Color(nsColor: .textBackgroundColor))
                                    .cornerRadius(4)
                            } else {
                                RoundedRectangle(cornerRadius: 4)
                                    .fill(Color(nsColor: .controlBackgroundColor))
                                    .frame(height: 60)
                                    .overlay {
                                        Text("No reference selected")
                                            .font(.caption)
                                            .foregroundColor(.secondary)
                                    }
                            }

                            VStack {
                                Button("Select Font...") {
                                    selectReferenceFont()
                                }
                                .buttonStyle(.bordered)

                                Button("Upload Image...") {
                                    selectReferenceImage()
                                }
                                .buttonStyle(.bordered)
                            }
                        }

                        if let style = referenceFontStyle {
                            styleInfoView(style)
                        }

                        Text("Select a font file or upload an image of the style you want to clone")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }

                Divider()

                // Model Status
                modelStatusSection

                Divider()

                // Generate Button
                VStack(spacing: 12) {
                    Button(action: startGeneration) {
                        HStack {
                            if isGenerating {
                                ProgressView()
                                    .scaleEffect(0.8)
                                    .padding(.trailing, 4)
                            } else {
                                Image(systemName: "sparkles")
                            }
                            Text(isGenerating ? "Generating..." : "Generate Glyphs")
                        }
                        .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
                    .disabled(isGenerating || !canGenerate)

                    if isGenerating {
                        VStack(spacing: 4) {
                            ProgressView(value: progress)
                            Text("\(generatedCount) / \(totalGlyphsToGenerate) glyphs")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }

                    if !generatedGlyphs.isEmpty && !isGenerating {
                        Button(action: addToProject) {
                            HStack {
                                Image(systemName: "plus.circle")
                                Text("Add to Project")
                            }
                            .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.large)
                    }
                }

                Spacer()
            }
            .padding()
        }
        .background(Color(nsColor: .controlBackgroundColor))
    }

    @ViewBuilder
    func styleInfoView(_ style: StyleEncoder.FontStyle) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Extracted Style:")
                .font(.caption)
                .fontWeight(.medium)

            HStack(spacing: 12) {
                styleMetric("Weight", value: style.strokeWeight)
                styleMetric("Contrast", value: style.strokeContrast)
                styleMetric("Round", value: style.roundness)
            }

            Text("Serif: \(style.serifStyle.rawValue)")
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .padding(8)
        .background(Color(nsColor: .controlBackgroundColor).opacity(0.5))
        .cornerRadius(4)
    }

    @ViewBuilder
    func styleMetric(_ label: String, value: Float) -> some View {
        VStack(spacing: 2) {
            Text(label)
                .font(.caption2)
                .foregroundColor(.secondary)
            Text(String(format: "%.1f", value))
                .font(.caption)
                .fontWeight(.medium)
        }
    }

    @ViewBuilder
    var previewPanel: some View {
        VStack(spacing: 0) {
            // Preview Header
            HStack {
                Text("Preview")
                    .font(.headline)
                Spacer()
                if !generatedGlyphs.isEmpty {
                    Text("\(generatedGlyphs.count) glyphs generated")
                        .font(.caption)
                        .foregroundColor(.secondary)
                } else {
                    Text("Generated glyphs will appear here")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
            .padding()
            .background(Color(nsColor: .controlBackgroundColor))

            // Preview Content
            if isGenerating || !generatedGlyphs.isEmpty {
                generationPreview
            } else {
                placeholderPreview
            }
        }
    }

    @ViewBuilder
    var placeholderPreview: some View {
        VStack(spacing: 16) {
            Image(systemName: "wand.and.stars.inverse")
                .font(.system(size: 64))
                .foregroundColor(.secondary.opacity(0.5))

            Text("AI Generation")
                .font(.title2)
                .foregroundColor(.secondary)

            Text("Configure your settings and click Generate to create glyphs using AI models.")
                .font(.body)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 40)

            Divider()
                .frame(width: 200)
                .padding(.vertical)

            VStack(alignment: .leading, spacing: 8) {
                featureRow(icon: "cpu", title: "Local Processing", description: "All generation runs on your Mac using Core ML")
                featureRow(icon: "lock.shield", title: "Private", description: "Your fonts and designs never leave your device")
                featureRow(icon: "bolt", title: "Fast", description: "Optimized for Apple Silicon with Neural Engine")
            }
            .padding()
            .background(Color(nsColor: .controlBackgroundColor).opacity(0.5))
            .cornerRadius(8)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color(nsColor: .textBackgroundColor))
    }

    @ViewBuilder
    func featureRow(icon: String, title: String, description: String) -> some View {
        HStack(spacing: 12) {
            Image(systemName: icon)
                .frame(width: 24)
                .foregroundColor(.accentColor)
            VStack(alignment: .leading) {
                Text(title)
                    .font(.subheadline)
                    .fontWeight(.medium)
                Text(description)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
    }

    @ViewBuilder
    var generationPreview: some View {
        ScrollView {
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 80))], spacing: 12) {
                // Show generated glyphs
                ForEach(Array(generatedGlyphs.keys.sorted()), id: \.self) { char in
                    if let glyph = generatedGlyphs[char] {
                        glyphPreviewCell(character: char, glyph: glyph)
                    }
                }

                // Placeholder for remaining characters during generation
                if isGenerating {
                    let remaining = charactersToGenerate.filter { !generatedGlyphs.keys.contains($0) }
                    ForEach(Array(remaining), id: \.self) { char in
                        VStack(spacing: 4) {
                            RoundedRectangle(cornerRadius: 4)
                                .fill(Color(nsColor: .controlBackgroundColor))
                                .frame(width: 60, height: 60)
                                .overlay {
                                    ProgressView()
                                        .scaleEffect(0.5)
                                }

                            Text(String(char))
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }
                }
            }
            .padding()
        }
        .background(Color(nsColor: .textBackgroundColor))
    }

    @ViewBuilder
    func glyphPreviewCell(character: Character, glyph: Glyph) -> some View {
        VStack(spacing: 4) {
            GlyphPreviewCanvas(glyph: glyph)
                .frame(width: 60, height: 60)
                .background(Color(nsColor: .textBackgroundColor))
                .cornerRadius(4)
                .overlay {
                    RoundedRectangle(cornerRadius: 4)
                        .stroke(Color.secondary.opacity(0.2), lineWidth: 1)
                }

            Text(String(character))
                .font(.caption)
                .foregroundColor(.secondary)
        }
    }

    @ViewBuilder
    var modelStatusSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("AI Models")
                .font(.headline)

            VStack(spacing: 6) {
                modelStatusRow(type: .glyphDiffusion)
                modelStatusRow(type: .styleEncoder)
                modelStatusRow(type: .kerningNet)
            }

            HStack {
                Button("Download All") {
                    Task {
                        for modelType in ModelManager.ModelType.allCases {
                            await modelManager.downloadModel(modelType)
                        }
                    }
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .disabled(modelManager.isLoading)

                if modelManager.allModelsReady {
                    Label("Ready", systemImage: "checkmark.circle.fill")
                        .font(.caption)
                        .foregroundColor(.green)
                }
            }

            Text("Models are required for AI generation. They will be downloaded and cached locally (~\(ModelManager.totalModelSize)).")
                .font(.caption)
                .foregroundColor(.secondary)
        }
    }

    @ViewBuilder
    func modelStatusRow(type: ModelManager.ModelType) -> some View {
        let status = modelManager.status(for: type)

        HStack {
            Circle()
                .fill(statusColor(for: status))
                .frame(width: 8, height: 8)

            Text(type.displayName)
                .font(.subheadline)

            Spacer()

            Text(type.estimatedSize)
                .font(.caption)
                .foregroundColor(.secondary)

            Text(status.displayText)
                .font(.caption)
                .foregroundColor(statusColor(for: status))
        }
    }

    private func statusColor(for status: ModelManager.ModelStatus) -> Color {
        switch status {
        case .notDownloaded: return .gray
        case .downloading: return .orange
        case .downloaded: return .yellow
        case .loading: return .orange
        case .loaded: return .green
        case .error: return .red
        }
    }

    private var modeDescription: String {
        switch selectedMode {
        case .completeFont:
            return "Generate a complete set of glyphs from scratch based on style description"
        case .missingGlyphs:
            return "Generate only the glyphs missing from your current font"
        case .styleTransfer:
            return "Clone the style from an existing font or image"
        case .variation:
            return "Create a variation (bold, italic, etc.) of existing glyphs"
        }
    }

    private var canGenerate: Bool {
        !charactersToGenerate.isEmpty
    }

    private var charactersToGenerate: [Character] {
        if selectedMode == .missingGlyphs, let project = appState.currentProject {
            // Only generate characters not already in the project
            return Array(selectedCharacterSet.characters).filter { project.glyphs[$0] == nil }
        }
        return Array(selectedCharacterSet.characters)
    }

    private var totalGlyphsToGenerate: Int {
        charactersToGenerate.count
    }

    private func startGeneration() {
        isGenerating = true
        progress = 0
        generatedCount = 0
        generatedGlyphs = [:]

        let characters = charactersToGenerate
        let metrics = appState.currentProject?.metrics ?? FontMetrics()

        Task {
            let generator = GlyphGenerator()
            let style = referenceFontStyle ?? StyleEncoder.FontStyle.default

            for (index, char) in characters.enumerated() {
                do {
                    let result = try await generator.generate(
                        character: char,
                        mode: .fromScratch(style: style),
                        metrics: metrics,
                        settings: .fast
                    )

                    await MainActor.run {
                        generatedGlyphs[char] = result.glyph
                        generatedCount = index + 1
                        progress = Double(index + 1) / Double(characters.count)
                    }
                } catch {
                    await MainActor.run {
                        errorMessage = "Failed to generate '\(char)': \(error.localizedDescription)"
                        showingError = true
                    }
                }
            }

            await MainActor.run {
                isGenerating = false
            }
        }
    }

    private func addToProject() {
        guard var project = appState.currentProject else { return }

        for (char, glyph) in generatedGlyphs {
            project.glyphs[char] = glyph
        }

        appState.currentProject = project
        generatedGlyphs = [:]
    }

    private func selectReferenceFont() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.font]
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false

        if panel.runModal() == .OK, let url = panel.url {
            Task {
                do {
                    let parser = FontParser()
                    let project = try await parser.parse(url: url)

                    let encoder = StyleEncoder()
                    let style = try await encoder.extractStyle(from: project)

                    await MainActor.run {
                        referenceFontStyle = style
                    }
                } catch {
                    await MainActor.run {
                        errorMessage = "Failed to analyze font: \(error.localizedDescription)"
                        showingError = true
                    }
                }
            }
        }
    }

    private func selectReferenceImage() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.image]
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false

        if panel.runModal() == .OK, let url = panel.url {
            referenceImage = NSImage(contentsOf: url)
        }
    }
}

// MARK: - Glyph Preview Canvas

struct GlyphPreviewCanvas: View {
    let glyph: Glyph

    var body: some View {
        Canvas { context, size in
            let bounds = glyph.outline.boundingBox
            guard bounds.width > 0 && bounds.height > 0 else { return }

            // Calculate scale and offset to center the glyph
            let scaleX = (size.width - 8) / CGFloat(bounds.width)
            let scaleY = (size.height - 8) / CGFloat(bounds.height)
            let scale = min(scaleX, scaleY)

            let scaledWidth = CGFloat(bounds.width) * scale
            let scaledHeight = CGFloat(bounds.height) * scale
            let offsetX = (size.width - scaledWidth) / 2 - CGFloat(bounds.minX) * scale
            let offsetY = (size.height - scaledHeight) / 2 - CGFloat(bounds.minY) * scale

            // Transform and draw
            var transform = CGAffineTransform.identity
            transform = transform.translatedBy(x: offsetX, y: size.height - offsetY)
            transform = transform.scaledBy(x: scale, y: -scale)

            let path = glyph.outline.cgPath
            let transformedPath = path.copy(using: &transform)

            if let transformedPath = transformedPath {
                context.fill(Path(transformedPath), with: .color(.primary))
            }
        }
    }
}

#Preview {
    GenerateView()
        .environmentObject(AppState())
        .frame(width: 900, height: 600)
}
