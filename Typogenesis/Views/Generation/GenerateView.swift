import SwiftUI

struct GenerateView: View {
    @EnvironmentObject var appState: AppState
    @State private var selectedMode: GenerationMode = .completeFont
    @State private var selectedCharacterSet: CharacterSet = .basicLatin
    @State private var stylePrompt: String = ""
    @State private var referenceImage: NSImage?
    @State private var isGenerating = false
    @State private var progress: Double = 0
    @State private var generatedCount = 0

    enum GenerationMode: String, CaseIterable {
        case completeFont = "Complete Font"
        case missingGlyphs = "Missing Glyphs"
        case styleTransfer = "Style Transfer"
        case variation = "Create Variation"
    }

    enum CharacterSet: String, CaseIterable {
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
                        ForEach(CharacterSet.allCases, id: \.self) { charset in
                            Text(charset.rawValue).tag(charset)
                        }
                    }
                    .labelsHidden()

                    if selectedCharacterSet != .custom {
                        Text("\(selectedCharacterSet.characters.count) characters")
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
                }

                Spacer()
            }
            .padding()
        }
        .background(Color(nsColor: .controlBackgroundColor))
    }

    @ViewBuilder
    var previewPanel: some View {
        VStack(spacing: 0) {
            // Preview Header
            HStack {
                Text("Preview")
                    .font(.headline)
                Spacer()
                Text("Generated glyphs will appear here")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            .padding()
            .background(Color(nsColor: .controlBackgroundColor))

            // Preview Content
            if isGenerating || generatedCount > 0 {
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
                ForEach(Array(selectedCharacterSet.characters.prefix(generatedCount)), id: \.self) { char in
                    VStack(spacing: 4) {
                        RoundedRectangle(cornerRadius: 4)
                            .fill(Color(nsColor: .textBackgroundColor))
                            .frame(width: 60, height: 60)
                            .overlay {
                                Text(String(char))
                                    .font(.system(size: 32))
                            }

                        Text(String(char))
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }

                // Placeholder for remaining characters
                if isGenerating {
                    ForEach(0..<(totalGlyphsToGenerate - generatedCount), id: \.self) { _ in
                        VStack(spacing: 4) {
                            RoundedRectangle(cornerRadius: 4)
                                .fill(Color(nsColor: .controlBackgroundColor))
                                .frame(width: 60, height: 60)
                                .overlay {
                                    ProgressView()
                                        .scaleEffect(0.5)
                                }

                            Text("...")
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
    var modelStatusSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("AI Models")
                .font(.headline)

            VStack(spacing: 6) {
                modelStatusRow(name: "Glyph Diffusion", status: .notLoaded, size: "~250 MB")
                modelStatusRow(name: "Style Encoder", status: .notLoaded, size: "~50 MB")
                modelStatusRow(name: "Kerning Predictor", status: .notLoaded, size: "~15 MB")
            }

            Button("Download Models") {
                // Would trigger model download
            }
            .buttonStyle(.bordered)
            .controlSize(.small)

            Text("Models are required for AI generation. They will be downloaded and cached locally.")
                .font(.caption)
                .foregroundColor(.secondary)
        }
    }

    enum ModelStatus {
        case notLoaded
        case downloading(Double)
        case loaded
        case error
    }

    @ViewBuilder
    func modelStatusRow(name: String, status: ModelStatus, size: String) -> some View {
        HStack {
            Circle()
                .fill(statusColor(status))
                .frame(width: 8, height: 8)

            Text(name)
                .font(.subheadline)

            Spacer()

            Text(size)
                .font(.caption)
                .foregroundColor(.secondary)

            Text(statusText(status))
                .font(.caption)
                .foregroundColor(statusColor(status))
        }
    }

    private func statusColor(_ status: ModelStatus) -> Color {
        switch status {
        case .notLoaded: return .gray
        case .downloading: return .orange
        case .loaded: return .green
        case .error: return .red
        }
    }

    private func statusText(_ status: ModelStatus) -> String {
        switch status {
        case .notLoaded: return "Not loaded"
        case .downloading(let progress): return "\(Int(progress * 100))%"
        case .loaded: return "Ready"
        case .error: return "Error"
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
        // Would check if models are loaded
        !selectedCharacterSet.characters.isEmpty || selectedCharacterSet == .custom
    }

    private var totalGlyphsToGenerate: Int {
        selectedCharacterSet.characters.count
    }

    private func startGeneration() {
        isGenerating = true
        progress = 0
        generatedCount = 0

        let total = totalGlyphsToGenerate

        // Simulate generation progress using async/await
        Task {
            for i in 1...total {
                try? await Task.sleep(nanoseconds: 100_000_000) // 0.1 seconds
                generatedCount = i
                progress = Double(i) / Double(total)
            }
            isGenerating = false
        }
    }

    private func selectReferenceFont() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.font]
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false

        if panel.runModal() == .OK {
            // Would load font and extract style
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

#Preview {
    GenerateView()
        .environmentObject(AppState())
        .frame(width: 900, height: 600)
}
