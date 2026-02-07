import SwiftUI
import UniformTypeIdentifiers

/// Multi-step wizard for cloning font styles
/// Allows users to import a reference font, analyze its style, and generate
/// new glyphs that match the visual characteristics of the reference.
struct CloneWizard: View {
    @EnvironmentObject var appState: AppState
    @StateObject private var viewModel = CloneWizardViewModel()

    var body: some View {
        HSplitView {
            // Left: Wizard steps
            wizardPanel
                .layoutPriority(0)

            // Right: Preview
            previewPanel
                .layoutPriority(1)
        }
        .accessibilityIdentifier(AccessibilityID.Clone.wizard)
    }

    // MARK: - Wizard Panel

    @ViewBuilder
    var wizardPanel: some View {
        VStack(spacing: 0) {
            // Step indicator
            stepIndicator
                .padding()
                .background(Color(nsColor: .controlBackgroundColor))

            Divider()

            // Step content
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    switch viewModel.currentStep {
                    case .selectFont:
                        selectFontStep
                    case .analyzeStyle:
                        analyzeStyleStep
                    case .selectCharacters:
                        selectCharactersStep
                    case .generate:
                        generateStep
                    case .complete:
                        completeStep
                    }
                }
                .padding()
            }

            Divider()

            // Navigation buttons
            navigationButtons
                .padding()
                .background(Color(nsColor: .controlBackgroundColor))
        }
    }

    // MARK: - Step Indicator

    @ViewBuilder
    var stepIndicator: some View {
        HStack(spacing: 8) {
            ForEach(CloneWizardViewModel.WizardStep.allCases, id: \.self) { step in
                HStack(spacing: 4) {
                    Circle()
                        .fill(stepColor(for: step))
                        .frame(width: 8, height: 8)

                    if step == viewModel.currentStep {
                        Text(step.title)
                            .font(.caption)
                            .fontWeight(.medium)
                    }
                }

                if step != CloneWizardViewModel.WizardStep.allCases.last {
                    Rectangle()
                        .fill(Color.secondary.opacity(0.3))
                        .frame(height: 1)
                        .frame(maxWidth: 20)
                }
            }
        }
        .accessibilityIdentifier(AccessibilityID.Clone.stepIndicator)
    }

    private func stepColor(for step: CloneWizardViewModel.WizardStep) -> Color {
        if step == viewModel.currentStep {
            return .accentColor
        } else if step.rawValue < viewModel.currentStep.rawValue {
            return .green
        } else {
            return .secondary.opacity(0.5)
        }
    }

    // MARK: - Step 1: Select Font

    @ViewBuilder
    var selectFontStep: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Select Reference Font")
                .font(.title2)
                .fontWeight(.semibold)

            Text("Choose a font file to clone. The style characteristics will be extracted and used to generate new glyphs that match the visual appearance.")
                .foregroundColor(.secondary)

            // Drop zone
            ZStack {
                RoundedRectangle(cornerRadius: 12)
                    .strokeBorder(style: StrokeStyle(lineWidth: 2, dash: [8]))
                    .foregroundColor(.secondary.opacity(0.5))

                VStack(spacing: 12) {
                    Image(systemName: "doc.badge.plus")
                        .font(.system(size: 48))
                        .foregroundColor(.secondary)

                    if let font = viewModel.referenceFont {
                        VStack(spacing: 4) {
                            Text(font.name)
                                .font(.headline)
                            Text("\(font.glyphs.count) glyphs")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    } else {
                        Text("Drop a font file here or click to browse")
                            .foregroundColor(.secondary)
                    }

                    Button("Select Font File...") {
                        selectReferenceFont()
                    }
                    .buttonStyle(.bordered)
                    .accessibilityIdentifier(AccessibilityID.Clone.selectFontButton)
                }
                .padding(40)
            }
            .frame(height: 200)
            .accessibilityIdentifier(AccessibilityID.Clone.uploadArea)
            .onDrop(of: [.fileURL], isTargeted: nil) { providers in
                handleDrop(providers: providers)
                return true
            }

            if let error = viewModel.errorMessage {
                Label(error, systemImage: "exclamationmark.triangle")
                    .foregroundColor(.red)
                    .font(.caption)
            }
        }
    }

    // MARK: - Step 2: Analyze Style

    @ViewBuilder
    var analyzeStyleStep: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Style Analysis")
                .font(.title2)
                .fontWeight(.semibold)

            if viewModel.isAnalyzing {
                VStack(spacing: 12) {
                    ProgressView()
                        .scaleEffect(1.5)
                    Text("Analyzing font style...")
                        .foregroundColor(.secondary)
                }
                .frame(maxWidth: .infinity)
                .padding(40)
                .accessibilityIdentifier(AccessibilityID.Clone.analyzingIndicator)
            } else if let style = viewModel.extractedStyle {
                styleAnalysisCard(style: style)
            }
        }
    }

    @ViewBuilder
    func styleAnalysisCard(style: StyleEncoder.FontStyle) -> some View {
        VStack(alignment: .leading, spacing: 16) {
            if let font = viewModel.referenceFont {
                HStack {
                    Image(systemName: "textformat")
                        .font(.title2)
                        .foregroundColor(.accentColor)
                    VStack(alignment: .leading) {
                        Text(font.family)
                            .font(.headline)
                        Text(font.style)
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    Spacer()
                    Text("\(font.glyphs.count) glyphs")
                        .font(.caption)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(Color.secondary.opacity(0.2))
                        .cornerRadius(4)
                }
            }

            Divider()

            Text("Extracted Style Properties")
                .font(.headline)

            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 12) {
                styleProperty("Stroke Weight", value: String(format: "%.0f%%", style.strokeWeight * 100))
                styleProperty("Stroke Contrast", value: String(format: "%.0f%%", style.strokeContrast * 100))
                styleProperty("x-Height Ratio", value: String(format: "%.0f%%", style.xHeightRatio * 100))
                styleProperty("Width Ratio", value: String(format: "%.0f%%", style.widthRatio * 100))
                styleProperty("Slant", value: String(format: "%.1f°", style.slant))
                styleProperty("Roundness", value: String(format: "%.0f%%", style.roundness * 100))
                styleProperty("Regularity", value: String(format: "%.0f%%", style.regularity * 100))
                styleProperty("Serif Style", value: style.serifStyle.rawValue)
            }
        }
        .padding()
        .background(Color(nsColor: .controlBackgroundColor))
        .cornerRadius(8)
        .accessibilityIdentifier(AccessibilityID.Clone.styleCard)
    }

    @ViewBuilder
    func styleProperty(_ label: String, value: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label)
                .font(.caption)
                .foregroundColor(.secondary)
            Text(value)
                .font(.subheadline)
                .fontWeight(.medium)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    // MARK: - Step 3: Select Characters

    @ViewBuilder
    var selectCharactersStep: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Select Characters to Generate")
                .font(.title2)
                .fontWeight(.semibold)

            Text("Choose which character sets to generate in the cloned style. These will be added to your current project.")
                .foregroundColor(.secondary)

            VStack(alignment: .leading, spacing: 8) {
                characterSetToggle("Uppercase Letters (A-Z)", isOn: $viewModel.generateUppercase, count: 26)
                characterSetToggle("Lowercase Letters (a-z)", isOn: $viewModel.generateLowercase, count: 26)
                characterSetToggle("Digits (0-9)", isOn: $viewModel.generateDigits, count: 10)
                characterSetToggle("Basic Punctuation", isOn: $viewModel.generatePunctuation, count: 14)
            }
            .padding()
            .background(Color(nsColor: .controlBackgroundColor))
            .cornerRadius(8)
            .accessibilityIdentifier(AccessibilityID.Clone.characterSetPicker)

            let totalCount = viewModel.selectedCharacterCount
            HStack {
                Text("Total characters to generate:")
                    .foregroundColor(.secondary)
                Text("\(totalCount)")
                    .fontWeight(.semibold)
            }

            if totalCount == 0 {
                Label("Select at least one character set", systemImage: "exclamationmark.triangle")
                    .foregroundColor(.orange)
                    .font(.caption)
            }
        }
    }

    @ViewBuilder
    func characterSetToggle(_ label: String, isOn: Binding<Bool>, count: Int) -> some View {
        Toggle(isOn: isOn) {
            HStack {
                Text(label)
                Spacer()
                Text("\(count)")
                    .foregroundColor(.secondary)
                    .font(.caption)
            }
        }
    }

    // MARK: - Step 4: Generate

    @ViewBuilder
    var generateStep: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Generating Glyphs")
                .font(.title2)
                .fontWeight(.semibold)

            if viewModel.isGenerating {
                VStack(spacing: 16) {
                    // Informational banner when using template fallback
                    if viewModel.isUsingTemplateFallback {
                        HStack(spacing: 8) {
                            Image(systemName: "info.circle")
                                .foregroundColor(.orange)
                            Text("AI models not loaded. Generating glyphs using geometric templates.")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        .padding(10)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(Color.orange.opacity(0.1))
                        .cornerRadius(6)
                    }

                    ProgressView(value: viewModel.generationProgress)
                        .progressViewStyle(.linear)

                    Text("Generating \(viewModel.currentGeneratingCharacter ?? "")...")
                        .foregroundColor(.secondary)

                    Text("\(viewModel.generatedGlyphs.count) / \(viewModel.selectedCharacterCount) complete")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .padding()
                .background(Color(nsColor: .controlBackgroundColor))
                .cornerRadius(8)
                .accessibilityIdentifier(AccessibilityID.Clone.generatingIndicator)
            } else if !viewModel.generatedGlyphs.isEmpty {
                VStack(alignment: .leading, spacing: 12) {
                    if viewModel.isUsingPlaceholders {
                        Label("Generation Complete", systemImage: "info.circle")
                            .foregroundColor(.orange)
                            .font(.headline)

                        Text("\(viewModel.generatedGlyphs.count) glyphs generated using geometric templates. AI models not yet loaded.")
                            .foregroundColor(.secondary)
                    } else {
                        Label("Generation Complete", systemImage: "checkmark.circle.fill")
                            .foregroundColor(.green)
                            .font(.headline)

                        Text("\(viewModel.generatedGlyphs.count) glyphs generated in the cloned style.")
                            .foregroundColor(.secondary)
                    }

                    // Grid of generated glyphs
                    ScrollView {
                        LazyVGrid(columns: [GridItem(.adaptive(minimum: 50))], spacing: 8) {
                            ForEach(viewModel.generatedGlyphs, id: \.character) { glyph in
                                generatedGlyphCell(glyph)
                            }
                        }
                    }
                    .frame(maxHeight: 200)
                }
                .padding()
                .background(Color(nsColor: .controlBackgroundColor))
                .cornerRadius(8)
            }
        }
    }

    @ViewBuilder
    func generatedGlyphCell(_ glyph: Glyph) -> some View {
        VStack(spacing: 2) {
            ZStack {
                RoundedRectangle(cornerRadius: 4)
                    .fill(Color(nsColor: .textBackgroundColor))

                if !glyph.outline.isEmpty {
                    GlyphPreview(outline: glyph.outline)
                        .padding(4)
                } else {
                    Text(String(glyph.character))
                        .font(.system(size: 20, design: .serif))
                        .foregroundColor(.secondary)
                }
            }
            .frame(width: 44, height: 44)

            Text(String(glyph.character))
                .font(.caption2)
                .foregroundColor(.secondary)
        }
    }

    // MARK: - Step 5: Complete

    @ViewBuilder
    var completeStep: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Clone Complete")
                .font(.title2)
                .fontWeight(.semibold)

            VStack(alignment: .leading, spacing: 12) {
                if viewModel.isUsingPlaceholders {
                    Label("Template Generation Complete", systemImage: "info.circle")
                        .foregroundColor(.orange)
                        .font(.headline)

                    Text("\(viewModel.generatedGlyphs.count) glyphs have been generated using geometric templates. These are placeholder shapes, not AI-cloned from \(viewModel.referenceFont?.family ?? "the reference font").")
                        .foregroundColor(.secondary)

                    HStack(spacing: 8) {
                        Image(systemName: "exclamationmark.triangle")
                            .foregroundColor(.orange)
                        Text("AI models are not yet loaded. Glyphs will not match the reference font style until models are available.")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    .padding(10)
                    .background(Color.orange.opacity(0.1))
                    .cornerRadius(6)
                } else {
                    Label("Success!", systemImage: "checkmark.circle.fill")
                        .foregroundColor(.green)
                        .font(.headline)

                    Text("\(viewModel.generatedGlyphs.count) glyphs have been generated in the style of \(viewModel.referenceFont?.family ?? "the reference font").")
                        .foregroundColor(.secondary)
                }

                Divider()

                Text("Click 'Apply to Project' to add these glyphs to your current font project.")
                    .foregroundColor(.secondary)
            }
            .padding()
            .background(Color(nsColor: .controlBackgroundColor))
            .cornerRadius(8)
        }
    }

    // MARK: - Preview Panel

    @ViewBuilder
    var previewPanel: some View {
        VStack(spacing: 0) {
            HStack {
                Text("Preview")
                    .font(.headline)
                Spacer()
            }
            .padding()
            .background(Color(nsColor: .controlBackgroundColor))

            Divider()

            if viewModel.referenceFont != nil || !viewModel.generatedGlyphs.isEmpty {
                ClonePreviewCanvas(
                    referenceFont: viewModel.referenceFont,
                    generatedGlyphs: viewModel.generatedGlyphs,
                    extractedStyle: viewModel.extractedStyle
                )
            } else {
                VStack(spacing: 12) {
                    Image(systemName: "doc.on.doc")
                        .font(.system(size: 48))
                        .foregroundColor(.secondary.opacity(0.5))
                    Text("Select a reference font to see preview")
                        .foregroundColor(.secondary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .accessibilityIdentifier(AccessibilityID.Clone.previewCanvas)
    }

    // MARK: - Navigation Buttons

    @ViewBuilder
    var navigationButtons: some View {
        HStack {
            if viewModel.currentStep != .selectFont {
                Button("Back") {
                    viewModel.previousStep()
                }
                .buttonStyle(.bordered)
                .accessibilityIdentifier(AccessibilityID.Clone.backButton)
            }

            Spacer()

            if viewModel.currentStep == .complete {
                Button("Apply to Project") {
                    applyToProject()
                }
                .buttonStyle(.borderedProminent)
                .disabled(viewModel.generatedGlyphs.isEmpty)
                .accessibilityIdentifier(AccessibilityID.Clone.applyButton)
            } else if viewModel.currentStep == .generate && !viewModel.isGenerating && viewModel.generatedGlyphs.isEmpty {
                Button("Generate") {
                    Task {
                        await viewModel.generateGlyphs()
                    }
                }
                .buttonStyle(.borderedProminent)
                .accessibilityIdentifier(AccessibilityID.Clone.generateButton)
            } else if viewModel.canProceed {
                Button("Next") {
                    if viewModel.currentStep == .selectFont {
                        Task {
                            await viewModel.analyzeStyle()
                        }
                    }
                    viewModel.nextStep()
                }
                .buttonStyle(.borderedProminent)
                .accessibilityIdentifier(AccessibilityID.Clone.nextButton)
            }
        }
    }

    // MARK: - Actions

    private func selectReferenceFont() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [
            UTType(filenameExtension: "ttf")!,
            UTType(filenameExtension: "otf")!
        ]
        panel.allowsMultipleSelection = false
        panel.message = "Select a font to clone"
        panel.prompt = "Select"

        guard panel.runModal() == .OK, let url = panel.url else { return }

        Task {
            await viewModel.loadReferenceFont(from: url)
        }
    }

    private func handleDrop(providers: [NSItemProvider]) {
        guard let provider = providers.first else { return }

        provider.loadItem(forTypeIdentifier: "public.file-url") { item, _ in
            guard let data = item as? Data,
                  let url = URL(dataRepresentation: data, relativeTo: nil) else {
                return
            }

            let ext = url.pathExtension.lowercased()
            guard ext == "ttf" || ext == "otf" else {
                Task { @MainActor in
                    viewModel.errorMessage = "Please drop a .ttf or .otf font file"
                }
                return
            }

            Task {
                await viewModel.loadReferenceFont(from: url)
            }
        }
    }

    private func applyToProject() {
        guard var project = appState.currentProject else {
            viewModel.errorMessage = "No font project is open. Create or open a project first."
            return
        }

        for glyph in viewModel.generatedGlyphs {
            project.setGlyph(glyph, for: glyph.character)
        }

        appState.currentProject = project
        appState.sidebarSelection = .glyphs

        // Reset wizard
        viewModel.reset()
    }
}

// MARK: - View Model

@MainActor
final class CloneWizardViewModel: ObservableObject {
    enum WizardStep: Int, CaseIterable {
        case selectFont
        case analyzeStyle
        case selectCharacters
        case generate
        case complete

        var title: String {
            switch self {
            case .selectFont: return "Select"
            case .analyzeStyle: return "Analyze"
            case .selectCharacters: return "Characters"
            case .generate: return "Generate"
            case .complete: return "Complete"
            }
        }
    }

    @Published var currentStep: WizardStep = .selectFont
    @Published var referenceFont: FontProject?
    @Published var extractedStyle: StyleEncoder.FontStyle?
    @Published var isAnalyzing = false
    @Published var isGenerating = false
    @Published var errorMessage: String?

    @Published var generateUppercase = true
    @Published var generateLowercase = true
    @Published var generateDigits = true
    @Published var generatePunctuation = false

    @Published var generatedGlyphs: [Glyph] = []
    @Published var generationProgress: Double = 0
    @Published var currentGeneratingCharacter: String?
    @Published var isUsingTemplateFallback = false

    private let fontParser = FontParser()

    var canProceed: Bool {
        switch currentStep {
        case .selectFont:
            return referenceFont != nil
        case .analyzeStyle:
            return extractedStyle != nil && !isAnalyzing
        case .selectCharacters:
            return selectedCharacterCount > 0
        case .generate:
            return !generatedGlyphs.isEmpty && !isGenerating
        case .complete:
            return true
        }
    }

    var selectedCharacterCount: Int {
        var count = 0
        if generateUppercase { count += 26 }
        if generateLowercase { count += 26 }
        if generateDigits { count += 10 }
        if generatePunctuation { count += 14 }
        return count
    }

    /// Whether the generated glyphs are template-based placeholders (not real AI)
    var isUsingPlaceholders: Bool {
        guard !generatedGlyphs.isEmpty else { return false }
        // Check if ANY glyph was generated by real AI or style transfer
        let hasRealAI = generatedGlyphs.contains { glyph in
            glyph.generatedBy == .aiGenerated || glyph.generatedBy == .styleTransfer
        }
        return !hasRealAI
    }

    var selectedCharacters: [Character] {
        var chars: [Character] = []
        if generateUppercase {
            chars.append(contentsOf: Array("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
        }
        if generateLowercase {
            chars.append(contentsOf: Array("abcdefghijklmnopqrstuvwxyz"))
        }
        if generateDigits {
            chars.append(contentsOf: Array("0123456789"))
        }
        if generatePunctuation {
            chars.append(contentsOf: Array(".,;:!?'\"()-–—…"))
        }
        return chars
    }

    func nextStep() {
        guard let nextIndex = WizardStep.allCases.firstIndex(where: { $0.rawValue == currentStep.rawValue + 1 }) else { return }
        currentStep = WizardStep.allCases[nextIndex]
    }

    func previousStep() {
        guard let prevIndex = WizardStep.allCases.firstIndex(where: { $0.rawValue == currentStep.rawValue - 1 }) else { return }
        currentStep = WizardStep.allCases[prevIndex]
    }

    func loadReferenceFont(from url: URL) async {
        errorMessage = nil

        do {
            referenceFont = try await fontParser.parse(url: url)
        } catch {
            errorMessage = "Failed to load font: \(error.localizedDescription)"
        }
    }

    func analyzeStyle() async {
        guard let font = referenceFont else { return }

        isAnalyzing = true
        errorMessage = nil

        // Run style extraction on a non-isolated context to avoid Sendable issues
        let result: Result<StyleEncoder.FontStyle, Error> = await Task.detached {
            let encoder = StyleEncoder()
            do {
                let style = try await encoder.extractStyle(from: font)
                return .success(style)
            } catch {
                return .failure(error)
            }
        }.value

        switch result {
        case .success(let style):
            extractedStyle = style
        case .failure(let error):
            errorMessage = "Failed to analyze style: \(error.localizedDescription)"
        }

        isAnalyzing = false
    }

    func generateGlyphs() async {
        guard let style = extractedStyle, let refFont = referenceFont else { return }

        isGenerating = true
        generatedGlyphs = []
        generationProgress = 0
        isUsingTemplateFallback = !GlyphGenerator.isModelAvailable()

        let characters = selectedCharacters
        let metrics = refFont.metrics

        // Generate using batch method on a non-isolated context to avoid Sendable issues
        let result: Result<[GlyphGenerator.GenerationResult], Error> = await Task.detached {
            let generator = GlyphGenerator()
            let mode = GlyphGenerator.GenerationMode.fromScratch(style: style)
            let settings = GlyphGenerator.GenerationSettings.default

            do {
                let results = try await generator.generateBatch(
                    characters: characters,
                    mode: mode,
                    metrics: metrics,
                    settings: settings
                ) { completed, total in
                    Task { @MainActor in
                        self.generationProgress = Double(completed) / Double(total)
                        if completed < characters.count {
                            self.currentGeneratingCharacter = String(characters[completed])
                        }
                    }
                }
                return .success(results)
            } catch {
                return .failure(error)
            }
        }.value

        switch result {
        case .success(let results):
            generatedGlyphs = results.map { $0.glyph }
        case .failure:
            // Create placeholders on failure
            generatedGlyphs = characters.map { character in
                Glyph(
                    character: character,
                    advanceWidth: metrics.unitsPerEm / 2,
                    leftSideBearing: metrics.unitsPerEm / 20,
                    generatedBy: .placeholder
                )
            }
        }

        currentGeneratingCharacter = nil
        isGenerating = false
        nextStep()  // Auto-advance to complete
    }

    func reset() {
        currentStep = .selectFont
        referenceFont = nil
        extractedStyle = nil
        generatedGlyphs = []
        generationProgress = 0
        errorMessage = nil
        isAnalyzing = false
        isGenerating = false
        isUsingTemplateFallback = false
    }
}

// MARK: - Preview Canvas

struct ClonePreviewCanvas: View {
    let referenceFont: FontProject?
    let generatedGlyphs: [Glyph]
    let extractedStyle: StyleEncoder.FontStyle?

    @State private var sampleText = "Hamburgevons"
    @State private var fontSize: CGFloat = 48

    var body: some View {
        VStack(spacing: 16) {
            TextField("Sample text", text: $sampleText)
                .textFieldStyle(.roundedBorder)
                .padding(.horizontal)

            ScrollView {
                VStack(alignment: .leading, spacing: 24) {
                    if let font = referenceFont {
                        VStack(alignment: .leading, spacing: 8) {
                            Text("Reference Font")
                                .font(.caption)
                                .foregroundColor(.secondary)

                            fontPreview(glyphs: font.glyphs, metrics: font.metrics)
                        }
                    }

                    if !generatedGlyphs.isEmpty, let refFont = referenceFont {
                        Divider()

                        VStack(alignment: .leading, spacing: 8) {
                            Text("Generated Glyphs")
                                .font(.caption)
                                .foregroundColor(.secondary)

                            let glyphDict = Dictionary(uniqueKeysWithValues: generatedGlyphs.map { ($0.character, $0) })
                            fontPreview(glyphs: glyphDict, metrics: refFont.metrics)
                        }
                    }
                }
                .padding()
            }

            HStack {
                Text("Size:")
                    .foregroundColor(.secondary)
                Slider(value: $fontSize, in: 24...96)
                    .frame(width: 120)
                Text("\(Int(fontSize))pt")
                    .frame(width: 50)
            }
            .padding()
            .background(Color(nsColor: .controlBackgroundColor))
        }
    }

    @ViewBuilder
    func fontPreview(glyphs: [Character: Glyph], metrics: FontMetrics) -> some View {
        Canvas { context, size in
            var xOffset: CGFloat = 20
            let baseline = size.height * 0.75
            let safeUnitsPerEm = max(CGFloat(metrics.unitsPerEm), 1)
            let scale = fontSize / safeUnitsPerEm

            for char in sampleText {
                if let glyph = glyphs[char] {
                    let path = glyph.outline.cgPath

                    var transform = CGAffineTransform.identity
                    transform = transform.translatedBy(x: xOffset, y: baseline)
                    transform = transform.scaledBy(x: scale, y: -scale)

                    if let transformedPath = path.copy(using: &transform) {
                        context.fill(Path(transformedPath), with: .color(.primary))
                    }

                    xOffset += CGFloat(glyph.advanceWidth) * scale
                } else if char == " " {
                    xOffset += fontSize * 0.3
                } else {
                    // Show placeholder for missing characters
                    let text = Text(String(char))
                        .font(.system(size: fontSize * 0.8))
                        .foregroundColor(.secondary)
                    context.draw(text, at: CGPoint(x: xOffset + fontSize * 0.3, y: baseline - fontSize * 0.3))
                    xOffset += fontSize * 0.6
                }
            }
        }
        .frame(height: fontSize * 1.5)
        .background(Color(nsColor: .textBackgroundColor))
        .cornerRadius(4)
    }
}

#Preview {
    CloneWizard()
        .environmentObject(AppState())
        .frame(width: 900, height: 600)
}
