import SwiftUI

struct KerningEditor: View {
    @EnvironmentObject var appState: AppState
    @State private var selectedPairIndex: Int?
    @State private var leftChar: String = ""
    @State private var rightChar: String = ""
    @State private var kerningValue: Int = 0
    @State private var previewText: String = "AVAST Wavy Type"
    @State private var showAddSheet = false
    @State private var showAutoKernSheet = false
    @State private var isAutoKerning = false
    @State private var autoKernProgress: Double = 0
    @State private var autoKernError: String?
    @State private var showAutoKernError = false

    var body: some View {
        HSplitView {
            kerningList
                .frame(minWidth: 250, maxWidth: 350)

            kerningPreview
        }
    }

    @ViewBuilder
    var kerningList: some View {
        VStack(spacing: 0) {
            // Header
            HStack {
                Text("Kerning Pairs")
                    .font(.headline)

                Spacer()

                Button(action: { showAutoKernSheet = true }) {
                    Image(systemName: "wand.and.stars")
                }
                .buttonStyle(.borderless)
                .help("Auto-generate kerning pairs")
                .disabled(appState.currentProject?.glyphs.isEmpty ?? true)

                Button(action: { showAddSheet = true }) {
                    Image(systemName: "plus")
                }
                .buttonStyle(.borderless)
                .help("Add kerning pair")
            }
            .padding()
            .background(Color(nsColor: .controlBackgroundColor))

            // Pair list
            if let project = appState.currentProject {
                if project.kerning.isEmpty {
                    emptyState
                } else {
                    List(selection: $selectedPairIndex) {
                        ForEach(Array(project.kerning.enumerated()), id: \.offset) { index, pair in
                            KerningPairRow(pair: pair)
                                .tag(index)
                                .contextMenu {
                                    Button("Delete") {
                                        deletePair(at: index)
                                    }
                                }
                        }
                        .onDelete { indexSet in
                            for index in indexSet {
                                deletePair(at: index)
                            }
                        }
                    }
                    .listStyle(.inset)
                }
            }

            Divider()

            // Quick add section
            quickAddSection
        }
        .sheet(isPresented: $showAddSheet) {
            AddKerningPairSheet { left, right, value in
                addPair(left: left, right: right, value: value)
            }
        }
        .sheet(isPresented: $showAutoKernSheet) {
            AutoKerningSheet(
                isGenerating: $isAutoKerning,
                progress: $autoKernProgress,
                onGenerate: generateAutoKerning,
                onApply: applyAutoKerning
            )
            .environmentObject(appState)
        }
        .alert("Auto-Kerning Error", isPresented: $showAutoKernError) {
            Button("OK") {}
        } message: {
            Text(autoKernError ?? "Unknown error")
        }
    }

    @ViewBuilder
    var emptyState: some View {
        VStack(spacing: 12) {
            Image(systemName: "character.textbox")
                .font(.system(size: 40))
                .foregroundColor(.secondary)

            Text("No Kerning Pairs")
                .font(.headline)
                .foregroundColor(.secondary)

            Text("Add kerning pairs to adjust spacing between specific character combinations.")
                .font(.caption)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal)

            Button("Add Pair") {
                showAddSheet = true
            }
            .buttonStyle(.bordered)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    @ViewBuilder
    var quickAddSection: some View {
        VStack(spacing: 8) {
            Text("Quick Add")
                .font(.caption)
                .foregroundColor(.secondary)
                .frame(maxWidth: .infinity, alignment: .leading)

            HStack(spacing: 8) {
                TextField("L", text: $leftChar)
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 40)
                    .multilineTextAlignment(.center)

                TextField("R", text: $rightChar)
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 40)
                    .multilineTextAlignment(.center)

                TextField("Val", value: $kerningValue, format: .number)
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 60)

                Stepper("", value: $kerningValue, step: 10)
                    .labelsHidden()

                Button(action: addQuickPair) {
                    Image(systemName: "plus.circle.fill")
                }
                .buttonStyle(.borderless)
                .disabled(leftChar.isEmpty || rightChar.isEmpty)
            }
        }
        .padding()
        .background(Color(nsColor: .controlBackgroundColor))
    }

    @ViewBuilder
    var kerningPreview: some View {
        VStack(spacing: 0) {
            // Preview header
            HStack {
                Text("Preview")
                    .font(.headline)

                Spacer()

                TextField("Sample text", text: $previewText)
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 200)
            }
            .padding()
            .background(Color(nsColor: .controlBackgroundColor))

            // Preview canvas
            KerningPreviewCanvas(
                text: previewText,
                project: appState.currentProject,
                highlightedPair: selectedPair
            )

            // Selected pair editor
            if let index = selectedPairIndex,
               let project = appState.currentProject,
               index < project.kerning.count {
                selectedPairEditor(project.kerning[index], index: index)
            }
        }
    }

    @ViewBuilder
    func selectedPairEditor(_ pair: KerningPair, index: Int) -> some View {
        HStack(spacing: 16) {
            Text("Selected: \(String(pair.left))\(String(pair.right))")
                .font(.headline)

            Spacer()

            Text("Value:")
            TextField("", value: Binding(
                get: { pair.value },
                set: { updatePairValue(at: index, value: $0) }
            ), format: .number)
            .textFieldStyle(.roundedBorder)
            .frame(width: 80)

            Stepper("", value: Binding(
                get: { pair.value },
                set: { updatePairValue(at: index, value: $0) }
            ), step: 10)
            .labelsHidden()

            Button("Delete") {
                deletePair(at: index)
            }
            .buttonStyle(.bordered)
            .foregroundColor(.red)
        }
        .padding()
        .background(Color(nsColor: .controlBackgroundColor))
    }

    private var selectedPair: KerningPair? {
        guard let index = selectedPairIndex,
              let project = appState.currentProject,
              index < project.kerning.count else { return nil }
        return project.kerning[index]
    }

    private func addQuickPair() {
        guard let left = leftChar.first,
              let right = rightChar.first else { return }
        addPair(left: left, right: right, value: kerningValue)
        leftChar = ""
        rightChar = ""
        kerningValue = 0
    }

    private func addPair(left: Character, right: Character, value: Int) {
        guard var project = appState.currentProject else { return }

        // Check if pair already exists
        if let existingIndex = project.kerning.firstIndex(where: { $0.left == left && $0.right == right }) {
            // Update existing pair
            project.kerning[existingIndex] = KerningPair(left: left, right: right, value: value)
        } else {
            // Add new pair
            project.kerning.append(KerningPair(left: left, right: right, value: value))
        }

        appState.currentProject = project
    }

    private func updatePairValue(at index: Int, value: Int) {
        guard var project = appState.currentProject,
              index < project.kerning.count else { return }
        let pair = project.kerning[index]
        project.kerning[index] = KerningPair(left: pair.left, right: pair.right, value: value)
        appState.currentProject = project
    }

    private func deletePair(at index: Int) {
        guard var project = appState.currentProject,
              index < project.kerning.count else { return }
        project.kerning.remove(at: index)
        appState.currentProject = project
        if selectedPairIndex == index {
            selectedPairIndex = nil
        } else if let selected = selectedPairIndex, selected > index {
            selectedPairIndex = selected - 1
        }
    }

    private func generateAutoKerning(
        settings: KerningPredictor.PredictionSettings
    ) async -> KerningPredictor.PredictionResult? {
        guard let project = appState.currentProject else { return nil }

        let predictor = KerningPredictor()
        do {
            let result = try await predictor.predictKerning(for: project, settings: settings)
            return result
        } catch {
            await MainActor.run {
                autoKernError = error.localizedDescription
                showAutoKernError = true
            }
            return nil
        }
    }

    private func applyAutoKerning(pairs: [KerningPair], replaceExisting: Bool) {
        guard var project = appState.currentProject else { return }

        if replaceExisting {
            project.kerning = pairs
        } else {
            // Merge: update existing pairs, add new ones
            for newPair in pairs {
                if let existingIndex = project.kerning.firstIndex(where: { $0.left == newPair.left && $0.right == newPair.right }) {
                    project.kerning[existingIndex] = newPair
                } else {
                    project.kerning.append(newPair)
                }
            }
        }

        // Sort by left character, then right
        project.kerning.sort { lhs, rhs in
            if lhs.left == rhs.left {
                return lhs.right < rhs.right
            }
            return lhs.left < rhs.left
        }

        appState.currentProject = project
    }
}

struct KerningPairRow: View {
    let pair: KerningPair

    var body: some View {
        HStack {
            Text("\(String(pair.left))\(String(pair.right))")
                .font(.system(.body, design: .monospaced))
                .frame(width: 50)

            Spacer()

            Text("\(pair.value)")
                .foregroundColor(pair.value < 0 ? .red : (pair.value > 0 ? .green : .secondary))
                .font(.system(.body, design: .monospaced))
        }
        .padding(.vertical, 2)
    }
}

struct KerningPreviewCanvas: View {
    let text: String
    let project: FontProject?
    let highlightedPair: KerningPair?

    var body: some View {
        GeometryReader { geometry in
            Canvas { context, size in
                guard let project = project else { return }

                let fontSize: CGFloat = 72
                let scale = fontSize / CGFloat(project.metrics.unitsPerEm)
                let baseline = size.height / 2 + fontSize / 3

                var xPosition: CGFloat = 20

                let chars = Array(text)
                for (index, char) in chars.enumerated() {
                    // Get glyph or use placeholder width
                    let glyph = project.glyph(for: char)
                    let advanceWidth = CGFloat(glyph?.advanceWidth ?? project.metrics.unitsPerEm / 2) * scale

                    // Draw character using system font as proxy
                    let charText = Text(String(char))
                        .font(.system(size: fontSize))
                    context.draw(charText, at: CGPoint(x: xPosition + advanceWidth / 2, y: baseline), anchor: .center)

                    // Apply kerning if there's a next character
                    if index < chars.count - 1 {
                        let nextChar = chars[index + 1]
                        if let kernPair = project.kerning.first(where: { $0.left == char && $0.right == nextChar }) {
                            let kernValue = CGFloat(kernPair.value) * scale

                            // Highlight if this is the selected pair
                            if let highlighted = highlightedPair,
                               highlighted.left == char && highlighted.right == nextChar {
                                let highlightRect = CGRect(
                                    x: xPosition + advanceWidth - 2,
                                    y: baseline - fontSize,
                                    width: max(4, abs(kernValue)),
                                    height: fontSize * 1.2
                                )
                                context.fill(Path(highlightRect), with: .color(.yellow.opacity(0.3)))
                            }

                            xPosition += kernValue
                        }
                    }

                    xPosition += advanceWidth
                }
            }
        }
        .background(Color(nsColor: .textBackgroundColor))
    }
}

struct AddKerningPairSheet: View {
    @Environment(\.dismiss) var dismiss
    @State private var leftChar: String = ""
    @State private var rightChar: String = ""
    @State private var value: Int = -50

    let onAdd: (Character, Character, Int) -> Void

    var body: some View {
        VStack(spacing: 20) {
            Text("Add Kerning Pair")
                .font(.title2)
                .fontWeight(.semibold)

            HStack(spacing: 20) {
                VStack {
                    Text("Left")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    TextField("A", text: $leftChar)
                        .textFieldStyle(.roundedBorder)
                        .frame(width: 60)
                        .font(.system(size: 24))
                        .multilineTextAlignment(.center)
                }

                VStack {
                    Text("Right")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    TextField("V", text: $rightChar)
                        .textFieldStyle(.roundedBorder)
                        .frame(width: 60)
                        .font(.system(size: 24))
                        .multilineTextAlignment(.center)
                }
            }

            VStack {
                Text("Kerning Value")
                    .font(.caption)
                    .foregroundColor(.secondary)
                HStack {
                    TextField("", value: $value, format: .number)
                        .textFieldStyle(.roundedBorder)
                        .frame(width: 100)
                    Stepper("", value: $value, step: 10)
                        .labelsHidden()
                }
                Text("Negative values move characters closer")
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }

            // Common pairs suggestions
            VStack(alignment: .leading, spacing: 8) {
                Text("Common Pairs")
                    .font(.caption)
                    .foregroundColor(.secondary)

                HStack(spacing: 8) {
                    ForEach(commonPairs, id: \.self) { pair in
                        Button(pair) {
                            leftChar = String(pair.first!)
                            rightChar = String(pair.last!)
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                    }
                }
            }

            Divider()

            HStack {
                Button("Cancel") {
                    dismiss()
                }
                .keyboardShortcut(.cancelAction)

                Spacer()

                Button("Add") {
                    if let left = leftChar.first, let right = rightChar.first {
                        onAdd(left, right, value)
                        dismiss()
                    }
                }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.defaultAction)
                .disabled(leftChar.isEmpty || rightChar.isEmpty)
            }
        }
        .padding(24)
        .frame(width: 350)
    }

    private var commonPairs: [String] {
        ["AV", "AW", "AT", "AY", "LT", "LV", "LY", "Ta", "Te", "To", "Tr", "Ty", "VA", "Vo", "WA", "Ya", "Yo"]
    }
}

// MARK: - Auto Kerning Sheet

struct AutoKerningSheet: View {
    @EnvironmentObject var appState: AppState
    @Environment(\.dismiss) var dismiss

    @Binding var isGenerating: Bool
    @Binding var progress: Double

    let onGenerate: (KerningPredictor.PredictionSettings) async -> KerningPredictor.PredictionResult?
    let onApply: ([KerningPair], Bool) -> Void

    @State private var spacingPreset: SpacingPreset = .default
    @State private var onlyCriticalPairs = false
    @State private var includePunctuation = true
    @State private var includeNumbers = true
    @State private var minKerningValue = 2
    @State private var replaceExisting = false

    @State private var generatedResult: KerningPredictor.PredictionResult?
    @State private var showingPreview = false

    enum SpacingPreset: String, CaseIterable {
        case tight = "Tight"
        case `default` = "Default"
        case loose = "Loose"

        var targetSpacing: Float {
            switch self {
            case .tight: return 0.3
            case .default: return 0.5
            case .loose: return 0.7
            }
        }
    }

    var body: some View {
        VStack(spacing: 20) {
            // Header
            HStack {
                Image(systemName: "wand.and.stars")
                    .font(.title2)
                    .foregroundColor(.accentColor)
                Text("Auto-Generate Kerning")
                    .font(.title2)
                    .fontWeight(.semibold)
            }

            if showingPreview, let result = generatedResult {
                previewSection(result: result)
            } else {
                settingsSection
            }
        }
        .padding(24)
        .frame(width: 450)
    }

    @ViewBuilder
    var settingsSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            // Font info
            if let project = appState.currentProject {
                HStack {
                    Text("Font:")
                    Text(project.name)
                        .fontWeight(.medium)
                    Spacer()
                    Text("\(project.glyphs.count) glyphs")
                        .foregroundColor(.secondary)
                }
                .padding()
                .background(Color(nsColor: .controlBackgroundColor))
                .cornerRadius(8)
            }

            // Spacing preset
            VStack(alignment: .leading, spacing: 8) {
                Text("Spacing")
                    .font(.headline)

                Picker("Spacing", selection: $spacingPreset) {
                    ForEach(SpacingPreset.allCases, id: \.self) { preset in
                        Text(preset.rawValue).tag(preset)
                    }
                }
                .pickerStyle(.segmented)

                Text(spacingDescription)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            // Options
            VStack(alignment: .leading, spacing: 8) {
                Text("Options")
                    .font(.headline)

                Toggle("Only critical pairs (faster)", isOn: $onlyCriticalPairs)

                Toggle("Include punctuation", isOn: $includePunctuation)
                    .disabled(onlyCriticalPairs)

                Toggle("Include numbers", isOn: $includeNumbers)
                    .disabled(onlyCriticalPairs)

                HStack {
                    Text("Minimum kerning value:")
                    TextField("", value: $minKerningValue, format: .number)
                        .textFieldStyle(.roundedBorder)
                        .frame(width: 60)
                    Stepper("", value: $minKerningValue, in: 1...50)
                        .labelsHidden()
                }
            }

            // Existing pairs handling
            if let project = appState.currentProject, !project.kerning.isEmpty {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Existing Pairs")
                        .font(.headline)

                    Picker("", selection: $replaceExisting) {
                        Text("Merge with existing (\(project.kerning.count) pairs)").tag(false)
                        Text("Replace all existing").tag(true)
                    }
                    .pickerStyle(.radioGroup)
                }
            }
        }

        Divider()

        // Actions
        HStack {
            Button("Cancel") {
                dismiss()
            }
            .keyboardShortcut(.cancelAction)

            Spacer()

            if isGenerating {
                ProgressView()
                    .scaleEffect(0.7)
                    .frame(width: 20, height: 20)
                Text("Analyzing...")
                    .foregroundColor(.secondary)
            } else {
                Button("Generate") {
                    Task {
                        await generateKerning()
                    }
                }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.defaultAction)
                .disabled(appState.currentProject?.glyphs.count ?? 0 < 2)
            }
        }
    }

    @ViewBuilder
    func previewSection(result: KerningPredictor.PredictionResult) -> some View {
        VStack(alignment: .leading, spacing: 16) {
            // Summary
            HStack {
                VStack(alignment: .leading) {
                    Text("\(result.pairs.count) kerning pairs generated")
                        .font(.headline)
                    Text("Confidence: \(Int(result.confidence * 100))%")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text("Time: \(String(format: "%.2f", result.predictionTime))s")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }

                Spacer()

                VStack(alignment: .trailing) {
                    let negative = result.pairs.filter { $0.value < 0 }.count
                    let positive = result.pairs.filter { $0.value > 0 }.count
                    Text("\(negative) negative, \(positive) positive")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
            .padding()
            .background(Color(nsColor: .controlBackgroundColor))
            .cornerRadius(8)

            // Preview list
            Text("Preview")
                .font(.headline)

            ScrollView {
                LazyVStack(spacing: 4) {
                    ForEach(Array(result.pairs.prefix(50).enumerated()), id: \.offset) { _, pair in
                        HStack {
                            Text("\(String(pair.left))\(String(pair.right))")
                                .font(.system(.body, design: .monospaced))
                                .frame(width: 50, alignment: .leading)

                            Spacer()

                            Text("\(pair.value)")
                                .foregroundColor(pair.value < 0 ? .red : (pair.value > 0 ? .green : .secondary))
                                .font(.system(.body, design: .monospaced))
                        }
                        .padding(.horizontal)
                        .padding(.vertical, 2)
                    }

                    if result.pairs.count > 50 {
                        Text("... and \(result.pairs.count - 50) more pairs")
                            .foregroundColor(.secondary)
                            .padding()
                    }
                }
            }
            .frame(height: 200)
            .background(Color(nsColor: .textBackgroundColor))
            .cornerRadius(8)
        }

        Divider()

        HStack {
            Button("Back") {
                showingPreview = false
                generatedResult = nil
            }

            Spacer()

            Button("Cancel") {
                dismiss()
            }

            Button("Apply") {
                onApply(result.pairs, replaceExisting)
                dismiss()
            }
            .buttonStyle(.borderedProminent)
            .keyboardShortcut(.defaultAction)
        }
    }

    private var spacingDescription: String {
        switch spacingPreset {
        case .tight: return "Tighter letter spacing for display text"
        case .default: return "Balanced spacing for body text"
        case .loose: return "More open spacing for readability"
        }
    }

    private func generateKerning() async {
        isGenerating = true

        let settings = KerningPredictor.PredictionSettings(
            minKerningValue: minKerningValue,
            targetOpticalSpacing: spacingPreset.targetSpacing,
            includePunctuation: includePunctuation,
            includeNumbers: includeNumbers,
            onlyCriticalPairs: onlyCriticalPairs
        )

        if let result = await onGenerate(settings) {
            await MainActor.run {
                generatedResult = result
                showingPreview = true
            }
        }

        await MainActor.run {
            isGenerating = false
        }
    }
}

#Preview {
    KerningEditor()
        .environmentObject({
            let state = AppState()
            state.createNewProject()
            // Add some test kerning pairs
            if var project = state.currentProject {
                project.kerning = [
                    KerningPair(left: "A", right: "V", value: -80),
                    KerningPair(left: "A", right: "W", value: -60),
                    KerningPair(left: "T", right: "o", value: -40),
                    KerningPair(left: "V", right: "a", value: -50)
                ]
                state.currentProject = project
            }
            return state
        }())
        .frame(width: 900, height: 600)
}
