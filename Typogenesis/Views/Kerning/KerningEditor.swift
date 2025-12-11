import SwiftUI

struct KerningEditor: View {
    @EnvironmentObject var appState: AppState
    @State private var selectedPairIndex: Int?
    @State private var leftChar: String = ""
    @State private var rightChar: String = ""
    @State private var kerningValue: Int = 0
    @State private var previewText: String = "AVAST Wavy Type"
    @State private var showAddSheet = false

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
