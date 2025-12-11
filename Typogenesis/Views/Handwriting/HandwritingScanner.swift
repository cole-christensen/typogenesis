import SwiftUI
import UniformTypeIdentifiers

struct HandwritingScanner: View {
    @EnvironmentObject var appState: AppState
    @State private var currentStep: ScannerStep = .upload
    @State private var uploadedImage: NSImage?
    @State private var processedImage: NSImage?
    @State private var detectedCharacters: [DetectedCharacter] = []
    @State private var selectedCharacterIndex: Int?
    @State private var isProcessing = false
    @State private var threshold: Double = 0.5
    @State private var simplification: Double = 2.0
    @State private var showSampleSheetInfo = false

    enum ScannerStep: Int, CaseIterable {
        case upload = 0
        case process = 1
        case assign = 2
        case import_ = 3

        var title: String {
            switch self {
            case .upload: return "Upload"
            case .process: return "Process"
            case .assign: return "Assign"
            case .import_: return "Import"
            }
        }

        var icon: String {
            switch self {
            case .upload: return "photo"
            case .process: return "wand.and.rays"
            case .assign: return "character.textbox"
            case .import_: return "square.and.arrow.down"
            }
        }
    }

    struct DetectedCharacter: Identifiable {
        let id = UUID()
        var boundingBox: CGRect
        var assignedCharacter: Character?
        var isSelected = false
    }

    var body: some View {
        VStack(spacing: 0) {
            // Step Indicator
            stepIndicator
                .padding()
                .background(Color(nsColor: .controlBackgroundColor))

            Divider()

            // Main Content
            HSplitView {
                leftPanel
                    .frame(minWidth: 300, maxWidth: 400)

                rightPanel
            }
        }
    }

    @ViewBuilder
    var stepIndicator: some View {
        HStack(spacing: 0) {
            ForEach(ScannerStep.allCases, id: \.self) { step in
                HStack(spacing: 8) {
                    ZStack {
                        Circle()
                            .fill(step.rawValue <= currentStep.rawValue ? Color.accentColor : Color.gray.opacity(0.3))
                            .frame(width: 32, height: 32)

                        if step.rawValue < currentStep.rawValue {
                            Image(systemName: "checkmark")
                                .font(.system(size: 14, weight: .bold))
                                .foregroundColor(.white)
                        } else {
                            Image(systemName: step.icon)
                                .font(.system(size: 14))
                                .foregroundColor(step.rawValue <= currentStep.rawValue ? .white : .gray)
                        }
                    }

                    Text(step.title)
                        .font(.subheadline)
                        .fontWeight(step == currentStep ? .semibold : .regular)
                        .foregroundColor(step.rawValue <= currentStep.rawValue ? .primary : .secondary)
                }

                if step != ScannerStep.allCases.last {
                    Rectangle()
                        .fill(step.rawValue < currentStep.rawValue ? Color.accentColor : Color.gray.opacity(0.3))
                        .frame(height: 2)
                        .frame(maxWidth: 60)
                }
            }
        }
    }

    @ViewBuilder
    var leftPanel: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                // Header
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Image(systemName: "pencil.and.scribble")
                            .font(.title)
                            .foregroundColor(.accentColor)
                        Text("Handwriting Scanner")
                            .font(.title2)
                            .fontWeight(.semibold)
                    }
                    Text("Convert handwritten samples to vector glyphs")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }

                Divider()

                switch currentStep {
                case .upload:
                    uploadStepControls
                case .process:
                    processStepControls
                case .assign:
                    assignStepControls
                case .import_:
                    importStepControls
                }

                Spacer()

                // Navigation Buttons
                navigationButtons
            }
            .padding()
        }
        .background(Color(nsColor: .controlBackgroundColor))
    }

    @ViewBuilder
    var uploadStepControls: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Step 1: Upload Image")
                .font(.headline)

            Text("Upload a photo or scan of your handwritten characters. For best results, use the sample sheet template.")
                .font(.caption)
                .foregroundColor(.secondary)

            VStack(spacing: 12) {
                Button {
                    selectImage()
                } label: {
                    HStack {
                        Image(systemName: "photo.badge.plus")
                        Text("Select Image...")
                    }
                    .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)

                Button {
                    // Would show camera interface
                } label: {
                    HStack {
                        Image(systemName: "camera")
                        Text("Take Photo...")
                    }
                    .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .controlSize(.large)
            }

            Divider()

            // Sample Sheet Info
            VStack(alignment: .leading, spacing: 8) {
                Button {
                    showSampleSheetInfo.toggle()
                } label: {
                    HStack {
                        Image(systemName: "doc.text")
                        Text("Sample Sheet Template")
                        Spacer()
                        Image(systemName: showSampleSheetInfo ? "chevron.up" : "chevron.down")
                    }
                }
                .buttonStyle(.plain)

                if showSampleSheetInfo {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("For best results, print and fill out a sample sheet:")
                            .font(.caption)
                            .foregroundColor(.secondary)

                        sampleSheetPreview

                        HStack {
                            Button("Download PDF") {
                                // Would download sample sheet
                            }
                            .buttonStyle(.bordered)
                            .controlSize(.small)

                            Button("Print") {
                                // Would print sample sheet
                            }
                            .buttonStyle(.bordered)
                            .controlSize(.small)
                        }
                    }
                    .padding()
                    .background(Color(nsColor: .textBackgroundColor))
                    .cornerRadius(8)
                }
            }

            // Supported Formats
            VStack(alignment: .leading, spacing: 4) {
                Text("Supported Formats")
                    .font(.caption)
                    .fontWeight(.medium)
                Text("PNG, JPEG, TIFF, HEIC, PDF")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
    }

    @ViewBuilder
    var sampleSheetPreview: some View {
        VStack(spacing: 4) {
            HStack(spacing: 4) {
                ForEach(["A", "B", "C", "D", "E"], id: \.self) { letter in
                    RoundedRectangle(cornerRadius: 2)
                        .stroke(Color.gray.opacity(0.5), lineWidth: 1)
                        .frame(width: 30, height: 30)
                        .overlay {
                            Text(letter)
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                }
            }
            HStack(spacing: 4) {
                ForEach(["F", "G", "H", "I", "J"], id: \.self) { letter in
                    RoundedRectangle(cornerRadius: 2)
                        .stroke(Color.gray.opacity(0.5), lineWidth: 1)
                        .frame(width: 30, height: 30)
                        .overlay {
                            Text(letter)
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                }
            }
            Text("...")
                .font(.caption)
                .foregroundColor(.secondary)
        }
    }

    @ViewBuilder
    var processStepControls: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Step 2: Process Image")
                .font(.headline)

            Text("Adjust the processing settings to optimize character detection and vectorization.")
                .font(.caption)
                .foregroundColor(.secondary)

            // Processing Settings
            VStack(alignment: .leading, spacing: 12) {
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text("Threshold")
                        Spacer()
                        Text("\(Int(threshold * 100))%")
                            .foregroundColor(.secondary)
                    }
                    .font(.subheadline)

                    Slider(value: $threshold, in: 0...1)

                    Text("Adjusts the black/white threshold for character detection")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }

                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text("Simplification")
                        Spacer()
                        Text(String(format: "%.1f", simplification))
                            .foregroundColor(.secondary)
                    }
                    .font(.subheadline)

                    Slider(value: $simplification, in: 0.5...5)

                    Text("Higher values result in smoother curves with fewer points")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }

            Divider()

            // Process Button
            Button {
                processImage()
            } label: {
                HStack {
                    if isProcessing {
                        ProgressView()
                            .scaleEffect(0.8)
                    } else {
                        Image(systemName: "wand.and.rays")
                    }
                    Text(isProcessing ? "Processing..." : "Detect Characters")
                }
                .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
            .disabled(uploadedImage == nil || isProcessing)

            if !detectedCharacters.isEmpty {
                Text("\(detectedCharacters.count) characters detected")
                    .font(.caption)
                    .foregroundColor(.green)
            }
        }
    }

    @ViewBuilder
    var assignStepControls: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Step 3: Assign Characters")
                .font(.headline)

            Text("Click on each detected shape and assign the corresponding character.")
                .font(.caption)
                .foregroundColor(.secondary)

            if let index = selectedCharacterIndex, index < detectedCharacters.count {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Selected Character")
                        .font(.subheadline)
                        .fontWeight(.medium)

                    HStack {
                        TextField("Character", text: Binding(
                            get: {
                                if let char = detectedCharacters[index].assignedCharacter {
                                    return String(char)
                                }
                                return ""
                            },
                            set: { newValue in
                                if let char = newValue.first {
                                    detectedCharacters[index].assignedCharacter = char
                                }
                            }
                        ))
                        .textFieldStyle(.roundedBorder)
                        .frame(width: 60)

                        // Quick assign buttons
                        HStack(spacing: 4) {
                            ForEach(["A", "B", "C", "a", "b", "c"], id: \.self) { char in
                                Button(char) {
                                    detectedCharacters[index].assignedCharacter = Character(char)
                                }
                                .buttonStyle(.bordered)
                                .controlSize(.small)
                            }
                        }
                    }
                }
                .padding()
                .background(Color(nsColor: .textBackgroundColor))
                .cornerRadius(8)
            } else {
                Text("Click on a detected character in the preview to assign it")
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .padding()
                    .frame(maxWidth: .infinity)
                    .background(Color(nsColor: .textBackgroundColor))
                    .cornerRadius(8)
            }

            // Assignment Progress
            let assigned = detectedCharacters.filter { $0.assignedCharacter != nil }.count
            VStack(alignment: .leading, spacing: 4) {
                ProgressView(value: Double(assigned), total: Double(max(1, detectedCharacters.count)))

                Text("\(assigned) of \(detectedCharacters.count) characters assigned")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            // Auto-detect from template
            Button {
                autoAssignFromTemplate()
            } label: {
                HStack {
                    Image(systemName: "wand.and.stars")
                    Text("Auto-Assign from Template")
                }
            }
            .buttonStyle(.bordered)
            .disabled(detectedCharacters.isEmpty)
        }
    }

    @ViewBuilder
    var importStepControls: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Step 4: Import Glyphs")
                .font(.headline)

            Text("Review the detected characters and import them into your font project.")
                .font(.caption)
                .foregroundColor(.secondary)

            // Summary
            VStack(alignment: .leading, spacing: 8) {
                Text("Summary")
                    .font(.subheadline)
                    .fontWeight(.medium)

                let assigned = detectedCharacters.filter { $0.assignedCharacter != nil }

                HStack {
                    Text("Characters to import:")
                    Spacer()
                    Text("\(assigned.count)")
                        .fontWeight(.semibold)
                }
                .font(.subheadline)

                if !assigned.isEmpty {
                    Text(assigned.compactMap { $0.assignedCharacter }.map { String($0) }.joined())
                        .font(.system(.body, design: .monospaced))
                        .padding(8)
                        .background(Color(nsColor: .textBackgroundColor))
                        .cornerRadius(4)
                }
            }

            // Import Options
            VStack(alignment: .leading, spacing: 8) {
                Text("Import Options")
                    .font(.subheadline)
                    .fontWeight(.medium)

                Toggle("Replace existing glyphs", isOn: .constant(false))
                Toggle("Auto-fit to metrics", isOn: .constant(true))
                Toggle("Generate kerning", isOn: .constant(false))
            }

            Divider()

            Button {
                importGlyphs()
            } label: {
                HStack {
                    Image(systemName: "square.and.arrow.down")
                    Text("Import to Font")
                }
                .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
            .disabled(detectedCharacters.filter { $0.assignedCharacter != nil }.isEmpty)
        }
    }

    @ViewBuilder
    var navigationButtons: some View {
        HStack {
            if currentStep != .upload {
                Button("Back") {
                    withAnimation {
                        if let index = ScannerStep.allCases.firstIndex(of: currentStep), index > 0 {
                            currentStep = ScannerStep.allCases[index - 1]
                        }
                    }
                }
                .buttonStyle(.bordered)
            }

            Spacer()

            if currentStep != .import_ {
                Button("Next") {
                    withAnimation {
                        if let index = ScannerStep.allCases.firstIndex(of: currentStep), index < ScannerStep.allCases.count - 1 {
                            currentStep = ScannerStep.allCases[index + 1]
                        }
                    }
                }
                .buttonStyle(.borderedProminent)
                .disabled(!canProceed)
            }
        }
    }

    @ViewBuilder
    var rightPanel: some View {
        VStack(spacing: 0) {
            // Preview Header
            HStack {
                Text("Preview")
                    .font(.headline)
                Spacer()

                if uploadedImage != nil {
                    Button {
                        uploadedImage = nil
                        processedImage = nil
                        detectedCharacters = []
                        currentStep = .upload
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                            .foregroundColor(.secondary)
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding()
            .background(Color(nsColor: .controlBackgroundColor))

            // Preview Content
            if let image = uploadedImage {
                imagePreview(image)
            } else {
                dropZone
            }
        }
    }

    @ViewBuilder
    func imagePreview(_ image: NSImage) -> some View {
        GeometryReader { geometry in
            ZStack {
                Image(nsImage: image)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)

                // Overlay detected characters
                ForEach(Array(detectedCharacters.enumerated()), id: \.element.id) { index, char in
                    let rect = scaleRect(char.boundingBox, to: geometry.size, imageSize: image.size)

                    Rectangle()
                        .stroke(
                            char.assignedCharacter != nil ? Color.green :
                                (selectedCharacterIndex == index ? Color.orange : Color.blue),
                            lineWidth: selectedCharacterIndex == index ? 3 : 2
                        )
                        .background(
                            Rectangle()
                                .fill((selectedCharacterIndex == index ? Color.orange : Color.blue).opacity(0.1))
                        )
                        .frame(width: rect.width, height: rect.height)
                        .position(x: rect.midX, y: rect.midY)
                        .onTapGesture {
                            selectedCharacterIndex = index
                        }

                    if let assigned = char.assignedCharacter {
                        Text(String(assigned))
                            .font(.caption)
                            .fontWeight(.bold)
                            .foregroundColor(.white)
                            .padding(2)
                            .background(Color.green)
                            .cornerRadius(2)
                            .position(x: rect.minX + 10, y: rect.minY + 10)
                    }
                }
            }
        }
        .background(Color(nsColor: .textBackgroundColor))
    }

    @ViewBuilder
    var dropZone: some View {
        VStack(spacing: 16) {
            Image(systemName: "arrow.down.doc")
                .font(.system(size: 48))
                .foregroundColor(.secondary)

            Text("Drop image here")
                .font(.title3)
                .foregroundColor(.secondary)

            Text("or click to select")
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color(nsColor: .textBackgroundColor))
        .onTapGesture {
            selectImage()
        }
        .onDrop(of: [.image], isTargeted: nil) { providers in
            if let provider = providers.first {
                _ = provider.loadDataRepresentation(for: .image) { data, error in
                    if let data = data, let image = NSImage(data: data) {
                        DispatchQueue.main.async {
                            uploadedImage = image
                            currentStep = .process
                        }
                    }
                }
                return true
            }
            return false
        }
    }

    private var canProceed: Bool {
        switch currentStep {
        case .upload:
            return uploadedImage != nil
        case .process:
            return !detectedCharacters.isEmpty
        case .assign:
            return detectedCharacters.contains { $0.assignedCharacter != nil }
        case .import_:
            return true
        }
    }

    private func selectImage() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.image, .png, .jpeg, .tiff, .heic, .pdf]
        panel.canChooseFiles = true
        panel.canChooseDirectories = false

        if panel.runModal() == .OK, let url = panel.url {
            uploadedImage = NSImage(contentsOf: url)
            if uploadedImage != nil {
                currentStep = .process
            }
        }
    }

    private func processImage() {
        guard uploadedImage != nil else { return }

        isProcessing = true

        // Simulate processing
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
            // Generate mock detected characters
            detectedCharacters = (0..<26).map { i in
                let row = i / 5
                let col = i % 5
                return DetectedCharacter(
                    boundingBox: CGRect(
                        x: 50 + col * 100,
                        y: 50 + row * 100,
                        width: 80,
                        height: 80
                    )
                )
            }

            isProcessing = false
            currentStep = .assign
        }
    }

    private func autoAssignFromTemplate() {
        let letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        for (index, char) in letters.enumerated() {
            if index < detectedCharacters.count {
                detectedCharacters[index].assignedCharacter = char
            }
        }
    }

    private func importGlyphs() {
        // Would import glyphs to font project
        // For now just close/reset
        uploadedImage = nil
        detectedCharacters = []
        currentStep = .upload
    }

    private func scaleRect(_ rect: CGRect, to viewSize: CGSize, imageSize: CGSize) -> CGRect {
        let scale = min(viewSize.width / imageSize.width, viewSize.height / imageSize.height)
        let offsetX = (viewSize.width - imageSize.width * scale) / 2
        let offsetY = (viewSize.height - imageSize.height * scale) / 2

        return CGRect(
            x: rect.origin.x * scale + offsetX,
            y: rect.origin.y * scale + offsetY,
            width: rect.width * scale,
            height: rect.height * scale
        )
    }
}

#Preview {
    HandwritingScanner()
        .environmentObject(AppState())
        .frame(width: 900, height: 600)
}
