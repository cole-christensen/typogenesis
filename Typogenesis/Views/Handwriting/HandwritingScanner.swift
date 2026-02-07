import SwiftUI
import UniformTypeIdentifiers
import PDFKit

struct HandwritingScanner: View {
    @EnvironmentObject var appState: AppState
    @State private var currentStep: ScannerStep = .upload
    @State private var uploadedImage: NSImage?
    @State private var detectedCharacters: [DetectedCharacter] = []
    @State private var selectedCharacterIndex: Int?
    @State private var isProcessing = false
    @State private var threshold: Double = 0.5
    @State private var simplification: Double = 2.0
    @State private var showSampleSheetInfo = false
    @State private var processingError: String?
    @State private var replaceExisting = false
    @State private var autoFitMetrics = true
    @State private var generateKerning = false
    @State private var showCameraAlert = false
    @State private var cameraAlertMessage = ""
    @State private var isGeneratingKerning = false

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
        var outline: GlyphOutline?  // Vectorized outline
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
                    .layoutPriority(0)

                rightPanel
                    .layoutPriority(1)
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

                // Error display (visible across all steps)
                if let error = processingError {
                    HStack {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundColor(.orange)
                        Text(error)
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    .padding(8)
                    .background(Color.orange.opacity(0.1))
                    .cornerRadius(4)
                }

                // Kerning generation progress indicator
                if isGeneratingKerning {
                    HStack(spacing: 8) {
                        ProgressView()
                            .scaleEffect(0.8)
                        Text("Generating kerning pairs...")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    .padding(8)
                    .background(Color.blue.opacity(0.1))
                    .cornerRadius(4)
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
                    takePhotoWithContinuityCamera()
                } label: {
                    HStack {
                        Image(systemName: "camera")
                        Text("Take Photo...")
                    }
                    .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .controlSize(.large)
                .alert("Camera Capture", isPresented: $showCameraAlert) {
                    Button("OK", role: .cancel) { }
                } message: {
                    Text(cameraAlertMessage)
                }
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
                                downloadSampleSheetPDF()
                            }
                            .buttonStyle(.bordered)
                            .controlSize(.small)

                            Button("Print") {
                                printSampleSheet()
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

                Toggle("Replace existing glyphs", isOn: $replaceExisting)
                Toggle("Auto-fit to metrics", isOn: $autoFitMetrics)
                Toggle("Generate kerning", isOn: $generateKerning)
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
            .disabled(detectedCharacters.filter { $0.assignedCharacter != nil }.isEmpty || isGeneratingKerning)
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

    private func takePhotoWithContinuityCamera() {
        // Try to use macOS Continuity Camera feature
        // This requires an iPhone with iOS 16+ and a Mac with macOS Ventura+
        // connected to the same iCloud account

        guard let window = NSApplication.shared.keyWindow,
              let contentView = window.contentView else {
            showContinuityCameraFallbackAlert()
            return
        }

        // Check if the system supports Continuity Camera (macOS 13+)
        if #available(macOS 13.0, *) {
            // Create a menu and populate it with Continuity Camera device options
            let menu = NSMenu(title: "Import from Device")

            // Use NSApp to populate the services submenu
            // This adds camera devices via the Continuity Camera API
            if let servicesMenu = NSApp.servicesMenu {
                // Look for existing Continuity Camera menu items
                var foundContinuityItems = false
                for item in servicesMenu.items {
                    if item.title.contains("iPhone") || item.title.contains("iPad") ||
                       item.title.contains("Import from") || item.title.contains("Take Photo") ||
                       item.title.contains("Scan Documents") {
                        let copiedItem = item.copy() as! NSMenuItem
                        menu.addItem(copiedItem)
                        foundContinuityItems = true
                    }
                }

                if foundContinuityItems {
                    // Show the menu at mouse location
                    let mouseLocation = NSEvent.mouseLocation
                    let windowLocation = window.convertPoint(fromScreen: mouseLocation)
                    menu.popUp(positioning: nil, at: windowLocation, in: contentView)
                    return
                }
            }

            // Fallback: show instructions if Continuity Camera items not found
            showContinuityCameraFallbackAlert()
        } else {
            showContinuityCameraFallbackAlert()
        }
    }

    private func showContinuityCameraFallbackAlert() {
        cameraAlertMessage = """
        To take a photo directly:

        1. Use Continuity Camera: Right-click in this app and look for "Import from iPhone or iPad" in the context menu.

        2. Or use your iPhone/iPad to take a photo and AirDrop it to your Mac, then use "Select Image..." to import it.

        3. You can also use the macOS Screenshot utility (Cmd+Shift+5) if you have a webcam connected.

        Tip: Make sure your iPhone is nearby and connected to the same iCloud account for Continuity Camera to work.
        """
        showCameraAlert = true
    }

    private func processImage() {
        guard let image = uploadedImage else { return }

        isProcessing = true
        processingError = nil

        Task {
            do {
                // Get metrics from current project or use defaults
                let metrics = appState.currentProject?.metrics ?? FontMetrics()

                // Create processing settings from UI values
                var settings = Vectorizer.VectorizationSettings.default
                settings.imageProcessing.threshold = threshold
                settings.tracing.simplificationTolerance = CGFloat(simplification)

                // Vectorize the image
                let result = try await Vectorizer.vectorize(
                    image: image,
                    metrics: metrics,
                    settings: settings
                )

                // Convert to detected characters
                detectedCharacters = result.characters.map { vectorized in
                    DetectedCharacter(
                        boundingBox: vectorized.bounds,
                        outline: vectorized.outline,
                        assignedCharacter: nil
                    )
                }

                isProcessing = false

                if detectedCharacters.isEmpty {
                    processingError = "No characters detected. Try adjusting the threshold."
                } else {
                    currentStep = .assign
                }
            } catch {
                isProcessing = false
                processingError = error.localizedDescription
            }
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
        guard var project = appState.currentProject else {
            processingError = "No font project is open. Create or open a project first."
            return
        }

        // Import each assigned character as a glyph
        for detected in detectedCharacters {
            guard let char = detected.assignedCharacter,
                  let outline = detected.outline else { continue }

            // Check if glyph already exists
            if let existingGlyph = project.glyphs[char] {
                if replaceExisting {
                    // Replace existing glyph
                    var glyph = existingGlyph
                    glyph.outline = outline
                    if autoFitMetrics {
                        let bounds = outline.boundingBox
                        glyph.advanceWidth = bounds.width + Int(CGFloat(project.metrics.unitsPerEm) * 0.2)
                    }
                    project.glyphs[char] = glyph
                }
                // If not replacing, skip
            } else {
                // Create new glyph
                let bounds = outline.boundingBox
                let advanceWidth = autoFitMetrics
                    ? bounds.width + Int(CGFloat(project.metrics.unitsPerEm) * 0.2)
                    : project.metrics.unitsPerEm / 2

                let glyph = Glyph(
                    character: char,
                    outline: outline,
                    advanceWidth: advanceWidth,
                    leftSideBearing: Int(CGFloat(project.metrics.unitsPerEm) * 0.1)
                )
                project.glyphs[char] = glyph
            }
        }

        // Update project with imported glyphs
        appState.currentProject = project

        // Generate kerning if enabled
        if generateKerning {
            isGeneratingKerning = true
            Task {
                await generateKerningPairs()
                await MainActor.run {
                    isGeneratingKerning = false
                }
            }
        }

        // Reset scanner
        uploadedImage = nil
        detectedCharacters = []
        currentStep = .upload
    }

    /// Generates kerning pairs for the imported glyphs using KerningPredictor
    private func generateKerningPairs() async {
        guard var project = appState.currentProject else { return }

        let predictor = KerningPredictor()
        let settings = KerningPredictor.PredictionSettings(
            minKerningValue: 2,
            targetOpticalSpacing: 0.5,
            includePunctuation: true,
            includeNumbers: true,
            onlyCriticalPairs: true  // Use critical pairs for faster generation
        )

        do {
            let result = try await predictor.predictKerning(for: project, settings: settings)

            // Merge new kerning pairs with existing ones
            for newPair in result.pairs {
                if let existingIndex = project.kerning.firstIndex(where: { $0.left == newPair.left && $0.right == newPair.right }) {
                    // Update existing pair
                    project.kerning[existingIndex] = newPair
                } else {
                    // Add new pair
                    project.kerning.append(newPair)
                }
            }

            // Sort kerning pairs by left character, then right
            project.kerning.sort { lhs, rhs in
                if lhs.left == rhs.left {
                    return lhs.right < rhs.right
                }
                return lhs.left < rhs.left
            }

            await MainActor.run {
                appState.currentProject = project
            }
        } catch {
            await MainActor.run {
                processingError = "Kerning generation failed: \(error.localizedDescription). Glyphs were imported successfully but without kerning data."
            }
        }
    }

    private func scaleRect(_ rect: CGRect, to viewSize: CGSize, imageSize: CGSize) -> CGRect {
        // Guard against division by zero
        let safeImageWidth = max(imageSize.width, 1)
        let safeImageHeight = max(imageSize.height, 1)
        let scale = min(viewSize.width / safeImageWidth, viewSize.height / safeImageHeight)
        let offsetX = (viewSize.width - safeImageWidth * scale) / 2
        let offsetY = (viewSize.height - safeImageHeight * scale) / 2

        return CGRect(
            x: rect.origin.x * scale + offsetX,
            y: rect.origin.y * scale + offsetY,
            width: rect.width * scale,
            height: rect.height * scale
        )
    }

    private func downloadSampleSheetPDF() {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.pdf]
        panel.nameFieldStringValue = "Typogenesis_Sample_Sheet.pdf"
        panel.title = "Save Sample Sheet"
        panel.message = "Choose where to save the sample sheet PDF"

        if panel.runModal() == .OK, let url = panel.url {
            let pdfData = generateSampleSheetPDFData()
            do {
                try pdfData.write(to: url)
            } catch {
                processingError = "Failed to save sample sheet PDF: \(error.localizedDescription)"
            }
        }
    }

    private func printSampleSheet() {
        let pdfData = generateSampleSheetPDFData()
        guard let pdfDocument = PDFDocument(data: pdfData) else {
            processingError = "Failed to create PDF document for printing. The sample sheet data may be corrupted."
            return
        }

        let printInfo = NSPrintInfo.shared.copy() as! NSPrintInfo
        printInfo.horizontalPagination = .fit
        printInfo.verticalPagination = .fit
        printInfo.isHorizontallyCentered = true
        printInfo.isVerticallyCentered = true

        let printOperation = pdfDocument.printOperation(for: printInfo, scalingMode: .pageScaleToFit, autoRotate: true)
        printOperation?.runModal(for: NSApp.mainWindow ?? NSWindow(), delegate: nil, didRun: nil, contextInfo: nil)
    }

    private func generateSampleSheetPDFData() -> Data {
        // US Letter size in points (72 points per inch)
        let pageWidth: CGFloat = 612
        let pageHeight: CGFloat = 792
        let pageRect = CGRect(x: 0, y: 0, width: pageWidth, height: pageHeight)

        // Layout constants
        let margin: CGFloat = 50
        let boxSize: CGFloat = 50
        let boxSpacing: CGFloat = 8
        let labelHeight: CGFloat = 14
        let sectionSpacing: CGFloat = 30

        // Characters to include
        let uppercase = Array("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        let lowercase = Array("abcdefghijklmnopqrstuvwxyz")
        let digits = Array("0123456789")

        // Calculate boxes per row
        let availableWidth = pageWidth - (2 * margin)
        let boxesPerRow = Int(availableWidth / (boxSize + boxSpacing))

        // Create PDF data
        let pdfData = NSMutableData()
        guard let consumer = CGDataConsumer(data: pdfData as CFMutableData),
              let context = CGContext(consumer: consumer, mediaBox: nil, nil) else {
            return Data()
        }

        // Begin page
        var mediaBox = pageRect
        context.beginPage(mediaBox: &mediaBox)

        // Flip coordinate system for easier text drawing
        context.translateBy(x: 0, y: pageHeight)
        context.scaleBy(x: 1, y: -1)

        // Create NSGraphicsContext for text drawing
        let nsContext = NSGraphicsContext(cgContext: context, flipped: true)

        // Draw title
        let titleFont = NSFont.boldSystemFont(ofSize: 24)
        let titleAttributes: [NSAttributedString.Key: Any] = [
            .font: titleFont,
            .foregroundColor: NSColor.black
        ]
        let title = "Typogenesis Sample Sheet"
        let titleString = NSAttributedString(string: title, attributes: titleAttributes)
        let titleSize = titleString.size()
        let titleX = (pageWidth - titleSize.width) / 2
        let titleY: CGFloat = 40

        NSGraphicsContext.saveGraphicsState()
        NSGraphicsContext.current = nsContext
        titleString.draw(at: NSPoint(x: titleX, y: titleY))
        NSGraphicsContext.restoreGraphicsState()

        // Draw subtitle/instructions
        let subtitleFont = NSFont.systemFont(ofSize: 11)
        let subtitleAttributes: [NSAttributedString.Key: Any] = [
            .font: subtitleFont,
            .foregroundColor: NSColor.darkGray
        ]
        let subtitle = "Write each character clearly within its box. Use a black pen for best results."
        let subtitleString = NSAttributedString(string: subtitle, attributes: subtitleAttributes)
        let subtitleSize = subtitleString.size()
        let subtitleX = (pageWidth - subtitleSize.width) / 2
        let subtitleY: CGFloat = 70

        NSGraphicsContext.saveGraphicsState()
        NSGraphicsContext.current = nsContext
        subtitleString.draw(at: NSPoint(x: subtitleX, y: subtitleY))
        NSGraphicsContext.restoreGraphicsState()

        var currentY: CGFloat = 100

        // Helper function to draw a section of character boxes
        func drawSection(sectionTitle: String, characters: [Character], startY: CGFloat) -> CGFloat {
            var y = startY

            // Section title
            let sectionFont = NSFont.boldSystemFont(ofSize: 14)
            let sectionAttributes: [NSAttributedString.Key: Any] = [
                .font: sectionFont,
                .foregroundColor: NSColor.black
            ]
            let sectionString = NSAttributedString(string: sectionTitle, attributes: sectionAttributes)

            NSGraphicsContext.saveGraphicsState()
            NSGraphicsContext.current = nsContext
            sectionString.draw(at: NSPoint(x: margin, y: y))
            NSGraphicsContext.restoreGraphicsState()

            y += 20

            // Draw character boxes
            let labelFont = NSFont.systemFont(ofSize: 10)
            let labelAttributes: [NSAttributedString.Key: Any] = [
                .font: labelFont,
                .foregroundColor: NSColor.darkGray
            ]

            for (index, char) in characters.enumerated() {
                let col = index % boxesPerRow
                let row = index / boxesPerRow

                let boxX = margin + CGFloat(col) * (boxSize + boxSpacing)
                let boxY = y + CGFloat(row) * (boxSize + labelHeight + boxSpacing)

                // Draw box
                context.setStrokeColor(NSColor.lightGray.cgColor)
                context.setLineWidth(1)
                context.stroke(CGRect(x: boxX, y: boxY, width: boxSize, height: boxSize))

                // Draw character label below box
                let labelString = NSAttributedString(string: String(char), attributes: labelAttributes)
                let labelSize = labelString.size()
                let labelX = boxX + (boxSize - labelSize.width) / 2
                let labelY = boxY + boxSize + 2

                NSGraphicsContext.saveGraphicsState()
                NSGraphicsContext.current = nsContext
                labelString.draw(at: NSPoint(x: labelX, y: labelY))
                NSGraphicsContext.restoreGraphicsState()
            }

            // Calculate total height used
            let rows = (characters.count + boxesPerRow - 1) / boxesPerRow
            return y + CGFloat(rows) * (boxSize + labelHeight + boxSpacing)
        }

        // Draw uppercase section
        currentY = drawSection(sectionTitle: "Uppercase (A-Z)", characters: uppercase, startY: currentY)
        currentY += sectionSpacing

        // Draw lowercase section
        currentY = drawSection(sectionTitle: "Lowercase (a-z)", characters: lowercase, startY: currentY)
        currentY += sectionSpacing

        // Draw digits section
        _ = drawSection(sectionTitle: "Digits (0-9)", characters: digits, startY: currentY)

        // Draw footer
        let footerFont = NSFont.systemFont(ofSize: 9)
        let footerAttributes: [NSAttributedString.Key: Any] = [
            .font: footerFont,
            .foregroundColor: NSColor.gray
        ]
        let footer = "Generated by Typogenesis"
        let footerString = NSAttributedString(string: footer, attributes: footerAttributes)
        let footerSize = footerString.size()
        let footerX = (pageWidth - footerSize.width) / 2
        let footerY = pageHeight - 30

        NSGraphicsContext.saveGraphicsState()
        NSGraphicsContext.current = nsContext
        footerString.draw(at: NSPoint(x: footerX, y: footerY))
        NSGraphicsContext.restoreGraphicsState()

        // End page and close PDF
        context.endPage()
        context.closePDF()

        return pdfData as Data
    }
}

#Preview {
    HandwritingScanner()
        .environmentObject(AppState())
        .frame(width: 900, height: 600)
}
