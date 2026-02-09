import SwiftUI
import UniformTypeIdentifiers
import PDFKit

@MainActor
final class HandwritingScannerViewModel: ObservableObject {

    // MARK: - Nested Types

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
        var outline: GlyphOutline?
        var assignedCharacter: Character?
        var isSelected = false
    }

    // MARK: - Published Properties

    @Published var currentStep: ScannerStep = .upload
    @Published var uploadedImage: NSImage?
    @Published var detectedCharacters: [DetectedCharacter] = []
    @Published var selectedCharacterIndex: Int?
    @Published var isProcessing = false
    @Published var threshold: Double = 0.5
    @Published var simplification: Double = 2.0
    @Published var showSampleSheetInfo = false
    @Published var processingError: String?
    @Published var replaceExisting = false
    @Published var autoFitMetrics = true
    @Published var generateKerning = false
    @Published var isGeneratingKerning = false

    // MARK: - Dependencies

    private let fileDialogService: FileDialogService

    // MARK: - Init

    init(fileDialogService: FileDialogService = NSPanelFileDialogService()) {
        self.fileDialogService = fileDialogService
    }

    // MARK: - Computed Properties

    var canProceed: Bool {
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

    func scaleRect(_ rect: CGRect, to viewSize: CGSize, imageSize: CGSize) -> CGRect {
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

    // MARK: - Actions

    func selectImage() async {
        let url = await fileDialogService.selectFile(
            types: [.image, .png, .jpeg, .tiff, .heic, .pdf],
            message: nil
        )
        guard let url else { return }
        let image = NSImage(contentsOf: url)
        uploadedImage = image
        if image != nil {
            currentStep = .process
        }
    }

    func processImage(metrics: FontMetrics) async {
        guard let image = uploadedImage else { return }

        isProcessing = true
        processingError = nil

        do {
            var settings = Vectorizer.VectorizationSettings.default
            settings.imageProcessing.threshold = threshold
            settings.tracing.simplificationTolerance = CGFloat(simplification)

            let result = try await Vectorizer.vectorize(
                image: image,
                metrics: metrics,
                settings: settings
            )

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

    func autoAssignFromTemplate() {
        let letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        for (index, char) in letters.enumerated() {
            if index < detectedCharacters.count {
                detectedCharacters[index].assignedCharacter = char
            }
        }
    }

    func importGlyphs(into project: inout FontProject) {
        for detected in detectedCharacters {
            guard let char = detected.assignedCharacter,
                  let outline = detected.outline else { continue }

            if let existingGlyph = project.glyphs[char] {
                if replaceExisting {
                    var glyph = existingGlyph
                    glyph.outline = outline
                    if autoFitMetrics {
                        let bounds = outline.boundingBox
                        glyph.advanceWidth = bounds.width + Int(CGFloat(project.metrics.unitsPerEm) * 0.2)
                    }
                    project.glyphs[char] = glyph
                }
            } else {
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
    }

    func generateKerningPairs(for project: inout FontProject) async {
        isGeneratingKerning = true
        defer { isGeneratingKerning = false }

        let predictor = KerningPredictor()
        let settings = KerningPredictor.PredictionSettings(
            minKerningValue: 2,
            targetOpticalSpacing: 0.5,
            includePunctuation: true,
            includeNumbers: true,
            onlyCriticalPairs: true
        )

        do {
            let result = try await predictor.predictKerning(for: project, settings: settings)

            for newPair in result.pairs {
                if let existingIndex = project.kerning.firstIndex(where: { $0.left == newPair.left && $0.right == newPair.right }) {
                    project.kerning[existingIndex] = newPair
                } else {
                    project.kerning.append(newPair)
                }
            }

            project.kerning.sort { lhs, rhs in
                if lhs.left == rhs.left {
                    return lhs.right < rhs.right
                }
                return lhs.left < rhs.left
            }
        } catch {
            processingError = "Kerning generation failed: \(error.localizedDescription). Glyphs were imported successfully but without kerning data."
        }
    }

    func downloadSampleSheetPDF() async {
        let url = await fileDialogService.selectSaveLocation(
            defaultName: "Typogenesis_Sample_Sheet.pdf",
            types: [.pdf],
            message: "Choose where to save the sample sheet PDF"
        )
        guard let url else { return }

        let pdfData = generateSampleSheetPDFData()
        do {
            try pdfData.write(to: url)
        } catch {
            processingError = "Failed to save sample sheet PDF: \(error.localizedDescription)"
        }
    }

    func resetScanner() {
        currentStep = .upload
        uploadedImage = nil
        detectedCharacters = []
        selectedCharacterIndex = nil
        isProcessing = false
        threshold = 0.5
        simplification = 2.0
        showSampleSheetInfo = false
        processingError = nil
        replaceExisting = false
        autoFitMetrics = true
        generateKerning = false
        isGeneratingKerning = false
    }

    // MARK: - PDF Generation

    func generateSampleSheetPDFData() -> Data {
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

                context.setStrokeColor(NSColor.lightGray.cgColor)
                context.setLineWidth(1)
                context.stroke(CGRect(x: boxX, y: boxY, width: boxSize, height: boxSize))

                let labelString = NSAttributedString(string: String(char), attributes: labelAttributes)
                let labelSize = labelString.size()
                let labelX = boxX + (boxSize - labelSize.width) / 2
                let labelY = boxY + boxSize + 2

                NSGraphicsContext.saveGraphicsState()
                NSGraphicsContext.current = nsContext
                labelString.draw(at: NSPoint(x: labelX, y: labelY))
                NSGraphicsContext.restoreGraphicsState()
            }

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
