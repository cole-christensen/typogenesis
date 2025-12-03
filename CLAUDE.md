# Typogenesis - AI Font Cloning, Editing & Generation

## Project Overview

Typogenesis is a native macOS application for creating, editing, and cloning fonts using AI. Scan your handwriting, clone existing typefaces, generate entirely new font families, and edit glyphs with AI assistance—all with a beautiful, professional interface.

**Philosophy**: Font creation should be accessible to everyone. AI handles the tedious work (consistent stroke weights, optical corrections, kerning) while you focus on creative expression.

## Technology Stack

### Application
- **Language**: Swift 6 + SwiftUI
- **Platform**: macOS 14+ (Sonoma)
- **Architecture**: MVVM with Swift Concurrency
- **Graphics**: Core Graphics, Metal (for AI inference)

### AI/ML
- **Local Inference**: Core ML + Metal Performance Shaders
- **Models**:
  - Glyph generation (diffusion-based)
  - Style transfer (for font cloning)
  - Stroke analysis (for vectorization)
  - Kerning prediction
- **Optional Cloud**: Ollama for enhanced generation

### Font Technology
- **Format Support**: TTF, OTF, WOFF, WOFF2, UFO
- **Libraries**:
  - fonttools (Python, via subprocess)
  - freetype (for rendering)
  - Custom Swift font table manipulation

## Project Structure

```
typogenesis/
├── Typogenesis/
│   ├── App/
│   │   ├── TypogenesisApp.swift
│   │   ├── AppState.swift
│   │   └── AppDelegate.swift
│   ├── Models/
│   │   ├── Font/
│   │   │   ├── FontProject.swift
│   │   │   ├── Glyph.swift
│   │   │   ├── GlyphOutline.swift
│   │   │   ├── Metrics.swift
│   │   │   ├── KerningPair.swift
│   │   │   └── FontFamily.swift
│   │   ├── AI/
│   │   │   ├── GenerationRequest.swift
│   │   │   ├── StyleTransfer.swift
│   │   │   └── TrainingData.swift
│   │   └── Project/
│   │       ├── ProjectDocument.swift
│   │       └── ProjectSettings.swift
│   ├── Views/
│   │   ├── Main/
│   │   │   ├── MainWindow.swift
│   │   │   ├── Sidebar.swift
│   │   │   └── Inspector.swift
│   │   ├── Editor/
│   │   │   ├── GlyphEditor.swift
│   │   │   ├── GlyphCanvas.swift
│   │   │   ├── BezierTools.swift
│   │   │   ├── PointEditor.swift
│   │   │   └── PreviewPanel.swift
│   │   ├── Generation/
│   │   │   ├── GenerateView.swift
│   │   │   ├── StylePicker.swift
│   │   │   ├── CharacterSetPicker.swift
│   │   │   └── ProgressView.swift
│   │   ├── Clone/
│   │   │   ├── CloneWizard.swift
│   │   │   ├── SampleUpload.swift
│   │   │   ├── StyleExtraction.swift
│   │   │   └── ClonePreview.swift
│   │   ├── Handwriting/
│   │   │   ├── HandwritingScanner.swift
│   │   │   ├── SampleSheet.swift
│   │   │   ├── CharacterSegmentation.swift
│   │   │   └── VectorizationView.swift
│   │   ├── Metrics/
│   │   │   ├── MetricsEditor.swift
│   │   │   ├── KerningEditor.swift
│   │   │   └── SpacingPreview.swift
│   │   └── Export/
│   │       ├── ExportSheet.swift
│   │       ├── FormatOptions.swift
│   │       └── WebFontPreview.swift
│   ├── ViewModels/
│   │   ├── FontProjectViewModel.swift
│   │   ├── GlyphEditorViewModel.swift
│   │   ├── GenerationViewModel.swift
│   │   └── CloneViewModel.swift
│   ├── Services/
│   │   ├── AI/
│   │   │   ├── GlyphGenerator.swift
│   │   │   ├── StyleTransferEngine.swift
│   │   │   ├── StrokeAnalyzer.swift
│   │   │   ├── KerningPredictor.swift
│   │   │   └── ModelManager.swift
│   │   ├── Font/
│   │   │   ├── FontParser.swift
│   │   │   ├── FontExporter.swift
│   │   │   ├── GlyphRenderer.swift
│   │   │   ├── OutlineProcessor.swift
│   │   │   └── OpenTypeBuilder.swift
│   │   ├── Image/
│   │   │   ├── ImageProcessor.swift
│   │   │   ├── Vectorizer.swift
│   │   │   ├── EdgeDetector.swift
│   │   │   └── ContourTracer.swift
│   │   └── Storage/
│   │       ├── ProjectStorage.swift
│   │       ├── ModelCache.swift
│   │       └── RecentProjects.swift
│   ├── Utilities/
│   │   ├── BezierMath.swift
│   │   ├── PathSimplifier.swift
│   │   ├── UnicodeHelper.swift
│   │   └── ColorScheme.swift
│   └── Resources/
│       ├── Assets.xcassets
│       ├── Models/               # Core ML models
│       │   ├── GlyphDiffusion.mlpackage
│       │   ├── StyleEncoder.mlpackage
│       │   └── KerningNet.mlpackage
│       └── SampleSheets/         # Printable templates
├── TypegenesisTests/
├── TypegenesisUITests/
├── scripts/
│   ├── train_models.py          # Model training scripts
│   ├── prepare_dataset.py       # Dataset preparation
│   └── convert_to_coreml.py     # Model conversion
├── docs/
│   └── JOURNAL.md
└── README.md
```

## Core Features

### 1. Handwriting to Font

Scan or photograph handwritten samples and convert to a digital font.

```swift
// Services/Image/Vectorizer.swift

class Vectorizer {
    private let edgeDetector: EdgeDetector
    private let contourTracer: ContourTracer
    private let pathSimplifier: PathSimplifier

    func vectorize(image: CGImage, settings: VectorizationSettings) async throws -> GlyphOutline {
        // 1. Preprocess image
        let preprocessed = try await preprocess(image)

        // 2. Detect edges
        let edges = try await edgeDetector.detect(preprocessed)

        // 3. Trace contours
        let contours = try await contourTracer.trace(edges)

        // 4. Convert to bezier curves
        let paths = try await fitBeziers(contours, tolerance: settings.tolerance)

        // 5. Simplify paths
        let simplified = pathSimplifier.simplify(paths, threshold: settings.simplificationThreshold)

        // 6. Normalize to glyph coordinates
        return normalizeToGlyphSpace(simplified)
    }

    private func fitBeziers(_ contours: [Contour], tolerance: CGFloat) async throws -> [BezierPath] {
        // Potrace-style curve fitting
        // ...
    }
}
```

### 2. Font Cloning (Style Transfer)

Clone the style of existing fonts to create new typefaces.

```swift
// Services/AI/StyleTransferEngine.swift

class StyleTransferEngine {
    private let styleEncoder: StyleEncoder
    private let glyphGenerator: GlyphGenerator

    struct FontStyle {
        let strokeWeight: Float
        let contrast: Float
        let xHeight: Float
        let serifStyle: SerifStyle
        let slant: Float
        let roundness: Float
        let embedding: [Float]  // Learned style vector
    }

    func extractStyle(from font: FontProject) async throws -> FontStyle {
        // Encode representative glyphs to extract style
        let representativeGlyphs = ["n", "o", "H", "O", "a", "g"]
        var embeddings: [[Float]] = []

        for char in representativeGlyphs {
            if let glyph = font.glyph(for: char) {
                let embedding = try await styleEncoder.encode(glyph)
                embeddings.append(embedding)
            }
        }

        // Average embeddings for stable style representation
        let styleEmbedding = averageEmbeddings(embeddings)

        return FontStyle(
            strokeWeight: analyzeStrokeWeight(font),
            contrast: analyzeContrast(font),
            xHeight: font.metrics.xHeight,
            serifStyle: classifySerifs(font),
            slant: measureSlant(font),
            roundness: measureRoundness(font),
            embedding: styleEmbedding
        )
    }

    func generateGlyph(character: Character, style: FontStyle) async throws -> Glyph {
        return try await glyphGenerator.generate(
            character: character,
            styleEmbedding: style.embedding,
            metrics: deriveMetrics(from: style)
        )
    }
}
```

### 3. AI Glyph Generation

Generate new glyphs from text descriptions or by completing partial designs.

```swift
// Services/AI/GlyphGenerator.swift

class GlyphGenerator {
    private let diffusionModel: GlyphDiffusionModel
    private let conditioningEncoder: ConditioningEncoder

    enum GenerationMode {
        case fromScratch(style: FontStyle)
        case completePartial(partial: GlyphOutline, style: FontStyle)
        case variation(base: Glyph, variationStrength: Float)
        case interpolate(glyphA: Glyph, glyphB: Glyph, t: Float)
    }

    func generate(
        character: Character,
        mode: GenerationMode,
        steps: Int = 50
    ) async throws -> Glyph {
        // Prepare conditioning
        let conditioning = try await prepareConditioning(character: character, mode: mode)

        // Run diffusion
        var latent = generateNoise(shape: diffusionModel.latentShape)

        for step in (0..<steps).reversed() {
            let t = Float(step) / Float(steps)
            latent = try await diffusionModel.denoise(
                latent: latent,
                conditioning: conditioning,
                timestep: t
            )
        }

        // Decode to glyph outline
        let outline = try await decodeLatent(latent)

        return Glyph(
            character: character,
            outline: outline,
            metrics: deriveMetrics(from: outline, style: conditioning.style)
        )
    }
}
```

### 4. Glyph Editor

Professional bezier editing with AI assistance.

```swift
// Views/Editor/GlyphCanvas.swift

struct GlyphCanvas: View {
    @ObservedObject var viewModel: GlyphEditorViewModel
    @State private var selectedPoints: Set<PointID> = []
    @State private var tool: EditorTool = .select

    enum EditorTool {
        case select
        case pen
        case bezier
        case knife
        case eraser
        case aiRefine
    }

    var body: some View {
        Canvas { context, size in
            // Draw grid
            drawGrid(context: context, size: size)

            // Draw guidelines
            drawGuidelines(context: context)

            // Draw glyph outline
            drawOutline(context: context, outline: viewModel.glyph.outline)

            // Draw control points
            if tool != .aiRefine {
                drawControlPoints(context: context)
            }

            // Draw AI suggestions overlay
            if tool == .aiRefine {
                drawAISuggestions(context: context)
            }
        }
        .gesture(editingGesture)
        .contextMenu { contextMenuItems }
    }

    @ViewBuilder
    var contextMenuItems: some View {
        Button("AI: Smooth Curve") {
            Task { await viewModel.aiSmoothSelection(selectedPoints) }
        }
        Button("AI: Match Style") {
            Task { await viewModel.aiMatchStyle(selectedPoints) }
        }
        Button("AI: Fix Consistency") {
            Task { await viewModel.aiFixConsistency() }
        }
        Divider()
        Button("Simplify Path") {
            viewModel.simplifyPath(selectedPoints)
        }
    }
}
```

### 5. Intelligent Kerning

AI-powered kerning suggestions with visual editor.

```swift
// Services/AI/KerningPredictor.swift

class KerningPredictor {
    private let kerningModel: KerningNet

    func predictKerning(for font: FontProject) async throws -> [KerningPair] {
        var pairs: [KerningPair] = []

        // Common kerning pairs
        let criticalPairs = generateCriticalPairs(font.characterSet)

        for (left, right) in criticalPairs {
            guard let leftGlyph = font.glyph(for: left),
                  let rightGlyph = font.glyph(for: right) else { continue }

            // Render pair at high resolution
            let pairImage = renderPair(leftGlyph, rightGlyph, spacing: 0)

            // Predict optimal kerning
            let kerning = try await kerningModel.predict(pairImage)

            if abs(kerning) > 1 {  // Only include significant adjustments
                pairs.append(KerningPair(left: left, right: right, value: Int(kerning)))
            }
        }

        return pairs
    }

    private func generateCriticalPairs(_ charset: CharacterSet) -> [(Character, Character)] {
        // Known problematic combinations
        let critical = [
            ("A", "V"), ("A", "W"), ("A", "Y"), ("A", "T"),
            ("T", "a"), ("T", "e"), ("T", "o"), ("T", "r"),
            ("V", "a"), ("V", "e"), ("V", "o"),
            ("W", "a"), ("W", "e"), ("W", "o"),
            ("Y", "a"), ("Y", "e"), ("Y", "o"),
            ("f", "f"), ("f", "i"), ("f", "l"),
            ("r", "a"), ("r", "e"), ("r", "o"),
            // ... more pairs
        ]

        return critical.compactMap { (l, r) in
            guard let lc = l.first, let rc = r.first,
                  charset.contains(lc), charset.contains(rc) else { return nil }
            return (lc, rc)
        }
    }
}
```

## Data Models

### Font Project

```swift
// Models/Font/FontProject.swift

struct FontProject: Identifiable, Codable {
    let id: UUID
    var name: String
    var family: String
    var style: String  // Regular, Bold, Italic, etc.

    var metrics: FontMetrics
    var glyphs: [Character: Glyph]
    var kerning: [KerningPair]

    var metadata: FontMetadata
    var settings: ProjectSettings

    // Computed
    var characterSet: CharacterSet {
        CharacterSet(charactersIn: String(glyphs.keys))
    }
}

struct FontMetrics: Codable {
    var unitsPerEm: Int = 1000
    var ascender: Int = 800
    var descender: Int = -200
    var xHeight: Int = 500
    var capHeight: Int = 700
    var lineGap: Int = 90
}

struct Glyph: Identifiable, Codable {
    let id: UUID
    var character: Character
    var unicodeScalars: [UInt32]
    var outline: GlyphOutline
    var advanceWidth: Int
    var leftSideBearing: Int

    // AI metadata
    var generatedBy: GenerationSource?
    var styleConfidence: Float?
}

struct GlyphOutline: Codable {
    var contours: [Contour]

    struct Contour: Codable {
        var points: [PathPoint]
        var isClosed: Bool
    }

    struct PathPoint: Codable {
        var position: CGPoint
        var type: PointType
        var controlIn: CGPoint?   // For curves
        var controlOut: CGPoint?

        enum PointType: String, Codable {
            case corner
            case smooth
            case symmetric
        }
    }
}
```

### AI Models

```swift
// Services/AI/ModelManager.swift

class ModelManager {
    static let shared = ModelManager()

    private var glyphDiffusion: GlyphDiffusionModel?
    private var styleEncoder: StyleEncoderModel?
    private var kerningNet: KerningNetModel?

    enum ModelStatus {
        case notLoaded
        case loading(Progress)
        case loaded
        case error(Error)
    }

    @Published var glyphDiffusionStatus: ModelStatus = .notLoaded
    @Published var styleEncoderStatus: ModelStatus = .notLoaded
    @Published var kerningNetStatus: ModelStatus = .notLoaded

    func loadModels() async {
        await withTaskGroup(of: Void.self) { group in
            group.addTask { await self.loadGlyphDiffusion() }
            group.addTask { await self.loadStyleEncoder() }
            group.addTask { await self.loadKerningNet() }
        }
    }

    private func loadGlyphDiffusion() async {
        glyphDiffusionStatus = .loading(Progress())
        do {
            let config = MLModelConfiguration()
            config.computeUnits = .cpuAndNeuralEngine
            glyphDiffusion = try await GlyphDiffusionModel.load(configuration: config)
            glyphDiffusionStatus = .loaded
        } catch {
            glyphDiffusionStatus = .error(error)
        }
    }
}
```

## Export Formats

```swift
// Services/Font/FontExporter.swift

class FontExporter {
    enum ExportFormat {
        case ttf
        case otf
        case woff
        case woff2
        case ufo
        case designSpace  // Variable font source
    }

    func export(
        project: FontProject,
        format: ExportFormat,
        options: ExportOptions
    ) async throws -> Data {
        switch format {
        case .ttf, .otf:
            return try await exportOpenType(project, isTTF: format == .ttf, options: options)
        case .woff:
            let otf = try await exportOpenType(project, isTTF: false, options: options)
            return try compressToWOFF(otf)
        case .woff2:
            let otf = try await exportOpenType(project, isTTF: false, options: options)
            return try compressToWOFF2(otf)
        case .ufo:
            return try exportUFO(project, options: options)
        case .designSpace:
            return try exportDesignSpace(project, options: options)
        }
    }

    private func exportOpenType(
        _ project: FontProject,
        isTTF: Bool,
        options: ExportOptions
    ) async throws -> Data {
        let builder = OpenTypeBuilder()

        // Required tables
        builder.addHead(project.metadata)
        builder.addHhea(project.metrics)
        builder.addMaxp(glyphCount: project.glyphs.count)
        builder.addOS2(project)
        builder.addName(project.metadata)
        builder.addCmap(project.glyphs)
        builder.addPost(project)

        // Glyph data
        if isTTF {
            builder.addGlyf(project.glyphs)
            builder.addLoca(project.glyphs)
        } else {
            builder.addCFF(project.glyphs)
        }

        // Metrics
        builder.addHmtx(project.glyphs)

        // Kerning
        if !project.kerning.isEmpty {
            builder.addKern(project.kerning)
            builder.addGPOS(project.kerning)  // Modern kerning
        }

        return builder.build()
    }
}
```

## Testing Requirements

### Unit Tests
```swift
@Test func testVectorization() async throws {
    let image = loadTestImage("handwriting_a.png")
    let vectorizer = Vectorizer()
    let outline = try await vectorizer.vectorize(image: image, settings: .default)

    #expect(outline.contours.count > 0)
    #expect(outline.contours.allSatisfy { $0.isClosed })
}

@Test func testStyleExtraction() async throws {
    let font = loadTestFont("Helvetica")
    let engine = StyleTransferEngine()
    let style = try await engine.extractStyle(from: font)

    #expect(style.serifStyle == .sansSerif)
    #expect(style.slant < 0.1)  // Upright
}

@Test func testKerningPrediction() async throws {
    let font = createTestFont(withGlyphs: ["A", "V", "a", "v"])
    let predictor = KerningPredictor()
    let pairs = try await predictor.predictKerning(for: font)

    let avPair = pairs.first { $0.left == "A" && $0.right == "V" }
    #expect(avPair != nil)
    #expect(avPair!.value < 0)  // AV should have negative kerning
}

@Test func testFontExportRoundtrip() async throws {
    let project = createTestProject()
    let exporter = FontExporter()
    let parser = FontParser()

    let ttfData = try await exporter.export(project: project, format: .ttf, options: .default)
    let parsed = try await parser.parse(data: ttfData)

    #expect(parsed.glyphs.count == project.glyphs.count)
    #expect(parsed.metrics.unitsPerEm == project.metrics.unitsPerEm)
}
```

### Integration Tests
- Full handwriting → font workflow
- Style cloning produces visually consistent results
- Exported fonts render correctly in macOS apps
- Large glyph sets (1000+ characters)

### Visual Regression Tests
- Glyph rendering comparison
- Kerning visual validation
- Export format fidelity

## Development Guidelines

### TDD Mandatory
Write tests first. Every feature must have:
1. Unit tests for core logic
2. Integration tests for workflows
3. Visual tests for rendering

### Code Style
- Swift: Follow Swift API Design Guidelines
- SwiftUI: Prefer small, composable views
- Use Swift Concurrency (async/await) throughout

### Commit Convention
```
type(scope): description

feat(editor): add bezier pen tool
fix(export): correct kerning table generation
perf(ai): optimize diffusion inference on M1
```

## Agent Decision Authority

### Can Decide (✅)
- Implementation details within established patterns
- UI polish and animations
- Performance optimizations
- Test structure and organization

### Should Ask (⚠️)
- New AI model architectures
- Changes to file format support
- Major UI workflow changes
- New export formats

### Must Ask (🚫)
- Core font data model changes
- Breaking changes to project file format
- Removing features
- Changes to AI training approach

## Commands

```bash
# Development
open Typogenesis.xcodeproj         # Open in Xcode
xcodebuild -scheme Typogenesis -destination 'platform=macOS' build
xcodebuild test -scheme Typogenesis

# Model Training (Python)
cd scripts
python train_models.py --model glyph_diffusion --epochs 100
python convert_to_coreml.py --input model.pt --output ../Typogenesis/Resources/Models/

# Release
xcodebuild archive -scheme Typogenesis -archivePath build/Typogenesis.xcarchive
```

## Performance Targets

- Glyph generation: < 2s per glyph on M1
- Style extraction: < 5s for full font
- Font export (TTF): < 1s for 256 glyphs
- Editor responsiveness: 60fps during editing
- App launch: < 1s to main window

## Inspiration & References

- **Glyphs.app** - Professional font editor
- **FontForge** - Open source editor
- **Calligraphr** - Handwriting to font
- **Prototypo** - Parametric font design
- **FontLab** - Industry standard tools
- **DeepFont** - Adobe's font recognition
