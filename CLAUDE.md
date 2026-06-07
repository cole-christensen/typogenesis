# Typogenesis - AI Font Cloning, Editing & Generation

---

## THE TEN COMMANDMENTS OF SOFTWARE DEVELOPMENT

*Carved in stone. Eternal. Non-negotiable.*

---

### I. THOU SHALT NOT LIE
If it doesn't work, say it doesn't work. If it's not implemented, don't pretend it is.
**The compiler accepts lies. The user suffers them.**

### II. THOU SHALT VERIFY THY OWN WORK
Never assume. Always check. Run it. Test it. Use it. Look at the output.
**If you didn't see it work with your own eyes, it doesn't work.**

### III. THOU SHALT NOT WORSHIP FALSE IDOLS OF PROGRESS
Passing tests, green CI, merged PRs—these are not progress if the software doesn't function.
**Progress is measured in working software, not in activity.**

### IV. THOU SHALT HONOR THY TESTS
Tests are a sacred contract. They define what "working" means.
Write tests that would fail if the code is broken.
**A test that cannot fail is not a test.**

### V. THOU SHALT NOT COMMIT FRAUD
Do not ship what does not work. Do not demo what is faked.
If the button says "Download" it must download. If it says "Generate" it must generate.
**The user's trust is not yours to squander.**

### VI. THOU SHALT KEEP THY CODE HONEST
Stubs must be obvious. TODOs must be tracked. Incomplete work must be marked.
**Code that silently fails is worse than code that loudly crashes.**

### VII. THOU SHALT NOT CONFUSE MOTION WITH ACTION
Writing code is not the same as solving problems.
**Ask not "what did I do?" but "what now works that didn't before?"**

### VIII. THOU SHALT FACE THY FAILURES
When something breaks, understand why. When tests fail, fix the code—not the tests.
**Every failure avoided is a lesson unlearned.**

### IX. THOU SHALT BUILD ON ROCK, NOT SAND
Do not write features that depend on unfinished work.
**A castle built on quicksand will sink, no matter how beautiful its towers.**

### X. THOU SHALT REMEMBER THE USER
Someone will use this software. They will trust that it works.
**Write code as if the user is watching, because eventually they will be.**

---

## Project Overview

Typogenesis is a native macOS application for creating, editing, and cloning fonts using AI. Scan your handwriting, clone existing typefaces, generate entirely new font families, and edit glyphs with AI assistance—all with a beautiful, professional interface.

**Philosophy**: Font creation should be accessible to everyone. AI handles the tedious work (consistent stroke weights, optical corrections, kerning) while you focus on creative expression.

**Current Status:** AI models are not yet available. All AI features fall back to geometric placeholder generation which creates visible shapes but not typographically correct glyphs.

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
Scan or photograph handwritten samples and convert to a digital font. Uses edge detection, contour tracing, and Potrace-style bezier curve fitting to vectorize handwritten characters.

### 2. Font Cloning (Style Transfer)
Clone the style of existing fonts to create new typefaces. Extracts style features (stroke weight, contrast, x-height, serif style, slant, roundness) from representative glyphs and uses learned style embeddings.

### 3. AI Glyph Generation
Generate new glyphs using diffusion-based generation. Supports multiple modes:
- **fromScratch**: Generate with a given style
- **completePartial**: Complete a partial glyph outline
- **variation**: Create variations of existing glyphs
- **interpolate**: Blend between two glyphs

### 4. Glyph Editor
Professional bezier editing with AI assistance. Tools include select, pen, bezier, knife, eraser, and AI refine. Context menu provides AI-powered curve smoothing, style matching, and consistency fixes.

### 5. Intelligent Kerning
AI-powered kerning suggestions for critical pairs (AV, AW, AY, AT, Ta, Te, To, Tr, Va, Ve, Vo, etc.). Renders glyph pairs and predicts optimal spacing adjustments.

## Data Models

### FontProject
Contains font name, family, style, metrics, glyphs dictionary, kerning pairs, metadata, and settings.

### FontMetrics
- unitsPerEm: 1000 (default)
- ascender: 800
- descender: -200
- xHeight: 500
- capHeight: 700
- lineGap: 90

### Glyph
Character, unicode scalars, outline (contours with bezier points), advance width, left side bearing, generation source, and style confidence.

### GlyphOutline
Array of contours, each containing path points with position, type (corner/smooth/symmetric), and optional control handles.

## Export Formats

- **TTF/OTF**: OpenType with TrueType or CFF outlines
- **WOFF/WOFF2**: Compressed web fonts
- **UFO**: Unified Font Object (source format)
- **DesignSpace**: Variable font source

Export builds required tables: head, hhea, maxp, OS/2, name, cmap, post, glyf/loca (TTF) or CFF (OTF), hmtx, kern, GPOS.

## Testing Requirements

### Unit Tests
- Vectorization produces closed contours
- Style extraction classifies serif style correctly
- Kerning prediction returns negative values for AV pairs
- Font export roundtrip preserves glyph count and metrics

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

## MVVM Architecture Guide

### ViewModel Pattern

All feature ViewModels follow this structure:

```swift
@MainActor
final class {Name}ViewModel: ObservableObject {
    @Published var someState: Type = defaultValue
    func doWork() async { let service = SomeService(); /* ... */ }
}
```

- Views own their ViewModel via `@StateObject`
- Services (FontParser, FontExporter, etc.) are created on demand inside methods, not stored as properties
- Use `FileDialogService` protocol for file dialogs (enables testability)
- No AppKit imports in ViewModels -- all platform code goes through protocols

### Sync Patterns

**Pattern A -- `.onChange` sync** (continuous edits):
Used by KerningEditor, VariableFontEditor, GlyphEditor. The ViewModel holds mutable state;
the View syncs changes back to AppState via `.onChange(of: viewModel.property)`.

**Pattern B -- One-shot callback** (button press):
Used by Generation, HandwritingScanner, Export, Import, Metrics. The ViewModel performs
an async action and exposes a result. The View applies the result to AppState on a
confirmation button press.

### Rules

1. Views must be purely declarative -- no business logic
2. No AppKit in ViewModels
3. Use `FileDialogService` protocol for file dialogs
4. Services created on demand, not stored as ViewModel properties
5. All ViewModels must have corresponding test files in `TypegenesisTests/`

### Test Pattern

```swift
@Suite("{Name}ViewModel Tests")
struct {Name}ViewModelTests {
    @Test("description")
    @MainActor func testSomething() {
        let vm = {Name}ViewModel()
        #expect(vm.state == expected)
    }
}
```

Inject `MockFileDialogService` for ViewModels that use file dialogs.

## Agent Decision Authority

### Can Decide
- Implementation details within established patterns
- UI polish and animations
- Performance optimizations
- Test structure and organization

### Should Ask
- New AI model architectures
- Changes to file format support
- Major UI workflow changes
- New export formats

### Must Ask
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
