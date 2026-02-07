import Foundation

/// Centralized accessibility identifiers for UI testing
/// Use these identifiers in views via .accessibilityIdentifier()
enum AccessibilityID {

    // MARK: - Welcome View
    enum Welcome {
        static let createNewFontButton = "welcome.createNewFont"
        static let importFontButton = "welcome.importFont"
        static let openProjectButton = "welcome.openProject"
        static let importingIndicator = "welcome.importingIndicator"
    }

    // MARK: - Sidebar
    enum Sidebar {
        static let glyphsItem = "sidebar.glyphs"
        static let metricsItem = "sidebar.metrics"
        static let kerningItem = "sidebar.kerning"
        static let previewItem = "sidebar.preview"
        static let variableItem = "sidebar.variable"
        static let generateItem = "sidebar.generate"
        static let handwritingItem = "sidebar.handwriting"
        static let cloneItem = "sidebar.clone"
    }

    // MARK: - Glyph Grid
    enum GlyphGrid {
        static let grid = "glyphGrid.grid"
        static let addGlyphButton = "glyphGrid.addGlyph"
        static let searchField = "glyphGrid.search"

        static func glyphCell(_ character: Character) -> String {
            "glyphGrid.cell.\(character)"
        }
    }

    // MARK: - Glyph Editor
    enum GlyphEditor {
        static let canvas = "glyphEditor.canvas"
        static let selectTool = "glyphEditor.tool.select"
        static let penTool = "glyphEditor.tool.pen"
        static let addPointTool = "glyphEditor.tool.addPoint"
        static let deletePointTool = "glyphEditor.tool.deletePoint"
        static let undoButton = "glyphEditor.undo"
        static let redoButton = "glyphEditor.redo"
        static let pathOperationsMenu = "glyphEditor.pathOperations"
    }

    // MARK: - Add Glyph Sheet
    enum AddGlyph {
        static let sheet = "addGlyph.sheet"
        static let keyboardInput = "addGlyph.keyboardInput"
        static let unicodeInput = "addGlyph.unicodeInput"
        static let presetAZ = "addGlyph.preset.AZ"
        static let presetaz = "addGlyph.preset.az"
        static let preset09 = "addGlyph.preset.09"
        static let presetPunctuation = "addGlyph.preset.punctuation"
        static let addButton = "addGlyph.addButton"
        static let cancelButton = "addGlyph.cancelButton"
    }

    // MARK: - Inspector
    enum Inspector {
        static let panel = "inspector.panel"
        static let fontFamilyField = "inspector.fontFamily"
        static let fontStyleField = "inspector.fontStyle"
        static let advanceWidthField = "inspector.advanceWidth"
        static let leftSideBearingField = "inspector.lsb"
    }

    // MARK: - Metrics Editor
    enum Metrics {
        static let editor = "metrics.editor"
        static let unitsPerEmField = "metrics.unitsPerEm"
        static let ascenderField = "metrics.ascender"
        static let descenderField = "metrics.descender"
        static let capHeightField = "metrics.capHeight"
        static let xHeightField = "metrics.xHeight"
        static let lineGapField = "metrics.lineGap"
        static let applyButton = "metrics.apply"
        static let resetButton = "metrics.reset"
    }

    // MARK: - Kerning Editor
    enum Kerning {
        static let editor = "kerning.editor"
        static let pairsList = "kerning.pairsList"
        static let addPairButton = "kerning.addPair"
        static let autoKernButton = "kerning.autoKern"
        static let leftCharField = "kerning.leftChar"
        static let rightCharField = "kerning.rightChar"
        static let valueField = "kerning.value"
        static let previewCanvas = "kerning.preview"
    }

    // MARK: - Preview Panel
    enum Preview {
        static let panel = "preview.panel"
        static let sampleTextField = "preview.sampleText"
        static let sizeSlider = "preview.sizeSlider"
        static let modeParagraph = "preview.mode.paragraph"
        static let modeWaterfall = "preview.mode.waterfall"
        static let modeGlyphProof = "preview.mode.glyphProof"
        static let modeKerning = "preview.mode.kerning"
    }

    // MARK: - Variable Font Editor
    enum Variable {
        static let editor = "variable.editor"
        static let enableToggle = "variable.enableToggle"
        static let addAxisButton = "variable.addAxis"
        static let addMasterButton = "variable.addMaster"
        static let addInstanceButton = "variable.addInstance"
        static let axesList = "variable.axesList"
        static let mastersList = "variable.mastersList"
        static let instancesList = "variable.instancesList"
        static let previewSlider = "variable.previewSlider"
    }

    // MARK: - Export Sheet
    enum Export {
        static let sheet = "export.sheet"
        static let formatTTF = "export.format.ttf"
        static let formatOTF = "export.format.otf"
        static let formatWOFF = "export.format.woff"
        static let formatUFO = "export.format.ufo"
        static let includeKerningToggle = "export.includeKerning"
        static let exportButton = "export.exportButton"
        static let cancelButton = "export.cancelButton"
    }

    // MARK: - Import Sheet
    enum Import {
        static let sheet = "import.sheet"
        static let analyzeButton = "import.analyzeButton"
        static let analyzingIndicator = "import.analyzingIndicator"
        static let importButton = "import.importButton"
        static let cancelButton = "import.cancelButton"
        static let backButton = "import.backButton"
        static let styleAnalysis = "import.styleAnalysis"
        static let fontFamily = "import.fontFamily"
        static let glyphCount = "import.glyphCount"
    }

    // MARK: - AI Generate View
    enum Generate {
        static let view = "generate.view"
        static let modeCompleteFont = "generate.mode.completeFont"
        static let modeMissingGlyphs = "generate.mode.missingGlyphs"
        static let modeStyleTransfer = "generate.mode.styleTransfer"
        static let modeVariation = "generate.mode.variation"
        static let generateButton = "generate.generateButton"
        static let addToProjectButton = "generate.addToProject"
        static let progressIndicator = "generate.progress"
    }

    // MARK: - Handwriting Scanner
    enum Handwriting {
        static let scanner = "handwriting.scanner"
        static let uploadArea = "handwriting.uploadArea"
        static let processButton = "handwriting.processButton"
        static let assignGrid = "handwriting.assignGrid"
        static let importButton = "handwriting.importButton"
        static let stepIndicator = "handwriting.stepIndicator"
    }

    // MARK: - Clone Wizard
    enum Clone {
        static let wizard = "clone.wizard"
        static let uploadArea = "clone.uploadArea"
        static let selectFontButton = "clone.selectFont"
        static let analyzeButton = "clone.analyzeButton"
        static let analyzingIndicator = "clone.analyzing"
        static let styleCard = "clone.styleCard"
        static let characterSetPicker = "clone.characterSet"
        static let generateButton = "clone.generateButton"
        static let generatingIndicator = "clone.generating"
        static let previewCanvas = "clone.preview"
        static let applyButton = "clone.applyButton"
        static let cancelButton = "clone.cancelButton"
        static let backButton = "clone.backButton"
        static let nextButton = "clone.nextButton"
        static let stepIndicator = "clone.stepIndicator"
    }
}
