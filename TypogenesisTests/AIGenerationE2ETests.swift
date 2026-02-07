import Testing
import Foundation
import CoreGraphics
@testable import Typogenesis

// MARK: - AI Generation E2E Tests

/// Comprehensive E2E tests for AI-powered font generation workflows.

@Suite("AI Generation E2E Tests")
struct AIGenerationE2ETests {

    @Test("Complete AI glyph generation workflow")
    func completeAIGenerationWorkflow() async throws {
        // =====================================================
        // PHASE 1: Create source font for style extraction
        // =====================================================
        var sourceProject = FontProject(name: "Source Font", family: "SourceSans", style: "Regular")
        sourceProject.metrics.unitsPerEm = 1000
        sourceProject.metrics.ascender = 800
        sourceProject.metrics.descender = -200
        sourceProject.metrics.capHeight = 700
        sourceProject.metrics.xHeight = 500

        // Add representative glyphs for style analysis
        let representativeChars: [Character] = ["n", "o", "H", "O", "a", "e"]
        for char in representativeChars {
            let outline = createSampleGlyphOutline(for: char)
            sourceProject.glyphs[char] = Glyph(
                character: char,
                outline: outline,
                advanceWidth: 500,
                leftSideBearing: 50
            )
        }

        #expect(sourceProject.glyphs.count == 6)

        // =====================================================
        // PHASE 2: Extract style from source font
        // =====================================================
        let styleEncoder = StyleEncoder()
        let extractedStyle = try await styleEncoder.extractStyle(from: sourceProject)

        // Verify style properties were extracted
        #expect(extractedStyle.strokeWeight >= 0 && extractedStyle.strokeWeight <= 1)
        #expect(extractedStyle.strokeContrast >= 0 && extractedStyle.strokeContrast <= 1)
        #expect(extractedStyle.xHeightRatio > 0)
        #expect(extractedStyle.roundness >= 0 && extractedStyle.roundness <= 1)
        #expect(extractedStyle.regularity >= 0 && extractedStyle.regularity <= 1)

        // =====================================================
        // PHASE 3: Generate new glyphs from scratch with style
        // =====================================================
        let generator = GlyphGenerator()
        let targetMetrics = sourceProject.metrics

        // Generate single glyph from scratch
        let resultA = try await generator.generate(
            character: "A",
            mode: .fromScratch(style: extractedStyle),
            metrics: targetMetrics,
            settings: .fast
        )

        #expect(resultA.glyph.character == "A")
        #expect(resultA.confidence >= 0 && resultA.confidence <= 1)
        #expect(resultA.generationTime > 0)

        // =====================================================
        // PHASE 4: Generate variation of existing glyph
        // =====================================================
        let baseGlyph = sourceProject.glyphs["H"]!

        let variationResult = try await generator.generate(
            character: "H",
            mode: .variation(base: baseGlyph, strength: 0.5),
            metrics: targetMetrics,
            settings: .fast
        )

        #expect(variationResult.glyph.character == "H")
        #expect(variationResult.generationTime > 0)

        // =====================================================
        // PHASE 5: Interpolate between two glyphs
        // =====================================================
        let glyphA = sourceProject.glyphs["o"]!
        let glyphB = sourceProject.glyphs["O"]!

        let interpolateResult = try await generator.generate(
            character: "0",  // Generate zero by interpolating o and O
            mode: .interpolate(glyphA: glyphA, glyphB: glyphB, t: 0.5),
            metrics: targetMetrics,
            settings: .fast
        )

        #expect(interpolateResult.glyph.character == "0")

        // =====================================================
        // PHASE 6: Complete partial glyph outline
        // =====================================================
        // Create a partial outline (just the top half of H)
        let partialOutline = GlyphOutline(contours: [
            Contour(points: [
                PathPoint(position: CGPoint(x: 50, y: 350), type: .corner),
                PathPoint(position: CGPoint(x: 150, y: 350), type: .corner),
                PathPoint(position: CGPoint(x: 150, y: 700), type: .corner),
                PathPoint(position: CGPoint(x: 50, y: 700), type: .corner)
            ], isClosed: true)
        ])

        let completedResult = try await generator.generate(
            character: "H",
            mode: .completePartial(partial: partialOutline, style: extractedStyle),
            metrics: targetMetrics,
            settings: .fast
        )

        #expect(completedResult.glyph.character == "H")

        // =====================================================
        // PHASE 7: Batch generation
        // =====================================================
        let batchChars: [Character] = ["B", "C", "D", "E", "F"]
        let progressCount = AtomicProgressCount()

        let batchResults = try await generator.generateBatch(
            characters: batchChars,
            mode: .fromScratch(style: extractedStyle),
            metrics: targetMetrics,
            settings: .fast,
            onProgress: { current, total in
                progressCount.increment()
            }
        )

        #expect(batchResults.count == 5)
        #expect(progressCount.value == 5, "Expected 5 progress callbacks (one per character), got \(progressCount.value)")
        for (index, result) in batchResults.enumerated() {
            #expect(result.glyph.character == batchChars[index])
        }

        // =====================================================
        // PHASE 8: Create new font project with generated glyphs
        // =====================================================
        var targetProject = FontProject(name: "Generated Font", family: "AIGenerated", style: "Regular")
        targetProject.metrics = targetMetrics

        // Add all generated glyphs
        targetProject.glyphs[resultA.glyph.character] = resultA.glyph
        targetProject.glyphs[variationResult.glyph.character] = variationResult.glyph
        targetProject.glyphs[interpolateResult.glyph.character] = interpolateResult.glyph
        targetProject.glyphs[completedResult.glyph.character] = completedResult.glyph
        for result in batchResults {
            targetProject.glyphs[result.glyph.character] = result.glyph
        }

        #expect(targetProject.glyphs.count >= 8)

        // =====================================================
        // PHASE 9: Add AI-predicted kerning
        // =====================================================
        let kerningPredictor = KerningPredictor()
        let kerningResult = try await kerningPredictor.predictKerning(for: targetProject)

        // Add predicted kerning to project
        for pair in kerningResult.pairs {
            targetProject.kerning.append(pair)
        }

        // =====================================================
        // PHASE 10: Export the AI-generated font
        // =====================================================
        let exporter = FontExporter()
        let ttfData = try await exporter.export(project: targetProject)

        #expect(!ttfData.isEmpty)

        // =====================================================
        // VERIFICATION
        // =====================================================
        #expect(targetProject.glyphs.count >= 8)
        // Verify all generated glyphs have non-empty outlines (not just metadata)
        for (char, glyph) in targetProject.glyphs {
            #expect(!glyph.outline.isEmpty, "Generated glyph '\(char)' should have non-empty outline")
        }
    }

    @Test("Complete font cloning workflow (style transfer)")
    func completeFontCloningWorkflow() async throws {
        // =====================================================
        // PHASE 1: Create source font to clone
        // =====================================================
        var sourceFont = FontProject(name: "Source Bold", family: "SourceBold", style: "Bold")
        sourceFont.metrics.unitsPerEm = 1000
        sourceFont.metrics.capHeight = 700
        sourceFont.metrics.xHeight = 500

        // Add glyphs with bold-like characteristics (thicker strokes)
        for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz" {
            let outline = createBoldGlyphOutline(for: char)
            sourceFont.glyphs[char] = Glyph(
                character: char,
                outline: outline,
                advanceWidth: 600,
                leftSideBearing: 50
            )
        }

        #expect(sourceFont.glyphs.count == 52)

        // =====================================================
        // PHASE 2: Extract style from source (the style to clone)
        // =====================================================
        let styleEncoder = StyleEncoder()
        let sourceStyle = try await styleEncoder.extractStyle(from: sourceFont)

        // Verify bold characteristics
        #expect(sourceStyle.strokeWeight > 0)
        #expect(sourceStyle.serifStyle == .sansSerif)

        // =====================================================
        // PHASE 3: Create second font for comparison
        // =====================================================
        var lightFont = FontProject(name: "Light Font", family: "LightFont", style: "Light")
        lightFont.metrics = sourceFont.metrics

        for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz" {
            let outline = createLightGlyphOutline(for: char)
            lightFont.glyphs[char] = Glyph(
                character: char,
                outline: outline,
                advanceWidth: 450,
                leftSideBearing: 50
            )
        }

        let lightStyle = try await styleEncoder.extractStyle(from: lightFont)

        // =====================================================
        // PHASE 4: Compare styles
        // =====================================================
        let similarity = styleEncoder.similarity(sourceStyle, lightStyle)
        #expect(similarity >= 0 && similarity <= 1)

        // Same style should be 100% similar
        let selfSimilarity = styleEncoder.similarity(sourceStyle, sourceStyle)
        #expect(selfSimilarity > 0.99)

        // =====================================================
        // PHASE 5: Interpolate between styles
        // =====================================================
        let mediumStyle = styleEncoder.interpolate(lightStyle, sourceStyle, t: 0.5)

        #expect(mediumStyle.strokeWeight >= min(lightStyle.strokeWeight, sourceStyle.strokeWeight))
        #expect(mediumStyle.strokeWeight <= max(lightStyle.strokeWeight, sourceStyle.strokeWeight))

        // =====================================================
        // PHASE 6: Generate new font with cloned style
        // =====================================================
        var clonedFont = FontProject(name: "Cloned Bold", family: "ClonedBold", style: "Bold")
        clonedFont.metrics = sourceFont.metrics

        let generator = GlyphGenerator()

        // Generate new characters that weren't in source (numbers, punctuation)
        let newChars: [Character] = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "!", "?", "@"]

        for char in newChars {
            let result = try await generator.generate(
                character: char,
                mode: .fromScratch(style: sourceStyle),
                metrics: clonedFont.metrics,
                settings: .fast
            )
            clonedFont.glyphs[char] = result.glyph
        }

        #expect(clonedFont.glyphs.count == 13)

        // =====================================================
        // PHASE 7: Export cloned font
        // =====================================================
        let exporter = FontExporter()
        let ttfData = try await exporter.export(project: clonedFont)
        #expect(!ttfData.isEmpty)

        // =====================================================
        // VERIFICATION
        // =====================================================
        #expect(clonedFont.glyphs.count == 13)
    }

    @Test("AI style analysis and comparison")
    func styleAnalysisWorkflow() async throws {
        // =====================================================
        // PHASE 1: Create fonts with different styles
        // =====================================================
        let styleEncoder = StyleEncoder()

        // Sans-serif style
        var sansSerifFont = FontProject(name: "Sans", family: "Sans", style: "Regular")
        sansSerifFont.metrics.unitsPerEm = 1000
        sansSerifFont.metrics.capHeight = 700
        sansSerifFont.metrics.xHeight = 500
        for char in "noHOae" {
            sansSerifFont.glyphs[char] = Glyph(
                character: char,
                outline: createSansSerifOutline(for: char),
                advanceWidth: 500,
                leftSideBearing: 50
            )
        }

        // Extract and verify sans-serif style
        let sansStyle = try await styleEncoder.extractStyle(from: sansSerifFont)
        #expect(sansStyle.serifStyle == .sansSerif)

        // =====================================================
        // PHASE 2: Test style interpolation
        // =====================================================
        let styleA = StyleEncoder.FontStyle(
            strokeWeight: 0.3,
            strokeContrast: 0.2,
            xHeightRatio: 0.7,
            widthRatio: 0.8,
            slant: 0,
            serifStyle: .sansSerif,
            roundness: 0.6,
            regularity: 0.9,
            embedding: []
        )

        let styleB = StyleEncoder.FontStyle(
            strokeWeight: 0.7,
            strokeContrast: 0.4,
            xHeightRatio: 0.65,
            widthRatio: 0.9,
            slant: 12,
            serifStyle: .modern,
            roundness: 0.3,
            regularity: 0.7,
            embedding: []
        )

        // Interpolate at different values
        let style25 = styleEncoder.interpolate(styleA, styleB, t: 0.25)
        let style50 = styleEncoder.interpolate(styleA, styleB, t: 0.5)
        let style75 = styleEncoder.interpolate(styleA, styleB, t: 0.75)

        // Verify interpolation is between source values
        #expect(style50.strokeWeight > styleA.strokeWeight)
        #expect(style50.strokeWeight < styleB.strokeWeight)
        #expect(style25.strokeWeight < style50.strokeWeight)
        #expect(style50.strokeWeight < style75.strokeWeight)

        // Serif style should switch at t=0.5
        #expect(style25.serifStyle == .sansSerif)
        #expect(style75.serifStyle == .modern)

        // =====================================================
        // PHASE 3: Style similarity matrix
        // =====================================================

        // Same style = 1.0
        #expect(styleEncoder.similarity(styleA, styleA) > 0.99)

        // Different styles < 1.0
        let similarityAB = styleEncoder.similarity(styleA, styleB)
        #expect(similarityAB < 1.0)
        #expect(similarityAB >= 0)

        // Interpolated style should be more similar to closer endpoint
        let sim25A = styleEncoder.similarity(style25, styleA)
        let sim25B = styleEncoder.similarity(style25, styleB)
        #expect(sim25A > sim25B)

        let sim75A = styleEncoder.similarity(style75, styleA)
        let sim75B = styleEncoder.similarity(style75, styleB)
        #expect(sim75B > sim75A)
    }

    // MARK: - Helper Methods

    private func createSampleGlyphOutline(for char: Character) -> GlyphOutline {
        // Create simple rectangular outline
        GlyphOutline(contours: [
            Contour(points: [
                PathPoint(position: CGPoint(x: 50, y: 0), type: .corner),
                PathPoint(position: CGPoint(x: 450, y: 0), type: .corner),
                PathPoint(position: CGPoint(x: 450, y: 700), type: .corner),
                PathPoint(position: CGPoint(x: 50, y: 700), type: .corner)
            ], isClosed: true)
        ])
    }

    private func createBoldGlyphOutline(for char: Character) -> GlyphOutline {
        // Create thicker rectangular outline (bold-like)
        GlyphOutline(contours: [
            Contour(points: [
                PathPoint(position: CGPoint(x: 30, y: 0), type: .corner),
                PathPoint(position: CGPoint(x: 470, y: 0), type: .corner),
                PathPoint(position: CGPoint(x: 470, y: 700), type: .corner),
                PathPoint(position: CGPoint(x: 30, y: 700), type: .corner)
            ], isClosed: true)
        ])
    }

    private func createLightGlyphOutline(for char: Character) -> GlyphOutline {
        // Create thinner rectangular outline (light-like)
        GlyphOutline(contours: [
            Contour(points: [
                PathPoint(position: CGPoint(x: 100, y: 0), type: .corner),
                PathPoint(position: CGPoint(x: 400, y: 0), type: .corner),
                PathPoint(position: CGPoint(x: 400, y: 700), type: .corner),
                PathPoint(position: CGPoint(x: 100, y: 700), type: .corner)
            ], isClosed: true)
        ])
    }

    private func createSansSerifOutline(for char: Character) -> GlyphOutline {
        // Create outline with smooth curves (sans-serif like)
        GlyphOutline(contours: [
            Contour(points: [
                PathPoint(position: CGPoint(x: 50, y: 0), type: .smooth),
                PathPoint(position: CGPoint(x: 450, y: 0), type: .smooth),
                PathPoint(position: CGPoint(x: 450, y: 700), type: .smooth),
                PathPoint(position: CGPoint(x: 50, y: 700), type: .smooth)
            ], isClosed: true)
        ])
    }
}

// MARK: - Thread-safe Progress Counter

/// Actor for safely recording progress updates from @Sendable callbacks
private actor ProgressCounter {
    private var updates: [(current: Int, total: Int)] = []

    func record(current: Int, total: Int) {
        updates.append((current, total))
    }

    func getUpdates() -> [(current: Int, total: Int)] {
        updates
    }
}

/// Synchronous thread-safe counter for @Sendable progress callbacks.
/// Uses os_unfair_lock for lock-free synchronous access (no await needed).
private final class AtomicProgressCount: @unchecked Sendable {
    private var _value: Int = 0
    private let lock = NSLock()

    var value: Int {
        lock.lock()
        defer { lock.unlock() }
        return _value
    }

    func increment() {
        lock.lock()
        _value += 1
        lock.unlock()
    }
}
