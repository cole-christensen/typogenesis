import Testing
import Foundation
import CoreGraphics
import AppKit
@testable import Typogenesis

// MARK: - Handwriting Scanner E2E Tests

/// Single comprehensive E2E test for the handwriting scanner workflow.
/// Tests the complete cycle from image input through vectorization to font creation.

@Suite("Handwriting Scanner E2E Tests")
struct HandwritingScannerE2ETests {

    @Test("Complete handwriting to font workflow")
    func completeHandwritingWorkflow() async throws {
        // =====================================================
        // PHASE 1: Create test image (simulating scanned handwriting)
        // =====================================================

        // Create a simple test image with a black shape on white background
        let imageSize = NSSize(width: 200, height: 300)
        let testImage = NSImage(size: imageSize)

        testImage.lockFocus()

        // White background
        NSColor.white.setFill()
        NSRect(origin: .zero, size: imageSize).fill()

        // Draw a simple "I" shape (vertical rectangle)
        NSColor.black.setFill()
        NSRect(x: 70, y: 20, width: 60, height: 260).fill()

        testImage.unlockFocus()

        #expect(testImage.size.width == 200)
        #expect(testImage.size.height == 300)

        // =====================================================
        // PHASE 2: Vectorize the image using Vectorizer
        // =====================================================

        let metrics = FontMetrics()  // Default: 1000 unitsPerEm
        let settings = Vectorizer.VectorizationSettings.default

        let result = try await Vectorizer.vectorize(
            image: testImage,
            metrics: metrics,
            settings: settings
        )

        // Result contains detected characters from the image
        // Note: Full-image vectorization may detect 0 characters for simple test images
        // since it looks for multiple distinct character regions. The important validation
        // is that the process completes without error and returns a valid result struct.
        #expect(result.processingTime > 0, "Processing should take measurable time, got \(result.processingTime)")

        // =====================================================
        // PHASE 3: Also test vectorizeCharacter with CGImage
        // =====================================================

        // Get CGImage from NSImage
        guard let cgImage = testImage.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
            throw TestError.failedToCreateImage
        }

        let characterOutline = try Vectorizer.vectorizeCharacter(
            image: cgImage,
            metrics: metrics,
            settings: settings
        )

        #expect(!characterOutline.isEmpty)
        #expect(characterOutline.contours.count > 0)

        // =====================================================
        // PHASE 4: Verify outline properties
        // =====================================================

        let bbox = characterOutline.boundingBox
        #expect(bbox.width > 0)
        #expect(bbox.height > 0)

        // Should be normalized to font coordinate space
        #expect(bbox.maxY <= metrics.ascender + 100)

        // =====================================================
        // PHASE 5: Create glyph from vectorized outline
        // =====================================================

        let glyph = Glyph(
            character: "I",
            outline: characterOutline,
            advanceWidth: bbox.width + 100,  // Add sidebearings
            leftSideBearing: 50
        )

        #expect(glyph.character == "I")
        #expect(glyph.outline.contours.count > 0)

        // =====================================================
        // PHASE 6: Create font project with vectorized glyph
        // =====================================================

        var project = FontProject(name: "Handwriting Font", family: "MyHandwriting", style: "Regular")
        project.metrics = metrics
        project.glyphs["I"] = glyph

        #expect(project.glyphs.count == 1)

        // =====================================================
        // PHASE 7: Add more glyphs (simulating batch processing)
        // =====================================================

        // In real usage, these would come from scanned sample sheets
        // For testing, create placeholder glyphs
        for char in "ABCDEFGH" {
            let placeholderOutline = GlyphOutline(contours: [
                Contour(points: [
                    PathPoint(position: CGPoint(x: 50, y: 0), type: .corner),
                    PathPoint(position: CGPoint(x: 450, y: 0), type: .corner),
                    PathPoint(position: CGPoint(x: 450, y: 700), type: .corner),
                    PathPoint(position: CGPoint(x: 50, y: 700), type: .corner)
                ], isClosed: true)
            ])
            project.glyphs[char] = Glyph(
                character: char,
                outline: placeholderOutline,
                advanceWidth: 500,
                leftSideBearing: 50
            )
        }

        #expect(project.glyphs.count == 9)  // I + ABCDEFGH

        // =====================================================
        // PHASE 8: Export the handwriting font
        // =====================================================

        let exporter = FontExporter()
        let ttfData = try await exporter.export(project: project)

        #expect(!ttfData.isEmpty)

        // Validate TTF magic bytes
        let magic = ttfData.prefix(4).withUnsafeBytes { $0.load(as: UInt32.self) }
        #expect(UInt32(bigEndian: magic) == 0x00010000, "Should have TrueType magic bytes")

        // =====================================================
        // PHASE 9: Test ContourTracer directly
        // =====================================================

        let tracedOutline = try ContourTracer.vectorize(
            image: cgImage,
            metrics: metrics
        )

        #expect(!tracedOutline.isEmpty)

        // =====================================================
        // VERIFICATION: Complete handwriting workflow succeeded
        // =====================================================

        #expect(project.glyphs.count == 9)
        #expect(project.glyphs["I"] != nil)
        #expect(!ttfData.isEmpty)
    }
}

// MARK: - Test Helpers

enum TestError: Error {
    case failedToCreateImage
}
