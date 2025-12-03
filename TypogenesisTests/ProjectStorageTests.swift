import Testing
import Foundation
import CoreGraphics
@testable import Typogenesis

@Suite("ProjectStorage Tests")
struct ProjectStorageTests {

    @Test("Save and load project round trip")
    func saveLoadRoundTrip() async throws {
        let storage = ProjectStorage()
        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("typogenesis")

        defer {
            try? FileManager.default.removeItem(at: tempURL)
        }

        var project = FontProject(
            name: "Round Trip Test",
            family: "Test Family",
            style: "Bold"
        )

        project.setGlyph(Glyph(character: "A", advanceWidth: 600, leftSideBearing: 50), for: "A")
        project.setGlyph(Glyph(character: "B", advanceWidth: 550, leftSideBearing: 45), for: "B")

        project.metrics = FontMetrics(
            unitsPerEm: 2048,
            ascender: 1800,
            descender: -400,
            xHeight: 1000,
            capHeight: 1400,
            lineGap: 100
        )

        try await storage.save(project, to: tempURL)
        let loaded = try await storage.load(from: tempURL)

        #expect(loaded.name == project.name)
        #expect(loaded.family == project.family)
        #expect(loaded.style == project.style)
        #expect(loaded.glyphs.count == project.glyphs.count)
        #expect(loaded.metrics == project.metrics)

        #expect(loaded.glyph(for: "A")?.advanceWidth == 600)
        #expect(loaded.glyph(for: "B")?.advanceWidth == 550)
    }

    @Test("Save project with kerning")
    func saveWithKerning() async throws {
        let storage = ProjectStorage()
        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("typogenesis")

        defer {
            try? FileManager.default.removeItem(at: tempURL)
        }

        var project = FontProject(name: "Kerning Test", family: "Test", style: "Regular")
        project.kerning = [
            KerningPair(left: "A", right: "V", value: -50),
            KerningPair(left: "T", right: "o", value: -30),
        ]

        try await storage.save(project, to: tempURL)
        let loaded = try await storage.load(from: tempURL)

        #expect(loaded.kerning.count == 2)
        #expect(loaded.kerning.first { $0.left == "A" && $0.right == "V" }?.value == -50)
    }

    @Test("Save project with glyph outlines")
    func saveWithOutlines() async throws {
        let storage = ProjectStorage()
        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("typogenesis")

        defer {
            try? FileManager.default.removeItem(at: tempURL)
        }

        let outline = GlyphOutline(contours: [
            Contour(points: [
                PathPoint(position: CGPoint(x: 100, y: 0), type: .corner),
                PathPoint(position: CGPoint(x: 400, y: 0), type: .corner),
                PathPoint(position: CGPoint(x: 250, y: 700), type: .smooth,
                         controlIn: CGPoint(x: 200, y: 600),
                         controlOut: CGPoint(x: 300, y: 600)),
            ], isClosed: true)
        ])

        var project = FontProject(name: "Outline Test", family: "Test", style: "Regular")
        project.setGlyph(Glyph(character: "A", outline: outline), for: "A")

        try await storage.save(project, to: tempURL)
        let loaded = try await storage.load(from: tempURL)

        let loadedGlyph = loaded.glyph(for: "A")
        #expect(loadedGlyph != nil)
        #expect(loadedGlyph?.outline.contours.count == 1)
        #expect(loadedGlyph?.outline.contours.first?.points.count == 3)

        let smoothPoint = loadedGlyph?.outline.contours.first?.points[2]
        #expect(smoothPoint?.type == .smooth)
        #expect(smoothPoint?.controlIn != nil)
        #expect(smoothPoint?.controlOut != nil)
    }

    @Test("Load nonexistent file throws error")
    func loadNonexistentFile() async {
        let storage = ProjectStorage()
        let fakeURL = URL(fileURLWithPath: "/nonexistent/path/file.typogenesis")

        do {
            _ = try await storage.load(from: fakeURL)
            Issue.record("Expected error to be thrown")
        } catch {
            // Expected
        }
    }
}
