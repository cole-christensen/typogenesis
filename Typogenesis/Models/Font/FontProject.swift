import Foundation

struct FontProject: Identifiable, Codable, Sendable {
    let id: UUID
    var name: String
    var family: String
    var style: String

    var metrics: FontMetrics
    var glyphs: [Character: Glyph]
    var kerning: [KerningPair]

    var metadata: FontMetadata
    var settings: ProjectSettings

    var characterSet: CharacterSet {
        CharacterSet(charactersIn: String(glyphs.keys))
    }

    init(
        id: UUID = UUID(),
        name: String,
        family: String,
        style: String,
        metrics: FontMetrics = FontMetrics(),
        glyphs: [Character: Glyph] = [:],
        kerning: [KerningPair] = [],
        metadata: FontMetadata = FontMetadata(),
        settings: ProjectSettings = ProjectSettings()
    ) {
        self.id = id
        self.name = name
        self.family = family
        self.style = style
        self.metrics = metrics
        self.glyphs = glyphs
        self.kerning = kerning
        self.metadata = metadata
        self.settings = settings
    }

    func glyph(for character: Character) -> Glyph? {
        glyphs[character]
    }

    func glyph(for string: String) -> Glyph? {
        guard let char = string.first else { return nil }
        return glyphs[char]
    }

    mutating func setGlyph(_ glyph: Glyph, for character: Character) {
        glyphs[character] = glyph
    }

    mutating func removeGlyph(for character: Character) {
        glyphs.removeValue(forKey: character)
    }
}

struct FontMetrics: Codable, Sendable, Equatable {
    var unitsPerEm: Int
    var ascender: Int
    var descender: Int
    var xHeight: Int
    var capHeight: Int
    var lineGap: Int

    init(
        unitsPerEm: Int = 1000,
        ascender: Int = 800,
        descender: Int = -200,
        xHeight: Int = 500,
        capHeight: Int = 700,
        lineGap: Int = 90
    ) {
        self.unitsPerEm = unitsPerEm
        self.ascender = ascender
        self.descender = descender
        self.xHeight = xHeight
        self.capHeight = capHeight
        self.lineGap = lineGap
    }

    var baseline: Int { 0 }

    var totalHeight: Int {
        ascender - descender
    }
}

struct FontMetadata: Codable, Sendable {
    var copyright: String
    var designer: String
    var designerURL: String
    var manufacturer: String
    var manufacturerURL: String
    var license: String
    var licenseURL: String
    var version: String
    var uniqueID: String
    var description: String

    init(
        copyright: String = "",
        designer: String = "",
        designerURL: String = "",
        manufacturer: String = "",
        manufacturerURL: String = "",
        license: String = "",
        licenseURL: String = "",
        version: String = "1.0",
        uniqueID: String = UUID().uuidString,
        description: String = ""
    ) {
        self.copyright = copyright
        self.designer = designer
        self.designerURL = designerURL
        self.manufacturer = manufacturer
        self.manufacturerURL = manufacturerURL
        self.license = license
        self.licenseURL = licenseURL
        self.version = version
        self.uniqueID = uniqueID
        self.description = description
    }
}
