import Foundation

struct Glyph: Identifiable, Codable, Sendable, Equatable {
    let id: UUID
    var character: Character
    var unicodeScalars: [UInt32]
    var outline: GlyphOutline
    var advanceWidth: Int
    var leftSideBearing: Int

    var generatedBy: GenerationSource?
    var styleConfidence: Float?

    init(
        id: UUID = UUID(),
        character: Character,
        unicodeScalars: [UInt32]? = nil,
        outline: GlyphOutline = GlyphOutline(),
        advanceWidth: Int = 500,
        leftSideBearing: Int = 50,
        generatedBy: GenerationSource? = nil,
        styleConfidence: Float? = nil
    ) {
        self.id = id
        self.character = character
        self.unicodeScalars = unicodeScalars ?? character.unicodeScalars.map { $0.value }
        self.outline = outline
        self.advanceWidth = advanceWidth
        self.leftSideBearing = leftSideBearing
        self.generatedBy = generatedBy
        self.styleConfidence = styleConfidence
    }

    var rightSideBearing: Int {
        advanceWidth - leftSideBearing - outline.boundingBox.width
    }

    var name: String {
        if let scalar = character.unicodeScalars.first {
            return String(format: "uni%04X", scalar.value)
        }
        return "unknown"
    }
}

enum GenerationSource: String, Codable, Sendable {
    case manual
    case handwriting
    case aiGenerated
    case styleTransfer
    case imported
    case placeholder  // Geometric placeholder - NOT AI generated
}

extension Character: @retroactive Codable {
    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        let string = try container.decode(String.self)
        guard let character = string.first, string.count == 1 else {
            throw DecodingError.dataCorruptedError(
                in: container,
                debugDescription: "Expected single character"
            )
        }
        self = character
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(String(self))
    }
}
