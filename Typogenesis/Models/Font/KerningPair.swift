import Foundation

struct KerningPair: Identifiable, Codable, Sendable, Equatable, Hashable {
    let id: UUID
    var left: Character
    var right: Character
    var value: Int

    init(id: UUID = UUID(), left: Character, right: Character, value: Int) {
        self.id = id
        self.left = left
        self.right = right
        self.value = value
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(left)
        hasher.combine(right)
    }
}

struct KerningClass: Identifiable, Codable, Sendable {
    let id: UUID
    var name: String
    var characters: [Character]

    init(id: UUID = UUID(), name: String, characters: [Character] = []) {
        self.id = id
        self.name = name
        self.characters = characters
    }
}

struct KerningTable: Codable, Sendable {
    var pairs: [KerningPair]
    var leftClasses: [KerningClass]
    var rightClasses: [KerningClass]
    var classKerning: [String: [String: Int]]

    init(
        pairs: [KerningPair] = [],
        leftClasses: [KerningClass] = [],
        rightClasses: [KerningClass] = [],
        classKerning: [String: [String: Int]] = [:]
    ) {
        self.pairs = pairs
        self.leftClasses = leftClasses
        self.rightClasses = rightClasses
        self.classKerning = classKerning
    }

    func kerning(for left: Character, right: Character) -> Int {
        if let pair = pairs.first(where: { $0.left == left && $0.right == right }) {
            return pair.value
        }

        let leftClassName = leftClasses.first { $0.characters.contains(left) }?.name
        let rightClassName = rightClasses.first { $0.characters.contains(right) }?.name

        if let leftClass = leftClassName,
           let rightClass = rightClassName,
           let value = classKerning[leftClass]?[rightClass] {
            return value
        }

        return 0
    }
}
