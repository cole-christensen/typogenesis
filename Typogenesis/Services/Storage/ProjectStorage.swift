import Foundation

/// Service for saving and loading FontProject files to disk using JSON serialization.
actor ProjectStorage {
    private let encoder = JSONEncoder()
    private let decoder = JSONDecoder()

    init() {
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    }

    func save(_ project: FontProject, to url: URL) async throws {
        let data = try encoder.encode(project)
        try data.write(to: url, options: .atomic)
    }

    func load(from url: URL) async throws -> FontProject {
        let data = try Data(contentsOf: url)
        return try decoder.decode(FontProject.self, from: data)
    }
}
