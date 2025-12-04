import SwiftUI
import Combine

@MainActor
final class AppState: ObservableObject {
    @Published var currentProject: FontProject?
    @Published var recentProjects: [URL] = []
    @Published var showExportSheet = false
    @Published var showNewProjectSheet = false
    @Published var selectedGlyph: Character?
    @Published var sidebarSelection: SidebarItem? = .glyphs

    enum SidebarItem: Hashable {
        case glyphs
        case metrics
        case kerning
        case generate
        case handwriting
    }

    private let projectStorage = ProjectStorage()

    init() {
        loadRecentProjects()
    }

    func createNewProject() {
        let project = FontProject(
            name: "Untitled Font",
            family: "Untitled",
            style: "Regular"
        )
        currentProject = project
        selectedGlyph = nil
    }

    func openProject() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.init(filenameExtension: "typogenesis")!]
        panel.allowsMultipleSelection = false

        if panel.runModal() == .OK, let url = panel.url {
            Task {
                do {
                    let project = try await projectStorage.load(from: url)
                    currentProject = project
                    addToRecentProjects(url)
                } catch {
                    print("Failed to open project: \(error)")
                }
            }
        }
    }

    func saveProject() {
        guard let project = currentProject else { return }

        let panel = NSSavePanel()
        panel.allowedContentTypes = [.init(filenameExtension: "typogenesis")!]
        panel.nameFieldStringValue = "\(project.name).typogenesis"

        if panel.runModal() == .OK, let url = panel.url {
            Task {
                do {
                    try await projectStorage.save(project, to: url)
                    addToRecentProjects(url)
                } catch {
                    print("Failed to save project: \(error)")
                }
            }
        }
    }

    private func loadRecentProjects() {
        if let paths = UserDefaults.standard.array(forKey: "recentProjects") as? [String] {
            recentProjects = paths.map { URL(fileURLWithPath: $0) }
        }
    }

    private func addToRecentProjects(_ url: URL) {
        recentProjects.removeAll { $0 == url }
        recentProjects.insert(url, at: 0)
        if recentProjects.count > 10 {
            recentProjects = Array(recentProjects.prefix(10))
        }
        UserDefaults.standard.set(recentProjects.map { $0.path }, forKey: "recentProjects")
    }

    // MARK: - Glyph Management

    func addGlyph(for character: Character) {
        guard currentProject != nil else { return }

        let glyph = Glyph(
            character: character,
            advanceWidth: currentProject!.metrics.unitsPerEm / 2,
            leftSideBearing: currentProject!.metrics.unitsPerEm / 20
        )
        currentProject!.setGlyph(glyph, for: character)
    }

    func updateGlyph(_ glyph: Glyph) {
        guard currentProject != nil else { return }
        currentProject!.setGlyph(glyph, for: glyph.character)
    }

    func deleteGlyph(for character: Character) {
        guard currentProject != nil else { return }
        currentProject!.removeGlyph(for: character)
        if selectedGlyph == character {
            selectedGlyph = nil
        }
    }
}
