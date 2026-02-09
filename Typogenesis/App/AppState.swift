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
    @Published var showImportSheet = false
    @Published var projectError: String?
    @Published var showProjectError = false

    /// URL of the currently open project file (nil for unsaved projects)
    @Published var projectURL: URL?

    enum SidebarItem: Hashable, CaseIterable {
        case glyphs
        case metrics
        case kerning
        case preview
        case variable
        case generate
        case handwriting
        case clone
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
        projectURL = nil
        selectedGlyph = nil
    }

    func openProject() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.init(filenameExtension: "typogenesis")!]
        panel.allowsMultipleSelection = false

        if panel.runModal() == .OK, let url = panel.url {
            loadProject(from: url)
        }
    }

    func loadProject(from url: URL) {
        Task {
            do {
                let project = try await projectStorage.load(from: url)
                currentProject = project
                projectURL = url
                selectedGlyph = nil
                addToRecentProjects(url)
            } catch {
                projectError = "Failed to open project: \(error.localizedDescription)"
                showProjectError = true
            }
        }
    }

    /// Save project. If project has a known URL, saves in place; otherwise shows Save As dialog.
    func saveProject() {
        guard let project = currentProject else { return }

        if let url = projectURL {
            // Save in place
            Task {
                do {
                    try await projectStorage.save(project, to: url)
                } catch {
                    projectError = "Failed to save project: \(error.localizedDescription)"
                    showProjectError = true
                }
            }
        } else {
            saveProjectAs()
        }
    }

    /// Always show Save As dialog for choosing a new file location.
    func saveProjectAs() {
        guard let project = currentProject else { return }

        let panel = NSSavePanel()
        panel.allowedContentTypes = [.init(filenameExtension: "typogenesis")!]
        panel.nameFieldStringValue = "\(project.name).typogenesis"

        if panel.runModal() == .OK, let url = panel.url {
            Task {
                do {
                    try await projectStorage.save(project, to: url)
                    projectURL = url
                    addToRecentProjects(url)
                } catch {
                    projectURL = nil
                    projectError = "Failed to save project: \(error.localizedDescription)"
                    showProjectError = true
                }
            }
        }
    }

    private func loadRecentProjects() {
        if let paths = UserDefaults.standard.array(forKey: "recentProjects") as? [String] {
            recentProjects = paths.map { URL(fileURLWithPath: $0) }
        }
    }

    /// Adds a URL to the recent projects list. Internal access required by ImportFontSheet.
    func addToRecentProjects(_ url: URL) {
        recentProjects.removeAll { $0 == url }
        recentProjects.insert(url, at: 0)
        if recentProjects.count > 10 {
            recentProjects = Array(recentProjects.prefix(10))
        }
        UserDefaults.standard.set(recentProjects.map { $0.path }, forKey: "recentProjects")
    }

    // MARK: - Glyph Management

    func addGlyph(for character: Character) {
        guard var project = currentProject else { return }

        let glyph = Glyph(
            character: character,
            advanceWidth: project.metrics.unitsPerEm / 2,
            leftSideBearing: project.metrics.unitsPerEm / 20
        )
        project.setGlyph(glyph, for: character)
        currentProject = project
    }

    func updateGlyph(_ glyph: Glyph) {
        guard var project = currentProject else { return }
        project.setGlyph(glyph, for: glyph.character)
        currentProject = project
    }

    func deleteGlyph(for character: Character) {
        guard var project = currentProject else { return }
        project.removeGlyph(for: character)
        currentProject = project
        if selectedGlyph == character {
            selectedGlyph = nil
        }
    }

    // MARK: - Font Import

    func importFont() {
        showImportSheet = true
    }

}
