import AppKit
import UniformTypeIdentifiers

/// Protocol for file dialog operations, enabling testability via mock implementations.
@MainActor
protocol FileDialogService {
    func selectFile(types: [UTType], message: String?) async -> URL?
    func selectFiles(types: [UTType], message: String?) async -> [URL]
    func selectSaveLocation(defaultName: String, types: [UTType], message: String?) async -> URL?
    func selectDirectory(message: String?) async -> URL?
}

/// Production implementation using NSOpenPanel / NSSavePanel.
@MainActor
final class NSPanelFileDialogService: FileDialogService {
    func selectFile(types: [UTType], message: String?) async -> URL? {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = types
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        if let message { panel.message = message }
        guard panel.runModal() == .OK else { return nil }
        return panel.url
    }

    func selectFiles(types: [UTType], message: String?) async -> [URL] {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = types
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = true
        if let message { panel.message = message }
        guard panel.runModal() == .OK else { return [] }
        return panel.urls
    }

    func selectSaveLocation(defaultName: String, types: [UTType], message: String?) async -> URL? {
        let panel = NSSavePanel()
        panel.nameFieldStringValue = defaultName
        panel.allowedContentTypes = types
        if let message { panel.message = message }
        guard panel.runModal() == .OK else { return nil }
        return panel.url
    }

    func selectDirectory(message: String?) async -> URL? {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.canCreateDirectories = true
        panel.allowsMultipleSelection = false
        if let message { panel.message = message }
        guard panel.runModal() == .OK else { return nil }
        return panel.url
    }
}
