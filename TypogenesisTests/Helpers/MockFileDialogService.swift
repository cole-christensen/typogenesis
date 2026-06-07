import Foundation
import UniformTypeIdentifiers
@testable import Typogenesis

@MainActor
final class MockFileDialogService: FileDialogService {
    var selectFileResult: URL?
    var selectFilesResult: [URL] = []
    var selectSaveLocationResult: URL?
    var selectDirectoryResult: URL?

    var selectFileCalled = false
    var selectFileCallCount = 0
    var selectFilesCalled = false
    var selectFilesCallCount = 0
    var selectSaveLocationCalled = false
    var selectSaveLocationCallCount = 0
    var selectDirectoryCalled = false
    var selectDirectoryCallCount = 0

    var lastSaveDefaultName: String?
    var lastSaveTypes: [UTType]?
    var lastTypes: [UTType]?
    var lastMessage: String?

    // Alias for backward compatibility
    var urlToReturn: URL? {
        get { selectFileResult }
        set { selectFileResult = newValue }
    }

    func selectFile(types: [UTType], message: String?) async -> URL? {
        selectFileCalled = true
        selectFileCallCount += 1
        lastTypes = types
        lastMessage = message
        return selectFileResult
    }

    func selectFiles(types: [UTType], message: String?) async -> [URL] {
        selectFilesCalled = true
        selectFilesCallCount += 1
        lastMessage = message
        return selectFilesResult
    }

    func selectSaveLocation(defaultName: String, types: [UTType], message: String?) async -> URL? {
        selectSaveLocationCalled = true
        selectSaveLocationCallCount += 1
        lastSaveDefaultName = defaultName
        lastSaveTypes = types
        lastMessage = message
        return selectSaveLocationResult
    }

    func selectDirectory(message: String?) async -> URL? {
        selectDirectoryCalled = true
        selectDirectoryCallCount += 1
        lastMessage = message
        return selectDirectoryResult
    }
}
