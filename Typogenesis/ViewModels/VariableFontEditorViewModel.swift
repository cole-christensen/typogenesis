import SwiftUI

@MainActor
final class VariableFontEditorViewModel: ObservableObject {
    // MARK: - Published Properties

    /// Local copy of variable font config, synced back to project via Pattern A (.onChange)
    @Published var variableConfig = VariableFontConfig()
    @Published var selectedMasterID: UUID?
    @Published var showAddAxisSheet = false
    @Published var showAddMasterSheet = false
    @Published var showAddInstanceSheet = false

    // MARK: - Loading

    func loadConfig(from project: FontProject) {
        variableConfig = project.variableConfig
    }

    // MARK: - Variable Font Toggle

    func toggleVariableFont(enabled: Bool) {
        variableConfig.isVariableFont = enabled
        if enabled && variableConfig.axes.isEmpty {
            variableConfig.axes = [.weight]
        }
    }

    // MARK: - Axis Management

    func addAxis(_ axis: VariationAxis) {
        variableConfig.axes.append(axis)
    }

    func removeAxis(id: UUID) {
        variableConfig.axes.removeAll { $0.id == id }
    }

    // MARK: - Master Management

    func addMaster(_ master: FontMaster) {
        variableConfig.masters.append(master)
    }

    func removeMaster(id: UUID) {
        variableConfig.masters.removeAll { $0.id == id }
    }

    // MARK: - Instance Management

    func addInstance(_ instance: NamedInstance) {
        variableConfig.instances.append(instance)
    }

    func removeInstance(id: UUID) {
        variableConfig.instances.removeAll { $0.id == id }
    }
}
