import SwiftUI

@MainActor
final class MetricsEditorViewModel: ObservableObject {
    @Published var editedMetrics: FontMetrics?
    @Published var showPreview = true

    private var loadedMetrics: FontMetrics?

    func loadMetrics(from project: FontProject) {
        if editedMetrics == nil {
            editedMetrics = project.metrics
            loadedMetrics = project.metrics
        }
    }

    func applyChanges() -> FontMetrics? {
        guard let edited = editedMetrics else { return nil }
        guard edited != loadedMetrics else { return nil }
        return edited
    }

    func hasChanges(vs currentMetrics: FontMetrics?) -> Bool {
        guard let current = currentMetrics, let edited = editedMetrics else { return false }
        return current != edited
    }

    func resetToDefaults() {
        editedMetrics = FontMetrics()
    }
}
