import Foundation

struct ProjectSettings: Codable, Sendable {
    var gridSize: Int
    var showMetricsLines: Bool
    var showGrid: Bool
    var snapToGrid: Bool
    var snapToMetrics: Bool
    var autoKerning: Bool
    var aiAssistEnabled: Bool

    init(
        gridSize: Int = 50,
        showMetricsLines: Bool = true,
        showGrid: Bool = true,
        snapToGrid: Bool = true,
        snapToMetrics: Bool = true,
        autoKerning: Bool = true,
        aiAssistEnabled: Bool = true
    ) {
        self.gridSize = gridSize
        self.showMetricsLines = showMetricsLines
        self.showGrid = showGrid
        self.snapToGrid = snapToGrid
        self.snapToMetrics = snapToMetrics
        self.autoKerning = autoKerning
        self.aiAssistEnabled = aiAssistEnabled
    }
}
