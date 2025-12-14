import SwiftUI

struct Sidebar: View {
    @EnvironmentObject var appState: AppState

    var body: some View {
        List(selection: $appState.sidebarSelection) {
            Section("Font") {
                Label("Glyphs", systemImage: "character")
                    .tag(AppState.SidebarItem.glyphs)
                    .accessibilityIdentifier(AccessibilityID.Sidebar.glyphsItem)

                Label("Metrics", systemImage: "ruler")
                    .tag(AppState.SidebarItem.metrics)
                    .accessibilityIdentifier(AccessibilityID.Sidebar.metricsItem)

                Label("Kerning", systemImage: "arrow.left.and.right.text.vertical")
                    .tag(AppState.SidebarItem.kerning)
                    .accessibilityIdentifier(AccessibilityID.Sidebar.kerningItem)

                Label("Preview", systemImage: "eye")
                    .tag(AppState.SidebarItem.preview)
                    .accessibilityIdentifier(AccessibilityID.Sidebar.previewItem)

                Label("Variable Font", systemImage: "slider.horizontal.below.rectangle")
                    .tag(AppState.SidebarItem.variable)
                    .accessibilityIdentifier(AccessibilityID.Sidebar.variableItem)
            }

            Section("Create") {
                Label("AI Generate", systemImage: "wand.and.stars")
                    .tag(AppState.SidebarItem.generate)
                    .accessibilityIdentifier(AccessibilityID.Sidebar.generateItem)

                Label("Handwriting", systemImage: "pencil.and.scribble")
                    .tag(AppState.SidebarItem.handwriting)
                    .accessibilityIdentifier(AccessibilityID.Sidebar.handwritingItem)
            }
        }
        .listStyle(.sidebar)
    }
}

#Preview {
    Sidebar()
        .environmentObject(AppState())
}
