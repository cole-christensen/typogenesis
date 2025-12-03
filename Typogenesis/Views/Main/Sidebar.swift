import SwiftUI

struct Sidebar: View {
    @EnvironmentObject var appState: AppState

    var body: some View {
        List(selection: $appState.sidebarSelection) {
            if appState.currentProject != nil {
                Section("Font") {
                    Label("Glyphs", systemImage: "character")
                        .tag(AppState.SidebarItem.glyphs)

                    Label("Metrics", systemImage: "ruler")
                        .tag(AppState.SidebarItem.metrics)

                    Label("Kerning", systemImage: "arrow.left.and.right.text.vertical")
                        .tag(AppState.SidebarItem.kerning)
                }

                Section("Create") {
                    Label("AI Generate", systemImage: "wand.and.stars")
                        .tag(AppState.SidebarItem.generate)

                    Label("Handwriting", systemImage: "pencil.and.scribble")
                        .tag(AppState.SidebarItem.handwriting)
                }
            } else {
                Text("No project open")
                    .foregroundColor(.secondary)
            }
        }
        .listStyle(.sidebar)
        .frame(minWidth: 180)
    }
}

#Preview {
    Sidebar()
        .environmentObject(AppState())
}
