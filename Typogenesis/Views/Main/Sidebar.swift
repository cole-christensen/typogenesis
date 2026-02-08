import SwiftUI

struct Sidebar: View {
    @EnvironmentObject var appState: AppState

    var body: some View {
        VStack(spacing: 0) {
            // Header with project name
            if let project = appState.currentProject {
                HStack {
                    Image(systemName: "textformat")
                        .foregroundColor(.accentColor)
                    Text(project.name)
                        .fontWeight(.semibold)
                        .lineLimit(1)
                    Spacer()
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 12)
                .background(Color(nsColor: .controlBackgroundColor))

                Divider()
            }

            // Navigation List
            List(selection: $appState.sidebarSelection) {
                Section("Font") {
                    SidebarItem(
                        title: "Glyphs",
                        icon: "character",
                        tag: .glyphs,
                        accessibilityID: AccessibilityID.Sidebar.glyphsItem
                    )

                    SidebarItem(
                        title: "Metrics",
                        icon: "ruler",
                        tag: .metrics,
                        accessibilityID: AccessibilityID.Sidebar.metricsItem
                    )

                    SidebarItem(
                        title: "Kerning",
                        icon: "arrow.left.and.right.text.vertical",
                        tag: .kerning,
                        accessibilityID: AccessibilityID.Sidebar.kerningItem
                    )

                    SidebarItem(
                        title: "Preview",
                        icon: "eye",
                        tag: .preview,
                        accessibilityID: AccessibilityID.Sidebar.previewItem
                    )

                    SidebarItem(
                        title: "Variable",
                        icon: "slider.horizontal.below.rectangle",
                        tag: .variable,
                        accessibilityID: AccessibilityID.Sidebar.variableItem
                    )
                }

                Section("Create") {
                    SidebarItem(
                        title: "AI Generate",
                        icon: "wand.and.stars",
                        tag: .generate,
                        accessibilityID: AccessibilityID.Sidebar.generateItem
                    )

                    SidebarItem(
                        title: "Handwriting",
                        icon: "pencil.and.scribble",
                        tag: .handwriting,
                        accessibilityID: AccessibilityID.Sidebar.handwritingItem
                    )

                    SidebarItem(
                        title: "Clone Font",
                        icon: "doc.on.doc",
                        tag: .clone,
                        accessibilityID: AccessibilityID.Sidebar.cloneItem
                    )
                }
            }
            .listStyle(.sidebar)
        }
    }
}

struct SidebarItem: View {
    let title: String
    let icon: String
    let tag: AppState.SidebarItem
    let accessibilityID: String

    var body: some View {
        Label(title, systemImage: icon)
            .tag(tag)
            .accessibilityIdentifier(accessibilityID)
    }
}

#Preview {
    Sidebar()
        .environmentObject(AppState())
}
