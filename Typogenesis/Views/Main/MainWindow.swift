import SwiftUI

struct MainWindow: View {
    @EnvironmentObject var appState: AppState

    var body: some View {
        NavigationSplitView {
            Sidebar()
        } detail: {
            if appState.currentProject != nil {
                ContentView()
            } else {
                WelcomeView()
            }
        }
        .frame(minWidth: 1000, minHeight: 700)
        .sheet(isPresented: $appState.showExportSheet) {
            if appState.currentProject != nil {
                ExportSheet()
            }
        }
    }
}

struct ContentView: View {
    @EnvironmentObject var appState: AppState

    var body: some View {
        HSplitView {
            mainContent
                .frame(minWidth: 500)

            Inspector()
                .frame(width: 280)
        }
    }

    @ViewBuilder
    var mainContent: some View {
        switch appState.sidebarSelection {
        case .glyphs:
            GlyphEditorContainer()
        case .metrics:
            MetricsEditorPlaceholder()
        case .kerning:
            KerningEditorPlaceholder()
        case .generate:
            GenerateViewPlaceholder()
        case .handwriting:
            HandwritingScannerPlaceholder()
        case .none:
            Text("Select an item from the sidebar")
                .foregroundColor(.secondary)
        }
    }
}

struct WelcomeView: View {
    @EnvironmentObject var appState: AppState

    var body: some View {
        VStack(spacing: 24) {
            Image(systemName: "textformat")
                .font(.system(size: 64))
                .foregroundColor(.accentColor)

            Text("Typogenesis")
                .font(.largeTitle)
                .fontWeight(.bold)

            Text("AI-Powered Font Creation")
                .font(.title2)
                .foregroundColor(.secondary)

            VStack(spacing: 12) {
                Button("Create New Font") {
                    appState.createNewProject()
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)

                Button("Open Existing Project...") {
                    appState.openProject()
                }
                .buttonStyle(.bordered)
                .controlSize(.large)
            }
            .padding(.top)

            if !appState.recentProjects.isEmpty {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Recent Projects")
                        .font(.headline)
                        .padding(.top)

                    ForEach(appState.recentProjects.prefix(5), id: \.self) { url in
                        Button(url.lastPathComponent) {
                            // Load project
                        }
                        .buttonStyle(.plain)
                        .foregroundColor(.accentColor)
                    }
                }
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color(nsColor: .windowBackgroundColor))
    }
}

struct GlyphEditorContainer: View {
    @EnvironmentObject var appState: AppState

    var body: some View {
        if let project = appState.currentProject {
            VSplitView {
                GlyphGrid(project: project, selectedGlyph: $appState.selectedGlyph)
                    .frame(minHeight: 150)

                if let character = appState.selectedGlyph,
                   let glyph = project.glyph(for: character) {
                    GlyphCanvas(glyph: glyph, metrics: project.metrics)
                        .frame(minHeight: 400)
                } else {
                    Text("Select a glyph to edit")
                        .foregroundColor(.secondary)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                }
            }
        }
    }
}

struct MetricsEditorPlaceholder: View {
    var body: some View {
        Text("Metrics Editor")
            .foregroundColor(.secondary)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

struct KerningEditorPlaceholder: View {
    var body: some View {
        Text("Kerning Editor")
            .foregroundColor(.secondary)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

struct GenerateViewPlaceholder: View {
    var body: some View {
        Text("AI Generation")
            .foregroundColor(.secondary)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

struct HandwritingScannerPlaceholder: View {
    var body: some View {
        Text("Handwriting Scanner")
            .foregroundColor(.secondary)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

#Preview {
    MainWindow()
        .environmentObject(AppState())
}
