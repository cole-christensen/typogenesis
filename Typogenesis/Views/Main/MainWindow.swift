import SwiftUI

struct MainWindow: View {
    @EnvironmentObject var appState: AppState

    var body: some View {
        Group {
            if appState.currentProject != nil {
                NavigationSplitView {
                    Sidebar()
                        .navigationSplitViewColumnWidth(min: 180, ideal: 220, max: 280)
                } detail: {
                    ContentView()
                }
                .navigationSplitViewStyle(.balanced)
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
        .sheet(isPresented: $appState.showImportSheet) {
            ImportFontSheet()
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
            MetricsEditor()
        case .kerning:
            KerningEditor()
        case .preview:
            if let project = appState.currentProject {
                FontPreviewPanel(project: project)
            }
        case .variable:
            VariableFontEditor()
        case .generate:
            GenerateView()
        case .handwriting:
            HandwritingScanner()
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
                .accessibilityIdentifier(AccessibilityID.Welcome.createNewFontButton)

                Button("Import Font (.ttf/.otf)...") {
                    appState.importFont()
                }
                .buttonStyle(.bordered)
                .controlSize(.large)
                .disabled(appState.isImporting)
                .accessibilityIdentifier(AccessibilityID.Welcome.importFontButton)

                Button("Open Existing Project...") {
                    appState.openProject()
                }
                .buttonStyle(.bordered)
                .controlSize(.large)
                .accessibilityIdentifier(AccessibilityID.Welcome.openProjectButton)
            }
            .padding(.top)

            if appState.isImporting {
                HStack(spacing: 8) {
                    ProgressView()
                        .scaleEffect(0.8)
                    Text("Importing font...")
                        .foregroundColor(.secondary)
                }
                .accessibilityIdentifier(AccessibilityID.Welcome.importingIndicator)
            }

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
        .alert("Import Error", isPresented: $appState.showImportError) {
            Button("OK") {}
        } message: {
            Text(appState.importError ?? "Unknown error")
        }
    }
}

struct GlyphEditorContainer: View {
    @EnvironmentObject var appState: AppState
    @State private var editorViewModel: GlyphEditorViewModel?
    @State private var showAddGlyphSheet = false

    var body: some View {
        if let project = appState.currentProject {
            VSplitView {
                GlyphGrid(
                    project: project,
                    selectedGlyph: $appState.selectedGlyph,
                    onAddGlyph: { showAddGlyphSheet = true }
                )
                .frame(minHeight: 150)

                if let character = appState.selectedGlyph,
                   let glyph = project.glyph(for: character) {
                    InteractiveGlyphCanvas(
                        viewModel: editorViewModel ?? GlyphEditorViewModel(glyph: glyph),
                        metrics: project.metrics
                    )
                    .frame(minHeight: 400)
                    .onChange(of: appState.selectedGlyph) { _, newChar in
                        if let char = newChar, let g = project.glyph(for: char) {
                            editorViewModel = GlyphEditorViewModel(glyph: g)
                        }
                    }
                    .onChange(of: editorViewModel?.glyph) { _, newGlyph in
                        if let glyph = newGlyph {
                            appState.updateGlyph(glyph)
                        }
                    }
                    .onAppear {
                        editorViewModel = GlyphEditorViewModel(glyph: glyph)
                    }
                } else {
                    VStack(spacing: 16) {
                        Text("Select a glyph to edit")
                            .foregroundColor(.secondary)

                        Button("Add New Glyph") {
                            showAddGlyphSheet = true
                        }
                        .buttonStyle(.bordered)
                        .accessibilityIdentifier(AccessibilityID.GlyphGrid.addGlyphButton)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                }
            }
            .sheet(isPresented: $showAddGlyphSheet) {
                AddGlyphSheet { character in
                    appState.addGlyph(for: character)
                    appState.selectedGlyph = character
                }
            }
        }
    }
}


#Preview {
    MainWindow()
        .environmentObject(AppState())
}
