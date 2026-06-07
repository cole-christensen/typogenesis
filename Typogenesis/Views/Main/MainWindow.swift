import SwiftUI

struct MainWindow: View {
    @EnvironmentObject var appState: AppState

    var body: some View {
        Group {
            if appState.currentProject != nil {
                ThreeColumnLayout()
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
        .alert("Project Error", isPresented: $appState.showProjectError) {
            Button("OK", role: .cancel) { }
        } message: {
            Text(appState.projectError ?? "Unknown error")
        }
    }
}

// MARK: - Three Column Layout

/// Content-driven macOS three-column layout
struct ThreeColumnLayout: View {
    @EnvironmentObject var appState: AppState

    var body: some View {
        HStack(spacing: 0) {
            // LEFT: Navigation Sidebar
            Sidebar()
                .frame(minWidth: 200, idealWidth: 250, maxWidth: 350)
                .background(Color(nsColor: .controlBackgroundColor))

            Divider()

            // CENTER: Main Content Area - expands to fill available space
            MainContentArea()
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .layoutPriority(1)
                .background(Color(nsColor: .textBackgroundColor))

            Divider()

            // RIGHT: Inspector Panel
            Inspector()
                .frame(minWidth: 200, idealWidth: 280, maxWidth: 400)
                .background(Color(nsColor: .controlBackgroundColor))
        }
    }
}

// MARK: - Main Content Area

struct MainContentArea: View {
    @EnvironmentObject var appState: AppState

    var body: some View {
        Group {
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
            case .clone:
                CloneWizard()
            case .none:
                EmptyContentView()
            }
        }
    }
}

struct EmptyContentView: View {
    var body: some View {
        VStack(spacing: 16) {
            Image(systemName: "arrow.left")
                .font(.system(size: 48))
                .foregroundColor(.secondary)
            Text("Select an item from the sidebar")
                .font(.title3)
                .foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color(nsColor: .textBackgroundColor))
    }
}

// MARK: - Glyph Editor Container

struct GlyphEditorContainer: View {
    @EnvironmentObject var appState: AppState
    @State private var showAddGlyphSheet = false

    var body: some View {
        if let project = appState.currentProject {
            VSplitView {
                GlyphGrid(
                    project: project,
                    selectedGlyph: $appState.selectedGlyph,
                    onAddGlyph: { showAddGlyphSheet = true },
                    onDeleteGlyph: { character in
                        appState.deleteGlyph(for: character)
                    }
                )

                if let character = appState.selectedGlyph,
                   let glyph = project.glyph(for: character) {
                    GlyphEditorSubview(glyph: glyph, metrics: project.metrics)
                        .id(character)
                } else {
                    GlyphEditorPlaceholder(onAddGlyph: { showAddGlyphSheet = true })
                }
            }
            .sheet(isPresented: $showAddGlyphSheet) {
                AddGlyphSheet(existingGlyphs: Set(project.glyphs.keys)) { character in
                    appState.addGlyph(for: character)
                    appState.selectedGlyph = character
                }
            }
        }
    }
}

/// Subview that owns its own GlyphEditorViewModel via @StateObject.
/// Using .id(character) on the parent forces recreation when the glyph changes.
struct GlyphEditorSubview: View {
    @EnvironmentObject var appState: AppState
    @StateObject private var viewModel: GlyphEditorViewModel
    let metrics: FontMetrics

    init(glyph: Glyph, metrics: FontMetrics) {
        _viewModel = StateObject(wrappedValue: GlyphEditorViewModel(glyph: glyph))
        self.metrics = metrics
    }

    var body: some View {
        InteractiveGlyphCanvas(
            viewModel: viewModel,
            metrics: metrics
        )
        .onChange(of: viewModel.glyph) { _, newGlyph in
            appState.updateGlyph(newGlyph)
        }
    }
}

struct GlyphEditorPlaceholder: View {
    var onAddGlyph: () -> Void

    var body: some View {
        VStack(spacing: 16) {
            Image(systemName: "square.on.square.dashed")
                .font(.system(size: 48))
                .foregroundColor(.secondary)
            Text("Select a glyph to edit")
                .font(.title3)
                .foregroundColor(.secondary)
            Button("Add New Glyph") {
                onAddGlyph()
            }
            .buttonStyle(.bordered)
            .accessibilityIdentifier(AccessibilityID.GlyphGrid.addGlyphButton)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color(nsColor: .textBackgroundColor))
    }
}

// MARK: - Welcome View

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
                .accessibilityIdentifier(AccessibilityID.Welcome.importFontButton)

                Button("Open Existing Project...") {
                    appState.openProject()
                }
                .buttonStyle(.bordered)
                .controlSize(.large)
                .accessibilityIdentifier(AccessibilityID.Welcome.openProjectButton)
            }
            .padding(.top)

            if !appState.recentProjects.isEmpty {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Recent Projects")
                        .font(.headline)
                        .padding(.top)

                    ForEach(appState.recentProjects.prefix(5), id: \.self) { url in
                        Button(url.lastPathComponent) {
                            appState.loadProject(from: url)
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

#Preview {
    MainWindow()
        .environmentObject(AppState())
}
