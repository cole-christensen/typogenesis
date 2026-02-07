import SwiftUI

struct SettingsView: View {
    @EnvironmentObject var appState: AppState
    @ObservedObject private var modelManager = ModelManager.shared

    @AppStorage("gridSize") private var gridSize = 50
    @AppStorage("showGrid") private var showGrid = true
    @AppStorage("showMetrics") private var showMetrics = true
    @AppStorage("snapToGrid") private var snapToGrid = true
    @AppStorage("aiAssistEnabled") private var aiAssistEnabled = true
    @AppStorage("showWelcomeOnLaunch") private var showWelcomeOnLaunch = true
    @AppStorage("checkForUpdatesAutomatically") private var checkForUpdatesAutomatically = true
    @AppStorage("defaultUnitsPerEm") private var defaultUnitsPerEm = 1000
    @AppStorage("autoSuggestKerning") private var autoSuggestKerning = true
    @AppStorage("styleConsistencyWarnings") private var styleConsistencyWarnings = true

    @State private var showingModelDownloadAlert = false

    var body: some View {
        TabView {
            generalSettings
                .tabItem {
                    Label("General", systemImage: "gear")
                }

            editorSettings
                .tabItem {
                    Label("Editor", systemImage: "pencil")
                }

            aiSettings
                .tabItem {
                    Label("AI", systemImage: "cpu")
                }
        }
        .frame(width: 450, height: 300)
    }

    private var generalSettings: some View {
        Form {
            Section("Application") {
                Toggle("Show welcome screen on launch", isOn: $showWelcomeOnLaunch)
                Toggle("Check for updates automatically", isOn: $checkForUpdatesAutomatically)
            }
        }
        .formStyle(.grouped)
        .padding()
    }

    private var editorSettings: some View {
        Form {
            Section("Canvas") {
                Stepper("Grid Size: \(gridSize)", value: $gridSize, in: 10...100, step: 10)
                Toggle("Show Grid", isOn: $showGrid)
                Toggle("Show Metrics", isOn: $showMetrics)
                Toggle("Snap to Grid", isOn: $snapToGrid)
            }

            Section("Defaults") {
                Picker("Default Units/Em", selection: $defaultUnitsPerEm) {
                    Text("1000").tag(1000)
                    Text("2048").tag(2048)
                }
            }
        }
        .formStyle(.grouped)
        .padding()
    }

    private var aiSettings: some View {
        Form {
            Section("AI Features") {
                Toggle("Enable AI Assistance", isOn: $aiAssistEnabled)
                Toggle("Auto-suggest kerning", isOn: $autoSuggestKerning)
                Toggle("Style consistency warnings", isOn: $styleConsistencyWarnings)
            }

            Section("Models") {
                LabeledContent("Glyph Generation", value: modelManager.glyphDiffusionStatus.displayText)
                LabeledContent("Style Encoder", value: modelManager.styleEncoderStatus.displayText)
                LabeledContent("Kerning Net", value: modelManager.kerningNetStatus.displayText)

                Button("Download Models...") {
                    showingModelDownloadAlert = true
                }
            }
        }
        .formStyle(.grouped)
        .padding()
        .alert("Models Not Available", isPresented: $showingModelDownloadAlert) {
            Button("OK", role: .cancel) { }
        } message: {
            Text("AI models are not yet available for download. In the meantime, AI features use geometric fallback generation to create placeholder glyphs.")
        }
    }
}

#Preview {
    SettingsView()
        .environmentObject(AppState())
}
