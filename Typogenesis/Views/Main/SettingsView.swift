import SwiftUI

struct SettingsView: View {
    @EnvironmentObject var appState: AppState

    @AppStorage("gridSize") private var gridSize = 50
    @AppStorage("showGrid") private var showGrid = true
    @AppStorage("showMetrics") private var showMetrics = true
    @AppStorage("snapToGrid") private var snapToGrid = true
    @AppStorage("aiAssistEnabled") private var aiAssistEnabled = true

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
                Toggle("Show welcome screen on launch", isOn: .constant(true))
                Toggle("Check for updates automatically", isOn: .constant(true))
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
                Picker("Default Units/Em", selection: .constant(1000)) {
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
                Toggle("Auto-suggest kerning", isOn: .constant(true))
                Toggle("Style consistency warnings", isOn: .constant(true))
            }

            Section("Models") {
                LabeledContent("Glyph Generation", value: "Not loaded")
                LabeledContent("Style Encoder", value: "Not loaded")
                LabeledContent("Kerning Net", value: "Not loaded")

                Button("Download Models...") {
                    // Model download
                }
            }
        }
        .formStyle(.grouped)
        .padding()
    }
}

#Preview {
    SettingsView()
        .environmentObject(AppState())
}
