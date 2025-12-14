import SwiftUI

@main
struct TypogenesisApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate
    @StateObject private var appState = AppState()

    var body: some Scene {
        WindowGroup {
            MainWindow()
                .environmentObject(appState)
        }
        .commands {
            CommandGroup(replacing: .newItem) {
                Button("New Font Project") {
                    appState.createNewProject()
                }
                .keyboardShortcut("n", modifiers: .command)

                Button("Open...") {
                    appState.openProject()
                }
                .keyboardShortcut("o", modifiers: .command)
            }

            CommandGroup(after: .saveItem) {
                Button("Export Font...") {
                    appState.showExportSheet = true
                }
                .keyboardShortcut("e", modifiers: [.command, .shift])
                .disabled(appState.currentProject == nil)
            }
        }

        Settings {
            SettingsView()
                .environmentObject(appState)
        }
    }
}
