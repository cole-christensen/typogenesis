import AppKit
import os.log

private let logger = Logger(subsystem: "com.typogenesis.app", category: "AppDelegate")

class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationWillFinishLaunching(_ notification: Notification) {
        logger.info("applicationWillFinishLaunching")
        print("DEBUG: applicationWillFinishLaunching")
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        logger.info("applicationDidFinishLaunching - window count: \(NSApp.windows.count)")
        print("DEBUG: applicationDidFinishLaunching - window count: \(NSApp.windows.count)")

        // Ensure the app activates and shows its window
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)

        // Make sure a window is visible
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
            print("DEBUG: Delayed check - window count: \(NSApp.windows.count)")
            for (index, window) in NSApp.windows.enumerated() {
                print("DEBUG: Window \(index): \(window.title), visible: \(window.isVisible), frame: \(window.frame)")
            }

            if let window = NSApp.windows.first {
                window.makeKeyAndOrderFront(nil)
                NSApp.activate(ignoringOtherApps: true)
                print("DEBUG: Activated window: \(window.title)")
            }
        }
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        return false  // Keep app running even if window is closed
    }

    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool) -> Bool {
        if !flag {
            // If no visible windows, show the first one
            NSApp.windows.first?.makeKeyAndOrderFront(nil)
        }
        return true
    }
}
