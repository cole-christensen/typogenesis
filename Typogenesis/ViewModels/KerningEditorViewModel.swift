import SwiftUI

@MainActor
final class KerningEditorViewModel: ObservableObject {
    // MARK: - Published Properties

    /// Local copy of kerning pairs, synced back to project via Pattern A (.onChange)
    @Published var kerningPairs: [KerningPair] = []
    @Published var selectedPairIndex: Int? = nil
    @Published var leftChar: String = ""
    @Published var rightChar: String = ""
    @Published var kerningValue: Int = 0
    @Published var previewText: String = "AVAST Wavy Type"
    @Published var showAddSheet = false
    @Published var showAutoKernSheet = false
    @Published var isAutoKerning = false
    @Published var autoKernProgress: Double = 0
    @Published var autoKernError: String? = nil
    @Published var showAutoKernError = false

    // MARK: - Computed Properties

    var selectedPair: KerningPair? {
        guard let index = selectedPairIndex,
              index < kerningPairs.count else { return nil }
        return kerningPairs[index]
    }

    // MARK: - Loading

    func loadPairs(from project: FontProject) {
        kerningPairs = project.kerning
    }

    // MARK: - Pair Management

    func addQuickPair() {
        guard let left = leftChar.first,
              let right = rightChar.first else { return }
        addPair(left: left, right: right, value: kerningValue)
        leftChar = ""
        rightChar = ""
        kerningValue = 0
    }

    func addPair(left: Character, right: Character, value: Int) {
        if let existingIndex = kerningPairs.firstIndex(where: { $0.left == left && $0.right == right }) {
            kerningPairs[existingIndex] = KerningPair(left: left, right: right, value: value)
            selectedPairIndex = existingIndex
        } else {
            kerningPairs.append(KerningPair(left: left, right: right, value: value))
            selectedPairIndex = kerningPairs.count - 1
        }
    }

    func updatePairValue(at index: Int, value: Int) {
        guard index < kerningPairs.count else { return }
        let pair = kerningPairs[index]
        kerningPairs[index] = KerningPair(left: pair.left, right: pair.right, value: value)
    }

    func deletePair(at index: Int) {
        guard index < kerningPairs.count else { return }
        kerningPairs.remove(at: index)
        if selectedPairIndex == index {
            selectedPairIndex = nil
        } else if let selected = selectedPairIndex, selected > index {
            selectedPairIndex = selected - 1
        }
    }

    // MARK: - Auto Kerning

    func generateAutoKerning(
        settings: KerningPredictor.PredictionSettings,
        project: FontProject
    ) async -> KerningPredictor.PredictionResult? {
        let predictor = KerningPredictor()
        do {
            let result = try await predictor.predictKerning(for: project, settings: settings)
            return result
        } catch {
            autoKernError = error.localizedDescription
            showAutoKernError = true
            return nil
        }
    }

    func applyAutoKerning(pairs: [KerningPair], replaceExisting: Bool) {
        if replaceExisting {
            kerningPairs = pairs
        } else {
            for newPair in pairs {
                if let existingIndex = kerningPairs.firstIndex(where: { $0.left == newPair.left && $0.right == newPair.right }) {
                    kerningPairs[existingIndex] = newPair
                } else {
                    kerningPairs.append(newPair)
                }
            }
        }

        kerningPairs.sort { lhs, rhs in
            if lhs.left == rhs.left {
                return lhs.right < rhs.right
            }
            return lhs.left < rhs.left
        }

        selectedPairIndex = nil
    }
}
