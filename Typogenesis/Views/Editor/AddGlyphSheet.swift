import SwiftUI

struct AddGlyphSheet: View {
    @Environment(\.dismiss) var dismiss
    let existingGlyphs: Set<Character>
    let onAdd: (Character) -> Void

    @State private var inputMode: InputMode = .keyboard
    @State private var characterInput = ""
    @State private var unicodeInput = ""
    @State private var selectedPreset: CharacterPreset?
    @State private var showDuplicateAlert = false
    @State private var pendingCharacter: Character?

    enum InputMode: String, CaseIterable {
        case keyboard = "Type Character"
        case unicode = "Unicode"
        case preset = "Character Set"
    }

    enum CharacterPreset: String, CaseIterable {
        case uppercaseLetters = "A-Z"
        case lowercaseLetters = "a-z"
        case digits = "0-9"
        case basicPunctuation = "Punctuation"
        case extendedLatin = "Extended Latin"

        var characters: [Character] {
            switch self {
            case .uppercaseLetters:
                return Array("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
            case .lowercaseLetters:
                return Array("abcdefghijklmnopqrstuvwxyz")
            case .digits:
                return Array("0123456789")
            case .basicPunctuation:
                return Array(".,;:!?'\"()-–—…")
            case .extendedLatin:
                return Array("ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ")
            }
        }
    }

    var body: some View {
        VStack(spacing: 20) {
            Text("Add Glyph")
                .font(.title2)
                .fontWeight(.semibold)

            Picker("Input Method", selection: $inputMode) {
                ForEach(InputMode.allCases, id: \.self) { mode in
                    Text(mode.rawValue).tag(mode)
                }
            }
            .pickerStyle(.segmented)

            switch inputMode {
            case .keyboard:
                keyboardInput
            case .unicode:
                unicodeInput_view
            case .preset:
                presetInput
            }

            Spacer()

            HStack {
                Button("Cancel") {
                    dismiss()
                }
                .keyboardShortcut(.cancelAction)

                Spacer()

                Button("Add") {
                    addCharacter()
                }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.defaultAction)
                .disabled(!canAdd)
            }
        }
        .padding(24)
        // Height increased from 350 to 520 to fit Basic Latin character grid without clipping
        .frame(width: 400, height: 520)
        .alert("Replace Existing Glyph?", isPresented: $showDuplicateAlert) {
            Button("Replace", role: .destructive) {
                if let character = pendingCharacter {
                    onAdd(character)
                    dismiss()
                }
            }
            Button("Cancel", role: .cancel) {
                pendingCharacter = nil
            }
        } message: {
            if let char = pendingCharacter {
                Text("A glyph for '\(String(char))' already exists. Adding it will replace the existing glyph.")
            }
        }
    }

    private var keyboardInput: some View {
        VStack(spacing: 12) {
            Text("Type or paste a character:")
                .foregroundColor(.secondary)

            TextField("Character", text: $characterInput)
                .textFieldStyle(.roundedBorder)
                .font(.system(size: 32))
                .multilineTextAlignment(.center)
                .frame(width: 100)
                .onChange(of: characterInput) { _, newValue in
                    if newValue.count > 1 {
                        characterInput = String(newValue.prefix(1))
                    }
                }

            if let char = characterInput.first {
                Text("Unicode: U+\(String(format: "%04X", char.unicodeScalars.first?.value ?? 0))")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
    }

    private var unicodeInput_view: some View {
        VStack(spacing: 12) {
            Text("Enter Unicode code point (hex):")
                .foregroundColor(.secondary)

            HStack {
                Text("U+")
                    .foregroundColor(.secondary)
                TextField("0041", text: $unicodeInput)
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 80)
                    .onChange(of: unicodeInput) { _, newValue in
                        unicodeInput = newValue.uppercased().filter { $0.isHexDigit }
                        if unicodeInput.count > 6 {
                            unicodeInput = String(unicodeInput.prefix(6))
                        }
                    }
            }

            if let codePoint = UInt32(unicodeInput, radix: 16),
               let scalar = Unicode.Scalar(codePoint) {
                Text("Character: \(String(Character(scalar)))")
                    .font(.system(size: 32))
            }
        }
    }

    private var presetInput: some View {
        VStack(spacing: 12) {
            Text("Select a character set:")
                .foregroundColor(.secondary)

            Picker("Preset", selection: $selectedPreset) {
                Text("Select...").tag(nil as CharacterPreset?)
                ForEach(CharacterPreset.allCases, id: \.self) { preset in
                    Text(preset.rawValue).tag(preset as CharacterPreset?)
                }
            }

            if let preset = selectedPreset {
                ScrollView {
                    LazyVGrid(columns: [GridItem(.adaptive(minimum: 40))], spacing: 8) {
                        ForEach(preset.characters, id: \.self) { char in
                            Button {
                                characterInput = String(char)
                                inputMode = .keyboard
                            } label: {
                                Text(String(char))
                                    .font(.system(size: 20))
                                    .frame(width: 36, height: 36)
                            }
                            .buttonStyle(.bordered)
                        }
                    }
                }
                .frame(maxHeight: 120)
            }
        }
    }

    private var canAdd: Bool {
        switch inputMode {
        case .keyboard:
            return characterInput.count == 1
        case .unicode:
            if let codePoint = UInt32(unicodeInput, radix: 16) {
                return Unicode.Scalar(codePoint) != nil
            }
            return false
        case .preset:
            return false  // Must select a character first
        }
    }

    private func addCharacter() {
        let char: Character?

        switch inputMode {
        case .keyboard:
            char = characterInput.first
        case .unicode:
            if let codePoint = UInt32(unicodeInput, radix: 16),
               let scalar = Unicode.Scalar(codePoint) {
                char = Character(scalar)
            } else {
                char = nil
            }
        case .preset:
            char = nil
        }

        if let character = char {
            if existingGlyphs.contains(character) {
                pendingCharacter = character
                showDuplicateAlert = true
            } else {
                onAdd(character)
                dismiss()
            }
        }
    }
}

#Preview {
    AddGlyphSheet(existingGlyphs: ["A", "B"]) { _ in }
}
