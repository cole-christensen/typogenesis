import SwiftUI

struct GlyphGrid: View {
    let project: FontProject
    @Binding var selectedGlyph: Character?

    @State private var searchText = ""
    @State private var displayMode: DisplayMode = .grid

    enum DisplayMode {
        case grid
        case list
    }

    private let defaultCharacters = Array("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")

    var filteredCharacters: [Character] {
        let chars = project.glyphs.isEmpty ? defaultCharacters : Array(project.glyphs.keys).sorted()

        if searchText.isEmpty {
            return chars
        } else {
            return chars.filter { String($0).localizedCaseInsensitiveContains(searchText) }
        }
    }

    var body: some View {
        VStack(spacing: 0) {
            toolbar

            Divider()

            ScrollView {
                switch displayMode {
                case .grid:
                    gridView
                case .list:
                    listView
                }
            }
        }
    }

    private var toolbar: some View {
        HStack {
            TextField("Search glyphs...", text: $searchText)
                .textFieldStyle(.roundedBorder)
                .frame(maxWidth: 200)

            Spacer()

            Picker("Display", selection: $displayMode) {
                Image(systemName: "square.grid.2x2").tag(DisplayMode.grid)
                Image(systemName: "list.bullet").tag(DisplayMode.list)
            }
            .pickerStyle(.segmented)
            .frame(width: 80)

            Button {
                // Add glyph
            } label: {
                Image(systemName: "plus")
            }
        }
        .padding(8)
    }

    private var gridView: some View {
        LazyVGrid(columns: [GridItem(.adaptive(minimum: 60, maximum: 80))], spacing: 8) {
            ForEach(filteredCharacters, id: \.self) { character in
                GlyphCell(
                    character: character,
                    glyph: project.glyph(for: character),
                    isSelected: selectedGlyph == character
                )
                .onTapGesture {
                    selectedGlyph = character
                }
            }
        }
        .padding()
    }

    private var listView: some View {
        LazyVStack(spacing: 2) {
            ForEach(filteredCharacters, id: \.self) { character in
                GlyphListRow(
                    character: character,
                    glyph: project.glyph(for: character),
                    isSelected: selectedGlyph == character
                )
                .onTapGesture {
                    selectedGlyph = character
                }
            }
        }
        .padding()
    }
}

struct GlyphCell: View {
    let character: Character
    let glyph: Glyph?
    let isSelected: Bool

    var body: some View {
        VStack(spacing: 4) {
            ZStack {
                RoundedRectangle(cornerRadius: 4)
                    .fill(isSelected ? Color.accentColor.opacity(0.2) : Color(nsColor: .controlBackgroundColor))
                    .overlay(
                        RoundedRectangle(cornerRadius: 4)
                            .stroke(isSelected ? Color.accentColor : Color.gray.opacity(0.3), lineWidth: isSelected ? 2 : 1)
                    )

                if let glyph = glyph, !glyph.outline.isEmpty {
                    GlyphPreview(outline: glyph.outline)
                        .padding(4)
                } else {
                    Text(String(character))
                        .font(.system(size: 28, design: .serif))
                        .foregroundColor(.secondary)
                }
            }
            .frame(width: 60, height: 60)

            Text(String(character))
                .font(.caption2)
                .foregroundColor(.secondary)
        }
    }
}

struct GlyphListRow: View {
    let character: Character
    let glyph: Glyph?
    let isSelected: Bool

    var body: some View {
        HStack {
            ZStack {
                RoundedRectangle(cornerRadius: 4)
                    .fill(Color(nsColor: .controlBackgroundColor))

                if let glyph = glyph, !glyph.outline.isEmpty {
                    GlyphPreview(outline: glyph.outline)
                        .padding(2)
                } else {
                    Text(String(character))
                        .font(.system(size: 20, design: .serif))
                        .foregroundColor(.secondary)
                }
            }
            .frame(width: 40, height: 40)

            VStack(alignment: .leading) {
                Text(String(character))
                    .font(.headline)

                if let glyph = glyph {
                    Text("Width: \(glyph.advanceWidth)")
                        .font(.caption)
                        .foregroundColor(.secondary)
                } else {
                    Text("Not defined")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }

            Spacer()

            if glyph != nil {
                Image(systemName: "checkmark.circle.fill")
                    .foregroundColor(.green)
            } else {
                Image(systemName: "circle")
                    .foregroundColor(.gray)
            }
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(isSelected ? Color.accentColor.opacity(0.2) : Color.clear)
        .cornerRadius(4)
    }
}

struct GlyphPreview: View {
    let outline: GlyphOutline

    var body: some View {
        GeometryReader { geometry in
            Canvas { context, size in
                guard !outline.isEmpty else { return }

                let boundingBox = outline.boundingBox
                let scaleX = size.width / CGFloat(max(boundingBox.width, 1))
                let scaleY = size.height / CGFloat(max(boundingBox.height, 1))
                let scale = min(scaleX, scaleY) * 0.9

                let offsetX = (size.width - CGFloat(boundingBox.width) * scale) / 2 - CGFloat(boundingBox.minX) * scale
                let offsetY = (size.height + CGFloat(boundingBox.height) * scale) / 2 + CGFloat(boundingBox.minY) * scale

                let transform = CGAffineTransform(translationX: offsetX, y: offsetY)
                    .scaledBy(x: scale, y: -scale)

                let cgPath = outline.toCGPath()
                let path = Path(cgPath).applying(transform)

                context.fill(path, with: .color(.primary))
            }
        }
    }
}

#Preview {
    GlyphGrid(project: FontProject(name: "Test", family: "Test", style: "Regular"), selectedGlyph: .constant(nil))
        .frame(width: 400, height: 300)
}
