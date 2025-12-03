# Typogenesis

**AI-Powered Font Creation for macOS**

Create, clone, and edit fonts using artificial intelligence. Turn your handwriting into a font, clone the style of existing typefaces, or generate entirely new font families—all with a beautiful native Mac experience.

## Features

### Handwriting to Font
Scan or photograph your handwriting and convert it to a fully functional digital font.

1. Print the template sheet
2. Write your characters
3. Scan or photograph
4. AI vectorizes and normalizes
5. Export your personal font

### Font Cloning
Extract the style essence from any font and apply it to generate new characters.

- Analyze stroke weight, contrast, x-height
- Extract serif style, roundness, slant
- Generate missing characters in the same style
- Create variations (lighter, bolder, condensed)

### AI Generation
Generate new glyphs from scratch or complete partial designs.

- Text-to-glyph generation
- Complete partial sketches
- Generate font variations
- Interpolate between styles

### Professional Editor
Full bezier curve editing with AI assistance.

- Pen and bezier tools
- Smart point snapping
- AI-powered curve smoothing
- Consistency checking
- Metrics and kerning editor

### Intelligent Kerning
Let AI analyze your font and suggest optimal letter spacing.

- Automatic kern pair detection
- Visual kerning editor
- Optical spacing suggestions
- Preview in real text

## Screenshots

```
┌─────────────────────────────────────────────────────────────────┐
│ Typogenesis                                        􀈭 􀆪 􀁢    │
├──────────┬─────────────────────────────────┬────────────────────┤
│ Glyphs   │                                 │ Inspector         │
│          │         ┌─────────────┐         │                    │
│ A B C D  │         │      /\     │         │ Character: A       │
│ E F G H  │         │     /  \    │         │ Unicode: U+0041    │
│ I J K L  │         │    /────\   │         │                    │
│ M N O P  │         │   /      \  │         │ Width: 722         │
│ Q R S T  │         │  /        \ │         │ LSB: 12            │
│ U V W X  │         │                       │                    │
│ Y Z      │         │  ● Control Points     │ ─────────────────  │
│          │         │  ○ On-curve Points    │ AI Actions         │
│ a b c d  │         │                       │ 􀋭 Smooth Curves   │
│ e f g h  │         └───────────────────────│ 􀅴 Match Style     │
│ ...      │                                 │ 􀍟 Fix Consistency │
└──────────┴─────────────────────────────────┴────────────────────┘
```

## Requirements

- macOS 14.0 (Sonoma) or later
- Apple Silicon recommended (M1/M2/M3)
- 4GB RAM minimum, 8GB recommended
- 2GB disk space for AI models

## Installation

### From Release
1. Download the latest `.dmg` from Releases
2. Drag to Applications
3. Open and allow in Security settings

### Build from Source
```bash
git clone https://github.com/notifd/typogenesis.git
cd typogenesis
open Typogenesis.xcodeproj
# Build with Cmd+B, Run with Cmd+R
```

## Quick Start

### Create from Handwriting

1. **File → New from Handwriting**
2. Print the sample sheet (or use iPad as input)
3. Write clearly in each box
4. Scan/photograph and import
5. Review AI vectorization
6. Adjust and export

### Clone a Font Style

1. **File → Clone Font Style**
2. Select source font (system or file)
3. Choose characters to generate
4. AI analyzes and generates
5. Review and refine
6. Export new font

### Generate New Font

1. **File → New AI Font**
2. Describe desired style or pick presets
3. AI generates initial character set
4. Edit in glyph editor
5. Auto-generate kerning
6. Export

## Supported Formats

| Format | Import | Export | Notes |
|--------|--------|--------|-------|
| TTF | ✅ | ✅ | TrueType |
| OTF | ✅ | ✅ | OpenType/CFF |
| WOFF | ✅ | ✅ | Web fonts |
| WOFF2 | ✅ | ✅ | Compressed web |
| UFO | ✅ | ✅ | Unified Font Object |
| .glyphs | 🔜 | 🔜 | Glyphs.app format |

## AI Models

Typogenesis uses on-device AI for privacy and speed:

- **Glyph Diffusion** - Generates new glyphs from style vectors
- **Style Encoder** - Extracts style characteristics from fonts
- **Kerning Net** - Predicts optimal letter spacing
- **Stroke Analyzer** - Analyzes stroke weights and consistency

All inference runs locally via Core ML and Metal.

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                     SwiftUI Frontend                      │
├──────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐ │
│  │ Glyph Editor │  │ Generation   │  │ Metrics/Kern   │ │
│  │ (Bezier)     │  │ Wizard       │  │ Editor         │ │
│  └──────────────┘  └──────────────┘  └────────────────┘ │
├──────────────────────────────────────────────────────────┤
│                      View Models                          │
├──────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐ │
│  │ AI Services  │  │ Font Parser  │  │ Font Exporter  │ │
│  │ (Core ML)    │  │ (OpenType)   │  │ (TTF/OTF/WOFF) │ │
│  └──────────────┘  └──────────────┘  └────────────────┘ │
├──────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐ │
│  │ Vectorizer   │  │ Path Math    │  │ Storage        │ │
│  │ (Potrace)    │  │ (Bezier)     │  │ (Documents)    │ │
│  └──────────────┘  └──────────────┘  └────────────────┘ │
└──────────────────────────────────────────────────────────┘
```

## Roadmap

### Phase 1: Core Editor
- [x] Project structure
- [ ] Glyph data model
- [ ] Basic bezier editor
- [ ] Font preview
- [ ] TTF/OTF export

### Phase 2: AI Foundation
- [ ] Core ML model integration
- [ ] Style extraction
- [ ] Basic glyph generation
- [ ] Kerning prediction

### Phase 3: Handwriting
- [ ] Sample sheet templates
- [ ] Image preprocessing
- [ ] Vectorization pipeline
- [ ] Character segmentation

### Phase 4: Polish
- [ ] Full character set support
- [ ] Variable font export
- [ ] Cloud sync (iCloud)
- [ ] Plugin system

## Contributing

1. Fork the repository
2. Create a feature branch
3. Write tests first (TDD)
4. Submit a pull request

## License

MIT

---

*"Every letter tells a story. Make yours unique."*
