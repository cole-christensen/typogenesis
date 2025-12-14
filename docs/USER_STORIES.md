# Typogenesis User Stories

This document describes how real users interact with Typogenesis. Each story represents a complete workflow that should work end-to-end. These stories form the basis for our UI tests.

---

## Story 1: Alex Creates Their First Font

**User**: Alex, a graphic designer who wants to create a custom display font for a client project.

**Goal**: Create a simple font with uppercase letters A-Z and export it as a TTF file.

### The Journey

1. **Launch and Welcome**
   - Alex launches Typogenesis
   - The welcome screen appears with options: "Create New Font", "Import Font", "Open Existing Project"
   - Alex clicks "Create New Font"

2. **Project Setup**
   - A new project is created with default settings
   - The main window appears with:
     - Sidebar on the left (Glyphs, Metrics, Kerning, Preview, Variable Font, AI Generate, Handwriting)
     - Empty glyph grid in the center
     - Inspector panel on the right
   - "Glyphs" is selected in the sidebar by default

3. **Adding the First Glyph**
   - Alex sees the empty glyph grid with an "Add New Glyph" button
   - Alex clicks "Add New Glyph"
   - The Add Glyph sheet appears with options:
     - Keyboard input field
     - Unicode input field
     - Preset buttons (A-Z, a-z, 0-9, Punctuation)
   - Alex clicks the "A-Z" preset
   - 26 empty glyphs (A through Z) are added to the grid

4. **Selecting a Glyph to Edit**
   - The glyph grid now shows 26 placeholder glyphs
   - Alex clicks on the "A" glyph
   - The glyph editor canvas appears below the grid
   - The canvas shows:
     - Grid lines
     - Baseline, x-height, cap-height guidelines
     - An empty glyph area ready for editing

5. **Drawing the Letter A**
   - Alex selects the "Pen" tool from the toolbar
   - Alex clicks to place the first point (bottom-left of the A)
   - Alex clicks to place more points, forming a triangular shape
   - Alex clicks on the first point to close the contour
   - The letter A outline appears on the canvas
   - Alex sees the points displayed as small squares (corner points)

6. **Refining the Glyph**
   - Alex switches to the "Select" tool
   - Alex clicks and drags a point to adjust its position
   - Alex shift-clicks to select multiple points
   - Alex drags the selection to move multiple points together
   - The glyph shape updates in real-time

7. **Adding the Crossbar**
   - Alex switches back to the "Pen" tool
   - Alex draws a horizontal rectangle for the A's crossbar
   - The A now has two contours (outer triangle, inner crossbar)

8. **Checking Metrics**
   - Alex clicks "Metrics" in the sidebar
   - The Metrics Editor appears showing:
     - Units Per Em: 1000
     - Ascender: 800
     - Descender: -200
     - Cap Height: 700
     - x-Height: 500
   - Alex leaves the defaults and clicks back to "Glyphs"

9. **Setting the Font Name**
   - Alex opens the Inspector panel (if not already visible)
   - Alex sees fields for:
     - Font Family: "Untitled"
     - Style: "Regular"
   - Alex changes Font Family to "AlexDisplay"
   - The title bar updates to show "AlexDisplay"

10. **Previewing the Font**
    - Alex clicks "Preview" in the sidebar
    - The preview panel shows sample text rendered with the current glyphs
    - Alex types "ALEX" in the sample text field
    - The preview updates to show the letters (with placeholders for missing glyphs)

11. **Exporting the Font**
    - Alex presses Cmd+E (or goes to File > Export)
    - The Export Sheet appears with format options:
      - TTF (selected)
      - OTF
      - WOFF
      - UFO
    - Alex keeps TTF selected
    - Alex clicks "Export"
    - A save dialog appears
    - Alex chooses a location and filename "AlexDisplay-Regular.ttf"
    - Alex clicks "Save"
    - A success message appears: "Font exported successfully"

12. **Verifying the Export**
    - Alex opens the exported .ttf file in Font Book (macOS)
    - The font appears with the name "AlexDisplay"
    - Alex can see the glyphs they created

### Success Criteria
- [ ] App launches to welcome screen
- [ ] "Create New Font" creates an empty project
- [ ] Glyph grid displays and allows adding glyphs
- [ ] Clicking a glyph opens the editor canvas
- [ ] Pen tool allows drawing closed contours
- [ ] Select tool allows moving points
- [ ] Multiple points can be selected and moved together
- [ ] Metrics editor shows and allows editing metrics
- [ ] Inspector allows changing font family name
- [ ] Preview shows rendered glyphs
- [ ] Export creates a valid TTF file
- [ ] Exported font can be opened in Font Book

---

## Story 2: Jordan Creates a Variable Weight Font

**User**: Jordan, a type designer who wants to create a variable font with weight variations.

**Goal**: Create a font with Light and Bold masters that interpolates between them.

### The Journey

1. **Starting with an Existing Project**
   - Jordan has already created a font project with basic uppercase letters
   - Jordan opens the project in Typogenesis

2. **Enabling Variable Font Mode**
   - Jordan clicks "Variable Font" in the sidebar
   - The Variable Font Editor appears
   - Jordan sees a toggle: "Enable Variable Font"
   - Jordan enables the toggle
   - New sections appear: Axes, Masters, Named Instances

3. **Adding the Weight Axis**
   - Jordan clicks "Add Axis" in the Axes section
   - The Add Axis sheet appears with presets:
     - Weight, Width, Slant, Italic, Optical Size, Custom
   - Jordan selects "Weight"
   - The weight axis is added with defaults:
     - Tag: wght
     - Min: 100
     - Default: 400
     - Max: 900

4. **Creating the Light Master**
   - Jordan clicks "Add Master" in the Masters section
   - The Add Master sheet appears
   - Jordan enters:
     - Name: "Light"
     - Weight: 300
   - Jordan clicks "Add"
   - The Light master appears in the list

5. **Creating the Bold Master**
   - Jordan clicks "Add Master" again
   - Jordan enters:
     - Name: "Bold"
     - Weight: 700
   - Jordan clicks "Add"
   - The Bold master appears in the list

6. **Drawing Glyphs for Light Master**
   - Jordan clicks "Glyphs" in the sidebar
   - Jordan selects the "A" glyph
   - The glyph editor shows the default (Regular) version
   - Jordan draws a thin version of the letter A with light strokes

7. **Drawing Glyphs for Bold Master**
   - Jordan needs to draw the Bold version
   - In the Variable Font Editor, Jordan selects the "Bold" master
   - Jordan goes back to Glyphs and draws a thick version of the letter A
   - The Bold A has the same number of points as the Light A (required for interpolation)

8. **Adding Named Instances**
   - Jordan clicks "Variable Font" in the sidebar
   - Jordan clicks "Add Instance" in Named Instances section
   - Jordan adds instances:
     - "Light" at weight 300
     - "Regular" at weight 400
     - "Medium" at weight 500
     - "SemiBold" at weight 600
     - "Bold" at weight 700

9. **Previewing Interpolation**
   - Jordan uses the weight slider in the preview panel
   - Jordan drags from 300 to 700
   - The preview shows the letter smoothly transitioning from Light to Bold
   - Jordan can see intermediate weights like 450 or 550

10. **Exporting the Variable Font**
    - Jordan presses Cmd+E
    - The Export Sheet shows TTF selected
    - Jordan sees "Export as variable font" is enabled
    - Jordan clicks Export
    - Jordan saves as "JordanSans-Variable.ttf"

11. **Verifying the Variable Font**
    - Jordan opens the font in a variable font tester
    - Jordan can adjust the weight slider
    - The font smoothly interpolates between Light and Bold

### Success Criteria
- [ ] Variable Font section appears in sidebar
- [ ] Enable Variable Font toggle works
- [ ] Weight axis can be added with correct defaults
- [ ] Masters can be created at specific axis locations
- [ ] Glyph editor allows editing glyphs per master
- [ ] Named instances can be added
- [ ] Preview shows interpolation between masters
- [ ] Export includes fvar, gvar, STAT, avar tables
- [ ] Exported font works in variable font testers

---

## Story 3: Sam Imports and Modifies an Existing Font

**User**: Sam, a developer who wants to modify an open-source font for their app.

**Goal**: Import an existing TTF, add a custom glyph, and export a modified version.

### The Journey

1. **Importing a Font**
   - Sam launches Typogenesis
   - Sam clicks "Import Font (.ttf/.otf)..." on the welcome screen
   - A file picker appears
   - Sam selects an open-source TTF file
   - A loading indicator appears: "Importing font..."

2. **Import Completion**
   - The Import Font Sheet appears showing:
     - Font name and style
     - Number of glyphs found
     - Style analysis (stroke weight, contrast, serif style)
   - Sam clicks "Import"
   - The font opens as a new project

3. **Exploring the Imported Font**
   - The glyph grid shows all imported glyphs
   - Sam scrolls through to see letters, numbers, punctuation
   - Sam clicks on the "A" glyph
   - The glyph editor shows the imported outline with all points and curves

4. **Adding a Custom Glyph**
   - Sam wants to add a custom logo glyph
   - Sam clicks "Add New Glyph"
   - Sam enters Unicode: U+E000 (Private Use Area)
   - An empty glyph is created at that code point

5. **Drawing the Custom Glyph**
   - Sam uses the Pen tool to draw a company logo
   - Sam uses curves (smooth points) for rounded shapes
   - Sam adjusts control handles for precise curves

6. **Setting Glyph Metrics**
   - In the Inspector, Sam sees:
     - Advance Width: 500
     - Left Side Bearing: 50
   - Sam adjusts the advance width to 600 for the wider logo

7. **Adding Kerning**
   - Sam clicks "Kerning" in the sidebar
   - Sam wants to adjust kerning between "A" and "V"
   - Sam clicks "Add Pair"
   - Sam enters Left: A, Right: V, Value: -40
   - The kerning preview shows the adjusted spacing

8. **Using Auto-Kerning**
   - Sam clicks "Auto-Kern" button
   - The Auto-Kerning sheet appears
   - Sam selects "Default" spacing preset
   - Sam clicks "Generate"
   - Multiple kerning pairs are generated automatically
   - Sam reviews and accepts the suggestions

9. **Exporting the Modified Font**
   - Sam presses Cmd+E
   - Sam selects OTF format this time
   - Sam exports as "ModifiedFont-Regular.otf"

10. **Testing in Their App**
    - Sam installs the font in their development environment
    - Sam can access the custom logo glyph using the Unicode code point
    - The kerning looks correct in their app's text

### Success Criteria
- [ ] Import button opens file picker
- [ ] TTF files can be imported
- [ ] Import shows loading indicator
- [ ] Import sheet displays font info and style analysis
- [ ] All glyphs from the font are imported
- [ ] Imported glyph outlines are editable
- [ ] Custom glyphs can be added at specific Unicode points
- [ ] Glyph metrics can be adjusted in Inspector
- [ ] Kerning pairs can be added manually
- [ ] Auto-kerning generates reasonable pairs
- [ ] OTF export works correctly
- [ ] Modified font works in external applications

---

## Story 4: Casey Uses AI to Generate Missing Glyphs

**User**: Casey, an indie game developer who needs a complete character set quickly.

**Goal**: Create a few glyphs manually, then use AI to generate the rest in the same style.

### The Journey

1. **Creating Reference Glyphs**
   - Casey creates a new font project
   - Casey manually draws: A, B, C, O, a, n, o
   - These establish the font's style (stroke weight, proportions, curves)

2. **Opening AI Generation**
   - Casey clicks "AI Generate" in the sidebar
   - The Generate View appears with options:
     - Generation Mode: Complete Font, Missing Glyphs, Style Transfer, Variation
     - Character Set selection
     - Style controls

3. **Selecting Missing Glyphs Mode**
   - Casey selects "Missing Glyphs" mode
   - The app analyzes the existing glyphs
   - A list shows which characters are missing

4. **Configuring Generation**
   - Casey selects character sets to generate:
     - [x] Remaining uppercase (D-Z except O)
     - [x] Remaining lowercase (except a, n, o)
     - [x] Numbers 0-9
     - [x] Basic punctuation
   - Casey sees: "127 glyphs to generate"

5. **Starting Generation**
   - Casey clicks "Generate"
   - A progress bar appears
   - Generated glyphs appear in a preview grid as they're created
   - Each glyph shows a confidence score

6. **Reviewing Generated Glyphs**
   - After generation completes, Casey reviews the results
   - Most glyphs look consistent with the hand-drawn ones
   - A few glyphs are flagged for review (low confidence)

7. **Refining Problem Glyphs**
   - Casey clicks on a flagged glyph (e.g., "S")
   - Casey can:
     - Accept as-is
     - Regenerate
     - Edit manually
   - Casey chooses to edit the S manually, adjusting a few points

8. **Adding to Project**
   - Casey clicks "Add to Project"
   - All generated glyphs are added to the font
   - Casey can now see them in the glyph grid

9. **Final Export**
   - Casey exports the complete font as TTF
   - The font now has a full character set
   - Casey uses it in their game

### Success Criteria
- [ ] AI Generate section appears in sidebar
- [ ] Multiple generation modes available
- [ ] Character set selection works
- [ ] Generation shows progress
- [ ] Generated glyphs appear in preview
- [ ] Confidence scores are displayed
- [ ] Individual glyphs can be regenerated
- [ ] Generated glyphs can be edited
- [ ] "Add to Project" adds glyphs to the font
- [ ] Complete font can be exported

---

## Story 5: Riley Converts Handwriting to a Font

**User**: Riley, a teacher who wants to create a font from their handwriting.

**Goal**: Scan handwritten characters and convert them to a digital font.

### The Journey

1. **Opening Handwriting Scanner**
   - Riley creates a new font project
   - Riley clicks "Handwriting" in the sidebar
   - The Handwriting Scanner wizard appears

2. **Step 1: Upload**
   - Riley sees instructions to upload a handwriting sample
   - Option to download a sample sheet template
   - Riley downloads and prints the template
   - Riley fills in the template with their handwriting
   - Riley scans/photographs the completed sheet
   - Riley drags the image into the upload area

3. **Step 2: Process**
   - The app shows the uploaded image
   - Processing settings appear:
     - Threshold slider
     - Simplification slider
   - Riley clicks "Process"
   - The app detects individual characters
   - Bounding boxes appear around detected characters

4. **Step 3: Assign**
   - Riley sees the detected characters in a grid
   - Each detected shape needs to be assigned to a character
   - Riley clicks on a shape, then types "A" to assign it
   - Or uses "Auto-Assign from Template" if using the standard template
   - All characters get assigned

5. **Step 4: Import**
   - Riley sees import options:
     - [x] Replace existing glyphs
     - [x] Auto-fit to metrics
     - [ ] Generate kerning
   - Riley clicks "Import"
   - The vectorized glyphs are added to the project

6. **Reviewing Vectorized Glyphs**
   - Riley goes to the Glyphs section
   - Riley clicks on the handwritten "A"
   - The editor shows the vectorized outline
   - The curves approximate Riley's handwriting

7. **Cleaning Up**
   - Some glyphs have extra noise or rough edges
   - Riley uses the Simplify Path option to smooth them
   - Riley manually adjusts a few problematic points

8. **Exporting the Handwriting Font**
   - Riley exports as TTF
   - The font captures Riley's personal handwriting style

### Success Criteria
- [ ] Handwriting section appears in sidebar
- [ ] Step-by-step wizard guides the user
- [ ] Template can be downloaded
- [ ] Images can be uploaded via drag-drop
- [ ] Processing settings adjust detection
- [ ] Characters are detected and bounded
- [ ] Characters can be assigned to glyphs
- [ ] Auto-assign works with template
- [ ] Vectorized glyphs are imported
- [ ] Imported glyphs can be edited
- [ ] Final font exports correctly

---

## Test Priority

Based on core functionality and risk, tests should be implemented in this order:

1. **Story 1 (Alex)** - Basic font creation is the core use case
2. **Story 3 (Sam)** - Import/export roundtrip tests file I/O
3. **Story 2 (Jordan)** - Variable fonts are a key feature
4. **Story 5 (Riley)** - Handwriting conversion tests image processing
5. **Story 4 (Casey)** - AI generation depends on model availability

Each story should have:
- A happy path test covering the full workflow
- Edge case tests for error conditions
- Regression tests for known bugs
