import SwiftUI

/// Editor for variable font configuration
struct VariableFontEditor: View {
    @EnvironmentObject var appState: AppState
    @State private var selectedMasterID: UUID?
    @State private var previewLocation: DesignSpaceLocation = [:]
    @State private var showAddAxisSheet = false
    @State private var showAddMasterSheet = false
    @State private var showAddInstanceSheet = false

    var body: some View {
        HSplitView {
            // Left side: Configuration
            configurationPanel
                .frame(minWidth: 300)

            // Right side: Preview
            previewPanel
                .frame(minWidth: 400)
        }
        .accessibilityIdentifier(AccessibilityID.Variable.editor)
    }

    // MARK: - Configuration Panel

    @ViewBuilder
    var configurationPanel: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                variableFontToggle
                if appState.currentProject?.variableConfig.isVariableFont == true {
                    axesSection
                    mastersSection
                    instancesSection
                }
            }
            .padding()
        }
        .background(Color(nsColor: .controlBackgroundColor))
    }

    @ViewBuilder
    var variableFontToggle: some View {
        VStack(alignment: .leading, spacing: 8) {
            Toggle("Variable Font", isOn: Binding(
                get: { appState.currentProject?.variableConfig.isVariableFont ?? false },
                set: { newValue in
                    appState.currentProject?.variableConfig.isVariableFont = newValue
                    if newValue && (appState.currentProject?.variableConfig.axes.isEmpty ?? true) {
                        // Add default weight axis
                        appState.currentProject?.variableConfig.axes = [.weight]
                    }
                }
            ))
            .font(.headline)
            .accessibilityIdentifier(AccessibilityID.Variable.enableToggle)

            Text("Variable fonts contain multiple styles that can be interpolated continuously.")
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .padding()
        .background(Color(nsColor: .textBackgroundColor))
        .cornerRadius(8)
    }

    // MARK: - Axes Section

    @ViewBuilder
    var axesSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Axes")
                    .font(.headline)
                Spacer()
                Button(action: { showAddAxisSheet = true }) {
                    Image(systemName: "plus")
                }
                .buttonStyle(.borderless)
                .accessibilityIdentifier(AccessibilityID.Variable.addAxisButton)
            }

            if let axes = appState.currentProject?.variableConfig.axes, !axes.isEmpty {
                ForEach(axes) { axis in
                    axisRow(axis)
                }
            } else {
                Text("No axes defined")
                    .foregroundColor(.secondary)
                    .font(.caption)
            }
        }
        .padding()
        .background(Color(nsColor: .textBackgroundColor))
        .cornerRadius(8)
        .sheet(isPresented: $showAddAxisSheet) {
            AddAxisSheet { axis in
                appState.currentProject?.variableConfig.axes.append(axis)
            }
        }
    }

    @ViewBuilder
    func axisRow(_ axis: VariationAxis) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(axis.name)
                    .font(.subheadline)
                    .fontWeight(.medium)
                Text("(\(axis.tag))")
                    .font(.caption)
                    .foregroundColor(.secondary)
                Spacer()
                Button(action: {
                    appState.currentProject?.variableConfig.axes.removeAll { $0.id == axis.id }
                }) {
                    Image(systemName: "trash")
                        .foregroundColor(.red)
                }
                .buttonStyle(.borderless)
            }

            HStack {
                Text("\(Int(axis.minValue))")
                    .font(.caption)
                    .frame(width: 40, alignment: .trailing)
                Slider(
                    value: Binding(
                        get: { previewLocation[axis.tag] ?? axis.defaultValue },
                        set: { previewLocation[axis.tag] = $0 }
                    ),
                    in: axis.minValue...axis.maxValue
                )
                Text("\(Int(axis.maxValue))")
                    .font(.caption)
                    .frame(width: 40, alignment: .leading)
            }

            Text("Current: \(Int(previewLocation[axis.tag] ?? axis.defaultValue))")
                .font(.caption2)
                .foregroundColor(.secondary)
        }
        .padding(8)
        .background(Color(nsColor: .controlBackgroundColor))
        .cornerRadius(6)
    }

    // MARK: - Masters Section

    @ViewBuilder
    var mastersSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Masters")
                    .font(.headline)
                Spacer()
                Button(action: { showAddMasterSheet = true }) {
                    Image(systemName: "plus")
                }
                .buttonStyle(.borderless)
                .accessibilityIdentifier(AccessibilityID.Variable.addMasterButton)
            }

            if let masters = appState.currentProject?.variableConfig.masters, !masters.isEmpty {
                ForEach(masters) { master in
                    masterRow(master)
                }
            } else {
                VStack(spacing: 8) {
                    Text("No masters defined")
                        .foregroundColor(.secondary)
                    Text("Masters are the source designs at specific points in the design space.")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
        }
        .padding()
        .background(Color(nsColor: .textBackgroundColor))
        .cornerRadius(8)
        .sheet(isPresented: $showAddMasterSheet) {
            AddMasterSheet(axes: appState.currentProject?.variableConfig.axes ?? []) { master in
                appState.currentProject?.variableConfig.masters.append(master)
            }
        }
    }

    @ViewBuilder
    func masterRow(_ master: FontMaster) -> some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text(master.name)
                    .font(.subheadline)
                    .fontWeight(.medium)

                Text(locationString(master.location))
                    .font(.caption)
                    .foregroundColor(.secondary)

                Text("\(master.glyphs.count) glyphs")
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }

            Spacer()

            Button("Edit") {
                selectedMasterID = master.id
            }
            .buttonStyle(.bordered)
            .controlSize(.small)

            Button(action: {
                appState.currentProject?.variableConfig.masters.removeAll { $0.id == master.id }
            }) {
                Image(systemName: "trash")
                    .foregroundColor(.red)
            }
            .buttonStyle(.borderless)
        }
        .padding(8)
        .background(selectedMasterID == master.id ?
                    Color.accentColor.opacity(0.1) :
                    Color(nsColor: .controlBackgroundColor))
        .cornerRadius(6)
    }

    // MARK: - Instances Section

    @ViewBuilder
    var instancesSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Named Instances")
                    .font(.headline)
                Spacer()
                Button(action: { showAddInstanceSheet = true }) {
                    Image(systemName: "plus")
                }
                .buttonStyle(.borderless)
                .accessibilityIdentifier(AccessibilityID.Variable.addInstanceButton)
            }

            if let instances = appState.currentProject?.variableConfig.instances, !instances.isEmpty {
                ForEach(instances) { instance in
                    instanceRow(instance)
                }
            } else {
                VStack(spacing: 8) {
                    Text("No instances defined")
                        .foregroundColor(.secondary)
                    Text("Instances are predefined locations like 'Bold' or 'Light'.")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
        }
        .padding()
        .background(Color(nsColor: .textBackgroundColor))
        .cornerRadius(8)
        .sheet(isPresented: $showAddInstanceSheet) {
            AddInstanceSheet(axes: appState.currentProject?.variableConfig.axes ?? []) { instance in
                appState.currentProject?.variableConfig.instances.append(instance)
            }
        }
    }

    @ViewBuilder
    func instanceRow(_ instance: NamedInstance) -> some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text(instance.name)
                    .font(.subheadline)

                Text(locationString(instance.location))
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            Spacer()

            Button("Preview") {
                previewLocation = instance.location
            }
            .buttonStyle(.bordered)
            .controlSize(.small)

            Button(action: {
                appState.currentProject?.variableConfig.instances.removeAll { $0.id == instance.id }
            }) {
                Image(systemName: "trash")
                    .foregroundColor(.red)
            }
            .buttonStyle(.borderless)
        }
        .padding(8)
        .background(Color(nsColor: .controlBackgroundColor))
        .cornerRadius(6)
    }

    // MARK: - Preview Panel

    @ViewBuilder
    var previewPanel: some View {
        VStack(spacing: 0) {
            // Header
            HStack {
                Text("Preview")
                    .font(.headline)
                Spacer()
                Text(locationString(previewLocation))
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            .padding()
            .background(Color(nsColor: .controlBackgroundColor))

            Divider()

            // Preview canvas
            if let project = appState.currentProject {
                VariableFontPreview(
                    project: project,
                    location: previewLocation
                )
            } else {
                Text("No project loaded")
                    .foregroundColor(.secondary)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
    }

    // MARK: - Helpers

    func locationString(_ location: DesignSpaceLocation) -> String {
        location.map { "\($0.key): \(Int($0.value))" }.joined(separator: ", ")
    }
}

// MARK: - Variable Font Preview

struct VariableFontPreview: View {
    let project: FontProject
    let location: DesignSpaceLocation

    @State private var sampleText = "AaBbCc"
    @State private var fontSize: CGFloat = 72

    var body: some View {
        VStack(spacing: 16) {
            // Sample text input
            TextField("Sample text", text: $sampleText)
                .textFieldStyle(.roundedBorder)
                .padding(.horizontal)

            // Preview canvas
            ScrollView {
                VStack(spacing: 24) {
                    // Large preview
                    Text("Preview at current location")
                        .font(.caption)
                        .foregroundColor(.secondary)

                    interpolatedGlyphsView
                        .frame(height: fontSize * 1.5)

                    Divider()

                    // Comparison with masters
                    if !project.variableConfig.masters.isEmpty {
                        Text("Master Comparison")
                            .font(.caption)
                            .foregroundColor(.secondary)

                        ForEach(project.variableConfig.masters) { master in
                            VStack(alignment: .leading, spacing: 4) {
                                Text(master.name)
                                    .font(.caption2)
                                    .foregroundColor(.secondary)
                                masterGlyphsView(master)
                                    .frame(height: fontSize * 0.8)
                            }
                        }
                    }
                }
                .padding()
            }

            // Size slider
            HStack {
                Text("Size:")
                    .foregroundColor(.secondary)
                Slider(value: $fontSize, in: 24...144)
                    .frame(width: 150)
                Text("\(Int(fontSize))pt")
                    .frame(width: 50)
            }
            .padding()
            .background(Color(nsColor: .controlBackgroundColor))
        }
    }

    @ViewBuilder
    var interpolatedGlyphsView: some View {
        Canvas { context, size in
            var xOffset: CGFloat = 20
            let baseline = size.height * 0.75
            let scale = fontSize / CGFloat(project.metrics.unitsPerEm)

            for char in sampleText {
                if let glyph = project.glyphs[char] {
                    // Apply interpolation based on location
                    let interpolatedGlyph = interpolateGlyph(glyph, at: location)
                    let path = interpolatedGlyph.outline.cgPath

                    var transform = CGAffineTransform.identity
                    transform = transform.translatedBy(x: xOffset, y: baseline)
                    transform = transform.scaledBy(x: scale, y: -scale)

                    if let transformedPath = path.copy(using: &transform) {
                        context.fill(Path(transformedPath), with: .color(.primary))
                    }

                    xOffset += CGFloat(interpolatedGlyph.advanceWidth) * scale
                } else if char == " " {
                    xOffset += fontSize * 0.3
                }
            }
        }
    }

    @ViewBuilder
    func masterGlyphsView(_ master: FontMaster) -> some View {
        Canvas { context, size in
            var xOffset: CGFloat = 20
            let baseline = size.height * 0.75
            let scale = (fontSize * 0.6) / CGFloat(master.metrics.unitsPerEm)

            for char in sampleText {
                if let glyph = master.glyphs[char] ?? project.glyphs[char] {
                    let path = glyph.outline.cgPath

                    var transform = CGAffineTransform.identity
                    transform = transform.translatedBy(x: xOffset, y: baseline)
                    transform = transform.scaledBy(x: scale, y: -scale)

                    if let transformedPath = path.copy(using: &transform) {
                        context.fill(Path(transformedPath), with: .color(.secondary))
                    }

                    xOffset += CGFloat(glyph.advanceWidth) * scale
                } else if char == " " {
                    xOffset += fontSize * 0.2
                }
            }
        }
    }

    func interpolateGlyph(_ glyph: Glyph, at location: DesignSpaceLocation) -> Glyph {
        // Simple weight-based interpolation for demonstration
        guard let weightValue = location[VariationAxis.weightTag] else {
            return glyph
        }

        // Simulate stroke weight change by scaling
        let weightFactor = (weightValue - 400) / 300  // -1 to +1.67 range
        var result = glyph

        // Modify stroke appearance (simplified - real implementation would modify outline)
        result.advanceWidth = Int(CGFloat(glyph.advanceWidth) * (1 + weightFactor * 0.1))

        return result
    }
}

// MARK: - Add Axis Sheet

struct AddAxisSheet: View {
    @Environment(\.dismiss) var dismiss
    let onAdd: (VariationAxis) -> Void

    @State private var selectedPreset: AxisPreset = .weight
    @State private var customTag = ""
    @State private var customName = ""
    @State private var minValue: Double = 0
    @State private var defaultValue: Double = 50
    @State private var maxValue: Double = 100

    enum AxisPreset: String, CaseIterable {
        case weight = "Weight"
        case width = "Width"
        case slant = "Slant"
        case italic = "Italic"
        case opticalSize = "Optical Size"
        case custom = "Custom"
    }

    var body: some View {
        VStack(spacing: 20) {
            Text("Add Variation Axis")
                .font(.title2)
                .fontWeight(.semibold)

            Picker("Preset", selection: $selectedPreset) {
                ForEach(AxisPreset.allCases, id: \.self) { preset in
                    Text(preset.rawValue).tag(preset)
                }
            }
            .pickerStyle(.segmented)

            if selectedPreset == .custom {
                Form {
                    TextField("Tag (4 chars)", text: $customTag)
                    TextField("Name", text: $customName)
                    HStack {
                        Text("Min:")
                        TextField("", value: $minValue, format: .number)
                            .frame(width: 80)
                    }
                    HStack {
                        Text("Default:")
                        TextField("", value: $defaultValue, format: .number)
                            .frame(width: 80)
                    }
                    HStack {
                        Text("Max:")
                        TextField("", value: $maxValue, format: .number)
                            .frame(width: 80)
                    }
                }
            }

            HStack {
                Button("Cancel") { dismiss() }
                    .keyboardShortcut(.cancelAction)

                Spacer()

                Button("Add") {
                    let axis = createAxis()
                    onAdd(axis)
                    dismiss()
                }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.defaultAction)
            }
        }
        .padding(24)
        .frame(width: 400)
    }

    func createAxis() -> VariationAxis {
        switch selectedPreset {
        case .weight: return .weight
        case .width: return .width
        case .slant: return .slant
        case .italic: return .italic
        case .opticalSize: return .opticalSize
        case .custom:
            return VariationAxis(
                tag: String(customTag.prefix(4)),
                name: customName,
                minValue: CGFloat(minValue),
                defaultValue: CGFloat(defaultValue),
                maxValue: CGFloat(maxValue)
            )
        }
    }
}

// MARK: - Add Master Sheet

struct AddMasterSheet: View {
    @Environment(\.dismiss) var dismiss
    let axes: [VariationAxis]
    let onAdd: (FontMaster) -> Void

    @State private var name = ""
    @State private var locationValues: [String: Double] = [:]

    var body: some View {
        VStack(spacing: 20) {
            Text("Add Master")
                .font(.title2)
                .fontWeight(.semibold)

            Form {
                TextField("Master Name", text: $name)

                ForEach(axes) { axis in
                    HStack {
                        Text(axis.name)
                        Spacer()
                        TextField("Value", value: Binding(
                            get: { locationValues[axis.tag] ?? Double(axis.defaultValue) },
                            set: { locationValues[axis.tag] = $0 }
                        ), format: .number)
                        .frame(width: 80)
                    }
                }
            }

            HStack {
                Button("Cancel") { dismiss() }
                    .keyboardShortcut(.cancelAction)

                Spacer()

                Button("Add") {
                    var location: DesignSpaceLocation = [:]
                    for (tag, value) in locationValues {
                        location[tag] = CGFloat(value)
                    }
                    let master = FontMaster(name: name, location: location)
                    onAdd(master)
                    dismiss()
                }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.defaultAction)
                .disabled(name.isEmpty)
            }
        }
        .padding(24)
        .frame(width: 400)
        .onAppear {
            // Initialize location with default values
            for axis in axes {
                locationValues[axis.tag] = Double(axis.defaultValue)
            }
        }
    }
}

// MARK: - Add Instance Sheet

struct AddInstanceSheet: View {
    @Environment(\.dismiss) var dismiss
    let axes: [VariationAxis]
    let onAdd: (NamedInstance) -> Void

    @State private var name = ""
    @State private var locationValues: [String: Double] = [:]

    var body: some View {
        VStack(spacing: 20) {
            Text("Add Named Instance")
                .font(.title2)
                .fontWeight(.semibold)

            Form {
                TextField("Instance Name (e.g., Bold)", text: $name)

                ForEach(axes) { axis in
                    HStack {
                        Text(axis.name)
                        Spacer()
                        Slider(
                            value: Binding(
                                get: { locationValues[axis.tag] ?? Double(axis.defaultValue) },
                                set: { locationValues[axis.tag] = $0 }
                            ),
                            in: Double(axis.minValue)...Double(axis.maxValue)
                        )
                        .frame(width: 150)
                        Text("\(Int(locationValues[axis.tag] ?? Double(axis.defaultValue)))")
                            .frame(width: 40)
                    }
                }
            }

            HStack {
                Button("Cancel") { dismiss() }
                    .keyboardShortcut(.cancelAction)

                Spacer()

                Button("Add") {
                    var location: DesignSpaceLocation = [:]
                    for (tag, value) in locationValues {
                        location[tag] = CGFloat(value)
                    }
                    let instance = NamedInstance(name: name, location: location)
                    onAdd(instance)
                    dismiss()
                }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.defaultAction)
                .disabled(name.isEmpty)
            }
        }
        .padding(24)
        .frame(width: 450)
        .onAppear {
            for axis in axes {
                locationValues[axis.tag] = Double(axis.defaultValue)
            }
        }
    }
}

#Preview {
    VariableFontEditor()
        .environmentObject(AppState())
        .frame(width: 900, height: 600)
}
