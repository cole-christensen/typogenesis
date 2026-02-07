import Foundation
import CoreGraphics

/// Exports FontProject with variable font configuration to DesignSpace XML format
/// DesignSpace is a directory-based format containing the .designspace XML file
/// and UFO masters, compatible with professional font editors and fontmake.
actor DesignSpaceExporter {

    enum DesignSpaceError: Error, LocalizedError {
        case noAxes
        case noMasters
        case insufficientMasters(count: Int)
        case failedToCreateDirectory
        case failedToWriteFile(String)
        case ufoExportFailed(masterName: String, underlyingError: Error)
        case notVariableFont

        var errorDescription: String? {
            switch self {
            case .noAxes:
                return "Variable font configuration has no axes defined"
            case .noMasters:
                return "Variable font configuration has no masters defined"
            case .insufficientMasters(let count):
                return "Variable font requires at least 2 masters, but only \(count) found"
            case .failedToCreateDirectory:
                return "Failed to create DesignSpace directory structure"
            case .failedToWriteFile(let name):
                return "Failed to write file: \(name)"
            case .ufoExportFailed(let masterName, let error):
                return "Failed to export master '\(masterName)': \(error.localizedDescription)"
            case .notVariableFont:
                return "Project is not configured as a variable font"
            }
        }
    }

    /// Export options for DesignSpace
    struct ExportOptions {
        var includeKerning: Bool = true
        var masterNamingStrategy: MasterNamingStrategy = .byMasterName

        enum MasterNamingStrategy {
            case byMasterName   // Uses master.name for UFO filename
            case byIndex        // Uses Master_0, Master_1, etc.
        }

        static let `default` = ExportOptions()
    }

    // MARK: - Public API

    /// Export a FontProject with variable font configuration to DesignSpace format
    /// - Parameters:
    ///   - project: The font project to export
    ///   - url: The URL for the .designspace directory (e.g., "FontFamily.designspace")
    ///   - options: Export options
    func export(project: FontProject, to url: URL, options: ExportOptions = .default) async throws {
        let config = project.variableConfig

        // Validate variable font configuration
        guard config.isVariableFont else {
            throw DesignSpaceError.notVariableFont
        }

        guard !config.axes.isEmpty else {
            throw DesignSpaceError.noAxes
        }

        guard !config.masters.isEmpty else {
            throw DesignSpaceError.noMasters
        }

        guard config.masters.count >= 2 else {
            throw DesignSpaceError.insufficientMasters(count: config.masters.count)
        }

        let fileManager = FileManager.default

        // Create the .designspace directory
        try fileManager.createDirectory(at: url, withIntermediateDirectories: true)

        // Export each master as a UFO
        let masterFilenames = try await exportMasters(
            project: project,
            config: config,
            to: url,
            options: options
        )

        // Generate and write the DesignSpace XML file
        let designSpaceXML = buildDesignSpaceXML(
            project: project,
            config: config,
            masterFilenames: masterFilenames
        )

        let xmlFilename = "\(project.family).designspace"
        let xmlURL = url.appendingPathComponent(xmlFilename)

        do {
            try designSpaceXML.write(to: xmlURL, atomically: true, encoding: .utf8)
        } catch {
            throw DesignSpaceError.failedToWriteFile(xmlFilename)
        }
    }

    // MARK: - Master Export

    /// Export all masters as UFO packages
    /// - Returns: Array of (master, ufo filename) pairs
    private func exportMasters(
        project: FontProject,
        config: VariableFontConfig,
        to baseURL: URL,
        options: ExportOptions
    ) async throws -> [(FontMaster, String)] {
        let ufoExporter = UFOExporter()
        let ufoOptions = UFOExporter.ExportOptions(includeKerning: options.includeKerning)

        var results: [(FontMaster, String)] = []

        var usedFilenames: Set<String> = []

        for (index, master) in config.masters.enumerated() {
            var filename: String
            switch options.masterNamingStrategy {
            case .byMasterName:
                // Sanitize master name for filename
                var safeName = master.name
                    .replacingOccurrences(of: " ", with: "_")
                    .replacingOccurrences(of: "/", with: "_")
                if safeName.isEmpty {
                    safeName = "Master_\(index)"
                }
                filename = "\(safeName).ufo"
            case .byIndex:
                filename = "Master_\(index).ufo"
            }

            // Ensure filename uniqueness
            if usedFilenames.contains(filename) {
                filename = "\(filename.dropLast(4))_\(index).ufo"
            }
            usedFilenames.insert(filename)

            let masterURL = baseURL.appendingPathComponent(filename)

            // Create a FontProject from this master
            let masterProject = createProjectFromMaster(master: master, baseProject: project)

            do {
                try await ufoExporter.export(project: masterProject, to: masterURL, options: ufoOptions)
                results.append((master, filename))
            } catch {
                throw DesignSpaceError.ufoExportFailed(masterName: master.name, underlyingError: error)
            }
        }

        return results
    }

    /// Create a FontProject from a FontMaster
    private func createProjectFromMaster(master: FontMaster, baseProject: FontProject) -> FontProject {
        // Use master's glyphs if available, otherwise fall back to base project glyphs
        let glyphs = master.glyphs.isEmpty ? baseProject.glyphs : master.glyphs

        // Extract style name from master name or location
        let styleName = extractStyleName(from: master)

        return FontProject(
            name: "\(baseProject.family) \(styleName)",
            family: baseProject.family,
            style: styleName,
            metrics: master.metrics,
            glyphs: glyphs,
            kerning: baseProject.kerning,
            metadata: baseProject.metadata,
            settings: baseProject.settings
        )
    }

    /// Extract a style name from master properties
    private func extractStyleName(from master: FontMaster) -> String {
        // If master has a clear name, use it
        if !master.name.isEmpty {
            // Remove common prefixes/suffixes
            return master.name
                .replacingOccurrences(of: " Master", with: "")
                .replacingOccurrences(of: "Master ", with: "")
        }

        // Build from location values
        var parts: [String] = []

        if let weight = master.location[VariationAxis.weightTag] {
            switch weight {
            case ...200: parts.append("Thin")
            case 201...350: parts.append("Light")
            case 351...450: parts.append("Regular")
            case 451...550: parts.append("Medium")
            case 551...650: parts.append("SemiBold")
            case 651...750: parts.append("Bold")
            case 751...850: parts.append("ExtraBold")
            default: parts.append("Black")
            }
        }

        if let width = master.location[VariationAxis.widthTag] {
            switch width {
            case ...75: parts.append("Condensed")
            case 76...90: parts.append("SemiCondensed")
            case 111...125: parts.append("SemiExpanded")
            case 126...: parts.append("Expanded")
            default: break  // Normal width, don't add anything
            }
        }

        return parts.isEmpty ? "Regular" : parts.joined(separator: " ")
    }

    // MARK: - XML Generation

    /// Build the DesignSpace XML document
    private func buildDesignSpaceXML(
        project: FontProject,
        config: VariableFontConfig,
        masterFilenames: [(FontMaster, String)]
    ) -> String {
        var xml = """
        <?xml version="1.0" encoding="UTF-8"?>
        <designspace format="5.0">

        """

        // Add axes section
        xml += buildAxesSection(axes: config.axes)

        // Add sources section (masters)
        xml += buildSourcesSection(
            project: project,
            masters: masterFilenames,
            axes: config.axes
        )

        // Add instances section
        xml += buildInstancesSection(
            project: project,
            instances: config.instances,
            axes: config.axes
        )

        xml += "</designspace>\n"

        return xml
    }

    /// Build the <axes> section
    private func buildAxesSection(axes: [VariationAxis]) -> String {
        var xml = "  <axes>\n"

        for axis in axes {
            xml += """
                <axis tag="\(escapeXML(axis.tag))" name="\(escapeXML(axis.name))" minimum="\(formatValue(axis.minValue))" default="\(formatValue(axis.defaultValue))" maximum="\(formatValue(axis.maxValue))"/>

            """
        }

        xml += "  </axes>\n\n"
        return xml
    }

    /// Build the <sources> section
    private func buildSourcesSection(
        project: FontProject,
        masters: [(FontMaster, String)],
        axes: [VariationAxis]
    ) -> String {
        var xml = "  <sources>\n"

        for (master, filename) in masters {
            let styleName = extractStyleName(from: master)

            xml += "    <source filename=\"\(escapeXML(filename))\" familyname=\"\(escapeXML(project.family))\" stylename=\"\(escapeXML(styleName))\">\n"
            xml += "      <location>\n"

            for axis in axes {
                if let value = master.location[axis.tag] {
                    xml += "        <dimension name=\"\(escapeXML(axis.name))\" xvalue=\"\(formatValue(value))\"/>\n"
                }
            }

            xml += "      </location>\n"
            xml += "    </source>\n"
        }

        xml += "  </sources>\n\n"
        return xml
    }

    /// Build the <instances> section
    private func buildInstancesSection(
        project: FontProject,
        instances: [NamedInstance],
        axes: [VariationAxis]
    ) -> String {
        var xml = "  <instances>\n"

        for instance in instances {
            xml += "    <instance familyname=\"\(escapeXML(project.family))\" stylename=\"\(escapeXML(instance.name))\">\n"
            xml += "      <location>\n"

            for axis in axes {
                if let value = instance.location[axis.tag] {
                    xml += "        <dimension name=\"\(escapeXML(axis.name))\" xvalue=\"\(formatValue(value))\"/>\n"
                }
            }

            xml += "      </location>\n"
            xml += "    </instance>\n"
        }

        xml += "  </instances>\n\n"
        return xml
    }

    // MARK: - Helper Methods

    /// Escape special XML characters
    private func escapeXML(_ string: String) -> String {
        var result = string
        result = result.replacingOccurrences(of: "&", with: "&amp;")
        result = result.replacingOccurrences(of: "<", with: "&lt;")
        result = result.replacingOccurrences(of: ">", with: "&gt;")
        result = result.replacingOccurrences(of: "\"", with: "&quot;")
        result = result.replacingOccurrences(of: "'", with: "&apos;")
        return result
    }

    /// Format a CGFloat value for XML output
    private func formatValue(_ value: CGFloat) -> String {
        if value.truncatingRemainder(dividingBy: 1) == 0 {
            return String(Int(value))
        }
        return String(format: "%.2f", value)
    }
}
