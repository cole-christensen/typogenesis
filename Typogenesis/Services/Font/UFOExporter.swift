import Foundation
import CoreGraphics

/// Exports FontProject to UFO (Unified Font Object) format
/// UFO is a directory-based format consisting of plist and glif (glyph) XML files
actor UFOExporter {

    enum UFOError: Error, LocalizedError {
        case failedToCreateDirectory
        case failedToWriteFile(String)
        case noGlyphs

        var errorDescription: String? {
            switch self {
            case .failedToCreateDirectory:
                return "Failed to create UFO directory structure"
            case .failedToWriteFile(let name):
                return "Failed to write file: \(name)"
            case .noGlyphs:
                return "Font has no glyphs to export"
            }
        }
    }

    /// Export options for UFO
    struct ExportOptions {
        var ufoVersion: Int = 3  // UFO version 3
        var includeKerning: Bool = true
        var includeLib: Bool = true  // Include lib.plist with custom data

        static let `default` = ExportOptions()
    }

    // MARK: - Public API

    /// Export a FontProject to UFO format at the specified URL
    func export(project: FontProject, to url: URL, options: ExportOptions = .default) async throws {
        guard !project.glyphs.isEmpty else {
            throw UFOError.noGlyphs
        }

        let fileManager = FileManager.default

        // Create UFO directory structure
        let ufoURL = url
        try fileManager.createDirectory(at: ufoURL, withIntermediateDirectories: true)

        let glyphsDir = ufoURL.appendingPathComponent("glyphs")
        try fileManager.createDirectory(at: glyphsDir, withIntermediateDirectories: true)

        // Write metainfo.plist
        let metainfo = buildMetainfo(version: options.ufoVersion)
        try metainfo.write(to: ufoURL.appendingPathComponent("metainfo.plist"), atomically: true, encoding: .utf8)

        // Write fontinfo.plist
        let fontinfo = buildFontinfo(project: project)
        try fontinfo.write(to: ufoURL.appendingPathComponent("fontinfo.plist"), atomically: true, encoding: .utf8)

        // Write lib.plist (custom data)
        if options.includeLib {
            let lib = buildLib(project: project)
            try lib.write(to: ufoURL.appendingPathComponent("lib.plist"), atomically: true, encoding: .utf8)
        }

        // Write kerning.plist
        if options.includeKerning && !project.kerning.isEmpty {
            let kerning = buildKerning(project: project)
            try kerning.write(to: ufoURL.appendingPathComponent("kerning.plist"), atomically: true, encoding: .utf8)
        }

        // Write groups.plist (empty for now)
        let groups = buildGroups()
        try groups.write(to: ufoURL.appendingPathComponent("groups.plist"), atomically: true, encoding: .utf8)

        // Write contents.plist and glyph files
        var glyphNameToFile: [(name: String, filename: String)] = []

        for (char, glyph) in project.glyphs {
            let glyphName = glyphNameFor(character: char)
            let filename = filenameFor(glyphName: glyphName)

            // Write glyph file
            let glifContent = buildGLIF(glyph: glyph, name: glyphName)
            try glifContent.write(to: glyphsDir.appendingPathComponent(filename), atomically: true, encoding: .utf8)

            glyphNameToFile.append((name: glyphName, filename: filename))
        }

        // Write glyphs/contents.plist
        let contents = buildGlyphContents(glyphs: glyphNameToFile)
        try contents.write(to: glyphsDir.appendingPathComponent("contents.plist"), atomically: true, encoding: .utf8)

        // Write layercontents.plist
        let layerContents = buildLayerContents()
        try layerContents.write(to: ufoURL.appendingPathComponent("layercontents.plist"), atomically: true, encoding: .utf8)
    }

    // MARK: - Plist Generation

    private func buildMetainfo(version: Int) -> String {
        return """
        <?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
        <plist version="1.0">
        <dict>
            <key>creator</key>
            <string>Typogenesis</string>
            <key>formatVersion</key>
            <integer>\(version)</integer>
        </dict>
        </plist>
        """
    }

    private func buildFontinfo(project: FontProject) -> String {
        var dict = [String: Any]()

        // Family info
        dict["familyName"] = project.family
        dict["styleName"] = project.style
        dict["styleMapFamilyName"] = project.family
        dict["styleMapStyleName"] = project.style.lowercased()

        // Metrics
        dict["unitsPerEm"] = project.metrics.unitsPerEm
        dict["ascender"] = project.metrics.ascender
        dict["descender"] = project.metrics.descender
        dict["xHeight"] = project.metrics.xHeight
        dict["capHeight"] = project.metrics.capHeight

        // Metadata
        if !project.metadata.copyright.isEmpty {
            dict["copyright"] = project.metadata.copyright
        }
        if !project.metadata.designer.isEmpty {
            dict["openTypeNameDesigner"] = project.metadata.designer
        }
        if !project.metadata.description.isEmpty {
            dict["openTypeNameDescription"] = project.metadata.description
        }
        if !project.metadata.license.isEmpty {
            dict["openTypeNameLicense"] = project.metadata.license
        }

        dict["versionMajor"] = 1
        dict["versionMinor"] = 0

        // OpenType-specific
        dict["openTypeHeadCreated"] = ISO8601DateFormatter().string(from: Date())
        dict["openTypeOS2WeightClass"] = 400  // Regular
        dict["openTypeOS2WidthClass"] = 5     // Medium

        return plistString(from: dict)
    }

    private func buildLib(project: FontProject) -> String {
        var dict = [String: Any]()

        // Store project-specific data
        dict["com.typogenesis.projectId"] = project.id.uuidString
        dict["public.glyphOrder"] = project.glyphs.keys.sorted().map { glyphNameFor(character: $0) }

        return plistString(from: dict)
    }

    private func buildKerning(project: FontProject) -> String {
        var kerningDict = [String: [String: Int]]()

        for pair in project.kerning {
            let leftName = glyphNameFor(character: pair.left)
            let rightName = glyphNameFor(character: pair.right)

            if kerningDict[leftName] == nil {
                kerningDict[leftName] = [:]
            }
            kerningDict[leftName]?[rightName] = pair.value
        }

        return plistString(from: kerningDict)
    }

    private func buildGroups() -> String {
        return """
        <?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
        <plist version="1.0">
        <dict>
        </dict>
        </plist>
        """
    }

    private func buildGlyphContents(glyphs: [(name: String, filename: String)]) -> String {
        var dict = [String: String]()
        for glyph in glyphs {
            dict[glyph.name] = glyph.filename
        }
        return plistString(from: dict)
    }

    private func buildLayerContents() -> String {
        return """
        <?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
        <plist version="1.0">
        <array>
            <array>
                <string>public.default</string>
                <string>glyphs</string>
            </array>
        </array>
        </plist>
        """
    }

    // MARK: - GLIF Generation

    private func buildGLIF(glyph: Glyph, name: String) -> String {
        var xml = """
        <?xml version="1.0" encoding="UTF-8"?>
        <glyph name="\(escapeXML(name))" format="2">
            <advance width="\(glyph.advanceWidth)"/>
            <unicode hex="\(unicodeHex(for: glyph.character))"/>
        """

        // Add outline if not empty
        if !glyph.outline.isEmpty {
            xml += "\n    <outline>"

            for contour in glyph.outline.contours {
                if contour.points.isEmpty { continue }

                xml += "\n        <contour>"

                for (index, point) in contour.points.enumerated() {
                    let prevPoint = contour.points[(index - 1 + contour.points.count) % contour.points.count]
                    let nextPoint = contour.points[(index + 1) % contour.points.count]

                    // Handle control points from previous point
                    if let controlOut = prevPoint.controlOut {
                        xml += """
                        \n            <point x="\(Int(controlOut.x))" y="\(Int(controlOut.y))"/>
                        """
                    }

                    // Handle incoming control point
                    if let controlIn = point.controlIn {
                        xml += """
                        \n            <point x="\(Int(controlIn.x))" y="\(Int(controlIn.y))"/>
                        """
                    }

                    // The on-curve point
                    let pointType: String
                    switch point.type {
                    case .corner:
                        if point.controlIn != nil || point.controlOut != nil {
                            pointType = "curve"
                        } else {
                            pointType = "line"
                        }
                    case .smooth, .symmetric:
                        pointType = "curve"
                    }

                    let smooth = (point.type == .smooth || point.type == .symmetric) ? " smooth=\"yes\"" : ""
                    xml += """
                    \n            <point x="\(Int(point.position.x))" y="\(Int(point.position.y))" type="\(pointType)"\(smooth)/>
                    """
                }

                xml += "\n        </contour>"
            }

            xml += "\n    </outline>"
        } else {
            xml += "\n    <outline/>"
        }

        xml += "\n</glyph>\n"
        return xml
    }

    // MARK: - Helper Methods

    private func glyphNameFor(character: Character) -> String {
        if let scalar = character.unicodeScalars.first {
            // Use standard names for common characters
            if let standardName = standardGlyphName(for: scalar) {
                return standardName
            }
            // Fall back to uni#### format
            return String(format: "uni%04X", scalar.value)
        }
        return "unknown"
    }

    private func filenameFor(glyphName: String) -> String {
        // UFO filename rules:
        // - Must be unique and case-insensitive unique
        // - Cannot start with a period
        // - Certain characters must be escaped with underscore
        var filename = glyphName

        // Handle uppercase collisions by appending underscore
        // For simplicity, just make the filename
        filename = filename.replacingOccurrences(of: "/", with: "_")
        filename = filename.replacingOccurrences(of: ":", with: "_")
        filename = filename.replacingOccurrences(of: "*", with: "_")
        filename = filename.replacingOccurrences(of: "?", with: "_")
        filename = filename.replacingOccurrences(of: "\"", with: "_")
        filename = filename.replacingOccurrences(of: "<", with: "_")
        filename = filename.replacingOccurrences(of: ">", with: "_")
        filename = filename.replacingOccurrences(of: "|", with: "_")

        return filename + ".glif"
    }

    private func unicodeHex(for character: Character) -> String {
        if let scalar = character.unicodeScalars.first {
            return String(format: "%04X", scalar.value)
        }
        return "0000"
    }

    private func standardGlyphName(for scalar: Unicode.Scalar) -> String? {
        let codepoint = scalar.value
        switch codepoint {
        case 0x0020: return "space"
        case 0x0021: return "exclam"
        case 0x0022: return "quotedbl"
        case 0x0023: return "numbersign"
        case 0x0024: return "dollar"
        case 0x0025: return "percent"
        case 0x0026: return "ampersand"
        case 0x0027: return "quotesingle"
        case 0x0028: return "parenleft"
        case 0x0029: return "parenright"
        case 0x002A: return "asterisk"
        case 0x002B: return "plus"
        case 0x002C: return "comma"
        case 0x002D: return "hyphen"
        case 0x002E: return "period"
        case 0x002F: return "slash"
        case 0x0030...0x0039: return ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"][Int(codepoint - 0x0030)]
        case 0x003A: return "colon"
        case 0x003B: return "semicolon"
        case 0x003C: return "less"
        case 0x003D: return "equal"
        case 0x003E: return "greater"
        case 0x003F: return "question"
        case 0x0040: return "at"
        case 0x0041...0x005A: return String(scalar)  // A-Z
        case 0x005B: return "bracketleft"
        case 0x005C: return "backslash"
        case 0x005D: return "bracketright"
        case 0x005E: return "asciicircum"
        case 0x005F: return "underscore"
        case 0x0060: return "grave"
        case 0x0061...0x007A: return String(scalar)  // a-z
        case 0x007B: return "braceleft"
        case 0x007C: return "bar"
        case 0x007D: return "braceright"
        case 0x007E: return "asciitilde"
        default: return nil
        }
    }

    private func escapeXML(_ string: String) -> String {
        var result = string
        result = result.replacingOccurrences(of: "&", with: "&amp;")
        result = result.replacingOccurrences(of: "<", with: "&lt;")
        result = result.replacingOccurrences(of: ">", with: "&gt;")
        result = result.replacingOccurrences(of: "\"", with: "&quot;")
        result = result.replacingOccurrences(of: "'", with: "&apos;")
        return result
    }

    private func plistString(from dict: [String: Any]) -> String {
        var xml = """
        <?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
        <plist version="1.0">
        <dict>
        """

        let sortedKeys = dict.keys.sorted()
        for key in sortedKeys {
            if let value = dict[key] {
                xml += "\n    <key>\(escapeXML(key))</key>"
                xml += plistValue(value, indent: 1)
            }
        }

        xml += "\n</dict>\n</plist>\n"
        return xml
    }

    private func plistValue(_ value: Any, indent: Int) -> String {
        let spaces = String(repeating: "    ", count: indent)

        switch value {
        case let string as String:
            return "\n\(spaces)<string>\(escapeXML(string))</string>"
        case let int as Int:
            return "\n\(spaces)<integer>\(int)</integer>"
        case let double as Double:
            return "\n\(spaces)<real>\(double)</real>"
        case let bool as Bool:
            return "\n\(spaces)<\(bool ? "true" : "false")/>"
        case let array as [Any]:
            var result = "\n\(spaces)<array>"
            for item in array {
                result += plistValue(item, indent: indent + 1)
            }
            result += "\n\(spaces)</array>"
            return result
        case let dict as [String: Any]:
            var result = "\n\(spaces)<dict>"
            let sortedKeys = dict.keys.sorted()
            for key in sortedKeys {
                if let v = dict[key] {
                    result += "\n\(spaces)    <key>\(escapeXML(key))</key>"
                    result += plistValue(v, indent: indent + 1)
                }
            }
            result += "\n\(spaces)</dict>"
            return result
        case let dict as [String: Int]:
            var result = "\n\(spaces)<dict>"
            let sortedKeys = dict.keys.sorted()
            for key in sortedKeys {
                if let v = dict[key] {
                    result += "\n\(spaces)    <key>\(escapeXML(key))</key>"
                    result += plistValue(v, indent: indent + 1)
                }
            }
            result += "\n\(spaces)</dict>"
            return result
        case let dict as [String: String]:
            var result = "\n\(spaces)<dict>"
            let sortedKeys = dict.keys.sorted()
            for key in sortedKeys {
                if let v = dict[key] {
                    result += "\n\(spaces)    <key>\(escapeXML(key))</key>"
                    result += plistValue(v, indent: indent + 1)
                }
            }
            result += "\n\(spaces)</dict>"
            return result
        case let dict as [String: [String: Int]]:
            var result = "\n\(spaces)<dict>"
            let sortedKeys = dict.keys.sorted()
            for key in sortedKeys {
                if let v = dict[key] {
                    result += "\n\(spaces)    <key>\(escapeXML(key))</key>"
                    result += plistValue(v, indent: indent + 1)
                }
            }
            result += "\n\(spaces)</dict>"
            return result
        default:
            return "\n\(spaces)<string>\(String(describing: value))</string>"
        }
    }
}
