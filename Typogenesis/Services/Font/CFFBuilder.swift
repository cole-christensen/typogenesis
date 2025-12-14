import Foundation
import CoreGraphics

/// Builds CFF (Compact Font Format) tables for OpenType fonts
/// CFF is required for .otf files and uses cubic bezier curves (vs TrueType's quadratic)
struct CFFBuilder {

    enum CFFError: Error, LocalizedError {
        case invalidGlyphData
        case encodingFailed(String)

        var errorDescription: String? {
            switch self {
            case .invalidGlyphData:
                return "Invalid glyph data for CFF encoding"
            case .encodingFailed(let reason):
                return "CFF encoding failed: \(reason)"
            }
        }
    }

    // MARK: - Public API

    /// Build a complete CFF table from a font project
    static func build(project: FontProject, glyphOrder: [(Character?, Glyph?)]) throws -> Data {
        var cff = Data()

        // CFF Header
        cff.append(buildHeader())

        // Name INDEX
        let nameIndex = buildNameIndex(name: project.family)
        cff.append(nameIndex)

        // Top DICT INDEX
        let topDictData = buildTopDict(project: project, glyphCount: glyphOrder.count)
        let topDictIndex = buildIndex([topDictData])
        cff.append(topDictIndex)

        // String INDEX
        let stringIndex = buildStringIndex(project: project)
        cff.append(stringIndex)

        // Global Subr INDEX (empty)
        let globalSubrIndex = buildIndex([])
        cff.append(globalSubrIndex)

        // CharStrings INDEX
        let charStrings = try buildCharStrings(glyphOrder: glyphOrder, metrics: project.metrics)
        cff.append(charStrings)

        // Private DICT (minimal)
        let privateDict = buildPrivateDict(project: project)
        cff.append(privateDict)

        return cff
    }

    // MARK: - CFF Header

    private static func buildHeader() -> Data {
        var header = Data()
        header.append(1)      // major version
        header.append(0)      // minor version
        header.append(4)      // header size
        header.append(4)      // offSize (4 bytes for offsets)
        return header
    }

    // MARK: - INDEX structures

    /// Build a CFF INDEX structure
    private static func buildIndex(_ items: [Data]) -> Data {
        var index = Data()

        let count = UInt16(items.count)
        index.writeUInt16(count)

        if count == 0 {
            return index
        }

        // Calculate offset size needed
        let totalDataSize = items.reduce(0) { $0 + $1.count }
        let offSize: UInt8
        if totalDataSize < 256 {
            offSize = 1
        } else if totalDataSize < 65536 {
            offSize = 2
        } else if totalDataSize < 16777216 {
            offSize = 3
        } else {
            offSize = 4
        }

        index.append(offSize)

        // Write offsets (1-based, first offset is always 1)
        var offset = 1
        writeOffset(&index, offset: offset, size: offSize)
        for item in items {
            offset += item.count
            writeOffset(&index, offset: offset, size: offSize)
        }

        // Write data
        for item in items {
            index.append(item)
        }

        return index
    }

    private static func writeOffset(_ data: inout Data, offset: Int, size: UInt8) {
        switch size {
        case 1:
            data.append(UInt8(offset))
        case 2:
            data.writeUInt16(UInt16(offset))
        case 3:
            data.append(UInt8((offset >> 16) & 0xFF))
            data.append(UInt8((offset >> 8) & 0xFF))
            data.append(UInt8(offset & 0xFF))
        case 4:
            data.writeUInt32(UInt32(offset))
        default:
            break
        }
    }

    // MARK: - Name INDEX

    private static func buildNameIndex(name: String) -> Data {
        let nameData = Data(name.utf8)
        return buildIndex([nameData])
    }

    // MARK: - Top DICT

    private static func buildTopDict(project: FontProject, glyphCount: Int) -> Data {
        var dict = Data()

        // version (SID)
        encodeDictNumber(&dict, value: 391)  // SID for version string
        dict.append(0)  // version operator

        // FullName (SID)
        encodeDictNumber(&dict, value: 392)
        dict.append(2)  // FullName operator

        // FamilyName (SID)
        encodeDictNumber(&dict, value: 393)
        dict.append(3)  // FamilyName operator

        // Weight (SID)
        encodeDictNumber(&dict, value: 394)
        dict.append(4)  // Weight operator

        // FontBBox
        encodeDictNumber(&dict, value: 0)
        encodeDictNumber(&dict, value: project.metrics.descender)
        encodeDictNumber(&dict, value: project.metrics.unitsPerEm)
        encodeDictNumber(&dict, value: project.metrics.ascender)
        dict.append(5)  // FontBBox operator

        // charset offset (will be standard)
        encodeDictNumber(&dict, value: 0)
        dict.append(15)  // charset operator

        // Encoding (standard)
        encodeDictNumber(&dict, value: 0)
        dict.append(16)  // Encoding operator

        // CharStrings offset (placeholder - would need to calculate actual offset)
        encodeDictNumber(&dict, value: 0)
        dict.append(17)  // CharStrings operator

        // Private DICT size and offset
        encodeDictNumber(&dict, value: 45)  // size
        encodeDictNumber(&dict, value: 0)   // offset (placeholder)
        dict.append(18)  // Private operator

        return dict
    }

    // MARK: - String INDEX

    private static func buildStringIndex(project: FontProject) -> Data {
        // Standard strings + custom strings
        let strings = [
            "1.0",                    // SID 391 - version
            project.name,             // SID 392 - FullName
            project.family,           // SID 393 - FamilyName
            project.style             // SID 394 - Weight
        ]

        let stringData = strings.map { Data($0.utf8) }
        return buildIndex(stringData)
    }

    // MARK: - CharStrings

    private static func buildCharStrings(
        glyphOrder: [(Character?, Glyph?)],
        metrics: FontMetrics
    ) throws -> Data {
        var charStringData: [Data] = []

        for (_, glyph) in glyphOrder {
            let charString: Data
            if let glyph = glyph, !glyph.outline.isEmpty {
                charString = try encodeGlyphToCharString(glyph: glyph, metrics: metrics)
            } else {
                // Empty/.notdef glyph
                charString = encodeEmptyCharString(width: metrics.unitsPerEm / 2)
            }
            charStringData.append(charString)
        }

        return buildIndex(charStringData)
    }

    private static func encodeGlyphToCharString(glyph: Glyph, metrics: FontMetrics) throws -> Data {
        var cs = Data()

        // Width (as difference from defaultWidthX)
        let width = glyph.advanceWidth
        encodeDictNumber(&cs, value: width)

        // Encode contours
        for contour in glyph.outline.contours where !contour.points.isEmpty {
            let points = contour.points

            // moveto first point
            if let first = points.first {
                let x = Int(first.position.x)
                let y = Int(first.position.y)
                encodeDictNumber(&cs, value: x)
                encodeDictNumber(&cs, value: y)
                cs.append(21)  // rmoveto
            }

            // Draw remaining points
            for i in 1..<points.count {
                let prev = points[i - 1]
                let curr = points[i]

                let dx = Int(curr.position.x) - Int(prev.position.x)
                let dy = Int(curr.position.y) - Int(prev.position.y)

                if let ctrlOut = prev.controlOut, let ctrlIn = curr.controlIn {
                    // Cubic bezier curve
                    let dx1 = Int(ctrlOut.x) - Int(prev.position.x)
                    let dy1 = Int(ctrlOut.y) - Int(prev.position.y)
                    let dx2 = Int(ctrlIn.x) - Int(ctrlOut.x)
                    let dy2 = Int(ctrlIn.y) - Int(ctrlOut.y)
                    let dx3 = Int(curr.position.x) - Int(ctrlIn.x)
                    let dy3 = Int(curr.position.y) - Int(ctrlIn.y)

                    encodeDictNumber(&cs, value: dx1)
                    encodeDictNumber(&cs, value: dy1)
                    encodeDictNumber(&cs, value: dx2)
                    encodeDictNumber(&cs, value: dy2)
                    encodeDictNumber(&cs, value: dx3)
                    encodeDictNumber(&cs, value: dy3)
                    cs.append(8)  // rrcurveto
                } else {
                    // Line
                    encodeDictNumber(&cs, value: dx)
                    encodeDictNumber(&cs, value: dy)
                    cs.append(5)  // rlineto
                }
            }

            // Close path if needed
            if contour.isClosed {
                // closepath is implicit before endchar
            }
        }

        cs.append(14)  // endchar
        return cs
    }

    private static func encodeEmptyCharString(width: Int) -> Data {
        var cs = Data()
        encodeDictNumber(&cs, value: width)
        cs.append(14)  // endchar
        return cs
    }

    // MARK: - Private DICT

    private static func buildPrivateDict(project: FontProject) -> Data {
        var dict = Data()

        // BlueValues (empty)
        encodeDictNumber(&dict, value: 0)
        encodeDictNumber(&dict, value: 0)
        dict.append(6)  // BlueValues operator

        // StdHW
        encodeDictNumber(&dict, value: 80)
        dict.append(10)  // StdHW operator

        // StdVW
        encodeDictNumber(&dict, value: 80)
        dict.append(11)  // StdVW operator

        // defaultWidthX
        encodeDictNumber(&dict, value: project.metrics.unitsPerEm / 2)
        dict.append(20)  // defaultWidthX operator

        // nominalWidthX
        encodeDictNumber(&dict, value: project.metrics.unitsPerEm / 2)
        dict.append(21)  // nominalWidthX operator

        return dict
    }

    // MARK: - Number Encoding

    /// Encode a number in CFF DICT format
    private static func encodeDictNumber(_ data: inout Data, value: Int) {
        if value >= -107 && value <= 107 {
            // Single byte encoding
            data.append(UInt8(value + 139))
        } else if value >= 108 && value <= 1131 {
            // Two byte positive encoding
            let adjusted = value - 108
            data.append(UInt8((adjusted >> 8) + 247))
            data.append(UInt8(adjusted & 0xFF))
        } else if value >= -1131 && value <= -108 {
            // Two byte negative encoding
            let adjusted = -value - 108
            data.append(UInt8((adjusted >> 8) + 251))
            data.append(UInt8(adjusted & 0xFF))
        } else if value >= -32768 && value <= 32767 {
            // Three byte encoding
            data.append(28)
            data.append(UInt8((value >> 8) & 0xFF))
            data.append(UInt8(value & 0xFF))
        } else {
            // Five byte encoding
            data.append(29)
            data.append(UInt8((value >> 24) & 0xFF))
            data.append(UInt8((value >> 16) & 0xFF))
            data.append(UInt8((value >> 8) & 0xFF))
            data.append(UInt8(value & 0xFF))
        }
    }
}

