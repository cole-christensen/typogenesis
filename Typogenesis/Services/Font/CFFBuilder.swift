import Foundation
import CoreGraphics

/// Builds CFF (Compact Font Format) tables for OpenType fonts
/// CFF is required for .otf files and uses cubic bezier curves (vs TrueType's quadratic)
struct CFFBuilder {

    enum CFFError: Error, LocalizedError {
        case invalidGlyphData
        case encodingFailed(String)
        case tooManyItems
        case valueOutOfRange(Int)

        var errorDescription: String? {
            switch self {
            case .invalidGlyphData:
                return "Invalid glyph data for CFF encoding"
            case .encodingFailed(let reason):
                return "CFF encoding failed: \(reason)"
            case .tooManyItems:
                return "CFF INDEX count exceeds maximum of 65535"
            case .valueOutOfRange(let value):
                return "CharString number \(value) outside encodable range -32768..32767"
            }
        }
    }

    // MARK: - Shared Constants

    /// Nominal width used for CharString width delta encoding.
    /// Both build() and buildPrivateDict() must agree on this value.
    /// Computed as safeUnitsPerEm / 2.
    private static func nominalWidth(for project: FontProject) -> Int {
        let safeUnitsPerEm = max(project.metrics.unitsPerEm, 1)
        return safeUnitsPerEm / 2
    }

    // MARK: - Public API

    /// Build a complete CFF table from a font project
    /// Uses two-pass approach: build CharStrings and Private DICT first,
    /// then compute real offsets for the Top DICT.
    static func build(project: FontProject, glyphOrder: [(Character?, Glyph?)]) throws -> Data {
        // N1: Use shared nominalWidth source of truth
        let nomWidth = nominalWidth(for: project)

        // --- Compute FontBBox from actual glyph bounding boxes ---
        let fontBBox = computeFontBBox(glyphOrder: glyphOrder, metrics: project.metrics)

        // --- Pass 1: Build all sections except Top DICT to determine sizes ---

        let header = buildHeader()
        let nameIndex = try buildNameIndex(name: project.family)
        let stringIndex = try buildStringIndex(project: project)
        let globalSubrIndex = try buildIndex([])
        let charStrings = try buildCharStrings(
            glyphOrder: glyphOrder,
            metrics: project.metrics,
            nominalWidth: nomWidth
        )
        let privateDict = buildPrivateDict(project: project)

        // --- Pass 2: Build Top DICT with real offsets ---

        // We need a placeholder Top DICT to measure its INDEX size.
        // Top DICT offsets depend on the Top DICT INDEX size, which depends on the Top DICT.
        // Resolve by computing with maximum-size placeholders first, then adjusting.

        // Calculate offset to CharStrings:
        // header + nameIndex + topDictIndex + stringIndex + globalSubrIndex
        // topDictIndex size depends on topDict data size, so we iterate.

        // Start with estimated offsets, then converge
        var charStringsOffset = 0
        var privateDictOffset = 0
        var topDictData = Data()

        for _ in 0..<3 {
            topDictData = buildTopDict(
                project: project,
                fontBBox: fontBBox,
                charStringsOffset: charStringsOffset,
                privateDictSize: privateDict.count,
                privateDictOffset: privateDictOffset
            )
            let topDictIndex = try buildIndex([topDictData])

            charStringsOffset = header.count + nameIndex.count + topDictIndex.count +
                stringIndex.count + globalSubrIndex.count
            privateDictOffset = charStringsOffset + charStrings.count
        }

        // C5: Verify convergence — recompute with final offsets and confirm they match
        let verifyTopDict = buildTopDict(
            project: project,
            fontBBox: fontBBox,
            charStringsOffset: charStringsOffset,
            privateDictSize: privateDict.count,
            privateDictOffset: privateDictOffset
        )
        let verifyTopDictIndex = try buildIndex([verifyTopDict])
        let verifyCharStringsOffset = header.count + nameIndex.count + verifyTopDictIndex.count +
            stringIndex.count + globalSubrIndex.count
        let verifyPrivateDictOffset = verifyCharStringsOffset + charStrings.count

        if verifyCharStringsOffset != charStringsOffset || verifyPrivateDictOffset != privateDictOffset {
            assertionFailure("CFF Top DICT offset convergence failed after 3 iterations: " +
                "charStrings \(charStringsOffset) vs \(verifyCharStringsOffset), " +
                "private \(privateDictOffset) vs \(verifyPrivateDictOffset)")
            #if DEBUG
            print("WARNING: CFF Top DICT offset convergence failed after 3 iterations: " +
                "charStrings \(charStringsOffset) vs \(verifyCharStringsOffset), " +
                "private \(privateDictOffset) vs \(verifyPrivateDictOffset)")
            #endif
        }

        let topDictIndex = try buildIndex([topDictData])

        // --- Assemble the final CFF ---

        var cff = Data()
        cff.append(header)
        cff.append(nameIndex)
        cff.append(topDictIndex)
        cff.append(stringIndex)
        cff.append(globalSubrIndex)
        cff.append(charStrings)
        cff.append(privateDict)

        return cff
    }

    // MARK: - FontBBox Computation

    /// Compute the font bounding box from actual glyph outlines.
    /// Falls back to metrics-based estimates if no glyphs have outlines.
    private static func computeFontBBox(
        glyphOrder: [(Character?, Glyph?)],
        metrics: FontMetrics
    ) -> (xMin: Int, yMin: Int, xMax: Int, yMax: Int) {
        var xMin = Int.max
        var yMin = Int.max
        var xMax = Int.min
        var yMax = Int.min
        var hasPoints = false

        for (_, glyph) in glyphOrder {
            guard let glyph = glyph else { continue }
            for contour in glyph.outline.contours {
                for point in contour.points {
                    hasPoints = true
                    let px = Int(point.position.x)
                    let py = Int(point.position.y)
                    xMin = min(xMin, px)
                    yMin = min(yMin, py)
                    xMax = max(xMax, px)
                    yMax = max(yMax, py)

                    // Include control points in bounding box
                    if let ctrlIn = point.controlIn {
                        xMin = min(xMin, Int(ctrlIn.x))
                        yMin = min(yMin, Int(ctrlIn.y))
                        xMax = max(xMax, Int(ctrlIn.x))
                        yMax = max(yMax, Int(ctrlIn.y))
                    }
                    if let ctrlOut = point.controlOut {
                        xMin = min(xMin, Int(ctrlOut.x))
                        yMin = min(yMin, Int(ctrlOut.y))
                        xMax = max(xMax, Int(ctrlOut.x))
                        yMax = max(yMax, Int(ctrlOut.y))
                    }
                }
            }
        }

        if hasPoints {
            return (xMin: xMin, yMin: yMin, xMax: xMax, yMax: yMax)
        }

        // Fallback to metrics-based estimate when no glyph outlines exist
        return (
            xMin: 0,
            yMin: metrics.descender,
            xMax: metrics.unitsPerEm,
            yMax: metrics.ascender
        )
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
    /// C6: Throws if items.count exceeds the CFF INDEX maximum of 65535.
    private static func buildIndex(_ items: [Data]) throws -> Data {
        guard items.count <= Int(UInt16.max) else {
            throw CFFError.tooManyItems
        }

        var index = Data()

        let count = UInt16(items.count)
        index.writeUInt16(count)

        if count == 0 {
            return index
        }

        // Calculate offset size needed
        // CFF offsets are 1-based, so max offset = totalDataSize + 1
        let totalDataSize = items.reduce(0) { $0 + $1.count }
        let maxOffset = totalDataSize + 1
        let offSize: UInt8
        if maxOffset <= 0xFF {
            offSize = 1
        } else if maxOffset <= 0xFFFF {
            offSize = 2
        } else if maxOffset <= 0xFFFFFF {
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

    private static func buildNameIndex(name: String) throws -> Data {
        let nameData = Data(name.utf8)
        return try buildIndex([nameData])
    }

    // MARK: - Top DICT

    private static func buildTopDict(
        project: FontProject,
        fontBBox: (xMin: Int, yMin: Int, xMax: Int, yMax: Int),
        charStringsOffset: Int,
        privateDictSize: Int,
        privateDictOffset: Int
    ) -> Data {
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

        // FontBBox (computed from actual glyph bounding boxes)
        encodeDictNumber(&dict, value: fontBBox.xMin)
        encodeDictNumber(&dict, value: fontBBox.yMin)
        encodeDictNumber(&dict, value: fontBBox.xMax)
        encodeDictNumber(&dict, value: fontBBox.yMax)
        dict.append(5)  // FontBBox operator

        // charset offset (will be standard)
        encodeDictNumber(&dict, value: 0)
        dict.append(15)  // charset operator

        // Encoding (standard)
        encodeDictNumber(&dict, value: 0)
        dict.append(16)  // Encoding operator

        // CharStrings offset (real computed offset)
        encodeDictNumber(&dict, value: charStringsOffset)
        dict.append(17)  // CharStrings operator

        // Private DICT size and offset (real computed values)
        encodeDictNumber(&dict, value: privateDictSize)
        encodeDictNumber(&dict, value: privateDictOffset)
        dict.append(18)  // Private operator

        return dict
    }

    // MARK: - String INDEX

    private static func buildStringIndex(project: FontProject) throws -> Data {
        // Standard strings + custom strings
        let strings = [
            "1.0",                    // SID 391 - version
            project.name,             // SID 392 - FullName
            project.family,           // SID 393 - FamilyName
            project.style             // SID 394 - Weight
        ]

        let stringData = strings.map { Data($0.utf8) }
        return try buildIndex(stringData)
    }

    // MARK: - CharStrings

    private static func buildCharStrings(
        glyphOrder: [(Character?, Glyph?)],
        metrics: FontMetrics,
        nominalWidth: Int
    ) throws -> Data {
        var charStringData: [Data] = []

        // Guard against division by zero
        let safeUnitsPerEm = max(metrics.unitsPerEm, 1)

        for (_, glyph) in glyphOrder {
            let charString: Data
            if let glyph = glyph, !glyph.outline.isEmpty {
                charString = try encodeGlyphToCharString(
                    glyph: glyph,
                    metrics: metrics,
                    nominalWidth: nominalWidth
                )
            } else {
                // Empty/.notdef glyph - width as delta from nominalWidth
                let defaultWidth = safeUnitsPerEm / 2
                charString = try encodeEmptyCharString(widthDelta: defaultWidth - nominalWidth)
            }
            charStringData.append(charString)
        }

        return try buildIndex(charStringData)
    }

    private static func encodeGlyphToCharString(
        glyph: Glyph,
        metrics: FontMetrics,
        nominalWidth: Int
    ) throws -> Data {
        var cs = Data()

        // B1: Track current point across the entire charstring for relative deltas.
        // Type 2 CharString operators (rmoveto, rlineto, rrcurveto) all use relative coordinates.
        var currentX = 0
        var currentY = 0

        // Width encoded as delta from nominalWidthX (Type 2 CharString convention)
        let widthDelta = glyph.advanceWidth - nominalWidth
        try encodeCharStringNumber(&cs, value: widthDelta)

        // Encode contours
        for contour in glyph.outline.contours where !contour.points.isEmpty {
            let points = contour.points

            // moveto first point — relative to current point (B1 fix)
            if let first = points.first {
                let targetX = Int(first.position.x.rounded())
                let targetY = Int(first.position.y.rounded())
                let dx = targetX - currentX
                let dy = targetY - currentY
                try encodeCharStringNumber(&cs, value: dx)
                try encodeCharStringNumber(&cs, value: dy)
                cs.append(21)  // rmoveto
                currentX = targetX
                currentY = targetY
            }

            // Draw remaining points
            for i in 1..<points.count {
                let prev = points[i - 1]
                let curr = points[i]

                if let ctrlOut = prev.controlOut, let ctrlIn = curr.controlIn {
                    // Cubic bezier curve — all deltas relative to current point
                    let dx1 = Int(ctrlOut.x.rounded()) - currentX
                    let dy1 = Int(ctrlOut.y.rounded()) - currentY
                    let dx2 = Int(ctrlIn.x.rounded()) - Int(ctrlOut.x.rounded())
                    let dy2 = Int(ctrlIn.y.rounded()) - Int(ctrlOut.y.rounded())
                    let dx3 = Int(curr.position.x.rounded()) - Int(ctrlIn.x.rounded())
                    let dy3 = Int(curr.position.y.rounded()) - Int(ctrlIn.y.rounded())

                    try encodeCharStringNumber(&cs, value: dx1)
                    try encodeCharStringNumber(&cs, value: dy1)
                    try encodeCharStringNumber(&cs, value: dx2)
                    try encodeCharStringNumber(&cs, value: dy2)
                    try encodeCharStringNumber(&cs, value: dx3)
                    try encodeCharStringNumber(&cs, value: dy3)
                    cs.append(8)  // rrcurveto
                    currentX = Int(curr.position.x.rounded())
                    currentY = Int(curr.position.y.rounded())
                } else if prev.controlOut != nil || curr.controlIn != nil {
                    // Quadratic bezier (one control point) — convert to cubic losslessly.
                    // For a quadratic with control point P, start S, end E:
                    //   cp1 = S + 2/3 * (P - S)
                    //   cp2 = E + 2/3 * (P - E)
                    let startX = CGFloat(currentX)
                    let startY = CGFloat(currentY)
                    let endX = curr.position.x
                    let endY = curr.position.y
                    let quadCtrl = prev.controlOut ?? curr.controlIn!

                    let cp1x = startX + 2.0 / 3.0 * (quadCtrl.x - startX)
                    let cp1y = startY + 2.0 / 3.0 * (quadCtrl.y - startY)
                    let cp2x = endX + 2.0 / 3.0 * (quadCtrl.x - endX)
                    let cp2y = endY + 2.0 / 3.0 * (quadCtrl.y - endY)

                    let dx1 = Int(cp1x.rounded()) - currentX
                    let dy1 = Int(cp1y.rounded()) - currentY
                    let dx2 = Int(cp2x.rounded()) - Int(cp1x.rounded())
                    let dy2 = Int(cp2y.rounded()) - Int(cp1y.rounded())
                    let dx3 = Int(endX.rounded()) - Int(cp2x.rounded())
                    let dy3 = Int(endY.rounded()) - Int(cp2y.rounded())

                    try encodeCharStringNumber(&cs, value: dx1)
                    try encodeCharStringNumber(&cs, value: dy1)
                    try encodeCharStringNumber(&cs, value: dx2)
                    try encodeCharStringNumber(&cs, value: dy2)
                    try encodeCharStringNumber(&cs, value: dx3)
                    try encodeCharStringNumber(&cs, value: dy3)
                    cs.append(8)  // rrcurveto
                    currentX = Int(endX.rounded())
                    currentY = Int(endY.rounded())
                } else {
                    // Line — relative to current point
                    let dx = Int(curr.position.x.rounded()) - currentX
                    let dy = Int(curr.position.y.rounded()) - currentY
                    try encodeCharStringNumber(&cs, value: dx)
                    try encodeCharStringNumber(&cs, value: dy)
                    cs.append(5)  // rlineto
                    currentX = Int(curr.position.x.rounded())
                    currentY = Int(curr.position.y.rounded())
                }
            }

            // B3: Emit explicit closing segment for closed contours.
            // CFF implicit closepath draws a straight line from current point to subpath start.
            // If the closing segment should be a curve, we must emit it explicitly.
            if contour.isClosed && points.count > 1 {
                let lastPoint = points[points.count - 1]
                let firstPoint = points[0]

                let firstX = Int(firstPoint.position.x.rounded())
                let firstY = Int(firstPoint.position.y.rounded())

                // Only emit closing segment if current point isn't already at the first point
                if currentX != firstX || currentY != firstY
                    || lastPoint.controlOut != nil || firstPoint.controlIn != nil {

                    if let ctrlOut = lastPoint.controlOut, let ctrlIn = firstPoint.controlIn {
                        // Closing segment is a cubic bezier curve
                        let dx1 = Int(ctrlOut.x.rounded()) - currentX
                        let dy1 = Int(ctrlOut.y.rounded()) - currentY
                        let dx2 = Int(ctrlIn.x.rounded()) - Int(ctrlOut.x.rounded())
                        let dy2 = Int(ctrlIn.y.rounded()) - Int(ctrlOut.y.rounded())
                        let dx3 = firstX - Int(ctrlIn.x.rounded())
                        let dy3 = firstY - Int(ctrlIn.y.rounded())

                        try encodeCharStringNumber(&cs, value: dx1)
                        try encodeCharStringNumber(&cs, value: dy1)
                        try encodeCharStringNumber(&cs, value: dx2)
                        try encodeCharStringNumber(&cs, value: dy2)
                        try encodeCharStringNumber(&cs, value: dx3)
                        try encodeCharStringNumber(&cs, value: dy3)
                        cs.append(8)  // rrcurveto
                        currentX = firstX
                        currentY = firstY
                    } else if lastPoint.controlOut != nil || firstPoint.controlIn != nil {
                        // Closing segment is a quadratic bezier (one control point) —
                        // convert to cubic losslessly using the same 2/3 rule.
                        let startX = CGFloat(currentX)
                        let startY = CGFloat(currentY)
                        let endX = CGFloat(firstX)
                        let endY = CGFloat(firstY)
                        let quadCtrl = lastPoint.controlOut ?? firstPoint.controlIn!

                        let cp1x = startX + 2.0 / 3.0 * (quadCtrl.x - startX)
                        let cp1y = startY + 2.0 / 3.0 * (quadCtrl.y - startY)
                        let cp2x = endX + 2.0 / 3.0 * (quadCtrl.x - endX)
                        let cp2y = endY + 2.0 / 3.0 * (quadCtrl.y - endY)

                        let dx1 = Int(cp1x.rounded()) - currentX
                        let dy1 = Int(cp1y.rounded()) - currentY
                        let dx2 = Int(cp2x.rounded()) - Int(cp1x.rounded())
                        let dy2 = Int(cp2y.rounded()) - Int(cp1y.rounded())
                        let dx3 = firstX - Int(cp2x.rounded())
                        let dy3 = firstY - Int(cp2y.rounded())

                        try encodeCharStringNumber(&cs, value: dx1)
                        try encodeCharStringNumber(&cs, value: dy1)
                        try encodeCharStringNumber(&cs, value: dx2)
                        try encodeCharStringNumber(&cs, value: dy2)
                        try encodeCharStringNumber(&cs, value: dx3)
                        try encodeCharStringNumber(&cs, value: dy3)
                        cs.append(8)  // rrcurveto
                        currentX = firstX
                        currentY = firstY
                    } else if currentX != firstX || currentY != firstY {
                        // Closing segment is a straight line (only if we're not already there)
                        let dx = firstX - currentX
                        let dy = firstY - currentY
                        try encodeCharStringNumber(&cs, value: dx)
                        try encodeCharStringNumber(&cs, value: dy)
                        cs.append(5)  // rlineto
                        currentX = firstX
                        currentY = firstY
                    }
                    // Note: implicit closepath will now draw a zero-length line (no-op)
                }
            }
        }

        cs.append(14)  // endchar
        return cs
    }

    private static func encodeEmptyCharString(widthDelta: Int) throws -> Data {
        var cs = Data()
        try encodeCharStringNumber(&cs, value: widthDelta)
        cs.append(14)  // endchar
        return cs
    }

    // MARK: - Private DICT

    private static func buildPrivateDict(project: FontProject) -> Data {
        var dict = Data()

        // N1: Use shared nominalWidth source of truth
        let nomWidth = nominalWidth(for: project)

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

        // N2: defaultWidthX and nominalWidthX are intentionally set to the same value.
        // This is technically valid per the CFF spec — it means glyphs whose actual width
        // equals the default don't need to encode a width operand at all. Setting them equal
        // wastes a small amount of space (the width delta is always encoded even when 0),
        // but simplifies the encoder. A future optimization could set defaultWidthX to the
        // most common glyph width and omit the width operand for matching glyphs.

        // defaultWidthX (two-byte operator: 12 20)
        encodeDictNumber(&dict, value: nomWidth)
        dict.append(12)  // escape byte for two-byte operator
        dict.append(20)  // defaultWidthX operator

        // nominalWidthX (two-byte operator: 12 21)
        encodeDictNumber(&dict, value: nomWidth)
        dict.append(12)  // escape byte for two-byte operator
        dict.append(21)  // nominalWidthX operator

        return dict
    }

    // MARK: - Number Encoding

    /// Encode a number in CFF DICT format
    /// Used for Top DICT and Private DICT operands only.
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
            // Five byte encoding (DICT uses byte 29)
            data.append(29)
            data.append(UInt8((value >> 24) & 0xFF))
            data.append(UInt8((value >> 16) & 0xFF))
            data.append(UInt8((value >> 8) & 0xFF))
            data.append(UInt8(value & 0xFF))
        }
    }

    /// Encode a number in Type 2 CharString format.
    /// CharStrings use a different 5-byte encoding (byte 255) than DICT (byte 29).
    ///
    /// B2: Per Adobe TN#5177, byte 255 in Type 2 CharStrings encodes a Fixed 16.16 number,
    /// NOT a plain 32-bit integer. Font coordinates should always fit in -32768..32767
    /// (the 3-byte encoding range). Values outside this range indicate a bug in the caller,
    /// so we throw an error rather than silently producing corrupt data.
    private static func encodeCharStringNumber(_ data: inout Data, value: Int) throws {
        if value >= -107 && value <= 107 {
            data.append(UInt8(value + 139))
        } else if value >= 108 && value <= 1131 {
            let adjusted = value - 108
            data.append(UInt8((adjusted >> 8) + 247))
            data.append(UInt8(adjusted & 0xFF))
        } else if value >= -1131 && value <= -108 {
            let adjusted = -value - 108
            data.append(UInt8((adjusted >> 8) + 251))
            data.append(UInt8(adjusted & 0xFF))
        } else if value >= -32768 && value <= 32767 {
            // Three byte encoding (same for both DICT and CharString)
            data.append(28)
            data.append(UInt8((value >> 8) & 0xFF))
            data.append(UInt8(value & 0xFF))
        } else {
            // B2: Values outside -32768..32767 cannot be correctly encoded as integers
            // in Type 2 CharStrings. Byte 255 encodes Fixed 16.16, not int32.
            // Font coordinates this large indicate a data error.
            throw CFFError.valueOutOfRange(value)
        }
    }
}
