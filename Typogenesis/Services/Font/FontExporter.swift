import Foundation
import CoreGraphics

/// Exports FontProject to TrueType (TTF) font files
actor FontExporter {

    enum FontExporterError: Error, LocalizedError {
        case noGlyphs
        case invalidGlyphData
        case exportFailed(String)

        var errorDescription: String? {
            switch self {
            case .noGlyphs: return "Font has no glyphs to export"
            case .invalidGlyphData: return "Invalid glyph data"
            case .exportFailed(let reason): return "Export failed: \(reason)"
            }
        }
    }

    enum ExportFormat {
        case ttf
        case otf  // Not yet implemented
    }

    struct ExportOptions {
        var format: ExportFormat = .ttf
        var includeKerning: Bool = true
        var hinting: Bool = false  // TrueType hinting not implemented

        static let `default` = ExportOptions()
    }

    // MARK: - Public Interface

    func export(project: FontProject, options: ExportOptions = .default) async throws -> Data {
        guard !project.glyphs.isEmpty else {
            throw FontExporterError.noGlyphs
        }

        switch options.format {
        case .ttf:
            return try await exportTrueType(project: project, options: options)
        case .otf:
            throw FontExporterError.exportFailed("OTF export not yet implemented")
        }
    }

    func export(project: FontProject, to url: URL, options: ExportOptions = .default) async throws {
        let data = try await export(project: project, options: options)
        try data.write(to: url)
    }

    // MARK: - TrueType Export

    private func exportTrueType(project: FontProject, options: ExportOptions) async throws -> Data {
        // Build glyph order (index 0 must be .notdef)
        var glyphOrder: [(Character?, Glyph?)] = [(nil, nil)]  // .notdef at index 0

        // Add all glyphs in sorted order for consistency
        let sortedChars = project.glyphs.keys.sorted()
        for char in sortedChars {
            if let glyph = project.glyphs[char] {
                glyphOrder.append((char, glyph))
            }
        }

        // Build character to glyph index mapping
        var charToGlyphIndex: [Character: UInt16] = [:]
        for (index, entry) in glyphOrder.enumerated() {
            if let char = entry.0 {
                charToGlyphIndex[char] = UInt16(index)
            }
        }

        // Build all tables
        let head = buildHeadTable(project: project, glyphOrder: glyphOrder)
        let hhea = buildHheaTable(project: project, glyphOrder: glyphOrder)
        let maxp = buildMaxpTable(glyphOrder: glyphOrder)
        let os2 = buildOS2Table(project: project, glyphOrder: glyphOrder)
        let name = buildNameTable(project: project)
        let cmap = buildCmapTable(charToGlyphIndex: charToGlyphIndex)
        let post = buildPostTable(project: project, glyphOrder: glyphOrder)
        let (glyf, loca) = buildGlyfAndLocaTables(project: project, glyphOrder: glyphOrder)
        let hmtx = buildHmtxTable(project: project, glyphOrder: glyphOrder)

        // Optionally build kerning
        var kern: Data? = nil
        if options.includeKerning && !project.kerning.isEmpty {
            kern = buildKernTable(project: project, charToGlyphIndex: charToGlyphIndex)
        }

        // Assemble font file
        return assembleFontFile(
            head: head,
            hhea: hhea,
            maxp: maxp,
            os2: os2,
            name: name,
            cmap: cmap,
            post: post,
            glyf: glyf,
            loca: loca,
            hmtx: hmtx,
            kern: kern
        )
    }

    // MARK: - Table Building

    private func buildHeadTable(project: FontProject, glyphOrder: [(Character?, Glyph?)]) -> Data {
        var data = Data()

        // Calculate bounding box for all glyphs
        var xMin: Int16 = 0
        var yMin: Int16 = 0
        var xMax: Int16 = 0
        var yMax: Int16 = 0

        for (_, glyph) in glyphOrder {
            if let glyph = glyph {
                let bbox = glyph.outline.boundingBox
                xMin = min(xMin, Int16(clamping: bbox.minX))
                yMin = min(yMin, Int16(clamping: bbox.minY))
                xMax = max(xMax, Int16(clamping: bbox.maxX))
                yMax = max(yMax, Int16(clamping: bbox.maxY))
            }
        }

        data.writeUInt16(1)  // majorVersion
        data.writeUInt16(0)  // minorVersion
        data.writeFixed(Fixed(value: 0x00010000))  // fontRevision 1.0
        data.writeUInt32(0)  // checksumAdjustment (will be calculated later)
        data.writeUInt32(0x5F0F3CF5)  // magicNumber
        data.writeUInt16(0x000B)  // flags
        data.writeUInt16(UInt16(project.metrics.unitsPerEm))
        data.writeInt64(LongDateTime().value)  // created
        data.writeInt64(LongDateTime().value)  // modified
        data.writeInt16(xMin)
        data.writeInt16(yMin)
        data.writeInt16(xMax)
        data.writeInt16(yMax)
        data.writeUInt16(0)  // macStyle
        data.writeUInt16(8)  // lowestRecPPEM
        data.writeInt16(2)   // fontDirectionHint
        data.writeInt16(1)   // indexToLocFormat (long offsets)
        data.writeInt16(0)   // glyphDataFormat

        return data
    }

    private func buildHheaTable(project: FontProject, glyphOrder: [(Character?, Glyph?)]) -> Data {
        var data = Data()

        // Calculate metrics
        var advanceWidthMax: UInt16 = 0
        var minLeftSideBearing: Int16 = Int16.max
        var minRightSideBearing: Int16 = Int16.max
        var xMaxExtent: Int16 = Int16.min

        for (_, glyph) in glyphOrder {
            if let glyph = glyph {
                advanceWidthMax = max(advanceWidthMax, UInt16(clamping: glyph.advanceWidth))
                minLeftSideBearing = min(minLeftSideBearing, Int16(clamping: glyph.leftSideBearing))

                let rsb = glyph.advanceWidth - glyph.leftSideBearing - glyph.outline.boundingBox.width
                minRightSideBearing = min(minRightSideBearing, Int16(clamping: rsb))

                let extent = glyph.leftSideBearing + glyph.outline.boundingBox.width
                xMaxExtent = max(xMaxExtent, Int16(clamping: extent))
            }
        }

        if minLeftSideBearing == Int16.max { minLeftSideBearing = 0 }
        if minRightSideBearing == Int16.max { minRightSideBearing = 0 }
        if xMaxExtent == Int16.min { xMaxExtent = 0 }

        data.writeUInt16(1)  // majorVersion
        data.writeUInt16(0)  // minorVersion
        data.writeInt16(Int16(clamping: project.metrics.ascender))
        data.writeInt16(Int16(clamping: project.metrics.descender))
        data.writeInt16(Int16(clamping: project.metrics.lineGap))
        data.writeUInt16(advanceWidthMax)
        data.writeInt16(minLeftSideBearing)
        data.writeInt16(minRightSideBearing)
        data.writeInt16(xMaxExtent)
        data.writeInt16(1)   // caretSlopeRise
        data.writeInt16(0)   // caretSlopeRun
        data.writeInt16(0)   // caretOffset
        data.writeInt16(0)   // reserved
        data.writeInt16(0)   // reserved
        data.writeInt16(0)   // reserved
        data.writeInt16(0)   // reserved
        data.writeInt16(0)   // metricDataFormat
        data.writeUInt16(UInt16(glyphOrder.count))  // numberOfHMetrics

        return data
    }

    private func buildMaxpTable(glyphOrder: [(Character?, Glyph?)]) -> Data {
        var data = Data()

        var maxPoints: UInt16 = 0
        var maxContours: UInt16 = 0

        for (_, glyph) in glyphOrder {
            if let glyph = glyph {
                var pointCount = 0
                for contour in glyph.outline.contours {
                    pointCount += contour.points.count
                }
                maxPoints = max(maxPoints, UInt16(clamping: pointCount))
                maxContours = max(maxContours, UInt16(clamping: glyph.outline.contours.count))
            }
        }

        data.writeFixed(Fixed(value: 0x00010000))  // version 1.0
        data.writeUInt16(UInt16(glyphOrder.count))  // numGlyphs
        data.writeUInt16(maxPoints)
        data.writeUInt16(maxContours)
        data.writeUInt16(0)  // maxCompositePoints
        data.writeUInt16(0)  // maxCompositeContours
        data.writeUInt16(2)  // maxZones
        data.writeUInt16(0)  // maxTwilightPoints
        data.writeUInt16(0)  // maxStorage
        data.writeUInt16(0)  // maxFunctionDefs
        data.writeUInt16(0)  // maxInstructionDefs
        data.writeUInt16(64) // maxStackElements
        data.writeUInt16(0)  // maxSizeOfInstructions
        data.writeUInt16(0)  // maxComponentElements
        data.writeUInt16(0)  // maxComponentDepth

        return data
    }

    private func buildOS2Table(project: FontProject, glyphOrder: [(Character?, Glyph?)]) -> Data {
        var data = Data()

        // Calculate average char width
        var totalWidth = 0
        var widthCount = 0
        var firstCharIndex: UInt16 = 0xFFFF
        var lastCharIndex: UInt16 = 0

        for (char, glyph) in glyphOrder {
            if let glyph = glyph {
                totalWidth += glyph.advanceWidth
                widthCount += 1
            }
            if let char = char, let scalar = char.unicodeScalars.first {
                let codepoint = UInt16(clamping: scalar.value)
                firstCharIndex = min(firstCharIndex, codepoint)
                lastCharIndex = max(lastCharIndex, codepoint)
            }
        }

        let avgCharWidth = widthCount > 0 ? totalWidth / widthCount : 500

        data.writeUInt16(4)  // version
        data.writeInt16(Int16(clamping: avgCharWidth))
        data.writeUInt16(400)  // usWeightClass (Normal)
        data.writeUInt16(5)    // usWidthClass (Medium)
        data.writeUInt16(0)    // fsType (Installable)
        data.writeInt16(650)   // ySubscriptXSize
        data.writeInt16(600)   // ySubscriptYSize
        data.writeInt16(0)     // ySubscriptXOffset
        data.writeInt16(75)    // ySubscriptYOffset
        data.writeInt16(650)   // ySuperscriptXSize
        data.writeInt16(600)   // ySuperscriptYSize
        data.writeInt16(0)     // ySuperscriptXOffset
        data.writeInt16(350)   // ySuperscriptYOffset
        data.writeInt16(50)    // yStrikeoutSize
        data.writeInt16(300)   // yStrikeoutPosition
        data.writeInt16(0)     // sFamilyClass
        data.writeBytes([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  // panose
        data.writeUInt32(0)    // ulUnicodeRange1
        data.writeUInt32(0)    // ulUnicodeRange2
        data.writeUInt32(0)    // ulUnicodeRange3
        data.writeUInt32(0)    // ulUnicodeRange4
        data.writeTag("TYPO")  // achVendID
        data.writeUInt16(0x0040)  // fsSelection (Regular)
        data.writeUInt16(firstCharIndex == 0xFFFF ? 0 : firstCharIndex)
        data.writeUInt16(lastCharIndex)
        data.writeInt16(Int16(clamping: project.metrics.ascender))
        data.writeInt16(Int16(clamping: project.metrics.descender))
        data.writeInt16(Int16(clamping: project.metrics.lineGap))
        data.writeUInt16(UInt16(clamping: project.metrics.ascender))  // usWinAscent
        data.writeUInt16(UInt16(clamping: abs(project.metrics.descender)))  // usWinDescent
        data.writeUInt32(0)    // ulCodePageRange1
        data.writeUInt32(0)    // ulCodePageRange2
        data.writeInt16(Int16(clamping: project.metrics.xHeight))
        data.writeInt16(Int16(clamping: project.metrics.capHeight))
        data.writeUInt16(0)    // usDefaultChar
        data.writeUInt16(32)   // usBreakChar (space)
        data.writeUInt16(0)    // usMaxContext

        return data
    }

    private func buildNameTable(project: FontProject) -> Data {
        var data = Data()
        var strings: [(UInt16, String)] = []

        // Add name records
        strings.append((0, project.metadata.copyright))  // Copyright
        strings.append((1, project.family))              // Font Family
        strings.append((2, project.style))               // Font Subfamily
        strings.append((3, project.metadata.uniqueID))   // Unique ID
        strings.append((4, "\(project.family) \(project.style)"))  // Full Name
        strings.append((5, "Version \(project.metadata.version)")) // Version
        strings.append((6, project.family.replacingOccurrences(of: " ", with: "")))  // PostScript Name
        strings.append((9, project.metadata.designer))   // Designer
        strings.append((10, project.metadata.description))  // Description
        strings.append((13, project.metadata.license))   // License

        // Filter out empty strings
        strings = strings.filter { !$0.1.isEmpty }

        // Build string storage
        var stringData = Data()
        var records: [(platformID: UInt16, encodingID: UInt16, languageID: UInt16, nameID: UInt16, offset: UInt16, length: UInt16)] = []

        for (nameID, string) in strings {
            // Windows platform (3), Unicode BMP (1), English US (0x0409)
            if let utf16Data = string.data(using: .utf16BigEndian) {
                records.append((3, 1, 0x0409, nameID, UInt16(stringData.count), UInt16(utf16Data.count)))
                stringData.append(utf16Data)
            }
        }

        // Write header
        data.writeUInt16(0)  // format
        data.writeUInt16(UInt16(records.count))
        data.writeUInt16(UInt16(6 + records.count * 12))  // stringOffset

        // Write records
        for record in records {
            data.writeUInt16(record.platformID)
            data.writeUInt16(record.encodingID)
            data.writeUInt16(record.languageID)
            data.writeUInt16(record.nameID)
            data.writeUInt16(record.length)
            data.writeUInt16(record.offset)
        }

        // Append string data
        data.append(stringData)

        return data
    }

    private func buildCmapTable(charToGlyphIndex: [Character: UInt16]) -> Data {
        var data = Data()

        // Build format 4 subtable data
        let format4 = buildCmapFormat4(charToGlyphIndex: charToGlyphIndex)

        // Header
        data.writeUInt16(0)  // version
        data.writeUInt16(1)  // numTables

        // Encoding record (Windows Unicode BMP)
        data.writeUInt16(3)  // platformID
        data.writeUInt16(1)  // encodingID
        data.writeUInt32(12) // offset to subtable

        // Append subtable
        data.append(format4)

        return data
    }

    private func buildCmapFormat4(charToGlyphIndex: [Character: UInt16]) -> Data {
        var data = Data()

        // Group consecutive characters into segments
        let sortedMappings = charToGlyphIndex.sorted { a, b in
            guard let aScalar = a.key.unicodeScalars.first,
                  let bScalar = b.key.unicodeScalars.first else { return false }
            return aScalar.value < bScalar.value
        }

        var segments: [(startCode: UInt16, endCode: UInt16, idDelta: Int16, idRangeOffset: UInt16, glyphIds: [UInt16])] = []

        if !sortedMappings.isEmpty {
            var currentStart: UInt16 = 0
            var currentEnd: UInt16 = 0
            var currentDelta: Int16 = 0
            var useIdDelta = true
            var glyphIds: [UInt16] = []

            for (index, mapping) in sortedMappings.enumerated() {
                guard let scalar = mapping.key.unicodeScalars.first else { continue }
                let charCode = UInt16(clamping: scalar.value)
                let glyphIndex = mapping.value

                if index == 0 {
                    currentStart = charCode
                    currentEnd = charCode
                    currentDelta = Int16(bitPattern: glyphIndex &- charCode)
                    glyphIds = [glyphIndex]
                } else {
                    let expectedGlyph = UInt16(bitPattern: Int16(bitPattern: charCode) &+ currentDelta)
                    let isConsecutive = charCode == currentEnd + 1
                    let deltaWorks = expectedGlyph == glyphIndex

                    if isConsecutive && (useIdDelta && deltaWorks) {
                        currentEnd = charCode
                        glyphIds.append(glyphIndex)
                    } else {
                        // Save current segment
                        if useIdDelta {
                            segments.append((currentStart, currentEnd, currentDelta, 0, []))
                        } else {
                            segments.append((currentStart, currentEnd, 0, 1, glyphIds))  // idRangeOffset will be calculated
                        }

                        // Start new segment
                        currentStart = charCode
                        currentEnd = charCode
                        currentDelta = Int16(bitPattern: glyphIndex &- charCode)
                        useIdDelta = true
                        glyphIds = [glyphIndex]
                    }
                }
            }

            // Save last segment
            if useIdDelta {
                segments.append((currentStart, currentEnd, currentDelta, 0, []))
            } else {
                segments.append((currentStart, currentEnd, 0, 1, glyphIds))
            }
        }

        // Add terminating segment
        segments.append((0xFFFF, 0xFFFF, 1, 0, []))

        let segCount = segments.count
        let segCountX2 = UInt16(segCount * 2)
        let searchRange = UInt16(truncatingIfNeeded: 1 << (Int(log2(Double(segCount))) + 1))
        let entrySelector = UInt16(log2(Double(searchRange / 2)))
        let rangeShift = segCountX2 - searchRange

        // Calculate table length
        let headerSize = 14
        let segmentArraysSize = segCount * 8  // 4 arrays of UInt16 each
        let reservedPadSize = 2
        var glyphIdArraySize = 0
        for seg in segments where seg.idRangeOffset != 0 {
            glyphIdArraySize += seg.glyphIds.count * 2
        }
        let length = headerSize + segmentArraysSize + reservedPadSize + glyphIdArraySize

        // Write header
        data.writeUInt16(4)  // format
        data.writeUInt16(UInt16(length))
        data.writeUInt16(0)  // language
        data.writeUInt16(segCountX2)
        data.writeUInt16(searchRange)
        data.writeUInt16(entrySelector)
        data.writeUInt16(rangeShift)

        // Write endCode array
        for seg in segments {
            data.writeUInt16(seg.endCode)
        }

        // Reserved pad
        data.writeUInt16(0)

        // Write startCode array
        for seg in segments {
            data.writeUInt16(seg.startCode)
        }

        // Write idDelta array
        for seg in segments {
            data.writeInt16(seg.idDelta)
        }

        // Write idRangeOffset array and glyph ID arrays
        // For simplicity, we use idDelta for all segments (idRangeOffset = 0)
        for _ in segments {
            data.writeUInt16(0)
        }

        return data
    }

    private func buildPostTable(project: FontProject, glyphOrder: [(Character?, Glyph?)]) -> Data {
        var data = Data()

        // Use version 2.0 with glyph names
        data.writeFixed(Fixed(value: 0x00020000))  // version 2.0
        data.writeFixed(Fixed(value: 0))  // italicAngle
        data.writeInt16(-100)  // underlinePosition
        data.writeInt16(50)    // underlineThickness
        data.writeUInt32(0)    // isFixedPitch
        data.writeUInt32(0)    // minMemType42
        data.writeUInt32(0)    // maxMemType42
        data.writeUInt32(0)    // minMemType1
        data.writeUInt32(0)    // maxMemType1

        // Write glyph name indices
        data.writeUInt16(UInt16(glyphOrder.count))

        var extraNames: [String] = []
        for (char, _) in glyphOrder {
            if let char = char, let scalar = char.unicodeScalars.first {
                let name = String(format: "uni%04X", scalar.value)
                if let stdIndex = standardNameToIndex(name) {
                    data.writeUInt16(stdIndex)
                } else {
                    data.writeUInt16(UInt16(258 + extraNames.count))
                    extraNames.append(name)
                }
            } else {
                data.writeUInt16(0)  // .notdef
            }
        }

        // Write extra names (Pascal strings)
        for name in extraNames {
            let utf8 = Array(name.utf8)
            data.writeUInt8(UInt8(utf8.count))
            data.writeBytes(utf8)
        }

        return data
    }

    private func buildGlyfAndLocaTables(project: FontProject, glyphOrder: [(Character?, Glyph?)]) -> (glyf: Data, loca: Data) {
        var glyfData = Data()
        var locaOffsets: [UInt32] = []

        for (_, glyph) in glyphOrder {
            locaOffsets.append(UInt32(glyfData.count))

            if let glyph = glyph, !glyph.outline.isEmpty {
                let glyphData = encodeSimpleGlyph(glyph: glyph)
                glyfData.append(glyphData)
            }
            // Empty glyph: offset stays same as previous (zero-length)
        }

        // Final offset (end of last glyph)
        locaOffsets.append(UInt32(glyfData.count))

        // Pad glyf to 4-byte boundary
        glyfData.pad(to: 4)

        // Build loca table (long format)
        var locaData = Data()
        for offset in locaOffsets {
            locaData.writeUInt32(offset)
        }

        return (glyfData, locaData)
    }

    private func encodeSimpleGlyph(glyph: Glyph) -> Data {
        var data = Data()

        let contours = glyph.outline.contours.filter { !$0.points.isEmpty }
        guard !contours.isEmpty else { return data }

        // Convert our cubic beziers to TrueType quadratic
        let ttContours = convertToTrueTypeContours(contours: contours)

        // Calculate bounding box
        let bbox = glyph.outline.boundingBox

        // Write header
        data.writeInt16(Int16(ttContours.count))  // numberOfContours
        data.writeInt16(Int16(clamping: bbox.minX))
        data.writeInt16(Int16(clamping: bbox.minY))
        data.writeInt16(Int16(clamping: bbox.maxX))
        data.writeInt16(Int16(clamping: bbox.maxY))

        // Write endPtsOfContours
        var pointIndex = 0
        for contour in ttContours {
            pointIndex += contour.count
            data.writeUInt16(UInt16(pointIndex - 1))
        }

        // Write instruction length (no instructions)
        data.writeUInt16(0)

        // Collect all points
        var allPoints: [(x: Int16, y: Int16, onCurve: Bool)] = []
        for contour in ttContours {
            for point in contour {
                allPoints.append(point)
            }
        }

        // Encode flags and coordinates
        var flags: [UInt8] = []
        var xCoords: [Int16] = []
        var yCoords: [Int16] = []

        var prevX: Int16 = 0
        var prevY: Int16 = 0

        for point in allPoints {
            let dx = point.x - prevX
            let dy = point.y - prevY

            var flag: UInt8 = point.onCurve ? 0x01 : 0x00

            // X coordinate encoding
            if dx == 0 {
                flag |= 0x10  // x is same
            } else if dx >= -255 && dx <= 255 {
                flag |= 0x02  // x is 1 byte
                if dx > 0 {
                    flag |= 0x10  // positive
                }
                xCoords.append(Int16(abs(dx)))
            } else {
                xCoords.append(dx)
            }

            // Y coordinate encoding
            if dy == 0 {
                flag |= 0x20  // y is same
            } else if dy >= -255 && dy <= 255 {
                flag |= 0x04  // y is 1 byte
                if dy > 0 {
                    flag |= 0x20  // positive
                }
                yCoords.append(Int16(abs(dy)))
            } else {
                yCoords.append(dy)
            }

            flags.append(flag)
            prevX = point.x
            prevY = point.y
        }

        // Write flags (no repeat optimization for simplicity)
        for flag in flags {
            data.writeUInt8(flag)
        }

        // Write x coordinates
        for (index, point) in allPoints.enumerated() {
            let flag = flags[index]
            if flag & 0x02 != 0 {
                // 1 byte
                let dx = point.x - (index > 0 ? allPoints[index - 1].x : 0)
                data.writeUInt8(UInt8(abs(dx)))
            } else if flag & 0x10 == 0 {
                // 2 bytes
                let dx = point.x - (index > 0 ? allPoints[index - 1].x : 0)
                data.writeInt16(dx)
            }
        }

        // Write y coordinates
        for (index, point) in allPoints.enumerated() {
            let flag = flags[index]
            if flag & 0x04 != 0 {
                // 1 byte
                let dy = point.y - (index > 0 ? allPoints[index - 1].y : 0)
                data.writeUInt8(UInt8(abs(dy)))
            } else if flag & 0x20 == 0 {
                // 2 bytes
                let dy = point.y - (index > 0 ? allPoints[index - 1].y : 0)
                data.writeInt16(dy)
            }
        }

        // Pad to 2-byte boundary
        if data.count % 2 != 0 {
            data.writeUInt8(0)
        }

        return data
    }

    private func convertToTrueTypeContours(contours: [Contour]) -> [[(x: Int16, y: Int16, onCurve: Bool)]] {
        var result: [[(x: Int16, y: Int16, onCurve: Bool)]] = []

        for contour in contours {
            var ttPoints: [(x: Int16, y: Int16, onCurve: Bool)] = []

            for point in contour.points {
                // Add on-curve point
                ttPoints.append((
                    x: Int16(clamping: Int(point.position.x)),
                    y: Int16(clamping: Int(point.position.y)),
                    onCurve: true
                ))

                // Add off-curve control point if present
                if let controlOut = point.controlOut {
                    ttPoints.append((
                        x: Int16(clamping: Int(controlOut.x)),
                        y: Int16(clamping: Int(controlOut.y)),
                        onCurve: false
                    ))
                }
            }

            if !ttPoints.isEmpty {
                result.append(ttPoints)
            }
        }

        return result
    }

    private func buildHmtxTable(project: FontProject, glyphOrder: [(Character?, Glyph?)]) -> Data {
        var data = Data()

        for (_, glyph) in glyphOrder {
            if let glyph = glyph {
                data.writeUInt16(UInt16(clamping: glyph.advanceWidth))
                data.writeInt16(Int16(clamping: glyph.leftSideBearing))
            } else {
                // .notdef glyph
                data.writeUInt16(500)
                data.writeInt16(0)
            }
        }

        return data
    }

    private func buildKernTable(project: FontProject, charToGlyphIndex: [Character: UInt16]) -> Data? {
        var pairs: [(left: UInt16, right: UInt16, value: Int16)] = []

        for kernPair in project.kerning {
            if let leftIndex = charToGlyphIndex[kernPair.left],
               let rightIndex = charToGlyphIndex[kernPair.right] {
                pairs.append((leftIndex, rightIndex, Int16(clamping: kernPair.value)))
            }
        }

        guard !pairs.isEmpty else { return nil }

        // Sort pairs for binary search
        pairs.sort { ($0.left, $0.right) < ($1.left, $1.right) }

        var data = Data()

        // Header
        data.writeUInt16(0)  // version
        data.writeUInt16(1)  // nTables

        // Subtable header
        let subtableLength = 14 + pairs.count * 6
        data.writeUInt16(0)  // version
        data.writeUInt16(UInt16(subtableLength))
        data.writeUInt16(1)  // coverage (horizontal, format 0)

        // Format 0 subtable
        data.writeUInt16(UInt16(pairs.count))
        let searchRange = UInt16(truncatingIfNeeded: (1 << Int(log2(Double(pairs.count)))) * 6)
        let entrySelector = UInt16(log2(Double(pairs.count)))
        let rangeShift = UInt16(pairs.count * 6) - searchRange
        data.writeUInt16(searchRange)
        data.writeUInt16(entrySelector)
        data.writeUInt16(rangeShift)

        // Pairs
        for pair in pairs {
            data.writeUInt16(pair.left)
            data.writeUInt16(pair.right)
            data.writeInt16(pair.value)
        }

        return data
    }

    // MARK: - Font Assembly

    private func assembleFontFile(
        head: Data,
        hhea: Data,
        maxp: Data,
        os2: Data,
        name: Data,
        cmap: Data,
        post: Data,
        glyf: Data,
        loca: Data,
        hmtx: Data,
        kern: Data?
    ) -> Data {
        // Tables in recommended order
        var tables: [(tag: String, data: Data)] = [
            ("cmap", cmap),
            ("glyf", glyf),
            ("head", head),
            ("hhea", hhea),
            ("hmtx", hmtx),
            ("loca", loca),
            ("maxp", maxp),
            ("name", name),
            ("OS/2", os2),
            ("post", post)
        ]

        if let kern = kern {
            tables.append(("kern", kern))
        }

        // Sort by tag for proper ordering
        tables.sort { $0.tag < $1.tag }

        let numTables = UInt16(tables.count)
        let searchRange = UInt16(truncatingIfNeeded: (1 << Int(log2(Double(numTables)))) * 16)
        let entrySelector = UInt16(log2(Double(numTables)))
        let rangeShift = numTables * 16 - searchRange

        var fontData = Data()

        // Offset table
        fontData.writeUInt32(OffsetTable.trueTypeVersion)
        fontData.writeUInt16(numTables)
        fontData.writeUInt16(searchRange)
        fontData.writeUInt16(entrySelector)
        fontData.writeUInt16(rangeShift)

        // Calculate table offsets
        let tableDirectorySize = 12 + Int(numTables) * 16
        var currentOffset = tableDirectorySize

        var tableRecords: [(tag: String, checksum: UInt32, offset: UInt32, length: UInt32)] = []

        for (tag, data) in tables {
            var paddedData = data
            paddedData.pad(to: 4)

            tableRecords.append((
                tag: tag,
                checksum: data.calculateTableChecksum(),
                offset: UInt32(currentOffset),
                length: UInt32(data.count)
            ))

            currentOffset += paddedData.count
        }

        // Write table directory
        for record in tableRecords {
            fontData.writeTag(record.tag)
            fontData.writeUInt32(record.checksum)
            fontData.writeUInt32(record.offset)
            fontData.writeUInt32(record.length)
        }

        // Write table data
        for (_, data) in tables {
            var paddedData = data
            paddedData.pad(to: 4)
            fontData.append(paddedData)
        }

        // Calculate and set checksumAdjustment in head table
        let fileChecksum = fontData.calculateTableChecksum()
        let checksumAdjustment = 0xB1B0AFBA &- fileChecksum

        // Find head table offset and update checksumAdjustment
        if let headRecord = tableRecords.first(where: { $0.tag == "head" }) {
            let checksumOffset = Int(headRecord.offset) + 8
            fontData.replaceSubrange(checksumOffset..<checksumOffset+4, with: withUnsafeBytes(of: checksumAdjustment.bigEndian) { Data($0) })
        }

        return fontData
    }

    // MARK: - Helpers

    private func standardNameToIndex(_ name: String) -> UInt16? {
        let standardNames = [
            ".notdef", ".null", "nonmarkingreturn", "space", "exclam", "quotedbl",
            "numbersign", "dollar", "percent", "ampersand", "quotesingle", "parenleft",
            "parenright", "asterisk", "plus", "comma", "hyphen", "period", "slash",
            "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
            "nine", "colon", "semicolon", "less", "equal", "greater", "question", "at",
            "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O",
            "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "bracketleft",
            "backslash", "bracketright", "asciicircum", "underscore", "grave", "a", "b",
            "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q",
            "r", "s", "t", "u", "v", "w", "x", "y", "z"
        ]

        if let index = standardNames.firstIndex(of: name) {
            return UInt16(index)
        }
        return nil
    }
}

extension Int16 {
    init(clamping value: Int) {
        if value > Int(Int16.max) {
            self = Int16.max
        } else if value < Int(Int16.min) {
            self = Int16.min
        } else {
            self = Int16(value)
        }
    }
}

extension UInt16 {
    init(clamping value: Int) {
        if value > Int(UInt16.max) {
            self = UInt16.max
        } else if value < 0 {
            self = 0
        } else {
            self = UInt16(value)
        }
    }

    init(clamping value: UInt32) {
        if value > UInt32(UInt16.max) {
            self = UInt16.max
        } else {
            self = UInt16(value)
        }
    }
}
