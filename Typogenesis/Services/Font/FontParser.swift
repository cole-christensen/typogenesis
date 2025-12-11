import Foundation
import CoreGraphics

/// Parses TrueType (TTF) and OpenType (OTF) font files
actor FontParser {

    enum FontParserError: Error, LocalizedError {
        case invalidData
        case unsupportedFormat
        case missingRequiredTable(String)
        case invalidTableData(String)
        case invalidGlyphData
        case invalidCmapFormat

        var errorDescription: String? {
            switch self {
            case .invalidData: return "Invalid font data"
            case .unsupportedFormat: return "Unsupported font format"
            case .missingRequiredTable(let table): return "Missing required table: \(table)"
            case .invalidTableData(let table): return "Invalid data in table: \(table)"
            case .invalidGlyphData: return "Invalid glyph data"
            case .invalidCmapFormat: return "Invalid or unsupported cmap format"
            }
        }
    }

    // MARK: - Public Interface

    func parse(data: Data) async throws -> FontProject {
        let offsetTable = try parseOffsetTable(data: data)
        let tableRecords = try parseTableRecords(data: data, numTables: Int(offsetTable.numTables))

        // Parse required tables
        guard let headRecord = tableRecords["head"] else {
            throw FontParserError.missingRequiredTable("head")
        }
        let head = try parseHeadTable(data: data, record: headRecord)

        guard let hheaRecord = tableRecords["hhea"] else {
            throw FontParserError.missingRequiredTable("hhea")
        }
        let hhea = try parseHheaTable(data: data, record: hheaRecord)

        guard let maxpRecord = tableRecords["maxp"] else {
            throw FontParserError.missingRequiredTable("maxp")
        }
        let maxp = try parseMaxpTable(data: data, record: maxpRecord)

        guard let cmapRecord = tableRecords["cmap"] else {
            throw FontParserError.missingRequiredTable("cmap")
        }
        let cmap = try parseCmapTable(data: data, record: cmapRecord)

        // Parse glyph data
        guard let locaRecord = tableRecords["loca"] else {
            throw FontParserError.missingRequiredTable("loca")
        }
        let loca = try parseLocaTable(
            data: data,
            record: locaRecord,
            numGlyphs: Int(maxp.numGlyphs),
            format: head.indexToLocFormat
        )

        guard let glyfRecord = tableRecords["glyf"] else {
            throw FontParserError.missingRequiredTable("glyf")
        }
        let glyf = try parseGlyfTable(
            data: data,
            record: glyfRecord,
            loca: loca,
            numGlyphs: Int(maxp.numGlyphs)
        )

        // Parse horizontal metrics
        guard let hmtxRecord = tableRecords["hmtx"] else {
            throw FontParserError.missingRequiredTable("hmtx")
        }
        let hmtx = try parseHmtxTable(
            data: data,
            record: hmtxRecord,
            numberOfHMetrics: Int(hhea.numberOfHMetrics),
            numGlyphs: Int(maxp.numGlyphs)
        )

        // Parse optional tables
        let name = tableRecords["name"].flatMap { try? parseNameTable(data: data, record: $0) }
        let os2 = tableRecords["OS/2"].flatMap { try? parseOS2Table(data: data, record: $0) }
        let post = tableRecords["post"].flatMap { try? parsePostTable(data: data, record: $0, numGlyphs: Int(maxp.numGlyphs)) }
        let kern = tableRecords["kern"].flatMap { try? parseKernTable(data: data, record: $0) }

        // Build FontProject
        return buildFontProject(
            head: head,
            hhea: hhea,
            maxp: maxp,
            cmap: cmap,
            glyf: glyf,
            hmtx: hmtx,
            name: name,
            os2: os2,
            post: post,
            kern: kern
        )
    }

    func parse(url: URL) async throws -> FontProject {
        let data = try Data(contentsOf: url)
        return try await parse(data: data)
    }

    // MARK: - Offset Table Parsing

    private func parseOffsetTable(data: Data) throws -> OffsetTable {
        guard data.count >= 12 else {
            throw FontParserError.invalidData
        }

        let sfntVersion = data.readUInt32(at: 0)

        guard sfntVersion == OffsetTable.trueTypeVersion || sfntVersion == OffsetTable.cffVersion else {
            throw FontParserError.unsupportedFormat
        }

        return OffsetTable(
            sfntVersion: sfntVersion,
            numTables: data.readUInt16(at: 4),
            searchRange: data.readUInt16(at: 6),
            entrySelector: data.readUInt16(at: 8),
            rangeShift: data.readUInt16(at: 10)
        )
    }

    private func parseTableRecords(data: Data, numTables: Int) throws -> [String: TableRecord] {
        var records: [String: TableRecord] = [:]
        var offset = 12

        for _ in 0..<numTables {
            guard offset + 16 <= data.count else {
                throw FontParserError.invalidData
            }

            let tag = data.readTag(at: offset)
            let record = TableRecord(
                tag: tag,
                checksum: data.readUInt32(at: offset + 4),
                offset: data.readUInt32(at: offset + 8),
                length: data.readUInt32(at: offset + 12)
            )
            records[tag] = record
            offset += 16
        }

        return records
    }

    // MARK: - Required Table Parsing

    private func parseHeadTable(data: Data, record: TableRecord) throws -> HeadTable {
        let offset = Int(record.offset)
        guard offset + 54 <= data.count else {
            throw FontParserError.invalidTableData("head")
        }

        return HeadTable(
            majorVersion: data.readUInt16(at: offset),
            minorVersion: data.readUInt16(at: offset + 2),
            fontRevision: data.readFixed(at: offset + 4),
            checksumAdjustment: data.readUInt32(at: offset + 8),
            magicNumber: data.readUInt32(at: offset + 12),
            flags: data.readUInt16(at: offset + 16),
            unitsPerEm: data.readUInt16(at: offset + 18),
            created: LongDateTime(value: data.readInt64(at: offset + 20)),
            modified: LongDateTime(value: data.readInt64(at: offset + 28)),
            xMin: data.readInt16(at: offset + 36),
            yMin: data.readInt16(at: offset + 38),
            xMax: data.readInt16(at: offset + 40),
            yMax: data.readInt16(at: offset + 42),
            macStyle: data.readUInt16(at: offset + 44),
            lowestRecPPEM: data.readUInt16(at: offset + 46),
            fontDirectionHint: data.readInt16(at: offset + 48),
            indexToLocFormat: data.readInt16(at: offset + 50),
            glyphDataFormat: data.readInt16(at: offset + 52)
        )
    }

    private func parseHheaTable(data: Data, record: TableRecord) throws -> HheaTable {
        let offset = Int(record.offset)
        guard offset + 36 <= data.count else {
            throw FontParserError.invalidTableData("hhea")
        }

        return HheaTable(
            majorVersion: data.readUInt16(at: offset),
            minorVersion: data.readUInt16(at: offset + 2),
            ascender: data.readInt16(at: offset + 4),
            descender: data.readInt16(at: offset + 6),
            lineGap: data.readInt16(at: offset + 8),
            advanceWidthMax: data.readUInt16(at: offset + 10),
            minLeftSideBearing: data.readInt16(at: offset + 12),
            minRightSideBearing: data.readInt16(at: offset + 14),
            xMaxExtent: data.readInt16(at: offset + 16),
            caretSlopeRise: data.readInt16(at: offset + 18),
            caretSlopeRun: data.readInt16(at: offset + 20),
            caretOffset: data.readInt16(at: offset + 22),
            reserved1: data.readInt16(at: offset + 24),
            reserved2: data.readInt16(at: offset + 26),
            reserved3: data.readInt16(at: offset + 28),
            reserved4: data.readInt16(at: offset + 30),
            metricDataFormat: data.readInt16(at: offset + 32),
            numberOfHMetrics: data.readUInt16(at: offset + 34)
        )
    }

    private func parseMaxpTable(data: Data, record: TableRecord) throws -> MaxpTable {
        let offset = Int(record.offset)
        guard offset + 6 <= data.count else {
            throw FontParserError.invalidTableData("maxp")
        }

        let version = data.readFixed(at: offset)
        let numGlyphs = data.readUInt16(at: offset + 4)

        // Version 1.0 has additional fields for TrueType
        if version.value == 0x00010000 && offset + 32 <= data.count {
            return MaxpTable(
                version: version,
                numGlyphs: numGlyphs,
                maxPoints: data.readUInt16(at: offset + 6),
                maxContours: data.readUInt16(at: offset + 8),
                maxCompositePoints: data.readUInt16(at: offset + 10),
                maxCompositeContours: data.readUInt16(at: offset + 12),
                maxZones: data.readUInt16(at: offset + 14),
                maxTwilightPoints: data.readUInt16(at: offset + 16),
                maxStorage: data.readUInt16(at: offset + 18),
                maxFunctionDefs: data.readUInt16(at: offset + 20),
                maxInstructionDefs: data.readUInt16(at: offset + 22),
                maxStackElements: data.readUInt16(at: offset + 24),
                maxSizeOfInstructions: data.readUInt16(at: offset + 26),
                maxComponentElements: data.readUInt16(at: offset + 28),
                maxComponentDepth: data.readUInt16(at: offset + 30)
            )
        }

        return MaxpTable(version: version, numGlyphs: numGlyphs)
    }

    private func parseCmapTable(data: Data, record: TableRecord) throws -> CmapTable {
        let offset = Int(record.offset)
        guard offset + 4 <= data.count else {
            throw FontParserError.invalidTableData("cmap")
        }

        let version = data.readUInt16(at: offset)
        let numTables = data.readUInt16(at: offset + 2)

        var subtables: [CmapTable.CmapSubtable] = []
        var encodingOffset = offset + 4

        for _ in 0..<numTables {
            guard encodingOffset + 8 <= data.count else { break }

            let platformID = data.readUInt16(at: encodingOffset)
            let encodingID = data.readUInt16(at: encodingOffset + 2)
            let subtableOffset = Int(data.readUInt32(at: encodingOffset + 4))

            let actualOffset = offset + subtableOffset
            if let mapping = try? parseCmapSubtable(data: data, offset: actualOffset) {
                let format = data.readUInt16(at: actualOffset)
                subtables.append(CmapTable.CmapSubtable(
                    platformID: platformID,
                    encodingID: encodingID,
                    format: format,
                    mapping: mapping
                ))
            }

            encodingOffset += 8
        }

        return CmapTable(version: version, subtables: subtables)
    }

    private func parseCmapSubtable(data: Data, offset: Int) throws -> [UInt32: UInt16] {
        guard offset + 2 <= data.count else {
            throw FontParserError.invalidCmapFormat
        }

        let format = data.readUInt16(at: offset)

        switch format {
        case 0:
            return try parseCmapFormat0(data: data, offset: offset)
        case 4:
            return try parseCmapFormat4(data: data, offset: offset)
        case 12:
            return try parseCmapFormat12(data: data, offset: offset)
        default:
            return [:]  // Skip unsupported formats
        }
    }

    private func parseCmapFormat0(data: Data, offset: Int) throws -> [UInt32: UInt16] {
        guard offset + 262 <= data.count else {
            throw FontParserError.invalidCmapFormat
        }

        var mapping: [UInt32: UInt16] = [:]
        for i in 0..<256 {
            let glyphIndex = data.readUInt8(at: offset + 6 + i)
            if glyphIndex > 0 {
                mapping[UInt32(i)] = UInt16(glyphIndex)
            }
        }
        return mapping
    }

    private func parseCmapFormat4(data: Data, offset: Int) throws -> [UInt32: UInt16] {
        guard offset + 14 <= data.count else {
            throw FontParserError.invalidCmapFormat
        }

        let segCountX2 = Int(data.readUInt16(at: offset + 6))
        let segCount = segCountX2 / 2

        let endCodesOffset = offset + 14
        let startCodesOffset = endCodesOffset + segCountX2 + 2  // +2 for reservedPad
        let idDeltaOffset = startCodesOffset + segCountX2
        let idRangeOffset = idDeltaOffset + segCountX2

        var mapping: [UInt32: UInt16] = [:]

        for i in 0..<segCount {
            let endCode = Int(data.readUInt16(at: endCodesOffset + i * 2))
            let startCode = Int(data.readUInt16(at: startCodesOffset + i * 2))
            let idDelta = Int(data.readInt16(at: idDeltaOffset + i * 2))
            let idRangeOffsetValue = Int(data.readUInt16(at: idRangeOffset + i * 2))

            if startCode == 0xFFFF { break }

            for charCode in startCode...endCode {
                var glyphIndex: UInt16

                if idRangeOffsetValue == 0 {
                    glyphIndex = UInt16((charCode + idDelta) & 0xFFFF)
                } else {
                    let glyphIdOffset = idRangeOffset + i * 2 + idRangeOffsetValue + (charCode - startCode) * 2
                    if glyphIdOffset + 2 <= data.count {
                        glyphIndex = data.readUInt16(at: glyphIdOffset)
                        if glyphIndex != 0 {
                            glyphIndex = UInt16((Int(glyphIndex) + idDelta) & 0xFFFF)
                        }
                    } else {
                        glyphIndex = 0
                    }
                }

                if glyphIndex != 0 {
                    mapping[UInt32(charCode)] = glyphIndex
                }
            }
        }

        return mapping
    }

    private func parseCmapFormat12(data: Data, offset: Int) throws -> [UInt32: UInt16] {
        guard offset + 16 <= data.count else {
            throw FontParserError.invalidCmapFormat
        }

        let numGroups = Int(data.readUInt32(at: offset + 12))
        var mapping: [UInt32: UInt16] = [:]
        var groupOffset = offset + 16

        for _ in 0..<numGroups {
            guard groupOffset + 12 <= data.count else { break }

            let startCharCode = data.readUInt32(at: groupOffset)
            let endCharCode = data.readUInt32(at: groupOffset + 4)
            let startGlyphID = data.readUInt32(at: groupOffset + 8)

            for charCode in startCharCode...min(endCharCode, startCharCode + 0xFFFF) {
                let glyphIndex = startGlyphID + (charCode - startCharCode)
                mapping[charCode] = UInt16(glyphIndex & 0xFFFF)
            }

            groupOffset += 12
        }

        return mapping
    }

    // MARK: - Glyph Table Parsing

    private func parseLocaTable(data: Data, record: TableRecord, numGlyphs: Int, format: Int16) throws -> LocaTable {
        let offset = Int(record.offset)
        var offsets: [UInt32] = []

        if format == 0 {
            // Short offsets (UInt16, divided by 2)
            for i in 0...numGlyphs {
                let shortOffset = data.readUInt16(at: offset + i * 2)
                offsets.append(UInt32(shortOffset) * 2)
            }
        } else {
            // Long offsets (UInt32)
            for i in 0...numGlyphs {
                offsets.append(data.readUInt32(at: offset + i * 4))
            }
        }

        return LocaTable(offsets: offsets, format: format)
    }

    private func parseGlyfTable(data: Data, record: TableRecord, loca: LocaTable, numGlyphs: Int) throws -> GlyfTable {
        let baseOffset = Int(record.offset)
        var glyphs: [GlyfTable.GlyphData] = []

        for i in 0..<numGlyphs {
            let glyphOffset = Int(loca.offsets[i])
            let nextOffset = Int(loca.offsets[i + 1])

            if glyphOffset == nextOffset {
                // Empty glyph (like space)
                glyphs.append(GlyfTable.GlyphData(
                    numberOfContours: 0,
                    xMin: 0, yMin: 0, xMax: 0, yMax: 0,
                    contours: [],
                    components: [],
                    instructions: []
                ))
                continue
            }

            let actualOffset = baseOffset + glyphOffset
            guard actualOffset + 10 <= data.count else {
                glyphs.append(GlyfTable.GlyphData(
                    numberOfContours: 0,
                    xMin: 0, yMin: 0, xMax: 0, yMax: 0,
                    contours: [],
                    components: [],
                    instructions: []
                ))
                continue
            }

            let numberOfContours = data.readInt16(at: actualOffset)
            let xMin = data.readInt16(at: actualOffset + 2)
            let yMin = data.readInt16(at: actualOffset + 4)
            let xMax = data.readInt16(at: actualOffset + 6)
            let yMax = data.readInt16(at: actualOffset + 8)

            if numberOfContours >= 0 {
                // Simple glyph
                let (contours, instructions) = try parseSimpleGlyph(
                    data: data,
                    offset: actualOffset + 10,
                    numberOfContours: Int(numberOfContours)
                )
                glyphs.append(GlyfTable.GlyphData(
                    numberOfContours: numberOfContours,
                    xMin: xMin, yMin: yMin, xMax: xMax, yMax: yMax,
                    contours: contours,
                    components: [],
                    instructions: instructions
                ))
            } else {
                // Composite glyph
                let (components, instructions) = try parseCompositeGlyph(data: data, offset: actualOffset + 10)
                glyphs.append(GlyfTable.GlyphData(
                    numberOfContours: numberOfContours,
                    xMin: xMin, yMin: yMin, xMax: xMax, yMax: yMax,
                    contours: [],
                    components: components,
                    instructions: instructions
                ))
            }
        }

        return GlyfTable(glyphs: glyphs)
    }

    private func parseSimpleGlyph(data: Data, offset: Int, numberOfContours: Int) throws -> ([GlyfTable.GlyphContour], [UInt8]) {
        guard numberOfContours > 0 else { return ([], []) }

        var currentOffset = offset

        // Read endPtsOfContours
        var endPtsOfContours: [Int] = []
        for _ in 0..<numberOfContours {
            endPtsOfContours.append(Int(data.readUInt16(at: currentOffset)))
            currentOffset += 2
        }

        let numPoints = (endPtsOfContours.last ?? -1) + 1

        // Read instruction length and instructions
        let instructionLength = Int(data.readUInt16(at: currentOffset))
        currentOffset += 2

        let instructions = data.readBytes(at: currentOffset, count: instructionLength)
        currentOffset += instructionLength

        // Read flags
        var flags: [UInt8] = []
        while flags.count < numPoints {
            let flag = data.readUInt8(at: currentOffset)
            currentOffset += 1
            flags.append(flag)

            // Check repeat flag
            if flag & 0x08 != 0 {
                let repeatCount = Int(data.readUInt8(at: currentOffset))
                currentOffset += 1
                for _ in 0..<repeatCount {
                    flags.append(flag)
                }
            }
        }

        // Read x coordinates
        var xCoordinates: [Int16] = []
        var x: Int16 = 0
        for i in 0..<numPoints {
            let flag = flags[i]
            if flag & 0x02 != 0 {
                // x is 1 byte
                let dx = Int16(data.readUInt8(at: currentOffset))
                currentOffset += 1
                x += (flag & 0x10 != 0) ? dx : -dx
            } else if flag & 0x10 == 0 {
                // x is 2 bytes signed
                x += data.readInt16(at: currentOffset)
                currentOffset += 2
            }
            // else: x is same as previous
            xCoordinates.append(x)
        }

        // Read y coordinates
        var yCoordinates: [Int16] = []
        var y: Int16 = 0
        for i in 0..<numPoints {
            let flag = flags[i]
            if flag & 0x04 != 0 {
                // y is 1 byte
                let dy = Int16(data.readUInt8(at: currentOffset))
                currentOffset += 1
                y += (flag & 0x20 != 0) ? dy : -dy
            } else if flag & 0x20 == 0 {
                // y is 2 bytes signed
                y += data.readInt16(at: currentOffset)
                currentOffset += 2
            }
            // else: y is same as previous
            yCoordinates.append(y)
        }

        // Build contours
        var contours: [GlyfTable.GlyphContour] = []
        var pointIndex = 0

        for contourIndex in 0..<numberOfContours {
            let endPoint = endPtsOfContours[contourIndex]
            var points: [GlyfTable.GlyphPoint] = []

            while pointIndex <= endPoint {
                let onCurve = flags[pointIndex] & 0x01 != 0
                points.append(GlyfTable.GlyphPoint(
                    x: xCoordinates[pointIndex],
                    y: yCoordinates[pointIndex],
                    onCurve: onCurve
                ))
                pointIndex += 1
            }

            contours.append(GlyfTable.GlyphContour(points: points))
        }

        return (contours, instructions)
    }

    private func parseCompositeGlyph(data: Data, offset: Int) throws -> ([GlyfTable.GlyphComponent], [UInt8]) {
        var components: [GlyfTable.GlyphComponent] = []
        var currentOffset = offset
        var hasMoreComponents = true
        var hasInstructions = false

        while hasMoreComponents {
            let flags = data.readUInt16(at: currentOffset)
            let glyphIndex = data.readUInt16(at: currentOffset + 2)
            currentOffset += 4

            var arg1: Int16 = 0
            var arg2: Int16 = 0

            if flags & 0x0001 != 0 {
                // ARG_1_AND_2_ARE_WORDS
                arg1 = data.readInt16(at: currentOffset)
                arg2 = data.readInt16(at: currentOffset + 2)
                currentOffset += 4
            } else {
                arg1 = Int16(data.readInt8(at: currentOffset))
                arg2 = Int16(data.readInt8(at: currentOffset + 1))
                currentOffset += 2
            }

            var scale: Float? = nil
            var scaleX: Float? = nil
            var scaleY: Float? = nil
            var scale01: Float? = nil
            var scale10: Float? = nil

            if flags & 0x0008 != 0 {
                // WE_HAVE_A_SCALE
                scale = Float(data.readInt16(at: currentOffset)) / 16384.0
                currentOffset += 2
            } else if flags & 0x0040 != 0 {
                // WE_HAVE_AN_X_AND_Y_SCALE
                scaleX = Float(data.readInt16(at: currentOffset)) / 16384.0
                scaleY = Float(data.readInt16(at: currentOffset + 2)) / 16384.0
                currentOffset += 4
            } else if flags & 0x0080 != 0 {
                // WE_HAVE_A_TWO_BY_TWO
                scaleX = Float(data.readInt16(at: currentOffset)) / 16384.0
                scale01 = Float(data.readInt16(at: currentOffset + 2)) / 16384.0
                scale10 = Float(data.readInt16(at: currentOffset + 4)) / 16384.0
                scaleY = Float(data.readInt16(at: currentOffset + 6)) / 16384.0
                currentOffset += 8
            }

            components.append(GlyfTable.GlyphComponent(
                flags: flags,
                glyphIndex: glyphIndex,
                argument1: arg1,
                argument2: arg2,
                scale: scale,
                scaleX: scaleX,
                scaleY: scaleY,
                scale01: scale01,
                scale10: scale10
            ))

            hasMoreComponents = flags & 0x0020 != 0
            hasInstructions = flags & 0x0100 != 0
        }

        var instructions: [UInt8] = []
        if hasInstructions {
            let instructionLength = Int(data.readUInt16(at: currentOffset))
            currentOffset += 2
            instructions = data.readBytes(at: currentOffset, count: instructionLength)
        }

        return (components, instructions)
    }

    // MARK: - Metrics Table Parsing

    private func parseHmtxTable(data: Data, record: TableRecord, numberOfHMetrics: Int, numGlyphs: Int) throws -> HmtxTable {
        let offset = Int(record.offset)
        var hMetrics: [HmtxTable.LongHorMetric] = []
        var leftSideBearings: [Int16] = []

        var currentOffset = offset

        // Read full metrics
        for _ in 0..<numberOfHMetrics {
            let advanceWidth = data.readUInt16(at: currentOffset)
            let lsb = data.readInt16(at: currentOffset + 2)
            hMetrics.append(HmtxTable.LongHorMetric(advanceWidth: advanceWidth, leftSideBearing: lsb))
            currentOffset += 4
        }

        // Read additional left side bearings
        let remainingGlyphs = numGlyphs - numberOfHMetrics
        for _ in 0..<remainingGlyphs {
            leftSideBearings.append(data.readInt16(at: currentOffset))
            currentOffset += 2
        }

        return HmtxTable(hMetrics: hMetrics, leftSideBearings: leftSideBearings)
    }

    // MARK: - Optional Table Parsing

    private func parseNameTable(data: Data, record: TableRecord) throws -> NameTable {
        let offset = Int(record.offset)
        guard offset + 6 <= data.count else {
            throw FontParserError.invalidTableData("name")
        }

        _ = data.readUInt16(at: offset)  // format (ignored)
        let count = Int(data.readUInt16(at: offset + 2))
        let stringOffset = Int(data.readUInt16(at: offset + 4))

        var records: [NameTable.NameRecord] = []
        var recordOffset = offset + 6

        for _ in 0..<count {
            guard recordOffset + 12 <= data.count else { break }

            let platformID = data.readUInt16(at: recordOffset)
            let encodingID = data.readUInt16(at: recordOffset + 2)
            let languageID = data.readUInt16(at: recordOffset + 4)
            let nameID = data.readUInt16(at: recordOffset + 6)
            let length = Int(data.readUInt16(at: recordOffset + 8))
            let strOffset = Int(data.readUInt16(at: recordOffset + 10))

            let stringStart = offset + stringOffset + strOffset
            if stringStart + length <= data.count {
                let stringData = data[stringStart..<stringStart + length]
                let encoding: String.Encoding = (platformID == 3 || platformID == 0) ? .utf16BigEndian : .utf8
                if let value = String(data: stringData, encoding: encoding) {
                    records.append(NameTable.NameRecord(
                        platformID: platformID,
                        encodingID: encodingID,
                        languageID: languageID,
                        nameID: nameID,
                        value: value
                    ))
                }
            }

            recordOffset += 12
        }

        return NameTable(records: records)
    }

    private func parseOS2Table(data: Data, record: TableRecord) throws -> OS2Table {
        let offset = Int(record.offset)
        guard offset + 78 <= data.count else {
            throw FontParserError.invalidTableData("OS/2")
        }

        var os2 = OS2Table()
        os2.version = data.readUInt16(at: offset)
        os2.xAvgCharWidth = data.readInt16(at: offset + 2)
        os2.usWeightClass = data.readUInt16(at: offset + 4)
        os2.usWidthClass = data.readUInt16(at: offset + 6)
        os2.fsType = data.readUInt16(at: offset + 8)
        os2.ySubscriptXSize = data.readInt16(at: offset + 10)
        os2.ySubscriptYSize = data.readInt16(at: offset + 12)
        os2.ySubscriptXOffset = data.readInt16(at: offset + 14)
        os2.ySubscriptYOffset = data.readInt16(at: offset + 16)
        os2.ySuperscriptXSize = data.readInt16(at: offset + 18)
        os2.ySuperscriptYSize = data.readInt16(at: offset + 20)
        os2.ySuperscriptXOffset = data.readInt16(at: offset + 22)
        os2.ySuperscriptYOffset = data.readInt16(at: offset + 24)
        os2.yStrikeoutSize = data.readInt16(at: offset + 26)
        os2.yStrikeoutPosition = data.readInt16(at: offset + 28)
        os2.sFamilyClass = data.readInt16(at: offset + 30)
        os2.panose = data.readBytes(at: offset + 32, count: 10)
        os2.ulUnicodeRange1 = data.readUInt32(at: offset + 42)
        os2.ulUnicodeRange2 = data.readUInt32(at: offset + 46)
        os2.ulUnicodeRange3 = data.readUInt32(at: offset + 50)
        os2.ulUnicodeRange4 = data.readUInt32(at: offset + 54)
        os2.achVendID = data.readTag(at: offset + 58)
        os2.fsSelection = data.readUInt16(at: offset + 62)
        os2.usFirstCharIndex = data.readUInt16(at: offset + 64)
        os2.usLastCharIndex = data.readUInt16(at: offset + 66)
        os2.sTypoAscender = data.readInt16(at: offset + 68)
        os2.sTypoDescender = data.readInt16(at: offset + 70)
        os2.sTypoLineGap = data.readInt16(at: offset + 72)
        os2.usWinAscent = data.readUInt16(at: offset + 74)
        os2.usWinDescent = data.readUInt16(at: offset + 76)

        if os2.version >= 1 && offset + 86 <= data.count {
            os2.ulCodePageRange1 = data.readUInt32(at: offset + 78)
            os2.ulCodePageRange2 = data.readUInt32(at: offset + 82)
        }

        if os2.version >= 2 && offset + 96 <= data.count {
            os2.sxHeight = data.readInt16(at: offset + 86)
            os2.sCapHeight = data.readInt16(at: offset + 88)
            os2.usDefaultChar = data.readUInt16(at: offset + 90)
            os2.usBreakChar = data.readUInt16(at: offset + 92)
            os2.usMaxContext = data.readUInt16(at: offset + 94)
        }

        return os2
    }

    private func parsePostTable(data: Data, record: TableRecord, numGlyphs: Int) throws -> PostTable {
        let offset = Int(record.offset)
        guard offset + 32 <= data.count else {
            throw FontParserError.invalidTableData("post")
        }

        var post = PostTable()
        post.version = data.readFixed(at: offset)
        post.italicAngle = data.readFixed(at: offset + 4)
        post.underlinePosition = data.readInt16(at: offset + 8)
        post.underlineThickness = data.readInt16(at: offset + 10)
        post.isFixedPitch = data.readUInt32(at: offset + 12)
        post.minMemType42 = data.readUInt32(at: offset + 16)
        post.maxMemType42 = data.readUInt32(at: offset + 20)
        post.minMemType1 = data.readUInt32(at: offset + 24)
        post.maxMemType1 = data.readUInt32(at: offset + 28)

        // Parse glyph names for version 2.0
        if post.version.value == 0x00020000 && offset + 34 <= data.count {
            let numberOfGlyphs = Int(data.readUInt16(at: offset + 32))
            var glyphNameIndices: [UInt16] = []
            var extraNames: [String] = []

            var currentOffset = offset + 34

            for _ in 0..<numberOfGlyphs {
                if currentOffset + 2 <= data.count {
                    glyphNameIndices.append(data.readUInt16(at: currentOffset))
                    currentOffset += 2
                }
            }

            // Read extra names (Pascal strings)
            while currentOffset < offset + Int(record.length) {
                let length = Int(data.readUInt8(at: currentOffset))
                currentOffset += 1
                if currentOffset + length <= data.count {
                    let nameData = data[currentOffset..<currentOffset + length]
                    if let name = String(data: nameData, encoding: .utf8) {
                        extraNames.append(name)
                    }
                    currentOffset += length
                }
            }

            // Build glyph names
            post.glyphNames = glyphNameIndices.map { index in
                if index < 258 {
                    return standardMacGlyphNames[Int(index)]
                } else {
                    let extraIndex = Int(index) - 258
                    return extraIndex < extraNames.count ? extraNames[extraIndex] : ".notdef"
                }
            }
        }

        return post
    }

    private func parseKernTable(data: Data, record: TableRecord) throws -> KernTable {
        let offset = Int(record.offset)
        guard offset + 4 <= data.count else {
            throw FontParserError.invalidTableData("kern")
        }

        let version = data.readUInt16(at: offset)
        let nTables = data.readUInt16(at: offset + 2)

        var subtables: [KernTable.KernSubtable] = []
        var currentOffset = offset + 4

        for _ in 0..<nTables {
            guard currentOffset + 6 <= data.count else { break }

            let subtableVersion = data.readUInt16(at: currentOffset)
            let length = data.readUInt16(at: currentOffset + 2)
            let coverage = data.readUInt16(at: currentOffset + 4)

            // Only parse format 0 (pair positioning)
            if coverage & 0xFF == 0 {
                let nPairs = Int(data.readUInt16(at: currentOffset + 6))
                var pairs: [KernTable.KernPair] = []
                var pairOffset = currentOffset + 14

                for _ in 0..<nPairs {
                    guard pairOffset + 6 <= data.count else { break }
                    let left = data.readUInt16(at: pairOffset)
                    let right = data.readUInt16(at: pairOffset + 2)
                    let value = data.readInt16(at: pairOffset + 4)
                    pairs.append(KernTable.KernPair(left: left, right: right, value: value))
                    pairOffset += 6
                }

                subtables.append(KernTable.KernSubtable(
                    version: subtableVersion,
                    length: length,
                    coverage: coverage,
                    pairs: pairs
                ))
            }

            currentOffset += Int(length)
        }

        return KernTable(version: version, subtables: subtables)
    }

    // MARK: - Build FontProject

    private func buildFontProject(
        head: HeadTable,
        hhea: HheaTable,
        maxp: MaxpTable,
        cmap: CmapTable,
        glyf: GlyfTable,
        hmtx: HmtxTable,
        name: NameTable?,
        os2: OS2Table?,
        post: PostTable?,
        kern: KernTable?
    ) -> FontProject {
        // Extract name strings
        let familyName = name?.records.first { $0.nameID == 1 }?.value ?? "Untitled"
        let styleName = name?.records.first { $0.nameID == 2 }?.value ?? "Regular"
        let copyright = name?.records.first { $0.nameID == 0 }?.value ?? ""
        let designer = name?.records.first { $0.nameID == 9 }?.value ?? ""
        let description = name?.records.first { $0.nameID == 10 }?.value ?? ""
        let license = name?.records.first { $0.nameID == 13 }?.value ?? ""
        let version = name?.records.first { $0.nameID == 5 }?.value ?? "1.0"
        let uniqueID: String = name?.records.first { $0.nameID == 3 }?.value ?? UUID().uuidString

        // Build metrics
        let metrics = FontMetrics(
            unitsPerEm: Int(head.unitsPerEm),
            ascender: Int(hhea.ascender),
            descender: Int(hhea.descender),
            xHeight: Int(os2?.sxHeight ?? 500),
            capHeight: Int(os2?.sCapHeight ?? 700),
            lineGap: Int(hhea.lineGap)
        )

        // Build character to glyph mapping
        let bestCmap = cmap.subtables.first { $0.platformID == 3 && $0.encodingID == 1 }
            ?? cmap.subtables.first { $0.platformID == 0 }
            ?? cmap.subtables.first

        var glyphs: [Character: Glyph] = [:]

        if let mapping = bestCmap?.mapping {
            for (codepoint, glyphIndex) in mapping {
                guard let scalar = Unicode.Scalar(codepoint),
                      Int(glyphIndex) < glyf.glyphs.count else { continue }

                let character = Character(scalar)
                let glyphData = glyf.glyphs[Int(glyphIndex)]

                // Get metrics
                let metricIndex = min(Int(glyphIndex), hmtx.hMetrics.count - 1)
                let hMetric = hmtx.hMetrics[metricIndex]

                // Convert TrueType contours to our GlyphOutline format
                let outline = convertToGlyphOutline(glyphData: glyphData)

                let glyph = Glyph(
                    character: character,
                    unicodeScalars: [codepoint],
                    outline: outline,
                    advanceWidth: Int(hMetric.advanceWidth),
                    leftSideBearing: Int(hMetric.leftSideBearing),
                    generatedBy: .imported
                )

                glyphs[character] = glyph
            }
        }

        // Build kerning pairs
        var kerningPairs: [KerningPair] = []
        if let kernTable = kern {
            // Build reverse mapping from glyph index to character
            var glyphToChar: [UInt16: Character] = [:]
            if let mapping = bestCmap?.mapping {
                for (codepoint, glyphIndex) in mapping {
                    if let scalar = Unicode.Scalar(codepoint) {
                        glyphToChar[glyphIndex] = Character(scalar)
                    }
                }
            }

            for subtable in kernTable.subtables {
                for pair in subtable.pairs {
                    if let leftChar = glyphToChar[pair.left],
                       let rightChar = glyphToChar[pair.right] {
                        kerningPairs.append(KerningPair(
                            left: leftChar,
                            right: rightChar,
                            value: Int(pair.value)
                        ))
                    }
                }
            }
        }

        return FontProject(
            name: familyName,
            family: familyName,
            style: styleName,
            metrics: metrics,
            glyphs: glyphs,
            kerning: kerningPairs,
            metadata: FontMetadata(
                copyright: copyright,
                designer: designer,
                license: license,
                version: version,
                uniqueID: uniqueID,
                description: description
            )
        )
    }

    private func convertToGlyphOutline(glyphData: GlyfTable.GlyphData) -> GlyphOutline {
        var contours: [Contour] = []

        for ttContour in glyphData.contours {
            guard !ttContour.points.isEmpty else { continue }

            var pathPoints: [PathPoint] = []
            let points = ttContour.points

            // TrueType uses quadratic beziers with on-curve and off-curve points
            // We need to convert to our cubic bezier representation
            var i = 0
            while i < points.count {
                let point = points[i]

                if point.onCurve {
                    // On-curve point - determine if it's a corner or smooth
                    let prevIndex = (i - 1 + points.count) % points.count
                    let nextIndex = (i + 1) % points.count
                    let prevPoint = points[prevIndex]
                    let nextPoint = points[nextIndex]

                    var controlIn: CGPoint? = nil
                    var controlOut: CGPoint? = nil

                    // Check if previous point is off-curve
                    if !prevPoint.onCurve {
                        controlIn = CGPoint(x: CGFloat(prevPoint.x), y: CGFloat(prevPoint.y))
                    }

                    // Check if next point is off-curve
                    if !nextPoint.onCurve {
                        controlOut = CGPoint(x: CGFloat(nextPoint.x), y: CGFloat(nextPoint.y))
                    }

                    let pointType: PathPoint.PointType
                    if controlIn != nil || controlOut != nil {
                        pointType = .smooth
                    } else {
                        pointType = .corner
                    }

                    pathPoints.append(PathPoint(
                        position: CGPoint(x: CGFloat(point.x), y: CGFloat(point.y)),
                        type: pointType,
                        controlIn: controlIn,
                        controlOut: controlOut
                    ))
                } else {
                    // Off-curve point - check if we need to insert an implicit on-curve point
                    let nextIndex = (i + 1) % points.count
                    let nextPoint = points[nextIndex]

                    if !nextPoint.onCurve {
                        // Two consecutive off-curve points: insert implicit on-curve point
                        let implicitX = (CGFloat(point.x) + CGFloat(nextPoint.x)) / 2
                        let implicitY = (CGFloat(point.y) + CGFloat(nextPoint.y)) / 2

                        pathPoints.append(PathPoint(
                            position: CGPoint(x: implicitX, y: implicitY),
                            type: .smooth,
                            controlIn: CGPoint(x: CGFloat(point.x), y: CGFloat(point.y)),
                            controlOut: CGPoint(x: CGFloat(nextPoint.x), y: CGFloat(nextPoint.y))
                        ))
                    }
                }

                i += 1
            }

            if !pathPoints.isEmpty {
                contours.append(Contour(points: pathPoints, isClosed: true))
            }
        }

        return GlyphOutline(contours: contours)
    }

    // Standard Mac glyph names (first 258)
    private let standardMacGlyphNames: [String] = [
        ".notdef", ".null", "nonmarkingreturn", "space", "exclam", "quotedbl", "numbersign",
        "dollar", "percent", "ampersand", "quotesingle", "parenleft", "parenright", "asterisk",
        "plus", "comma", "hyphen", "period", "slash", "zero", "one", "two", "three", "four",
        "five", "six", "seven", "eight", "nine", "colon", "semicolon", "less", "equal", "greater",
        "question", "at", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N",
        "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "bracketleft", "backslash",
        "bracketright", "asciicircum", "underscore", "grave", "a", "b", "c", "d", "e", "f", "g",
        "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y",
        "z", "braceleft", "bar", "braceright", "asciitilde", "Adieresis", "Aring", "Ccedilla",
        "Eacute", "Ntilde", "Odieresis", "Udieresis", "aacute", "agrave", "acircumflex",
        "adieresis", "atilde", "aring", "ccedilla", "eacute", "egrave", "ecircumflex", "edieresis",
        "iacute", "igrave", "icircumflex", "idieresis", "ntilde", "oacute", "ograve", "ocircumflex",
        "odieresis", "otilde", "uacute", "ugrave", "ucircumflex", "udieresis", "dagger", "degree",
        "cent", "sterling", "section", "bullet", "paragraph", "germandbls", "registered",
        "copyright", "trademark", "acute", "dieresis", "notequal", "AE", "Oslash", "infinity",
        "plusminus", "lessequal", "greaterequal", "yen", "mu", "partialdiff", "summation",
        "product", "pi", "integral", "ordfeminine", "ordmasculine", "Omega", "ae", "oslash",
        "questiondown", "exclamdown", "logicalnot", "radical", "florin", "approxequal", "Delta",
        "guillemotleft", "guillemotright", "ellipsis", "nonbreakingspace", "Agrave", "Atilde",
        "Otilde", "OE", "oe", "endash", "emdash", "quotedblleft", "quotedblright", "quoteleft",
        "quoteright", "divide", "lozenge", "ydieresis", "Ydieresis", "fraction", "currency",
        "guilsinglleft", "guilsinglright", "fi", "fl", "daggerdbl", "periodcentered",
        "quotesinglbase", "quotedblbase", "perthousand", "Acircumflex", "Ecircumflex", "Aacute",
        "Edieresis", "Egrave", "Iacute", "Icircumflex", "Idieresis", "Igrave", "Oacute",
        "Ocircumflex", "apple", "Ograve", "Uacute", "Ucircumflex", "Ugrave", "dotlessi",
        "circumflex", "tilde", "macron", "breve", "dotaccent", "ring", "cedilla", "hungarumlaut",
        "ogonek", "caron", "Lslash", "lslash", "Scaron", "scaron", "Zcaron", "zcaron",
        "brokenbar", "Eth", "eth", "Yacute", "yacute", "Thorn", "thorn", "minus", "multiply",
        "onesuperior", "twosuperior", "threesuperior", "onehalf", "onequarter", "threequarters",
        "franc", "Gbreve", "gbreve", "Idotaccent", "Scedilla", "scedilla", "Cacute", "cacute",
        "Ccaron", "ccaron", "dcroat"
    ]
}
