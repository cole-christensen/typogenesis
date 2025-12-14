import Foundation
import CoreGraphics

/// Exports variable font tables (fvar, gvar, STAT) for OpenType variable fonts
actor VariableFontExporter {

    enum VariableFontError: Error, LocalizedError {
        case noAxes
        case noMasters
        case incompatibleGlyphs(Character)
        case invalidAxisTag(String)

        var errorDescription: String? {
            switch self {
            case .noAxes: return "Variable font has no axes defined"
            case .noMasters: return "Variable font has no masters defined"
            case .incompatibleGlyphs(let char): return "Glyph '\(char)' has incompatible structure across masters"
            case .invalidAxisTag(let tag): return "Invalid axis tag: \(tag)"
            }
        }
    }

    // MARK: - fvar Table (Font Variations)

    /// Build the fvar table which describes variation axes and named instances
    /// fvar table structure:
    /// - Header (16 bytes)
    /// - Axis records (20 bytes each)
    /// - Instance records (variable size)
    func buildFvarTable(config: VariableFontConfig) throws -> Data {
        guard !config.axes.isEmpty else {
            throw VariableFontError.noAxes
        }

        var data = Data()

        let axisCount = UInt16(config.axes.count)
        let instanceCount = UInt16(config.instances.count)
        let axisSize: UInt16 = 20  // Each axis record is 20 bytes
        let instanceSize = UInt16(4 + config.axes.count * 4)  // nameID + coordinates

        // fvar header
        data.writeUInt16(1)  // majorVersion
        data.writeUInt16(0)  // minorVersion
        data.writeUInt16(16) // axesArrayOffset (immediately after header)
        data.writeUInt16(2)  // reserved
        data.writeUInt16(axisCount)
        data.writeUInt16(axisSize)
        data.writeUInt16(instanceCount)
        data.writeUInt16(instanceSize)

        // Write axis records
        for axis in config.axes {
            try writeAxisRecord(axis: axis, to: &data)
        }

        // Write instance records
        for (index, instance) in config.instances.enumerated() {
            writeInstanceRecord(instance: instance, axes: config.axes, nameID: UInt16(256 + index), to: &data)
        }

        return data
    }

    private func writeAxisRecord(axis: VariationAxis, to data: inout Data) throws {
        // Validate 4-character tag
        guard axis.tag.count == 4 else {
            throw VariableFontError.invalidAxisTag(axis.tag)
        }

        // Tag (4 bytes)
        data.writeTag(axis.tag)

        // Min value (Fixed 16.16)
        data.writeFixed(Fixed(value: Int32(axis.minValue * 65536)))

        // Default value (Fixed 16.16)
        data.writeFixed(Fixed(value: Int32(axis.defaultValue * 65536)))

        // Max value (Fixed 16.16)
        data.writeFixed(Fixed(value: Int32(axis.maxValue * 65536)))

        // Flags (UInt16) - 0 for standard axes
        data.writeUInt16(0)

        // Axis name ID (UInt16) - we'll use IDs starting at 256
        // In a real implementation, these would reference the name table
        let nameID: UInt16 = axis.tag == VariationAxis.weightTag ? 256 :
                             axis.tag == VariationAxis.widthTag ? 257 :
                             axis.tag == VariationAxis.slantTag ? 258 :
                             axis.tag == VariationAxis.italicTag ? 259 :
                             axis.tag == VariationAxis.opticalSizeTag ? 260 : 261
        data.writeUInt16(nameID)
    }

    private func writeInstanceRecord(instance: NamedInstance, axes: [VariationAxis], nameID: UInt16, to data: inout Data) {
        // Subfamily name ID
        data.writeUInt16(nameID)

        // Flags
        data.writeUInt16(0)

        // Coordinates for each axis
        for axis in axes {
            let value = instance.location[axis.tag] ?? axis.defaultValue
            data.writeFixed(Fixed(value: Int32(value * 65536)))
        }
    }

    // MARK: - gvar Table (Glyph Variations)

    /// Build the gvar table which stores per-glyph variation data
    /// gvar table structure:
    /// - Header (20 bytes)
    /// - Glyph variation data offsets
    /// - Shared tuple coordinates (optional)
    /// - Per-glyph variation data
    func buildGvarTable(
        project: FontProject,
        glyphOrder: [(Character?, Glyph?)],
        config: VariableFontConfig
    ) throws -> Data {
        guard !config.axes.isEmpty else {
            throw VariableFontError.noAxes
        }
        guard !config.masters.isEmpty else {
            throw VariableFontError.noMasters
        }

        var data = Data()
        let glyphCount = UInt16(glyphOrder.count)
        let axisCount = UInt16(config.axes.count)

        // Build shared tuples (normalized axis coordinates for each master)
        let sharedTuples = buildSharedTuples(config: config)
        let sharedTupleCount = UInt16(sharedTuples.count / (Int(axisCount) * 2))

        // Build per-glyph variation data
        var glyphVariationData: [Data] = []
        var glyphVariationOffsets: [UInt32] = []
        var currentOffset: UInt32 = 0

        for (char, _) in glyphOrder {
            glyphVariationOffsets.append(currentOffset)

            if let char = char {
                let variationData = try buildGlyphVariationData(
                    character: char,
                    config: config,
                    axisCount: Int(axisCount)
                )
                glyphVariationData.append(variationData)
                currentOffset += UInt32(variationData.count)
            }
            // .notdef and empty glyphs get no variation data
        }
        glyphVariationOffsets.append(currentOffset)  // Final offset

        // Calculate header offsets
        let headerSize: UInt32 = 20
        let offsetsSize = UInt32((glyphCount + 1) * 4)  // Long offsets
        let sharedTuplesOffset = headerSize + offsetsSize
        let glyphVariationDataOffset = sharedTuplesOffset + UInt32(sharedTuples.count)

        // Write header
        data.writeUInt16(1)  // majorVersion
        data.writeUInt16(0)  // minorVersion
        data.writeUInt16(axisCount)
        data.writeUInt16(sharedTupleCount)
        data.writeUInt32(sharedTuplesOffset)
        data.writeUInt16(glyphCount)
        data.writeUInt16(0x0001)  // flags: long offsets
        data.writeUInt32(glyphVariationDataOffset)

        // Write glyph variation data offsets
        for offset in glyphVariationOffsets {
            data.writeUInt32(offset)
        }

        // Write shared tuples
        data.append(sharedTuples)

        // Write glyph variation data
        for variationData in glyphVariationData {
            data.append(variationData)
        }

        return data
    }

    private func buildSharedTuples(config: VariableFontConfig) -> Data {
        var data = Data()

        // Create a shared tuple for each master (normalized coordinates)
        for master in config.masters {
            for axis in config.axes {
                let value = master.location[axis.tag] ?? axis.defaultValue
                // Normalize to -1.0 to 1.0 range (F2Dot14 format)
                let normalized = normalizeAxisValue(value: value, axis: axis)
                let f2dot14 = Int16(normalized * 16384)
                data.writeInt16(f2dot14)
            }
        }

        return data
    }

    private func normalizeAxisValue(value: CGFloat, axis: VariationAxis) -> CGFloat {
        if value == axis.defaultValue {
            return 0
        } else if value < axis.defaultValue {
            return (value - axis.defaultValue) / (axis.defaultValue - axis.minValue)
        } else {
            return (value - axis.defaultValue) / (axis.maxValue - axis.defaultValue)
        }
    }

    private func buildGlyphVariationData(
        character: Character,
        config: VariableFontConfig,
        axisCount: Int
    ) throws -> Data {
        var data = Data()

        // Find the default master
        guard let defaultMaster = config.masters.first(where: { master in
            config.axes.allSatisfy { axis in
                master.location[axis.tag] == axis.defaultValue
            }
        }) ?? config.masters.first else {
            return data  // No masters, no variation data
        }

        // Get the glyph from the default master
        guard let defaultGlyph = defaultMaster.glyphs[character] else {
            return data  // Glyph not in default master
        }

        // Collect variations from other masters
        var tupleVariationHeaders: [(tupleIndex: UInt16, deltas: [[PointDelta]])] = []

        for (masterIndex, master) in config.masters.enumerated() {
            if master.id == defaultMaster.id { continue }

            guard let targetGlyph = master.glyphs[character] else { continue }

            // Calculate deltas
            if let variation = GlyphVariation.calculate(
                character: character,
                source: defaultGlyph,
                target: targetGlyph,
                sourceMasterID: defaultMaster.id,
                targetMasterID: master.id
            ) {
                tupleVariationHeaders.append((UInt16(masterIndex), variation.pointDeltas))
            }
        }

        guard !tupleVariationHeaders.isEmpty else {
            return data  // No variations
        }

        // Write tuple variation count and data offset
        let tupleCount = UInt16(tupleVariationHeaders.count)
        data.writeUInt16(tupleCount | 0x8000)  // High bit = data follows immediately

        // Calculate serialized tuple data offset
        let headerSize = 4 + tupleCount * 4  // count/offset + headers
        data.writeUInt16(UInt16(headerSize))

        // Write tuple variation headers
        var tupleDataOffset: UInt16 = 0
        var serializedTupleData: [Data] = []

        for (tupleIndex, deltas) in tupleVariationHeaders {
            // Serialize the delta data
            let deltaData = serializePointDeltas(deltas: deltas)
            serializedTupleData.append(deltaData)

            // Write header: data size, tuple index
            data.writeUInt16(UInt16(deltaData.count))
            data.writeUInt16(tupleIndex | 0x2000)  // Use shared tuple coordinates

            tupleDataOffset += UInt16(deltaData.count)
        }

        // Append serialized tuple data
        for tupleData in serializedTupleData {
            data.append(tupleData)
        }

        return data
    }

    private func serializePointDeltas(deltas: [[PointDelta]]) -> Data {
        var data = Data()

        // Flatten deltas
        var xDeltas: [Int16] = []
        var yDeltas: [Int16] = []

        for contourDeltas in deltas {
            for delta in contourDeltas {
                xDeltas.append(Int16(clamping: Int(delta.dx)))
                yDeltas.append(Int16(clamping: Int(delta.dy)))
            }
        }

        // Write point count (all points)
        let pointCount = xDeltas.count
        if pointCount < 128 {
            data.writeUInt8(UInt8(pointCount))
        } else {
            data.writeUInt8(UInt8(0x80 | (pointCount >> 8)))
            data.writeUInt8(UInt8(pointCount & 0xFF))
        }

        // Serialize X deltas using packed delta encoding
        data.append(packDeltas(xDeltas))

        // Serialize Y deltas
        data.append(packDeltas(yDeltas))

        return data
    }

    private func packDeltas(_ deltas: [Int16]) -> Data {
        var data = Data()
        var i = 0

        while i < deltas.count {
            let delta = deltas[i]

            // Check for zero runs
            var zeroRun = 0
            while i + zeroRun < deltas.count && deltas[i + zeroRun] == 0 && zeroRun < 64 {
                zeroRun += 1
            }

            if zeroRun >= 2 {
                // Encode zero run: 0x80 | (count - 1)
                data.writeUInt8(0x80 | UInt8(zeroRun - 1))
                i += zeroRun
                continue
            }

            // Check if we can use byte deltas
            if delta >= -128 && delta <= 127 {
                // Count consecutive byte-sized deltas
                var byteRun = 0
                while i + byteRun < deltas.count &&
                      deltas[i + byteRun] >= -128 &&
                      deltas[i + byteRun] <= 127 &&
                      byteRun < 64 {
                    byteRun += 1
                }

                // Encode byte run: 0x00 | (count - 1)
                data.writeUInt8(UInt8(byteRun - 1))
                for j in 0..<byteRun {
                    data.writeInt8(Int8(deltas[i + j]))
                }
                i += byteRun
            } else {
                // Need word-sized deltas
                var wordRun = 0
                while i + wordRun < deltas.count &&
                      (deltas[i + wordRun] < -128 || deltas[i + wordRun] > 127) &&
                      wordRun < 64 {
                    wordRun += 1
                }

                if wordRun == 0 { wordRun = 1 }

                // Encode word run: 0x40 | (count - 1)
                data.writeUInt8(0x40 | UInt8(wordRun - 1))
                for j in 0..<wordRun {
                    data.writeInt16(deltas[i + j])
                }
                i += wordRun
            }
        }

        return data
    }

    // MARK: - STAT Table (Style Attributes)

    /// Build the STAT table which provides a complete font style overview
    func buildSTATTable(config: VariableFontConfig, fontFamily: String, fontStyle: String) throws -> Data {
        guard !config.axes.isEmpty else {
            throw VariableFontError.noAxes
        }

        var data = Data()

        let axisCount = UInt16(config.axes.count)

        // Calculate offsets
        let headerSize: UInt16 = 20
        let designAxisRecordSize: UInt16 = 8
        let designAxesOffset = headerSize
        let axisValueOffset = designAxesOffset + axisCount * designAxisRecordSize

        // Write header
        data.writeUInt16(1)  // majorVersion
        data.writeUInt16(2)  // minorVersion (version 1.2)
        data.writeUInt16(designAxisRecordSize)
        data.writeUInt16(axisCount)
        data.writeUInt32(UInt32(designAxesOffset))
        data.writeUInt16(0)  // axisValueCount (simplified: no axis values)
        data.writeUInt32(0)  // axisValueArrayOffset
        data.writeUInt16(2)  // elidedFallbackNameID (Subfamily name)

        // Write design axis records
        for (index, axis) in config.axes.enumerated() {
            // Tag
            data.writeTag(axis.tag)

            // Name ID
            data.writeUInt16(UInt16(256 + index))

            // Axis ordering
            data.writeUInt16(UInt16(index))
        }

        return data
    }

    // MARK: - avar Table (Axis Variations)

    /// Build the avar table for axis value normalization
    /// This table allows non-linear axis mapping
    func buildAvarTable(config: VariableFontConfig) throws -> Data {
        guard !config.axes.isEmpty else {
            throw VariableFontError.noAxes
        }

        var data = Data()

        // Header
        data.writeUInt16(1)  // majorVersion
        data.writeUInt16(0)  // minorVersion
        data.writeUInt16(0)  // reserved
        data.writeUInt16(UInt16(config.axes.count))

        // Write segment maps for each axis (identity mapping - linear)
        for _ in config.axes {
            // Each segment map has a position count and pairs
            // Simple identity mapping: 3 points (min, default, max)
            data.writeUInt16(3)  // positionMapCount

            // Point 1: min (-1.0, -1.0)
            data.writeInt16(-16384)  // fromCoordinate (F2Dot14)
            data.writeInt16(-16384)  // toCoordinate (F2Dot14)

            // Point 2: default (0.0, 0.0)
            data.writeInt16(0)
            data.writeInt16(0)

            // Point 3: max (1.0, 1.0)
            data.writeInt16(16384)
            data.writeInt16(16384)
        }

        return data
    }

    // MARK: - Name Table Additions

    /// Get additional name table entries needed for variable fonts
    func getVariableFontNameEntries(config: VariableFontConfig) -> [(nameID: UInt16, value: String)] {
        var entries: [(nameID: UInt16, value: String)] = []

        // Axis names
        for (index, axis) in config.axes.enumerated() {
            entries.append((UInt16(256 + index), axis.name))
        }

        // Instance names
        for (index, instance) in config.instances.enumerated() {
            entries.append((UInt16(256 + config.axes.count + index), instance.name))
        }

        return entries
    }
}
