import Foundation
import Compression

/// Exports fonts to web formats (WOFF, WOFF2)
actor WebFontExporter {

    enum WebFontError: Error, LocalizedError {
        case compressionFailed
        case invalidInput
        case tooManyTables(Int)

        var errorDescription: String? {
            switch self {
            case .compressionFailed:
                return "Failed to compress font data"
            case .invalidInput:
                return "Invalid font data"
            case .tooManyTables(let count):
                return "Font has \(count) tables, exceeding the maximum of 65535"
            }
        }
    }

    // MARK: - WOFF Export

    /// Convert TTF data to WOFF format
    func exportWOFF(ttfData: Data) throws -> Data {
        guard ttfData.count >= 12 else {
            throw WebFontError.invalidInput
        }

        // Parse TTF structure
        let tables = try parseTTFTables(from: ttfData)

        guard tables.count <= Int(UInt16.max) else {
            throw WebFontError.tooManyTables(tables.count)
        }

        // WOFF Header
        var woffData = Data()

        // Signature: "wOFF"
        woffData.writeTag("wOFF")

        // Flavor (original font format)
        let flavor = ttfData.prefix(4)
        woffData.append(flavor)

        // Length (will be calculated later)
        let lengthOffset = woffData.count
        woffData.writeUInt32(0)  // Placeholder

        // Number of tables
        woffData.writeUInt16(UInt16(tables.count))

        // Reserved
        woffData.writeUInt16(0)

        // Total size of uncompressed font data
        woffData.writeUInt32(UInt32(ttfData.count))

        // Major/minor version
        woffData.writeUInt16(1)
        woffData.writeUInt16(0)

        // Metadata offset, length, originalLength (no metadata)
        woffData.writeUInt32(0)
        woffData.writeUInt32(0)
        woffData.writeUInt32(0)

        // Private data offset, length (no private data)
        woffData.writeUInt32(0)
        woffData.writeUInt32(0)

        // Calculate compressed table data
        var compressedTables: [(tag: String, checksum: UInt32, origLength: UInt32, compLength: UInt32, data: Data)] = []

        for table in tables {
            let compressed = try compressData(table.data)

            // Only use compressed if it's actually smaller
            if compressed.count < table.data.count {
                compressedTables.append((
                    tag: table.tag,
                    checksum: table.checksum,
                    origLength: UInt32(table.data.count),
                    compLength: UInt32(compressed.count),
                    data: compressed
                ))
            } else {
                compressedTables.append((
                    tag: table.tag,
                    checksum: table.checksum,
                    origLength: UInt32(table.data.count),
                    compLength: UInt32(table.data.count),
                    data: table.data
                ))
            }
        }

        // Calculate table directory size
        let tableDirectorySize = tables.count * 20
        var dataOffset = UInt32(woffData.count + tableDirectorySize)

        // Write table directory
        for table in compressedTables {
            woffData.writeTag(table.tag)
            woffData.writeUInt32(dataOffset)
            woffData.writeUInt32(table.compLength)
            woffData.writeUInt32(table.origLength)
            woffData.writeUInt32(table.checksum)  // origChecksum

            // Pad to 4-byte boundary
            let paddedLength = (table.compLength + 3) & ~3
            dataOffset += paddedLength
        }

        // Write compressed table data
        for table in compressedTables {
            woffData.append(table.data)

            // Pad to 4-byte boundary
            let padding = (4 - (table.data.count % 4)) % 4
            for _ in 0..<padding {
                woffData.writeUInt8(0)
            }
        }

        // Update length field
        let totalLength = UInt32(woffData.count)
        woffData.replaceSubrange(lengthOffset..<lengthOffset+4, with: withUnsafeBytes(of: totalLength.bigEndian) { Data($0) })

        return woffData
    }

    /// Convert TTF data to WOFF2 format
    ///
    /// WOFF2 uses Brotli compression (available via COMPRESSION_BROTLI on macOS 14+).
    /// This is a simplified v1 implementation that compresses all tables together
    /// without applying WOFF2 table transformations (glyf/loca transforms).
    func exportWOFF2(ttfData: Data) throws -> Data {
        guard ttfData.count >= 12 else {
            throw WebFontError.invalidInput
        }

        // Parse TTF structure
        let tables = try parseTTFTables(from: ttfData)

        // WOFF2 known table tags in order (index 0-62)
        // Tables with these tags use flag-based encoding instead of literal tag
        let knownTags = [
            "cmap", "head", "hhea", "hmtx", "maxp", "name", "OS/2", "post",
            "cvt ", "fpgm", "glyf", "loca", "prep", "CFF ", "VORG", "EBDT",
            "EBLC", "gasp", "hdmx", "kern", "LTSH", "PCLT", "VDMX", "vhea",
            "vmtx", "BASE", "GDEF", "GPOS", "GSUB", "EBSC", "JSTF", "MATH",
            "CBDT", "CBLC", "COLR", "CPAL", "SVG ", "sbix", "acnt", "avar",
            "bdat", "bloc", "bsln", "cvar", "fdsc", "feat", "fmtx", "fvar",
            "gvar", "hsty", "just", "lcar", "mort", "morx", "opbd", "prop",
            "trak", "Zapf", "Silf", "Glat", "Gloc", "Feat", "Sill"
        ]

        guard tables.count <= Int(UInt16.max) else {
            throw WebFontError.tooManyTables(tables.count)
        }

        // Sort tables according to WOFF2 recommended order
        let sortedTables = tables.sorted { t1, t2 in
            let idx1 = knownTags.firstIndex(of: t1.tag) ?? 999
            let idx2 = knownTags.firstIndex(of: t2.tag) ?? 999
            if idx1 != idx2 {
                return idx1 < idx2
            }
            return t1.tag < t2.tag
        }

        // Build the uncompressed data stream (all tables concatenated WITHOUT padding).
        // Per the W3C WOFF2 spec: "There MUST NOT be any extraneous data between the
        // table entries in the decompressed data stream." Padding only exists in the
        // reconstructed sfnt, not in the compressed data.
        var uncompressedStream = Data()
        var tableInfos: [(tag: String, origLength: UInt32)] = []

        for table in sortedTables {
            tableInfos.append((
                tag: table.tag,
                origLength: UInt32(table.data.count)
            ))
            uncompressedStream.append(table.data)
        }

        // Compress the entire stream with Brotli
        let compressedStream = try compressDataBrotli(uncompressedStream)

        // Calculate totalSfntSize: sfnt header (12 bytes) + table directory (16 bytes per table)
        // + sum of each table's data padded to 4-byte alignment (except the last table,
        // which does not need trailing padding in the reconstructed sfnt)
        var totalSfntSize: UInt32 = 12 + UInt32(sortedTables.count) * 16
        for (index, table) in sortedTables.enumerated() {
            if index < sortedTables.count - 1 {
                totalSfntSize += UInt32((table.data.count + 3) & ~3)
            } else {
                totalSfntSize += UInt32(table.data.count)
            }
        }

        // Build WOFF2 file
        var woff2Data = Data()

        // === WOFF2 Header (48 bytes) ===

        // Signature: "wOF2"
        woff2Data.writeTag("wOF2")

        // Flavor: original font flavor (TrueType or CFF)
        let flavor = ttfData.prefix(4)
        woff2Data.append(flavor)

        // Length: total WOFF2 file size (placeholder, update later)
        let lengthOffset = woff2Data.count
        woff2Data.writeUInt32(0)

        // NumTables
        woff2Data.writeUInt16(UInt16(sortedTables.count))

        // Reserved
        woff2Data.writeUInt16(0)

        // TotalSfntSize: uncompressed font size including sfnt header
        woff2Data.writeUInt32(totalSfntSize)

        // TotalCompressedSize: size of compressed data block
        woff2Data.writeUInt32(UInt32(compressedStream.count))

        // MajorVersion
        woff2Data.writeUInt16(1)

        // MinorVersion
        woff2Data.writeUInt16(0)

        // MetaOffset, MetaLength, MetaOrigLength (no metadata)
        woff2Data.writeUInt32(0)
        woff2Data.writeUInt32(0)
        woff2Data.writeUInt32(0)

        // PrivOffset, PrivLength (no private data)
        woff2Data.writeUInt32(0)
        woff2Data.writeUInt32(0)

        // === Table Directory ===
        // Each entry is variable-length with UIntBase128 encoding

        for info in tableInfos {
            // Flags byte: bits 0-5 = known table index or 63 for arbitrary tag
            // bits 6-7 = transformation version
            // For glyf/loca: version 3 (0xC0) = null transform (no transformation applied)
            // For other tables: version 0 = no transform
            if let knownIndex = knownTags.firstIndex(of: info.tag) {
                var flagByte = UInt8(knownIndex)
                // glyf and loca require transformation version 3 (null transform)
                // when writing raw untransformed table data
                if info.tag == "glyf" || info.tag == "loca" {
                    flagByte |= 0xC0  // Set bits 6-7 to 11 (version 3)
                }
                woff2Data.writeUInt8(flagByte)
            } else {
                // Unknown table: flag = 63, followed by 4-byte tag
                woff2Data.writeUInt8(63)
                woff2Data.writeTag(info.tag)
            }

            // origLength as UIntBase128
            writeUIntBase128(&woff2Data, value: info.origLength)

            // transformLength is only written for transformation version 0 (actual transform).
            // For version 3 (null transform, 0xC0), transformLength is NOT written.
            // Since we use version 3 for glyf/loca and version 0 for others (no transform),
            // no transformLength field is needed for any table.
        }

        // === Compressed Data Stream ===
        woff2Data.append(compressedStream)

        // Update length field with actual total size
        let totalLength = UInt32(woff2Data.count)
        woff2Data.replaceSubrange(
            lengthOffset..<lengthOffset + 4,
            with: withUnsafeBytes(of: totalLength.bigEndian) { Data($0) }
        )

        return woff2Data
    }

    // MARK: - WOFF2 Helpers

    /// Compress data using Brotli (COMPRESSION_BROTLI)
    /// WOFF2 spec requires Brotli compression always - no raw data bypass.
    private func compressDataBrotli(_ data: Data) throws -> Data {
        let sourceSize = data.count

        // Allocate destination buffer sized for Brotli worst case.
        // Per the Brotli spec, worst-case output is: input_size + (input_size >> 14) + 11 + 1.
        // We use +1024 instead of +12 for comfortable margin.
        let destinationSize = sourceSize + (sourceSize >> 14) + 1024
        let destinationBuffer = UnsafeMutablePointer<UInt8>.allocate(capacity: destinationSize)
        defer { destinationBuffer.deallocate() }

        let compressedSize = data.withUnsafeBytes { sourceBuffer -> Int in
            guard let sourcePtr = sourceBuffer.bindMemory(to: UInt8.self).baseAddress else {
                return 0
            }

            return compression_encode_buffer(
                destinationBuffer,
                destinationSize,
                sourcePtr,
                sourceSize,
                nil,
                COMPRESSION_BROTLI
            )
        }

        if compressedSize <= 0 {
            throw WebFontError.compressionFailed
        }

        return Data(bytes: destinationBuffer, count: compressedSize)
    }

    /// Write a UInt32 value using WOFF2's UIntBase128 variable-length encoding
    /// Values are encoded in big-endian order with 7 bits per byte, MSB indicating continuation
    private func writeUIntBase128(_ data: inout Data, value: UInt32) {
        if value == 0 {
            data.writeUInt8(0)
            return
        }

        // Calculate how many bytes we need (1-5 bytes for UInt32)
        var bytes: [UInt8] = []
        var remaining = value

        while remaining > 0 {
            bytes.insert(UInt8(remaining & 0x7F), at: 0)
            remaining >>= 7
        }

        // Set continuation bit (0x80) on all bytes except the last
        for i in 0..<bytes.count - 1 {
            bytes[i] |= 0x80
        }

        data.append(contentsOf: bytes)
    }

    // MARK: - Helper Methods

    private struct TTFTable {
        let tag: String
        let checksum: UInt32
        let offset: UInt32
        let length: UInt32
        let data: Data
    }

    private func parseTTFTables(from data: Data) throws -> [TTFTable] {
        guard data.count >= 12 else {
            throw WebFontError.invalidInput
        }

        // Read number of tables
        let numTables = data.readUInt16(at: 4)

        var tables: [TTFTable] = []

        // Parse table directory
        for i in 0..<Int(numTables) {
            let recordOffset = 12 + i * 16

            guard recordOffset + 16 <= data.count else {
                throw WebFontError.invalidInput
            }

            let tag = data.readTag(at: recordOffset)
            let checksum = data.readUInt32(at: recordOffset + 4)
            let offset = data.readUInt32(at: recordOffset + 8)
            let length = data.readUInt32(at: recordOffset + 12)

            guard Int(offset) + Int(length) <= data.count else {
                throw WebFontError.invalidInput
            }

            let tableData = data.subdata(in: Int(offset)..<Int(offset)+Int(length))

            tables.append(TTFTable(
                tag: tag,
                checksum: checksum,
                offset: offset,
                length: length,
                data: tableData
            ))
        }

        return tables
    }

    private func compressData(_ data: Data) throws -> Data {
        // Use zlib compression (DEFLATE)
        let sourceSize = data.count

        // Skip compression for tiny tables (< 32 bytes). The zlib header/trailer overhead
        // (~11 bytes for raw deflate, more with wrapper) means compression of very small
        // inputs often produces output larger than the original, wasting CPU for no benefit.
        if sourceSize < 32 {
            return data
        }

        // Allocate destination buffer with extra space
        // Compression can produce larger output for incompressible data
        let destinationSize = sourceSize + 1024  // Extra headroom for zlib overhead
        let destinationBuffer = UnsafeMutablePointer<UInt8>.allocate(capacity: destinationSize)
        defer { destinationBuffer.deallocate() }

        let compressedSize = data.withUnsafeBytes { sourceBuffer -> Int in
            guard let sourcePtr = sourceBuffer.bindMemory(to: UInt8.self).baseAddress else {
                return 0
            }

            return compression_encode_buffer(
                destinationBuffer,
                destinationSize,
                sourcePtr,
                sourceSize,
                nil,
                COMPRESSION_ZLIB
            )
        }

        // Negative or zero return from compression_encode_buffer indicates compression
        // could not produce output. WOFF allows storing tables uncompressed, so return
        // the original data rather than failing the entire export.
        if compressedSize <= 0 {
            return data
        }

        // If compressed output is larger than or equal to the original, compression expanded
        // the data (common for already-compressed or incompressible tables). Return original.
        if compressedSize >= sourceSize {
            return data
        }

        return Data(bytes: destinationBuffer, count: compressedSize)
    }
}
