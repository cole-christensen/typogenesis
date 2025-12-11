import Foundation
import Compression

/// Exports fonts to web formats (WOFF, WOFF2)
actor WebFontExporter {

    enum WebFontError: Error, LocalizedError {
        case compressionFailed
        case invalidInput
        case woff2NotSupported

        var errorDescription: String? {
            switch self {
            case .compressionFailed:
                return "Failed to compress font data"
            case .invalidInput:
                return "Invalid font data"
            case .woff2NotSupported:
                return "WOFF2 export requires Brotli compression (not yet implemented)"
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
        var compressedTables: [(tag: String, origLength: UInt32, compLength: UInt32, data: Data)] = []

        for table in tables {
            let compressed = try compressData(table.data)

            // Only use compressed if it's actually smaller
            if compressed.count < table.data.count {
                compressedTables.append((
                    tag: table.tag,
                    origLength: UInt32(table.data.count),
                    compLength: UInt32(compressed.count),
                    data: compressed
                ))
            } else {
                compressedTables.append((
                    tag: table.tag,
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
            woffData.writeUInt32(table.origLength)  // origChecksum (using origLength as placeholder)

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
    func exportWOFF2(ttfData: Data) throws -> Data {
        // WOFF2 requires Brotli compression which isn't available in Foundation
        // For now, throw an error indicating it's not supported
        throw WebFontError.woff2NotSupported

        // Future implementation would:
        // 1. Parse TTF tables
        // 2. Apply WOFF2 preprocessing (transform tables)
        // 3. Compress with Brotli
        // 4. Build WOFF2 structure
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
        let destinationBuffer = UnsafeMutablePointer<UInt8>.allocate(capacity: sourceSize)
        defer { destinationBuffer.deallocate() }

        let compressedSize = data.withUnsafeBytes { sourceBuffer -> Int in
            guard let sourcePtr = sourceBuffer.bindMemory(to: UInt8.self).baseAddress else {
                return 0
            }

            return compression_encode_buffer(
                destinationBuffer,
                sourceSize,
                sourcePtr,
                sourceSize,
                nil,
                COMPRESSION_ZLIB
            )
        }

        guard compressedSize > 0 else {
            throw WebFontError.compressionFailed
        }

        return Data(bytes: destinationBuffer, count: compressedSize)
    }
}
