import Foundation

// MARK: - OpenType File Structure

/// OpenType/TrueType font file offset table
struct OffsetTable {
    let sfntVersion: UInt32  // 0x00010000 for TrueType, 'OTTO' for CFF
    let numTables: UInt16
    let searchRange: UInt16
    let entrySelector: UInt16
    let rangeShift: UInt16

    static let trueTypeVersion: UInt32 = 0x00010000
    static let cffVersion: UInt32 = 0x4F54544F  // 'OTTO'

    var isTrueType: Bool { sfntVersion == Self.trueTypeVersion }
    var isCFF: Bool { sfntVersion == Self.cffVersion }
}

/// Table directory entry
struct TableRecord {
    let tag: String  // 4-character tag
    let checksum: UInt32
    let offset: UInt32
    let length: UInt32
}

// MARK: - Required Tables

/// 'head' - Font Header Table
struct HeadTable {
    var majorVersion: UInt16 = 1
    var minorVersion: UInt16 = 0
    var fontRevision: Fixed = Fixed(value: 0x00010000)  // 1.0
    var checksumAdjustment: UInt32 = 0
    var magicNumber: UInt32 = 0x5F0F3CF5
    var flags: UInt16 = 0x000B  // Baseline at y=0, LSB at x=0, instructions depend on point size
    var unitsPerEm: UInt16 = 1000
    var created: LongDateTime = LongDateTime()
    var modified: LongDateTime = LongDateTime()
    var xMin: Int16 = 0
    var yMin: Int16 = 0
    var xMax: Int16 = 0
    var yMax: Int16 = 0
    var macStyle: UInt16 = 0
    var lowestRecPPEM: UInt16 = 8
    var fontDirectionHint: Int16 = 2  // Mixed directional glyphs
    var indexToLocFormat: Int16 = 1   // 0 for short offsets, 1 for long
    var glyphDataFormat: Int16 = 0
}

/// 'hhea' - Horizontal Header Table
struct HheaTable {
    var majorVersion: UInt16 = 1
    var minorVersion: UInt16 = 0
    var ascender: Int16 = 800
    var descender: Int16 = -200
    var lineGap: Int16 = 90
    var advanceWidthMax: UInt16 = 0
    var minLeftSideBearing: Int16 = 0
    var minRightSideBearing: Int16 = 0
    var xMaxExtent: Int16 = 0
    var caretSlopeRise: Int16 = 1
    var caretSlopeRun: Int16 = 0
    var caretOffset: Int16 = 0
    var reserved1: Int16 = 0
    var reserved2: Int16 = 0
    var reserved3: Int16 = 0
    var reserved4: Int16 = 0
    var metricDataFormat: Int16 = 0
    var numberOfHMetrics: UInt16 = 0
}

/// 'maxp' - Maximum Profile Table
struct MaxpTable {
    var version: Fixed = Fixed(value: 0x00010000)  // 1.0 for TrueType
    var numGlyphs: UInt16 = 0
    var maxPoints: UInt16 = 0
    var maxContours: UInt16 = 0
    var maxCompositePoints: UInt16 = 0
    var maxCompositeContours: UInt16 = 0
    var maxZones: UInt16 = 2
    var maxTwilightPoints: UInt16 = 0
    var maxStorage: UInt16 = 0
    var maxFunctionDefs: UInt16 = 0
    var maxInstructionDefs: UInt16 = 0
    var maxStackElements: UInt16 = 0
    var maxSizeOfInstructions: UInt16 = 0
    var maxComponentElements: UInt16 = 0
    var maxComponentDepth: UInt16 = 0
}

/// 'OS/2' - OS/2 and Windows Metrics Table
struct OS2Table {
    var version: UInt16 = 4
    var xAvgCharWidth: Int16 = 500
    var usWeightClass: UInt16 = 400  // Normal
    var usWidthClass: UInt16 = 5     // Medium
    var fsType: UInt16 = 0           // Installable embedding
    var ySubscriptXSize: Int16 = 650
    var ySubscriptYSize: Int16 = 600
    var ySubscriptXOffset: Int16 = 0
    var ySubscriptYOffset: Int16 = 75
    var ySuperscriptXSize: Int16 = 650
    var ySuperscriptYSize: Int16 = 600
    var ySuperscriptXOffset: Int16 = 0
    var ySuperscriptYOffset: Int16 = 350
    var yStrikeoutSize: Int16 = 50
    var yStrikeoutPosition: Int16 = 300
    var sFamilyClass: Int16 = 0
    var panose: [UInt8] = Array(repeating: 0, count: 10)
    var ulUnicodeRange1: UInt32 = 0
    var ulUnicodeRange2: UInt32 = 0
    var ulUnicodeRange3: UInt32 = 0
    var ulUnicodeRange4: UInt32 = 0
    var achVendID: String = "TYPO"
    var fsSelection: UInt16 = 0x0040  // Regular
    var usFirstCharIndex: UInt16 = 0
    var usLastCharIndex: UInt16 = 0
    var sTypoAscender: Int16 = 800
    var sTypoDescender: Int16 = -200
    var sTypoLineGap: Int16 = 90
    var usWinAscent: UInt16 = 1000
    var usWinDescent: UInt16 = 200
    var ulCodePageRange1: UInt32 = 0
    var ulCodePageRange2: UInt32 = 0
    var sxHeight: Int16 = 500
    var sCapHeight: Int16 = 700
    var usDefaultChar: UInt16 = 0
    var usBreakChar: UInt16 = 32
    var usMaxContext: UInt16 = 0
}

/// 'name' - Naming Table
struct NameTable {
    var records: [NameRecord] = []

    struct NameRecord {
        var platformID: UInt16
        var encodingID: UInt16
        var languageID: UInt16
        var nameID: UInt16
        var value: String
    }

    enum NameID: UInt16 {
        case copyright = 0
        case fontFamily = 1
        case fontSubfamily = 2
        case uniqueID = 3
        case fullFontName = 4
        case versionString = 5
        case postScriptName = 6
        case trademark = 7
        case manufacturer = 8
        case designer = 9
        case description = 10
        case urlVendor = 11
        case urlDesigner = 12
        case licenseDescription = 13
        case licenseURL = 14
        case preferredFamily = 16
        case preferredSubfamily = 17
        case sampleText = 19
    }
}

/// 'cmap' - Character to Glyph Index Mapping Table
struct CmapTable {
    var version: UInt16 = 0
    var subtables: [CmapSubtable] = []

    struct CmapSubtable {
        var platformID: UInt16
        var encodingID: UInt16
        var format: UInt16
        var mapping: [UInt32: UInt16]  // Unicode codepoint to glyph index
    }
}

/// 'post' - PostScript Table
struct PostTable {
    var version: Fixed = Fixed(value: 0x00020000)  // Version 2.0
    var italicAngle: Fixed = Fixed(value: 0)
    var underlinePosition: Int16 = -100
    var underlineThickness: Int16 = 50
    var isFixedPitch: UInt32 = 0
    var minMemType42: UInt32 = 0
    var maxMemType42: UInt32 = 0
    var minMemType1: UInt32 = 0
    var maxMemType1: UInt32 = 0
    var glyphNames: [String] = []
}

// MARK: - Glyph Tables

/// 'glyf' - Glyph Data Table (TrueType outlines)
struct GlyfTable {
    var glyphs: [GlyphData] = []

    struct GlyphData {
        var numberOfContours: Int16  // -1 for composite glyphs
        var xMin: Int16
        var yMin: Int16
        var xMax: Int16
        var yMax: Int16
        var contours: [GlyphContour]  // For simple glyphs
        var components: [GlyphComponent]  // For composite glyphs
        var instructions: [UInt8]
    }

    struct GlyphContour {
        var points: [GlyphPoint]
    }

    struct GlyphPoint {
        var x: Int16
        var y: Int16
        var onCurve: Bool
    }

    struct GlyphComponent {
        var flags: UInt16
        var glyphIndex: UInt16
        var argument1: Int16
        var argument2: Int16
        var scale: Float?
        var scaleX: Float?
        var scaleY: Float?
        var scale01: Float?
        var scale10: Float?
    }
}

/// 'loca' - Index to Location Table
struct LocaTable {
    var offsets: [UInt32] = []  // Offsets into glyf table
    var format: Int16 = 1  // 0 = short (UInt16/2), 1 = long (UInt32)
}

/// 'hmtx' - Horizontal Metrics Table
struct HmtxTable {
    var hMetrics: [LongHorMetric] = []
    var leftSideBearings: [Int16] = []  // For glyphs beyond numberOfHMetrics

    struct LongHorMetric {
        var advanceWidth: UInt16
        var leftSideBearing: Int16
    }
}

// MARK: - OpenType Feature Tables

/// 'kern' - Kerning Table (legacy)
struct KernTable {
    var version: UInt16 = 0
    var subtables: [KernSubtable] = []

    struct KernSubtable {
        var version: UInt16
        var length: UInt16
        var coverage: UInt16
        var pairs: [KernPair]
    }

    struct KernPair {
        var left: UInt16
        var right: UInt16
        var value: Int16
    }
}

/// 'GPOS' - Glyph Positioning Table
struct GPOSTable {
    var majorVersion: UInt16 = 1
    var minorVersion: UInt16 = 0
    var scriptList: [ScriptRecord] = []
    var featureList: [FeatureRecord] = []
    var lookupList: [Lookup] = []

    struct ScriptRecord {
        var tag: String
        var offset: UInt16
    }

    struct FeatureRecord {
        var tag: String
        var offset: UInt16
    }

    struct Lookup {
        var lookupType: UInt16
        var lookupFlag: UInt16
        var subtables: [Data]
    }
}

// MARK: - OpenType Data Types

struct Fixed {
    var value: Int32

    init(value: Int32) {
        self.value = value
    }

    init(integer: Int16, fraction: UInt16) {
        self.value = (Int32(integer) << 16) | Int32(fraction)
    }

    var floatValue: Float {
        Float(value) / 65536.0
    }
}

struct LongDateTime {
    var value: Int64

    init() {
        // Seconds since January 1, 1904
        let referenceDate = Date(timeIntervalSince1970: -2082844800)
        self.value = Int64(Date().timeIntervalSince(referenceDate))
    }

    init(value: Int64) {
        self.value = value
    }

    var date: Date {
        let referenceDate = Date(timeIntervalSince1970: -2082844800)
        return referenceDate.addingTimeInterval(TimeInterval(value))
    }
}

// MARK: - Binary Data Reading/Writing Extensions

extension Data {
    func readUInt8(at offset: Int) -> UInt8 {
        self[offset]
    }

    func readInt8(at offset: Int) -> Int8 {
        Int8(bitPattern: self[offset])
    }

    func readUInt16(at offset: Int) -> UInt16 {
        UInt16(self[offset]) << 8 | UInt16(self[offset + 1])
    }

    func readInt16(at offset: Int) -> Int16 {
        Int16(bitPattern: readUInt16(at: offset))
    }

    func readUInt32(at offset: Int) -> UInt32 {
        UInt32(self[offset]) << 24 |
        UInt32(self[offset + 1]) << 16 |
        UInt32(self[offset + 2]) << 8 |
        UInt32(self[offset + 3])
    }

    func readInt32(at offset: Int) -> Int32 {
        Int32(bitPattern: readUInt32(at: offset))
    }

    func readInt64(at offset: Int) -> Int64 {
        Int64(readUInt32(at: offset)) << 32 | Int64(readUInt32(at: offset + 4))
    }

    func readFixed(at offset: Int) -> Fixed {
        Fixed(value: readInt32(at: offset))
    }

    func readTag(at offset: Int) -> String {
        String(bytes: self[offset..<offset+4], encoding: .ascii) ?? "????"
    }

    func readBytes(at offset: Int, count: Int) -> [UInt8] {
        Array(self[offset..<offset+count])
    }
}

extension Data {
    mutating func writeUInt8(_ value: UInt8) {
        append(value)
    }

    mutating func writeInt8(_ value: Int8) {
        append(UInt8(bitPattern: value))
    }

    mutating func writeUInt16(_ value: UInt16) {
        append(UInt8(value >> 8))
        append(UInt8(value & 0xFF))
    }

    mutating func writeInt16(_ value: Int16) {
        writeUInt16(UInt16(bitPattern: value))
    }

    mutating func writeUInt32(_ value: UInt32) {
        append(UInt8(value >> 24))
        append(UInt8((value >> 16) & 0xFF))
        append(UInt8((value >> 8) & 0xFF))
        append(UInt8(value & 0xFF))
    }

    mutating func writeInt32(_ value: Int32) {
        writeUInt32(UInt32(bitPattern: value))
    }

    mutating func writeInt64(_ value: Int64) {
        writeUInt32(UInt32(value >> 32))
        writeUInt32(UInt32(value & 0xFFFFFFFF))
    }

    mutating func writeFixed(_ value: Fixed) {
        writeInt32(value.value)
    }

    mutating func writeTag(_ value: String) {
        let padded = value.padding(toLength: 4, withPad: " ", startingAt: 0)
        append(contentsOf: padded.utf8.prefix(4))
    }

    mutating func writeBytes(_ bytes: [UInt8]) {
        append(contentsOf: bytes)
    }

    mutating func pad(to alignment: Int) {
        let remainder = count % alignment
        if remainder > 0 {
            append(contentsOf: [UInt8](repeating: 0, count: alignment - remainder))
        }
    }
}

// MARK: - Checksum Calculation

extension Data {
    func calculateTableChecksum() -> UInt32 {
        var sum: UInt32 = 0
        var paddedData = self
        paddedData.pad(to: 4)

        for i in stride(from: 0, to: paddedData.count, by: 4) {
            sum = sum &+ paddedData.readUInt32(at: i)
        }
        return sum
    }
}
