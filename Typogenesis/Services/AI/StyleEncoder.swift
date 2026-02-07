import Foundation
import CoreGraphics
import AppKit
@preconcurrency import CoreML
import CoreImage
import os.log

/// Service for extracting style features from fonts and glyphs
///
/// ## Thread Safety
/// This class conforms to `@unchecked Sendable` because:
/// - `embeddingCache` is protected by `cacheLock` (NSLock)
/// - `representativeChars` is immutable (let) and set at init
/// - All other state is either immutable or accessed through lock-protected methods
final class StyleEncoder: @unchecked Sendable {

    /// Logger for tracking encoding method and fallback usage
    private static let logger = Logger(subsystem: "com.typogenesis", category: "StyleEncoder")

    // MARK: - Constants

    /// Parameters for style encoding
    private enum EncodingParams {
        /// Input image size for the model
        static let inputImageSize: Int = 64
        /// Embedding dimension
        static let embeddingDimension: Int = 128
        /// Maximum cache size for embeddings
        static let maxCacheSize: Int = 500
    }

    enum StyleEncoderError: Error, LocalizedError {
        case modelNotLoaded
        case invalidInput
        case encodingFailed(String)

        var errorDescription: String? {
            switch self {
            case .modelNotLoaded:
                return "Style encoder model is not loaded"
            case .invalidInput:
                return "Invalid input for style encoding"
            case .encodingFailed(let reason):
                return "Style encoding failed: \(reason)"
            }
        }
    }

    /// Extracted style features from a font or glyph
    struct FontStyle: Codable, Equatable {
        // Measured properties
        var strokeWeight: Float       // Average stroke thickness (0-1 normalized)
        var strokeContrast: Float     // Variation in stroke thickness (0-1)
        var xHeightRatio: Float       // x-height / cap-height ratio
        var widthRatio: Float         // Average glyph width / height
        var slant: Float              // Italicization angle in degrees

        // Classified properties
        var serifStyle: SerifStyle
        var roundness: Float          // Geometric vs organic curves (0-1)
        var regularity: Float         // Consistency of strokes (0-1)

        // Learned embedding (for ML models)
        var embedding: [Float]

        static let `default` = FontStyle(
            strokeWeight: 0.5,
            strokeContrast: 0.3,
            xHeightRatio: 0.7,
            widthRatio: 0.8,
            slant: 0,
            serifStyle: .sansSerif,
            roundness: 0.5,
            regularity: 0.8,
            embedding: []
        )
    }

    enum SerifStyle: String, Codable, CaseIterable {
        case sansSerif = "Sans Serif"
        case oldStyle = "Old Style"
        case transitional = "Transitional"
        case modern = "Modern"
        case slab = "Slab Serif"
        case script = "Script"
        case decorative = "Decorative"
    }

    // MARK: - Private Properties

    // Representative characters for style analysis
    private let representativeChars: [Character] = [
        "n", "o", "H", "O", "a", "g", "e", "p", "d", "b"
    ]

    /// Cache for computed embeddings (key: hash of image data, value: embedding)
    private var embeddingCache: [Int: [Float]] = [:]

    /// Lock for thread-safe cache access
    private let cacheLock = NSLock()

    // MARK: - Public API

    /// Extract style from a font project
    func extractStyle(from project: FontProject) async throws -> FontStyle {
        // Analyze geometric properties
        let strokeWeight = analyzeStrokeWeight(project)
        let strokeContrast = analyzeStrokeContrast(project)
        let xHeightRatio = project.metrics.capHeight > 0
            ? Float(project.metrics.xHeight) / Float(project.metrics.capHeight)
            : 0.5  // Default ratio when capHeight is unset
        let widthRatio = analyzeWidthRatio(project)
        let slant = analyzeSlant(project)
        let serifStyle = classifySerifStyle(project)
        let roundness = analyzeRoundness(project)
        let regularity = analyzeRegularity(project)

        // Get ML embedding if model is available
        var embedding: [Float] = []
        let hasModel = await MainActor.run { ModelManager.shared.styleEncoder != nil }
        if hasModel {
            embedding = try await encodeWithModel(project)
        }

        return FontStyle(
            strokeWeight: strokeWeight,
            strokeContrast: strokeContrast,
            xHeightRatio: xHeightRatio,
            widthRatio: widthRatio,
            slant: slant,
            serifStyle: serifStyle,
            roundness: roundness,
            regularity: regularity,
            embedding: embedding
        )
    }

    /// Encode a single glyph to a style embedding
    internal func encodeGlyph(_ glyph: Glyph) async throws -> [Float] {
        let hasModel = await MainActor.run { ModelManager.shared.styleEncoder != nil }
        guard hasModel else {
            throw StyleEncoderError.modelNotLoaded
        }

        // Render glyph to image; fall back to geometric analysis if rendering fails
        guard let glyphImage = renderGlyphForEncoding(glyph) else {
            Self.logger.info("Failed to render glyph for encoding, falling back to geometric analysis")
            return analyzeGlyphGeometry(glyph)
        }

        // Run through model; fall back to geometric analysis if encoding fails
        do {
            return try await encodeImage(glyphImage)
        } catch {
            Self.logger.warning("Glyph encoding failed: \(error.localizedDescription), falling back to geometric analysis")
            return analyzeGlyphGeometry(glyph)
        }
    }

    /// Geometric analysis fallback for glyph encoding when image-based encoding fails
    private func analyzeGlyphGeometry(_ glyph: Glyph) -> [Float] {
        var embedding = [Float](repeating: 0, count: EncodingParams.embeddingDimension)

        let bounds = glyph.outline.boundingBox
        let pointCount = glyph.outline.contours.reduce(0) { $0 + $1.points.count }
        let contourCount = glyph.outline.contours.count

        // Encode basic geometric features
        embedding[0] = bounds.height > 0 ? Float(bounds.width) / Float(bounds.height) : 0.5
        embedding[1] = Float(pointCount) / 100.0
        embedding[2] = Float(contourCount) / 10.0
        embedding[3] = estimateStrokeWeight(glyph.outline)
        embedding[4] = Float(bounds.width) / 1000.0
        embedding[5] = Float(bounds.height) / 1000.0

        // Analyze curve vs corner ratio
        var curves = 0
        var corners = 0
        for contour in glyph.outline.contours {
            for point in contour.points {
                switch point.type {
                case .corner: corners += 1
                case .smooth, .symmetric: curves += 1
                }
            }
        }
        let total = curves + corners
        embedding[6] = total > 0 ? Float(curves) / Float(total) : 0.5

        return embedding
    }

    /// Compute similarity between two styles (0-1, 1 = identical)
    func similarity(_ styleA: FontStyle, _ styleB: FontStyle) -> Float {
        var totalSimilarity: Float = 0
        var weights: Float = 0

        // Geometric properties
        totalSimilarity += (1 - abs(styleA.strokeWeight - styleB.strokeWeight)) * 2
        weights += 2
        totalSimilarity += (1 - abs(styleA.strokeContrast - styleB.strokeContrast))
        weights += 1
        totalSimilarity += (1 - abs(styleA.xHeightRatio - styleB.xHeightRatio))
        weights += 1
        totalSimilarity += (1 - abs(styleA.widthRatio - styleB.widthRatio))
        weights += 1
        totalSimilarity += (1 - abs(styleA.slant - styleB.slant) / 45)  // Normalize by max reasonable slant
        weights += 1
        totalSimilarity += (1 - abs(styleA.roundness - styleB.roundness))
        weights += 1

        // Serif style (binary match)
        totalSimilarity += (styleA.serifStyle == styleB.serifStyle) ? 2 : 0
        weights += 2

        // Embedding similarity (if both have embeddings)
        if !styleA.embedding.isEmpty && !styleB.embedding.isEmpty &&
           styleA.embedding.count == styleB.embedding.count {
            let cosineSim = cosineSimilarity(styleA.embedding, styleB.embedding)
            totalSimilarity += cosineSim * 3  // Weight embeddings heavily
            weights += 3
        }

        let result = totalSimilarity / weights
        return max(0, min(1, result))
    }

    /// Interpolate between two styles
    func interpolate(_ styleA: FontStyle, _ styleB: FontStyle, t: Float) -> FontStyle {
        let clampedT = max(0, min(1, t))

        return FontStyle(
            strokeWeight: lerp(styleA.strokeWeight, styleB.strokeWeight, clampedT),
            strokeContrast: lerp(styleA.strokeContrast, styleB.strokeContrast, clampedT),
            xHeightRatio: lerp(styleA.xHeightRatio, styleB.xHeightRatio, clampedT),
            widthRatio: lerp(styleA.widthRatio, styleB.widthRatio, clampedT),
            slant: lerp(styleA.slant, styleB.slant, clampedT),
            serifStyle: clampedT < 0.5 ? styleA.serifStyle : styleB.serifStyle,
            roundness: lerp(styleA.roundness, styleB.roundness, clampedT),
            regularity: lerp(styleA.regularity, styleB.regularity, clampedT),
            embedding: interpolateEmbedding(styleA.embedding, styleB.embedding, clampedT)
        )
    }

    // MARK: - Geometric Analysis

    private func analyzeStrokeWeight(_ project: FontProject) -> Float {
        var totalWeight: Float = 0
        var count: Float = 0

        for char in representativeChars {
            guard let glyph = project.glyphs[char] else { continue }
            let weight = estimateStrokeWeight(glyph.outline)
            totalWeight += weight
            count += 1
        }

        return count > 0 ? totalWeight / count : 0.5
    }

    private func analyzeStrokeContrast(_ project: FontProject) -> Float {
        var weights: [Float] = []

        for char in representativeChars {
            guard let glyph = project.glyphs[char] else { continue }
            weights.append(estimateStrokeWeight(glyph.outline))
        }

        guard weights.count >= 2 else { return 0 }

        let minWeight = weights.min() ?? 0
        let maxWeight = weights.max() ?? 1

        return maxWeight > minWeight ? (maxWeight - minWeight) / maxWeight : 0
    }

    private func analyzeWidthRatio(_ project: FontProject) -> Float {
        var totalRatio: Float = 0
        var count: Float = 0

        for char in representativeChars {
            guard let glyph = project.glyphs[char] else { continue }
            let bounds = glyph.outline.boundingBox
            if bounds.height > 0 {
                totalRatio += Float(bounds.width) / Float(bounds.height)
                count += 1
            }
        }

        return count > 0 ? min(2, totalRatio / count) / 2 : 0.5  // Normalize to 0-1
    }

    private func analyzeSlant(_ project: FontProject) -> Float {
        // Analyze average slant angle from vertical strokes
        // For now, return 0 (upright)
        // A full implementation would analyze vertical stroke angles
        return 0
    }

    private func classifySerifStyle(_ project: FontProject) -> SerifStyle {
        // Analyze serif presence and style
        // For now, do simple heuristic based on terminal shapes

        guard let glyph = project.glyphs["n"] ?? project.glyphs["H"] else {
            return .sansSerif
        }

        // Check for serif-like features (endpoints with perpendicular extensions)
        // This is a simplified heuristic
        let hasSerifs = detectSerifs(in: glyph.outline)

        if hasSerifs {
            // Further classify serif type based on shape
            return .transitional  // Default serif type
        }

        return .sansSerif
    }

    private func analyzeRoundness(_ project: FontProject) -> Float {
        // Analyze ratio of curves to corners
        var totalCurves = 0
        var totalCorners = 0

        for char in representativeChars {
            guard let glyph = project.glyphs[char] else { continue }
            for contour in glyph.outline.contours {
                for point in contour.points {
                    switch point.type {
                    case .corner:
                        totalCorners += 1
                    case .smooth, .symmetric:
                        totalCurves += 1
                    }
                }
            }
        }

        let total = totalCurves + totalCorners
        return total > 0 ? Float(totalCurves) / Float(total) : 0.5
    }

    private func analyzeRegularity(_ project: FontProject) -> Float {
        // Analyze consistency of glyph proportions
        // Higher regularity = more consistent width/height ratios
        var ratios: [Float] = []

        for char in representativeChars {
            guard let glyph = project.glyphs[char] else { continue }
            let bounds = glyph.outline.boundingBox
            if bounds.height > 0 {
                ratios.append(Float(bounds.width) / Float(bounds.height))
            }
        }

        guard ratios.count >= 2 else { return 0.5 }

        // Calculate standard deviation
        let mean = ratios.reduce(0, +) / Float(ratios.count)
        let variance = ratios.map { pow($0 - mean, 2) }.reduce(0, +) / Float(ratios.count)
        let stdDev = sqrt(variance)

        // Convert to regularity score (lower stdDev = higher regularity)
        return max(0, min(1, 1 - stdDev))
    }

    // MARK: - Helper Methods

    private func estimateStrokeWeight(_ outline: GlyphOutline) -> Float {
        // Estimate stroke weight from outline area / perimeter ratio
        // This is a rough heuristic
        let bounds = outline.boundingBox
        guard bounds.width > 0 && bounds.height > 0 else { return 0.5 }

        // Larger bounding box relative to point count suggests thicker strokes
        let pointCount = outline.contours.reduce(0) { $0 + $1.points.count }
        guard pointCount > 0 else { return 0.5 }

        let area = Float(bounds.width * bounds.height)
        let density = Float(pointCount) / area * 1000  // Normalize

        return max(0, min(1, 1 - density / 10))  // Invert: more points = finer detail = thinner
    }

    private func detectSerifs(in outline: GlyphOutline) -> Bool {
        // Simplified serif detection
        // Look for terminal points with handles roughly perpendicular to stroke direction

        for contour in outline.contours where contour.isClosed {
            let points = contour.points
            guard points.count >= 3 else { continue }

            for i in 0..<points.count {
                let point = points[i]
                guard point.type == .corner else { continue }

                // Check for perpendicular handles (serif indicator)
                if let controlIn = point.controlIn,
                   let controlOut = point.controlOut {
                    let inAngle = atan2(point.position.y - controlIn.y, point.position.x - controlIn.x)
                    let outAngle = atan2(controlOut.y - point.position.y, controlOut.x - point.position.x)
                    let angleDiff = abs(inAngle - outAngle)

                    // Near perpendicular suggests serif
                    if angleDiff > .pi * 0.4 && angleDiff < .pi * 0.6 {
                        return true
                    }
                }
            }
        }

        return false
    }

    private func renderGlyphForEncoding(_ glyph: Glyph) -> CGImage? {
        // Render glyph to a small image for model input
        let size = 64

        guard let context = CGContext(
            data: nil,
            width: size,
            height: size,
            bitsPerComponent: 8,
            bytesPerRow: size * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            return nil
        }

        // White background
        context.setFillColor(CGColor.white)
        context.fill(CGRect(x: 0, y: 0, width: size, height: size))

        // Draw glyph outline
        let bounds = glyph.outline.boundingBox
        guard bounds.width > 0 && bounds.height > 0 else {
            return context.makeImage()
        }

        let scaleX = CGFloat(size - 8) / CGFloat(bounds.width)
        let scaleY = CGFloat(size - 8) / CGFloat(bounds.height)
        let glyphScale = min(scaleX, scaleY)

        context.translateBy(x: 4, y: 4)
        context.scaleBy(x: glyphScale, y: glyphScale)
        context.translateBy(x: CGFloat(-bounds.minX), y: CGFloat(-bounds.minY))

        // Draw filled path
        context.setFillColor(CGColor.black)
        let path = glyph.outline.cgPath
        context.addPath(path)
        context.fillPath()

        return context.makeImage()
    }

    private func encodeWithModel(_ project: FontProject) async throws -> [Float] {
        // Encode representative glyphs and average embeddings
        var embeddings: [[Float]] = []
        var encodingFailures: [(Character, Error)] = []

        for char in representativeChars {
            guard let glyph = project.glyphs[char] else { continue }
            guard let image = renderGlyphForEncoding(glyph) else { continue }

            do {
                let embedding = try await encodeImage(image)
                embeddings.append(embedding)
            } catch {
                // Log failures instead of silently swallowing
                encodingFailures.append((char, error))
            }
        }

        // Report any encoding failures for debugging
        if !encodingFailures.isEmpty {
            Self.logger.warning("Failed to encode \(encodingFailures.count) glyphs")
            for (char, error) in encodingFailures {
                Self.logger.warning("  '\(String(char))': \(error.localizedDescription)")
            }
        }

        guard !embeddings.isEmpty else {
            return []
        }

        // Average all embeddings
        return averageEmbeddings(embeddings)
    }

    /// Encode image to style embedding using CoreML model
    /// Falls back to statistical embedding if model unavailable
    private func encodeImage(_ image: CGImage) async throws -> [Float] {
        // Check cache first
        let cacheKey = computeImageHash(image)
        if let cached = getCachedEmbedding(for: cacheKey) {
            return cached
        }

        // Get model from ModelManager
        let model = await MainActor.run { ModelManager.shared.styleEncoder }

        let embedding: [Float]
        if let styleModel = model {
            // Use CoreML model for encoding
            Self.logger.debug("Using CoreML StyleEncoder model for embedding")
            embedding = try await encodeWithCoreML(image: image, model: styleModel)
        } else {
            // Fall back to statistical embedding
            Self.logger.info("Using statistical fallback for style encoding - AI model not loaded")
            embedding = computeStatisticalEmbedding(image)
        }

        // Guard against degenerate (zero-vector) embeddings that would cause NaN in cosine similarity
        let safedEmbedding = ensureNonZeroEmbedding(embedding)

        // Cache the result
        cacheEmbedding(safedEmbedding, for: cacheKey)

        return safedEmbedding
    }

    /// Ensure embedding is not a zero vector, which would produce NaN in cosine similarity.
    /// Returns a small default embedding if the magnitude is near zero.
    private func ensureNonZeroEmbedding(_ embedding: [Float]) -> [Float] {
        let magnitude = sqrt(embedding.reduce(0) { $0 + $1 * $1 })
        if magnitude < 1e-8 {
            // Return a small uniform embedding instead of zeros
            return [Float](repeating: 1e-4, count: embedding.count)
        }
        return embedding
    }

    /// Encode image using CoreML StyleEncoder model
    private func encodeWithCoreML(image: CGImage, model: MLModel) async throws -> [Float] {
        // Convert CGImage to CVPixelBuffer
        let pixelBuffer = try createPixelBuffer(from: image, size: EncodingParams.inputImageSize)

        // Create model input using MLDictionaryFeatureProvider (model-generated types unavailable until models compiled)
        let inputFeatures = try MLDictionaryFeatureProvider(dictionary: [
            "image": MLFeatureValue(pixelBuffer: pixelBuffer)
        ])

        // Run inference - model.prediction is async in Swift 6
        let prediction = try await model.prediction(from: inputFeatures)

        // Extract embedding from output
        guard let embeddingFeature = prediction.featureValue(for: "embedding"),
              let embeddingArray = embeddingFeature.multiArrayValue else {
            throw StyleEncoderError.encodingFailed("Model did not produce valid embedding output")
        }

        // Convert MLMultiArray to [Float]
        var embedding = [Float](repeating: 0, count: EncodingParams.embeddingDimension)
        for i in 0..<min(embeddingArray.count, EncodingParams.embeddingDimension) {
            embedding[i] = embeddingArray[i].floatValue
        }

        return embedding
    }

    /// Create CVPixelBuffer from CGImage for model input
    private func createPixelBuffer(from image: CGImage, size: Int) throws -> CVPixelBuffer {
        // Create pixel buffer with correct format
        var pixelBuffer: CVPixelBuffer?
        let attrs: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true
        ]

        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            size,
            size,
            kCVPixelFormatType_32BGRA,
            attrs as CFDictionary,
            &pixelBuffer
        )

        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            throw StyleEncoderError.invalidInput
        }

        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }

        guard let context = CGContext(
            data: CVPixelBufferGetBaseAddress(buffer),
            width: size,
            height: size,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
        ) else {
            throw StyleEncoderError.invalidInput
        }

        // Draw image scaled to target size
        context.interpolationQuality = .high
        context.draw(image, in: CGRect(x: 0, y: 0, width: size, height: size))

        return buffer
    }

    /// Compute statistical embedding as fallback (no ML model)
    /// Uses image statistics to create a deterministic embedding
    private func computeStatisticalEmbedding(_ image: CGImage) -> [Float] {
        var embedding = [Float](repeating: 0, count: EncodingParams.embeddingDimension)

        // Get pixel data
        guard let dataProvider = image.dataProvider,
              let data = dataProvider.data as Data? else {
            return embedding
        }

        let pixels = [UInt8](data)
        let width = image.width
        let height = image.height
        let bytesPerPixel = image.bitsPerPixel / 8
        let bytesPerRow = image.bytesPerRow

        // Prevent out-of-bounds access on grayscale or other sub-3-channel images
        guard bytesPerPixel >= 3 else {
            return Array(repeating: 0.0, count: 128)
        }

        guard pixels.count >= bytesPerRow * height else {
            return embedding
        }

        // Compute statistical features
        var sum: Float = 0
        var sumSquares: Float = 0
        var pixelCount: Float = 0

        // Compute mean and variance (grayscale equivalent)
        for y in 0..<height {
            for x in 0..<width {
                let offset = y * bytesPerRow + x * bytesPerPixel
                // Convert to grayscale using luminance weights
                let r = Float(pixels[offset])
                let g = Float(pixels[offset + 1])
                let b = Float(pixels[offset + 2])
                let gray = 0.299 * r + 0.587 * g + 0.114 * b

                sum += gray
                sumSquares += gray * gray
                pixelCount += 1
            }
        }

        let mean = sum / max(pixelCount, 1)
        let variance = (sumSquares / max(pixelCount, 1)) - (mean * mean)
        let stdDev = sqrt(max(variance, 0))

        // Compute horizontal and vertical stroke density
        var horizontalDensity: [Float] = []
        var verticalDensity: [Float] = []

        // Sample rows for horizontal density
        for y in stride(from: 0, to: height, by: max(height / 16, 1)) {
            var rowSum: Float = 0
            for x in 0..<width {
                let offset = y * bytesPerRow + x * bytesPerPixel
                let gray = (Float(pixels[offset]) + Float(pixels[offset + 1]) + Float(pixels[offset + 2])) / 3.0
                rowSum += (255 - gray) / 255.0
            }
            horizontalDensity.append(rowSum / Float(width))
        }

        // Sample columns for vertical density
        for x in stride(from: 0, to: width, by: max(width / 16, 1)) {
            var colSum: Float = 0
            for y in 0..<height {
                let offset = y * bytesPerRow + x * bytesPerPixel
                let gray = (Float(pixels[offset]) + Float(pixels[offset + 1]) + Float(pixels[offset + 2])) / 3.0
                colSum += (255 - gray) / 255.0
            }
            verticalDensity.append(colSum / Float(height))
        }

        // Fill embedding vector with statistical features
        var idx = 0

        // Global statistics (0-3)
        embedding[idx] = mean / 255.0; idx += 1
        embedding[idx] = stdDev / 128.0; idx += 1
        embedding[idx] = variance / 10000.0; idx += 1
        embedding[idx] = Float(pixelCount) / Float(width * height); idx += 1

        // Horizontal density profile (4-19)
        for i in 0..<min(16, horizontalDensity.count) {
            embedding[idx] = horizontalDensity[i]; idx += 1
        }
        idx = 20

        // Vertical density profile (20-35)
        for i in 0..<min(16, verticalDensity.count) {
            embedding[idx] = verticalDensity[i]; idx += 1
        }
        idx = 36

        // Fill remaining with derived features
        // Pixel intensity sampling at grid positions
        for i in 36..<128 {
            let sampleX = (i - 36) % width
            let sampleY = ((i - 36) / width) % height
            let offset = sampleY * bytesPerRow + sampleX * bytesPerPixel
            if offset + 2 < pixels.count {
                embedding[i] = Float(pixels[offset]) / 255.0
            }
        }

        return embedding
    }

    /// Compute hash of image for caching
    private func computeImageHash(_ image: CGImage) -> Int {
        var hasher = Hasher()
        hasher.combine(image.width)
        hasher.combine(image.height)
        hasher.combine(image.bitsPerPixel)

        // Sample pixels spread across the image for a more representative hash
        if let dataProvider = image.dataProvider,
           let data = dataProvider.data as Data? {
            let bytes = [UInt8](data)
            let stride = max(bytes.count / 256, 1)  // ~256 samples across image
            for i in Swift.stride(from: 0, to: bytes.count, by: stride) {
                hasher.combine(bytes[i])
            }
        }

        return hasher.finalize()
    }

    /// Get cached embedding if available
    private func getCachedEmbedding(for key: Int) -> [Float]? {
        cacheLock.lock()
        defer { cacheLock.unlock() }
        return embeddingCache[key]
    }

    /// Cache embedding with size limit enforcement
    private func cacheEmbedding(_ embedding: [Float], for key: Int) {
        cacheLock.lock()
        defer { cacheLock.unlock() }

        // Enforce cache size limit
        if embeddingCache.count >= EncodingParams.maxCacheSize {
            // Remove arbitrary entries (Dictionary has no ordering; this is not LRU eviction)
            let keysToRemove = Array(embeddingCache.keys.prefix(EncodingParams.maxCacheSize / 2))
            for k in keysToRemove {
                embeddingCache.removeValue(forKey: k)
            }
        }

        embeddingCache[key] = embedding
    }

    /// Clear the embedding cache
    func clearCache() {
        cacheLock.lock()
        defer { cacheLock.unlock() }
        embeddingCache.removeAll()
    }

    private func averageEmbeddings(_ embeddings: [[Float]]) -> [Float] {
        guard let first = embeddings.first else { return [] }

        var result = [Float](repeating: 0, count: first.count)

        for embedding in embeddings {
            for i in 0..<min(result.count, embedding.count) {
                result[i] += embedding[i]
            }
        }

        let count = Float(embeddings.count)
        let averaged = result.map { $0 / count }
        return ensureNonZeroEmbedding(averaged)
    }

    private func interpolateEmbedding(_ a: [Float], _ b: [Float], _ t: Float) -> [Float] {
        guard a.count == b.count else {
            return a.isEmpty ? b : a
        }

        return zip(a, b).map { lerp($0, $1, t) }
    }

    private func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count else { return 0 }

        var dot: Float = 0
        var normA: Float = 0
        var normB: Float = 0

        for i in 0..<a.count {
            dot += a[i] * b[i]
            normA += a[i] * a[i]
            normB += b[i] * b[i]
        }

        let denom = sqrt(normA) * sqrt(normB)
        return denom > 0 ? dot / denom : 0
    }

    private func lerp(_ a: Float, _ b: Float, _ t: Float) -> Float {
        a + (b - a) * t
    }
}
