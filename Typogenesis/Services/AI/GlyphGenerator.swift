import Foundation
import CoreGraphics
@preconcurrency import CoreML
import CoreImage
import AppKit
import Accelerate
import os.log

/// Service for generating glyphs using AI diffusion or template-based fallback.
///
/// When a trained GlyphDiffusion CoreML model is available, this service runs a
/// flow-matching diffusion loop (50 Euler steps) to generate glyph images, then
/// converts them to vector outlines via threshold → contour trace → bezier fit.
///
/// When no model is loaded, it falls back to parametric stroke templates that
/// produce recognizable letterforms with honest confidence=0.0.
///
/// ## Confidence Values
/// - 0.0: Template/algorithmic generation (fallback)
/// - 0.9: Real AI model generation
///
/// ## Thread Safety
/// Actor isolation provides automatic thread safety for all mutable state.
/// `styleEncoder`, `templateLibrary`, and `strokeBuilder` are immutable after init.
actor GlyphGenerator {

    enum GeneratorError: Error, LocalizedError {
        case modelNotLoaded
        case generationFailed(String)
        case invalidStyle
        case cancelled
        case shapeMismatch(got: Int, expected: Int, shape: [NSNumber])
        case invalidDataType(expected: String, got: String)
        case outputSizeMismatch(got: Int, expected: Int)

        var errorDescription: String? {
            switch self {
            case .modelNotLoaded:
                return "Glyph generation model is not loaded"
            case .generationFailed(let reason):
                return "Generation failed: \(reason)"
            case .invalidStyle:
                return "Invalid style configuration"
            case .cancelled:
                return "Generation was cancelled"
            case .shapeMismatch(let got, let expected, let shape):
                return "Array count \(got) doesn't match shape \(shape) (expected \(expected))"
            case .invalidDataType(let expected, let got):
                return "Expected \(expected) MLMultiArray, got \(got)"
            case .outputSizeMismatch(let got, let expected):
                return "UNet output size \(got) doesn't match expected \(expected)"
            }
        }
    }

    /// Generation mode
    enum GenerationMode {
        /// Generate from scratch with a given style
        case fromScratch(style: StyleEncoder.FontStyle)

        /// Complete a partial glyph outline
        case completePartial(partial: GlyphOutline, style: StyleEncoder.FontStyle)

        /// Create a variation of an existing glyph
        case variation(base: Glyph, strength: Float)

        /// Interpolate between two glyphs
        case interpolate(glyphA: Glyph, glyphB: Glyph, t: Float)
    }

    /// Settings for glyph generation
    struct GenerationSettings {
        var steps: Int = 50               // Diffusion steps
        var guidanceScale: Float = 7.5    // Classifier-free guidance scale
        var seed: UInt64? = nil           // Random seed (nil for random)
        var temperature: Float = 1.0      // Sampling temperature

        static let `default` = GenerationSettings()

        /// Faster generation with fewer steps
        static let fast = GenerationSettings(steps: 20)

        /// Higher quality with more steps
        static let quality = GenerationSettings(steps: 100)
    }

    /// Result of glyph generation
    struct GenerationResult {
        let glyph: Glyph
        let confidence: Float
        let generationTime: TimeInterval
    }

    // MARK: - Constants

    /// Typographic spacing ratios (as percentage of unitsPerEm)
    private enum SpacingRatio {
        /// Left sidebearing - space before the glyph (10% of em)
        static let leftSideBearing: CGFloat = 0.1
        /// Right sidebearing - space after the glyph (20% of em)
        static let rightSideBearing: CGFloat = 0.2
        /// Minimum advance width as fraction of em (50%)
        static let minAdvanceWidth: CGFloat = 0.5
    }

    /// Confidence scores for generation quality assessment
    private enum ConfidenceScore {
        /// High confidence when using real trained AI model
        static let aiModel: Float = 0.9
        /// Zero confidence for template/placeholder generation (honest)
        static let placeholder: Float = 0.0
    }

    /// Geometric placeholder parameters
    private enum PlaceholderParams {
        /// Height for punctuation/symbols as fraction of em
        static let punctuationHeight: CGFloat = 0.5
        /// Margin inside placeholder rectangle as fraction of em
        static let margin: CGFloat = 0.05
        /// Stroke width for hollow rectangle as fraction of em
        static let strokeWidth: CGFloat = 0.04
        /// Variation scaling factor for glyph variations
        static let variationScale: Double = 0.1
    }

    // MARK: - Private Properties

    private let styleEncoder = StyleEncoder()
    private let templateLibrary = GlyphTemplateLibrary.shared
    private let strokeBuilder = StrokeBuilder()

    /// Logger for tracking generation method and fallback usage
    private static let logger = Logger(subsystem: "com.typogenesis", category: "GlyphGenerator")

    /// Track statistics about generation method usage (actor isolation provides thread safety)
    private var _generationStats = GenerationStats()

    /// Read access to generation stats (protected by actor isolation)
    var generationStats: GenerationStats {
        _generationStats
    }

    /// Record stat updates (protected by actor isolation)
    private func recordFallback() {
        _generationStats.fallbackGenerations += 1
    }

    private func recordAI() {
        _generationStats.aiGenerations += 1
    }

    private func recordTemplate() {
        _generationStats.templateGenerations += 1
    }

    private func recordFailed() {
        _generationStats.failedGenerations += 1
    }

    /// Statistics about generation method usage
    struct GenerationStats {
        var aiGenerations: Int = 0
        var templateGenerations: Int = 0
        var fallbackGenerations: Int = 0
        var failedGenerations: Int = 0

        var totalGenerations: Int {
            aiGenerations + templateGenerations + fallbackGenerations
        }

        var aiUsagePercentage: Double {
            guard totalGenerations > 0 else { return 0 }
            return Double(aiGenerations) / Double(totalGenerations) * 100
        }

        mutating func reset() {
            aiGenerations = 0
            templateGenerations = 0
            fallbackGenerations = 0
            failedGenerations = 0
        }
    }

    // MARK: - Diffusion Model Constants

    /// Parameters for the UNet diffusion model
    private enum DiffusionParams {
        /// Image size for UNet input/output (must match Python training)
        static let imageSize: Int = 64
        /// Style embedding dimension (must match StyleEncoder output)
        static let styleEmbedDim: Int = 128
        /// Number of supported characters (a-z + A-Z + 0-9)
        static let numCharacters: Int = 62
        /// Null class index for classifier-free guidance (one past valid classes)
        static let nullClassIndex: Int32 = 62
        /// Threshold for binarizing model output (0-255 scale)
        static let binaryThreshold: UInt8 = 128
        /// Minimum contour length in pixels
        static let minContourLength: Int = 10
    }

    /// Character-to-index mapping matching Python's `config.py` ordering:
    /// `abcdefghijklmnopqrstuvwxyz` + `ABCDEFGHIJKLMNOPQRSTUVWXYZ` + `0123456789` → 0-61
    nonisolated static let charToIndex: [Character: Int32] = {
        let chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        var map = [Character: Int32]()
        for (i, c) in chars.enumerated() {
            map[c] = Int32(i)
        }
        return map
    }()

    // MARK: - Public API

    /// Check if model is available (must be called from MainActor)
    @MainActor
    static func isModelAvailable() -> Bool {
        ModelManager.shared.glyphDiffusion != nil
    }

    /// Generate a single glyph
    func generate(
        character: Character,
        mode: GenerationMode,
        metrics: FontMetrics,
        settings: GenerationSettings = .default
    ) async throws -> GenerationResult {
        // Check model availability on MainActor
        let hasModel = await MainActor.run { Self.isModelAvailable() }
        guard hasModel else {
            // Log fallback usage
            Self.logger.info("Using template fallback for '\(String(character))' - AI model not loaded")
            // Note: recordFallback()/recordTemplate() is called inside generatePlaceholder -> generateFromTemplate,
            // so we do NOT call recordFallback() here to avoid double-counting.

            // Return placeholder when model not available
            return try await generatePlaceholder(
                character: character,
                mode: mode,
                metrics: metrics
            )
        }

        // Extract style from mode and delegate to generateWithModel
        let style: StyleEncoder.FontStyle
        switch mode {
        case .fromScratch(let s):
            style = s
        case .completePartial(_, let s):
            // TODO: Implement inpainting using rasterizeOutlineToMask for partial completion conditioning.
            // Currently falls through to from-scratch generation with the same style.
            Self.logger.warning("completePartial mode not yet implemented for AI generation; falling through to from-scratch generation")
            style = s
        case .variation(_, _):
            // TODO: Implement variation generation that uses the base glyph to condition the diffusion model.
            Self.logger.warning("variation mode not yet implemented for AI generation; falling through to from-scratch generation")
            style = StyleEncoder.FontStyle.default
        case .interpolate(_, _, _):
            // TODO: Implement interpolation that blends two glyph embeddings in the diffusion model.
            Self.logger.warning("interpolate mode not yet implemented for AI generation; falling through to from-scratch generation")
            style = StyleEncoder.FontStyle.default
        }

        return try await generateWithModel(
            character: character,
            style: style,
            metrics: metrics,
            settings: settings
        )
    }

    /// Generate multiple glyphs for a character set
    func generateBatch(
        characters: [Character],
        mode: GenerationMode,
        metrics: FontMetrics,
        settings: GenerationSettings = .default,
        onProgress: (@Sendable (Int, Int) -> Void)? = nil
    ) async throws -> [GenerationResult] {
        var results: [GenerationResult] = []

        for (index, char) in characters.enumerated() {
            try Task.checkCancellation()
            let result = try await generate(
                character: char,
                mode: mode,
                metrics: metrics,
                settings: settings
            )
            results.append(result)

            onProgress?(index + 1, characters.count)
        }

        return results
    }

    /// Check if generation is available.
    /// Returns true because template-based generation always works as fallback.
    /// For real AI model availability, use `GlyphGenerator.isModelAvailable()`.
    nonisolated var isAvailable: Bool {
        true  // Template fallback is always available
    }

    // MARK: - Model-Based Generation

    /// Generate a glyph using the CoreML diffusion model.
    /// Falls back to template generation if model is unavailable or fails.
    internal func generateWithModel(
        character: Character,
        style: StyleEncoder.FontStyle,
        metrics: FontMetrics,
        settings: GenerationSettings = .default
    ) async throws -> GenerationResult {
        let startTime = Date()

        // Check model availability on MainActor
        let model = await MainActor.run { ModelManager.shared.glyphDiffusion }
        guard let diffusionModel = model else {
            // Fall back to template generation
            return try await generatePlaceholder(
                character: character,
                mode: .fromScratch(style: style),
                metrics: metrics
            )
        }

        do {
            // Get style embedding (use from FontStyle or default)
            let styleEmbedding = style.embedding.isEmpty ?
                [Float](repeating: 0, count: DiffusionParams.styleEmbedDim) : style.embedding

            // Run diffusion sampling loop
            let outputImage = try await runDiffusionSampling(
                model: diffusionModel,
                character: character,
                styleEmbedding: styleEmbedding,
                settings: settings
            )

            recordAI()

            // Post-process: convert model output to glyph outline
            let outline = try postProcessModelOutput(
                output: outputImage,
                metrics: metrics
            )

            // Create glyph with proper metrics
            let bounds = outline.boundingBox
            let glyph = Glyph(
                character: character,
                outline: outline,
                advanceWidth: bounds.width + Int(CGFloat(metrics.unitsPerEm) * SpacingRatio.rightSideBearing),
                leftSideBearing: Int(CGFloat(metrics.unitsPerEm) * SpacingRatio.leftSideBearing),
                generatedBy: .aiGenerated,
                styleConfidence: ConfidenceScore.aiModel
            )

            let generationTime = Date().timeIntervalSince(startTime)

            return GenerationResult(
                glyph: glyph,
                confidence: ConfidenceScore.aiModel,
                generationTime: generationTime
            )
        } catch {
            // Model inference failed - fall back to template generation
            // WARNING: User will receive a geometric placeholder, NOT AI-generated output
            recordFailed()
            Self.logger.error("AI model inference failed: \(error.localizedDescription). Returning PLACEHOLDER glyph (not AI output). User will see geometric fallback.")
            return try await generatePlaceholder(
                character: character,
                mode: .fromScratch(style: style),
                metrics: metrics
            )
        }
    }

    // MARK: - Diffusion Sampling

    /// Run the full flow-matching diffusion sampling loop.
    ///
    /// This is the core AI generation path. It:
    /// 1. Starts with Gaussian noise
    /// 2. Iterates through Euler steps, calling the UNet at each timestep
    /// 3. Optionally applies classifier-free guidance
    /// 4. Converts the final sample to a grayscale CGImage
    ///
    /// - Parameters:
    ///   - model: The compiled GlyphDiffusion CoreML model.
    ///   - character: The character to generate.
    ///   - styleEmbedding: 128-dim style vector from StyleEncoder.
    ///   - settings: Generation settings (steps, guidance scale, seed).
    ///   - mask: Optional mask for partial completion (nil = zeros = from scratch).
    /// - Returns: A 64x64 grayscale CGImage of the generated glyph.
    private func runDiffusionSampling(
        model: MLModel,
        character: Character,
        styleEmbedding: [Float],
        settings: GenerationSettings,
        mask: [Float]? = nil
    ) async throws -> CGImage {
        let size = DiffusionParams.imageSize
        let pixelCount = size * size

        // Clamp guidance scale to minimum 1.0 (no guidance) to prevent negative/inverted guidance
        let guidanceScale = max(1.0, settings.guidanceScale)

        // Look up character index
        let charIndex = Self.charToIndex[character] ?? DiffusionParams.nullClassIndex

        // Create scheduler
        let scheduler = FlowMatchingScheduler(numSteps: settings.steps)

        // Seed the RNG
        var rng: any RandomNumberGenerator = if let seed = settings.seed {
            SeededRNG(seed: seed) as any RandomNumberGenerator
        } else {
            SystemRandomNumberGenerator() as any RandomNumberGenerator
        }

        // Generate initial Gaussian noise
        var sample = generateGaussianNoise(count: pixelCount, using: &rng)

        // Prepare fixed inputs
        let styleArray = try floatArrayToMLMultiArray(styleEmbedding, shape: [1, NSNumber(value: DiffusionParams.styleEmbedDim)])
        let maskArray = try floatArrayToMLMultiArray(
            mask ?? [Float](repeating: 0, count: pixelCount),
            shape: [1, 1, NSNumber(value: size), NSNumber(value: size)]
        )

        // Sampling loop
        for stepIndex in 0..<settings.steps {
            try Task.checkCancellation()

            let t = scheduler.timesteps[stepIndex]

            // Prepare UNet inputs for this step
            let velocity = try await callUNet(
                model: model,
                sample: sample,
                timestep: t,
                charIndex: charIndex,
                styleEmbed: styleArray,
                mask: maskArray
            )

            // Classifier-free guidance
            var finalVelocity = velocity
            if guidanceScale > 1.0 {
                let uncondVelocity = try await callUNet(
                    model: model,
                    sample: sample,
                    timestep: t,
                    charIndex: DiffusionParams.nullClassIndex,
                    styleEmbed: styleArray,
                    mask: maskArray
                )
                // guided = uncond + scale * (cond - uncond)
                for i in 0..<pixelCount {
                    finalVelocity[i] = uncondVelocity[i] + guidanceScale * (velocity[i] - uncondVelocity[i])
                }
            }

            // Euler step
            let stepResult = scheduler.step(velocity: finalVelocity, stepIndex: stepIndex, sample: sample)
            sample = stepResult.prevSample
        }

        // Convert final sample [-1, 1] → grayscale CGImage [0, 255]
        return try sampleToGrayscaleImage(sample, width: size, height: size)
    }

    /// Call the UNet CoreML model for a single forward pass.
    private func callUNet(
        model: MLModel,
        sample: [Float],
        timestep: Float,
        charIndex: Int32,
        styleEmbed: MLMultiArray,
        mask: MLMultiArray
    ) async throws -> [Float] {
        let size = DiffusionParams.imageSize

        let xArray = try floatArrayToMLMultiArray(sample, shape: [1, 1, NSNumber(value: size), NSNumber(value: size)])
        let tArray = try MLMultiArray(shape: [1], dataType: .float32)
        tArray[0] = NSNumber(value: timestep)
        let charArray = try MLMultiArray(shape: [1], dataType: .int32)
        charArray[0] = NSNumber(value: charIndex)

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "x": MLFeatureValue(multiArray: xArray),
            "timesteps": MLFeatureValue(multiArray: tArray),
            "char_indices": MLFeatureValue(multiArray: charArray),
            "style_embed": MLFeatureValue(multiArray: styleEmbed),
            "mask": MLFeatureValue(multiArray: mask),
        ])

        let prediction = try await model.prediction(from: input)

        // Extract velocity output
        guard let velocityFeature = prediction.featureValue(for: "velocity"),
              let velocityMLArray = velocityFeature.multiArrayValue else {
            throw GeneratorError.generationFailed("UNet did not produce velocity output")
        }

        let velocityArray = try mlMultiArrayToFloatArray(velocityMLArray)
        let pixelCount = size * size
        guard velocityArray.count == pixelCount else {
            throw GeneratorError.outputSizeMismatch(got: velocityArray.count, expected: pixelCount)
        }

        return velocityArray
    }

    // MARK: - Array / MLMultiArray Conversion Helpers

    /// Convert a flat Float array to an MLMultiArray with the given shape.
    private func floatArrayToMLMultiArray(_ array: [Float], shape: [NSNumber]) throws -> MLMultiArray {
        let expectedCount = shape.reduce(1) { $0 * $1.intValue }
        guard array.count == expectedCount else {
            throw GeneratorError.shapeMismatch(got: array.count, expected: expectedCount, shape: shape)
        }
        let mlArray = try MLMultiArray(shape: shape, dataType: .float32)
        mlArray.withUnsafeMutableBufferPointer(ofType: Float.self) { buffer, _ in
            for i in 0..<array.count {
                buffer[i] = array[i]
            }
        }
        return mlArray
    }

    /// Convert an MLMultiArray to a flat Float array.
    private func mlMultiArrayToFloatArray(_ mlArray: MLMultiArray) throws -> [Float] {
        guard mlArray.dataType == .float32 else {
            throw GeneratorError.invalidDataType(expected: "float32", got: "\(mlArray.dataType)")
        }
        return mlArray.withUnsafeBufferPointer(ofType: Float.self) { buffer in
            Array(buffer)
        }
    }

    // MARK: - Noise Generation

    /// Generate Gaussian noise using Box-Muller transform.
    private func generateGaussianNoise(count: Int, using rng: inout RandomNumberGenerator) -> [Float] {
        var noise = [Float](repeating: 0, count: count)
        // Box-Muller: generate pairs
        let pairCount = (count + 1) / 2
        for i in 0..<pairCount {
            let u1 = max(Float.random(in: 0..<1, using: &rng), Float.leastNormalMagnitude)
            let u2 = Float.random(in: 0..<1, using: &rng)
            let mag = (-2.0 * log(u1)).squareRoot()
            let angle = 2.0 * Float.pi * u2
            noise[i * 2] = mag * cos(angle)
            if i * 2 + 1 < count {
                noise[i * 2 + 1] = mag * sin(angle)
            }
        }
        return noise
    }

    // MARK: - Image Conversion

    /// Convert a diffusion sample in [-1, 1] range to a grayscale CGImage.
    /// Denormalization: pixel = clamp((x + 1) * 127.5, 0, 255)
    private func sampleToGrayscaleImage(_ sample: [Float], width: Int, height: Int) throws -> CGImage {
        guard sample.count == width * height else {
            throw GeneratorError.outputSizeMismatch(got: sample.count, expected: width * height)
        }

        var pixels = [UInt8](repeating: 0, count: width * height)
        for i in 0..<sample.count {
            let value = (sample[i] + 1.0) * 127.5
            pixels[i] = UInt8(min(max(value, 0), 255))
        }

        guard let provider = CGDataProvider(data: Data(pixels) as CFData),
              let image = CGImage(
                width: width,
                height: height,
                bitsPerComponent: 8,
                bitsPerPixel: 8,
                bytesPerRow: width,
                space: CGColorSpaceCreateDeviceGray(),
                bitmapInfo: CGBitmapInfo(rawValue: 0),
                provider: provider,
                decode: nil,
                shouldInterpolate: false,
                intent: .defaultIntent
              ) else {
            throw GeneratorError.generationFailed("Failed to create CGImage from diffusion output")
        }

        return image
    }

    // MARK: - Post-Processing Pipeline

    /// Convert model output image to GlyphOutline
    /// Pipeline: grayscale → threshold → contour trace → bezier fit
    private func postProcessModelOutput(
        output: CGImage,
        metrics: FontMetrics
    ) throws -> GlyphOutline {
        // Step 1: Convert to grayscale pixel data
        let pixelData = try ImageProcessor.getPixelData(output)

        // Step 2: Apply threshold to create binary image
        let binaryData = pixelData.toBinary(threshold: DiffusionParams.binaryThreshold)

        // Step 3: Trace contours using existing ContourTracer
        let tracingSettings = ContourTracer.TracingSettings(
            simplificationTolerance: 2.0,
            minContourLength: DiffusionParams.minContourLength,
            detectCorners: true,
            cornerAngleThreshold: 60,
            smoothCurves: true
        )

        let tracedContours = try ContourTracer.trace(
            binary: binaryData,
            width: pixelData.width,
            height: pixelData.height,
            settings: tracingSettings
        )

        guard !tracedContours.isEmpty else {
            throw GeneratorError.generationFailed("No contours detected in model output")
        }

        // Step 4: Fit bezier curves to traced contours
        let outline = fitBeziersToModelOutput(
            contours: tracedContours,
            imageSize: CGSize(width: pixelData.width, height: pixelData.height),
            metrics: metrics
        )

        return outline
    }

    /// Fit bezier curves to model output contours and scale to font metrics
    private func fitBeziersToModelOutput(
        contours: [ContourTracer.TracedContour],
        imageSize: CGSize,
        metrics: FontMetrics
    ) -> GlyphOutline {
        var glyphContours: [Contour] = []

        // Collect all points to find bounds for normalization
        var allPoints: [CGPoint] = []
        for contour in contours {
            allPoints.append(contentsOf: contour.points)
        }

        guard !allPoints.isEmpty else {
            return GlyphOutline()
        }

        // Find bounding box
        let minX = allPoints.map { $0.x }.min()!
        let maxX = allPoints.map { $0.x }.max()!
        let minY = allPoints.map { $0.y }.min()!
        let maxY = allPoints.map { $0.y }.max()!

        let width = maxX - minX
        let height = maxY - minY

        guard width > 0, height > 0 else {
            return GlyphOutline()
        }

        // Calculate scale to fit cap height
        let targetHeight = CGFloat(metrics.capHeight)
        let scale = targetHeight / height
        let margin = CGFloat(metrics.unitsPerEm) * 0.1

        for tracedContour in contours {
            // Transform points to glyph coordinates
            let transformedPoints = tracedContour.points.map { point in
                CGPoint(
                    x: (point.x - minX) * scale + margin,
                    y: (maxY - point.y) * scale // Flip Y
                )
            }

            // Fit bezier curves
            let bezierSettings = BezierFitter.FittingSettings(
                errorThreshold: 4.0,
                maxIterations: 4,
                cornerThreshold: 60
            )

            let segments = BezierFitter.fitCurves(
                to: transformedPoints,
                isClosed: tracedContour.isClosed,
                settings: bezierSettings
            )

            // Convert to PathPoints
            let pathPoints = BezierFitter.toPathPoints(segments: segments)

            if !pathPoints.isEmpty {
                glyphContours.append(Contour(points: pathPoints, isClosed: tracedContour.isClosed))
            }
        }

        return GlyphOutline(contours: glyphContours)
    }

    // MARK: - Private Methods

    /// Rasterize a GlyphOutline to a binary mask for partial completion conditioning.
    /// Returns a flat array of size*size floats where 1.0 = glyph, 0.0 = background.
    ///
    /// Uses Core Graphics to properly fill contour interiors rather than just marking
    /// individual on-curve points. The outline's coordinate system is scaled to fit the
    /// image size with Y flipped (glyph coords have Y-up, image has Y-down).
    private func rasterizeOutlineToMask(_ outline: GlyphOutline, size: Int) -> [Float] {
        let bounds = outline.boundingBox
        guard bounds.width > 0, bounds.height > 0 else {
            return [Float](repeating: 0, count: size * size)
        }

        // Create a grayscale 8-bit CGContext
        let bytesPerRow = size
        var pixelBuffer = [UInt8](repeating: 0, count: size * size)
        guard let context = CGContext(
            data: &pixelBuffer,
            width: size,
            height: size,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: CGColorSpaceCreateDeviceGray(),
            bitmapInfo: CGImageAlphaInfo.none.rawValue
        ) else {
            return [Float](repeating: 0, count: size * size)
        }

        // Calculate scale to fit the outline into the image, preserving aspect ratio
        let scaleX = CGFloat(size) / CGFloat(bounds.width)
        let scaleY = CGFloat(size) / CGFloat(bounds.height)
        let scale = min(scaleX, scaleY)

        // Build a CGPath from the outline contours
        let path = CGMutablePath()
        for contour in outline.contours {
            guard !contour.points.isEmpty else { continue }

            for (index, point) in contour.points.enumerated() {
                // Transform: translate to origin, scale, flip Y
                let x = (point.position.x - CGFloat(bounds.minX)) * scale
                let y = (CGFloat(bounds.maxY) - point.position.y) * scale

                if index == 0 {
                    path.move(to: CGPoint(x: x, y: y))
                } else {
                    path.addLine(to: CGPoint(x: x, y: y))
                }
            }

            if contour.isClosed {
                path.closeSubpath()
            }
        }

        // Fill the path with white (1.0)
        context.setFillColor(CGColor.white)
        context.addPath(path)
        context.fillPath()

        // Convert pixel buffer to [Float] (0.0 or 1.0)
        return pixelBuffer.map { $0 >= 128 ? Float(1.0) : Float(0.0) }
    }

    private func generatePlaceholder(
        character: Character,
        mode: GenerationMode,
        metrics: FontMetrics
    ) async throws -> GenerationResult {
        let startTime = Date()

        // Yield to allow UI updates, but no fake delay
        await Task.yield()

        // Extract style parameters from mode
        let style: StyleEncoder.FontStyle
        switch mode {
        case .fromScratch(let s):
            style = s
        case .completePartial(_, let s):
            style = s
        case .variation(_, _):
            // Extract style from base glyph metrics
            style = StyleEncoder.FontStyle.default
        case .interpolate(_, _, let t):
            // Interpolate styles (use default as fallback)
            style = styleEncoder.interpolate(.default, .default, t: t)
        }

        // Create outline based on mode
        let outline: GlyphOutline
        var advanceWidth: Int
        var leftSideBearing: Int
        var generationSource: GenerationSource = .placeholder  // Template-based placeholder, not real AI

        switch mode {
        case .variation(let base, let strength):
            // Return base with slight scaling
            outline = scaleOutline(base.outline, by: 1.0 + Double(strength) * PlaceholderParams.variationScale)
            advanceWidth = Int(Double(base.advanceWidth) * (1.0 + Double(strength) * PlaceholderParams.variationScale))
            leftSideBearing = base.leftSideBearing

        case .interpolate(let glyphA, let glyphB, let t):
            // Interpolate between the two outlines
            outline = interpolateOutlines(glyphA.outline, glyphB.outline, t: Double(t))
            advanceWidth = Int(Double(glyphA.advanceWidth) * Double(1 - t) + Double(glyphB.advanceWidth) * Double(t))
            leftSideBearing = Int(Double(glyphA.leftSideBearing) * Double(1 - t) + Double(glyphB.leftSideBearing) * Double(t))

        case .completePartial(let partial, _):
            // Return the partial as-is (user provided it)
            outline = partial
            let bounds = partial.boundingBox
            advanceWidth = bounds.width + Int(CGFloat(metrics.unitsPerEm) * SpacingRatio.rightSideBearing)
            leftSideBearing = Int(CGFloat(metrics.unitsPerEm) * SpacingRatio.leftSideBearing)
            generationSource = .manual  // User-provided

        case .fromScratch:
            // Generate using template system
            let result = generateFromTemplate(
                character: character,
                metrics: metrics,
                style: style
            )
            outline = result.outline
            advanceWidth = result.advanceWidth
            leftSideBearing = result.leftSideBearing
        }

        let glyph = Glyph(
            character: character,
            outline: outline,
            advanceWidth: advanceWidth,
            leftSideBearing: leftSideBearing,
            generatedBy: generationSource
        )

        // Confidence based on generation quality
        // Placeholder generation gets 0.0 (honest about lack of real AI)
        // Only real AI generation should have non-zero confidence
        let confidence: Float = generationSource == .aiGenerated ? ConfidenceScore.aiModel : ConfidenceScore.placeholder

        return GenerationResult(
            glyph: glyph,
            confidence: confidence,
            generationTime: Date().timeIntervalSince(startTime)
        )
    }

    /// Generate a glyph from the template library
    private func generateFromTemplate(
        character: Character,
        metrics: FontMetrics,
        style: StyleEncoder.FontStyle
    ) -> (outline: GlyphOutline, advanceWidth: Int, leftSideBearing: Int) {
        // Try to get a template for this character
        if let template = templateLibrary.template(for: character) {
            recordTemplate()
            let styleParams = StrokeBuilder.StyleParams(from: style)

            let outline = strokeBuilder.buildGlyph(
                from: template,
                metrics: metrics,
                style: styleParams
            )

            let advanceWidth = strokeBuilder.calculateAdvanceWidth(
                from: template,
                metrics: metrics,
                style: styleParams
            )

            let leftSideBearing = strokeBuilder.calculateLeftSideBearing(
                from: template,
                metrics: metrics,
                style: styleParams
            )

            return (outline, advanceWidth, leftSideBearing)
        }

        // Fallback to geometric placeholder for unsupported characters
        recordFallback()
        let outline = createGeometricPlaceholder(for: character, metrics: metrics)
        let bounds = outline.boundingBox
        let advanceWidth = max(bounds.width + Int(CGFloat(metrics.unitsPerEm) * SpacingRatio.rightSideBearing), Int(CGFloat(metrics.unitsPerEm) * SpacingRatio.minAdvanceWidth))
        let leftSideBearing = Int(CGFloat(metrics.unitsPerEm) * SpacingRatio.leftSideBearing)

        return (outline, advanceWidth, leftSideBearing)
    }

    /// Creates a fallback geometric placeholder for characters without templates.
    /// Used only when the template library doesn't have a definition for the character.
    /// This creates a simple rectangle that's clearly identifiable as a placeholder.
    private func createGeometricPlaceholder(for character: Character, metrics: FontMetrics) -> GlyphOutline {
        let em = CGFloat(metrics.unitsPerEm)
        let isUppercase = character.isUppercase
        let isDigit = character.isNumber

        // Determine height based on character type
        let height: CGFloat
        if isUppercase || isDigit {
            height = CGFloat(metrics.capHeight)
        } else if character.isLowercase {
            height = CGFloat(metrics.xHeight)
        } else {
            height = em * PlaceholderParams.punctuationHeight
        }

        // Base width varies by character to look slightly different
        let charValue = character.unicodeScalars.first?.value ?? 65
        let widthVariation = CGFloat((charValue % 20)) / 100.0  // 0-20% variation
        let width = em * (0.4 + widthVariation)

        // Create a rectangle outline (not filled) to indicate missing glyph
        // This is clearly identifiable as a placeholder
        let margin = em * PlaceholderParams.margin
        let strokeWidth = em * PlaceholderParams.strokeWidth

        // Outer rectangle
        let outerContour = Contour(
            points: [
                PathPoint(position: CGPoint(x: margin, y: 0), type: .corner),
                PathPoint(position: CGPoint(x: width - margin, y: 0), type: .corner),
                PathPoint(position: CGPoint(x: width - margin, y: height), type: .corner),
                PathPoint(position: CGPoint(x: margin, y: height), type: .corner)
            ],
            isClosed: true
        )

        // Inner rectangle (creates hollow effect)
        let innerContour = Contour(
            points: [
                PathPoint(position: CGPoint(x: margin + strokeWidth, y: strokeWidth), type: .corner),
                PathPoint(position: CGPoint(x: margin + strokeWidth, y: height - strokeWidth), type: .corner),
                PathPoint(position: CGPoint(x: width - margin - strokeWidth, y: height - strokeWidth), type: .corner),
                PathPoint(position: CGPoint(x: width - margin - strokeWidth, y: strokeWidth), type: .corner)
            ],
            isClosed: true
        )

        return GlyphOutline(contours: [outerContour, innerContour])
    }

    private func scaleOutline(_ outline: GlyphOutline, by scale: Double) -> GlyphOutline {
        guard !outline.isEmpty else { return outline }

        let bounds = outline.boundingBox
        let centerX = CGFloat(bounds.minX + bounds.width / 2)
        let centerY = CGFloat(bounds.minY + bounds.height / 2)

        var scaledContours: [Contour] = []
        for contour in outline.contours {
            var scaledPoints: [PathPoint] = []
            for point in contour.points {
                let newX = centerX + (point.position.x - centerX) * scale
                let newY = centerY + (point.position.y - centerY) * scale
                var newPoint = point
                newPoint.position = CGPoint(x: newX, y: newY)

                if let controlIn = point.controlIn {
                    let newCIX = centerX + (controlIn.x - centerX) * scale
                    let newCIY = centerY + (controlIn.y - centerY) * scale
                    newPoint.controlIn = CGPoint(x: newCIX, y: newCIY)
                }
                if let controlOut = point.controlOut {
                    let newCOX = centerX + (controlOut.x - centerX) * scale
                    let newCOY = centerY + (controlOut.y - centerY) * scale
                    newPoint.controlOut = CGPoint(x: newCOX, y: newCOY)
                }

                scaledPoints.append(newPoint)
            }
            scaledContours.append(Contour(points: scaledPoints, isClosed: contour.isClosed))
        }

        return GlyphOutline(contours: scaledContours)
    }

    private func interpolateOutlines(_ a: GlyphOutline, _ b: GlyphOutline, t: Double) -> GlyphOutline {
        // Simple interpolation: if structures match, interpolate points
        // Otherwise, crossfade based on t
        guard a.contours.count == b.contours.count else {
            return t < 0.5 ? a : b
        }

        var interpolatedContours: [Contour] = []
        for (contourA, contourB) in zip(a.contours, b.contours) {
            guard contourA.points.count == contourB.points.count else {
                interpolatedContours.append(t < 0.5 ? contourA : contourB)
                continue
            }

            var interpolatedPoints: [PathPoint] = []
            for (pointA, pointB) in zip(contourA.points, contourB.points) {
                let newX = pointA.position.x * (1 - t) + pointB.position.x * t
                let newY = pointA.position.y * (1 - t) + pointB.position.y * t
                var newPoint = PathPoint(
                    position: CGPoint(x: newX, y: newY),
                    type: t < 0.5 ? pointA.type : pointB.type
                )

                // Interpolate controlIn: if one side lacks a handle, use its on-curve position
                let ciA = pointA.controlIn ?? pointA.position
                let ciB = pointB.controlIn ?? pointB.position
                if pointA.controlIn != nil || pointB.controlIn != nil {
                    newPoint.controlIn = CGPoint(
                        x: ciA.x * (1 - t) + ciB.x * t,
                        y: ciA.y * (1 - t) + ciB.y * t
                    )
                }

                // Interpolate controlOut: if one side lacks a handle, use its on-curve position
                let coA = pointA.controlOut ?? pointA.position
                let coB = pointB.controlOut ?? pointB.position
                if pointA.controlOut != nil || pointB.controlOut != nil {
                    newPoint.controlOut = CGPoint(
                        x: coA.x * (1 - t) + coB.x * t,
                        y: coA.y * (1 - t) + coB.y * t
                    )
                }

                interpolatedPoints.append(newPoint)
            }
            interpolatedContours.append(Contour(
                points: interpolatedPoints,
                isClosed: contourA.isClosed || contourB.isClosed
            ))
        }

        return GlyphOutline(contours: interpolatedContours)
    }

}

// MARK: - Seeded Random Number Generator

/// Deterministic RNG for reproducible diffusion sampling.
/// Uses a simple xoshiro256** algorithm seeded from the user-provided seed.
private struct SeededRNG: RandomNumberGenerator {
    private var state: (UInt64, UInt64, UInt64, UInt64)

    init(seed: UInt64) {
        // SplitMix64 to expand seed into state
        var s = seed
        func next() -> UInt64 {
            s &+= 0x9e3779b97f4a7c15
            var z = s
            z = (z ^ (z >> 30)) &* 0xbf58476d1ce4e5b9
            z = (z ^ (z >> 27)) &* 0x94d049bb133111eb
            return z ^ (z >> 31)
        }
        state = (next(), next(), next(), next())
    }

    mutating func next() -> UInt64 {
        let result = rotl(state.1 &* 5, 7) &* 9
        let t = state.1 << 17
        state.2 ^= state.0
        state.3 ^= state.1
        state.1 ^= state.2
        state.0 ^= state.3
        state.2 ^= t
        state.3 = rotl(state.3, 45)
        return result
    }

    private func rotl(_ x: UInt64, _ k: Int) -> UInt64 {
        (x << k) | (x >> (64 - k))
    }
}
