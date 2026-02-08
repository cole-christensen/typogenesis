import Foundation
import CoreGraphics
@preconcurrency import CoreML
import CoreImage
import AppKit
import os.log

/// Service for generating glyphs using template-based algorithmic generation.
///
/// **HONEST STATUS:** This service uses parametric stroke templates to generate
/// recognizable letterforms. There is NO AI/ML involved - the code path for
/// diffusion-based generation exists but requires trained models that don't exist.
///
/// ## What Works Now (Template Generation)
/// - A-Z, a-z, 0-9, and common punctuation have stroke-based templates
/// - Style parameters (weight, contrast, roundness, slant) affect output
/// - Proper uppercase/lowercase height differentiation
/// - Serif style variations (slab, bracketed, hairline)
///
/// ## What Would Be Needed for Real AI
/// See `runDiffusion()` for detailed requirements. Summary:
/// - 10,000+ font training dataset
/// - Diffusion model architecture design
/// - 2-4 weeks GPU training time
/// - CoreML conversion and optimization
///
/// ## Confidence Values
/// - 0.0: Template/algorithmic generation (honest - this is what we do)
/// - 0.9: Would be used for real AI generation (unreachable currently)
///
/// ## Thread Safety
/// This class conforms to `@unchecked Sendable` because:
/// - `styleEncoder` and `strokeBuilder` are set at init and read-only thereafter (no mutation after init)
/// - `templateLibrary` is a shared singleton with its own thread safety
/// - `_generationStats` is protected by `statsLock` (NSLock)
final class GlyphGenerator: @unchecked Sendable {

    enum GeneratorError: Error, LocalizedError {
        case modelNotLoaded
        case generationFailed(String)
        case invalidStyle
        case cancelled

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

    /// Lock for thread-safe stats access
    private let statsLock = NSLock()

    /// Track statistics about generation method usage (access via recordStat/getStats for thread safety)
    private var _generationStats = GenerationStats()

    /// Thread-safe read access to generation stats
    var generationStats: GenerationStats {
        statsLock.lock()
        defer { statsLock.unlock() }
        return _generationStats
    }

    /// Thread-safe stat recording
    private func recordFallback() {
        statsLock.lock()
        defer { statsLock.unlock() }
        _generationStats.fallbackGenerations += 1
    }

    private func recordAI() {
        statsLock.lock()
        defer { statsLock.unlock() }
        _generationStats.aiGenerations += 1
    }

    private func recordTemplate() {
        statsLock.lock()
        defer { statsLock.unlock() }
        _generationStats.templateGenerations += 1
    }

    private func recordFailed() {
        statsLock.lock()
        defer { statsLock.unlock() }
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

    // MARK: - Model Inference Constants

    /// Parameters for model-based generation
    private enum ModelParams {
        /// Size of the output image from the diffusion model
        static let outputImageSize: Int = 256
        /// Threshold for binarizing model output (0-255 scale)
        static let binaryThreshold: UInt8 = 128
        /// Minimum contour length in pixels
        static let minContourLength: Int = 10
    }

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

        let startTime = Date()

        do {
            // Prepare conditioning based on mode
            let conditioning = try prepareConditioning(
                character: character,
                mode: mode,
                metrics: metrics
            )

            // Generate latent representation
            let outline = try await runDiffusion(
                conditioning: conditioning,
                settings: settings
            )

            // Create glyph
            let bounds = outline.boundingBox
            let glyph = Glyph(
                character: character,
                outline: outline,
                advanceWidth: bounds.width + Int(CGFloat(metrics.unitsPerEm) * SpacingRatio.rightSideBearing),
                leftSideBearing: Int(CGFloat(metrics.unitsPerEm) * SpacingRatio.leftSideBearing)
            )

            let generationTime = Date().timeIntervalSince(startTime)

            return GenerationResult(
                glyph: glyph,
                confidence: ConfidenceScore.aiModel,
                generationTime: generationTime
            )
        } catch {
            recordFailed()
            throw error
        }
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
    var isAvailable: Bool {
        true  // Template fallback is always available
    }

    // MARK: - Model-Based Generation

    /// Generate a glyph using the CoreML diffusion model
    /// Falls back to template generation if model is unavailable or fails
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
            // Prepare model input
            let inputFeatures = try prepareModelInput(
                character: character,
                style: style,
                seed: settings.seed
            )

            // Run inference
            let output = try await runModelInference(
                model: diffusionModel,
                input: inputFeatures,
                steps: settings.steps,
                guidanceScale: settings.guidanceScale
            )

            // Post-process: convert model output to glyph outline
            let outline = try postProcessModelOutput(
                output: output,
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
            recordFailed()
            print("[GlyphGenerator] Model inference failed: \(error.localizedDescription), falling back to template")
            return try await generatePlaceholder(
                character: character,
                mode: .fromScratch(style: style),
                metrics: metrics
            )
        }
    }

    // MARK: - Model Input Preparation

    /// Prepare input features for the diffusion model
    private func prepareModelInput(
        character: Character,
        style: StyleEncoder.FontStyle,
        seed: UInt64?
    ) throws -> MLFeatureProvider {
        // Create character embedding (one-hot or learned)
        let charEmbedding = createCharacterEmbedding(character)

        // Use style embedding or default
        let styleEmbedding = style.embedding.isEmpty ?
            createDefaultStyleEmbedding() : style.embedding

        // Create random seed for generation
        let actualSeed = seed ?? UInt64.random(in: 0..<UInt64.max)

        // Create MLMultiArray for embeddings
        let charArray = try MLMultiArray(shape: [128], dataType: .float32)
        for (i, value) in charEmbedding.enumerated() {
            charArray[i] = NSNumber(value: value)
        }

        let styleArray = try MLMultiArray(shape: [128], dataType: .float32)
        for i in 0..<min(styleEmbedding.count, 128) {
            styleArray[i] = NSNumber(value: styleEmbedding[i])
        }

        let seedArray = try MLMultiArray(shape: [1], dataType: .float32)
        seedArray[0] = NSNumber(value: Float(Double(actualSeed) / Double(UInt64.max)))

        // Use MLDictionaryFeatureProvider since actual model types aren't available until models are compiled
        return try MLDictionaryFeatureProvider(dictionary: [
            "characterEmbedding": MLFeatureValue(multiArray: charArray),
            "styleEmbedding": MLFeatureValue(multiArray: styleArray),
            "seed": MLFeatureValue(multiArray: seedArray)
        ])
    }

    /// Run diffusion model inference
    private func runModelInference(
        model: MLModel,
        input: MLFeatureProvider,
        steps: Int,
        guidanceScale: Float
    ) async throws -> CGImage {
        // Run prediction - model.prediction is async in Swift 6
        let prediction = try await model.prediction(from: input)

        // Extract output image from prediction
        guard let outputFeature = prediction.featureValue(for: "outputImage"),
              let pixelBuffer = outputFeature.imageBufferValue else {
            throw GeneratorError.generationFailed("Model did not produce valid image output")
        }

        // Convert CVPixelBuffer to CGImage
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let context = CIContext()
        guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else {
            throw GeneratorError.generationFailed("Failed to convert model output to image")
        }

        return cgImage
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
        let binaryData = pixelData.toBinary(threshold: ModelParams.binaryThreshold)

        // Step 3: Trace contours using existing ContourTracer
        let tracingSettings = ContourTracer.TracingSettings(
            simplificationTolerance: 2.0,
            minContourLength: ModelParams.minContourLength,
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

    private struct Conditioning {
        let characterEmbedding: [Float]
        let styleEmbedding: [Float]
        let targetMetrics: FontMetrics
        let baseOutline: GlyphOutline?
    }

    private func prepareConditioning(
        character: Character,
        mode: GenerationMode,
        metrics: FontMetrics
    ) throws -> Conditioning {
        // Create character embedding (one-hot or learned)
        let charEmbedding = createCharacterEmbedding(character)

        // Extract style embedding based on mode
        let styleEmbedding: [Float]
        var baseOutline: GlyphOutline? = nil

        switch mode {
        case .fromScratch(let style):
            styleEmbedding = style.embedding.isEmpty ?
                createDefaultStyleEmbedding() : style.embedding

        case .completePartial(let partial, let style):
            styleEmbedding = style.embedding.isEmpty ?
                createDefaultStyleEmbedding() : style.embedding
            baseOutline = partial

        case .variation(let base, _):
            styleEmbedding = createEmbeddingFromGlyph(base)
            baseOutline = base.outline

        case .interpolate(let glyphA, let glyphB, let t):
            let embeddingA = createEmbeddingFromGlyph(glyphA)
            let embeddingB = createEmbeddingFromGlyph(glyphB)
            styleEmbedding = interpolateEmbeddings(embeddingA, embeddingB, t: t)
        }

        return Conditioning(
            characterEmbedding: charEmbedding,
            styleEmbedding: styleEmbedding,
            targetMetrics: metrics,
            baseOutline: baseOutline
        )
    }

    private func runDiffusion(
        conditioning: Conditioning,
        settings: GenerationSettings
    ) async throws -> GlyphOutline {
        // ============================================================================
        // REAL AI IMPLEMENTATION REQUIREMENTS
        // ============================================================================
        //
        // To implement actual diffusion-based glyph generation, you need:
        //
        // 1. TRAINING DATA (~6 months of work)
        //    - 10,000+ fonts with consistent glyph labeling
        //    - Normalized outlines (all scaled to same units-per-em)
        //    - Style annotations (weight, contrast, serif style, x-height ratio, etc.)
        //    - Character class labels
        //
        // 2. MODEL ARCHITECTURE
        //    - Diffusion model conditioned on character class and style embedding
        //    - Input: noise + conditioning (character one-hot + style vector)
        //    - Output: sequence of control points forming glyph outline
        //    - Recommended: DDPM or flow-matching for cleaner outlines
        //
        // 3. TRAINING INFRASTRUCTURE
        //    - GPU cluster (8+ A100s for reasonable training time)
        //    - 2-4 weeks of training
        //    - PyTorch + custom training loop
        //
        // 4. COREML CONVERSION
        //    - Export trained model to ONNX
        //    - Convert ONNX → CoreML using coremltools
        //    - Optimize for Apple Neural Engine
        //
        // 5. INFERENCE PIPELINE (what this function would do)
        //    - Load CoreML model via ModelManager
        //    - Prepare input tensor from conditioning
        //    - Run denoising loop (settings.steps iterations)
        //    - Decode output points to GlyphOutline
        //
        // This is a multi-month ML research project requiring:
        //    - ML engineering expertise
        //    - Access to font datasets (Google Fonts, Adobe Fonts)
        //    - Significant compute resources
        //
        // For now, this path is unreachable (model never loads), and generation
        // falls back to the honest template-based system in generatePlaceholder().
        // ============================================================================

        // This code would run if a model was loaded - but no model exists
        guard let base = conditioning.baseOutline else {
            // Would run diffusion here if model existed
            throw GeneratorError.modelNotLoaded
        }

        return base
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

    private func createCharacterEmbedding(_ character: Character) -> [Float] {
        // Create a simple embedding from Unicode code point
        var embedding = [Float](repeating: 0, count: 128)

        for (index, scalar) in character.unicodeScalars.enumerated() {
            let value = Float(scalar.value)
            let normalized = value / 65536.0

            if index < embedding.count {
                embedding[index] = normalized
            }
        }

        return embedding
    }

    private func createDefaultStyleEmbedding() -> [Float] {
        // Default neutral style embedding
        [Float](repeating: 0.5, count: 128)
    }

    private func createEmbeddingFromGlyph(_ glyph: Glyph) -> [Float] {
        // Create embedding from glyph properties
        var embedding = [Float](repeating: 0, count: 128)

        let bounds = glyph.outline.boundingBox
        let pointCount = glyph.outline.contours.reduce(0) { $0 + $1.points.count }

        // Encode basic geometric features
        embedding[0] = Float(bounds.width) / 1000.0
        embedding[1] = Float(bounds.height) / 1000.0
        embedding[2] = Float(pointCount) / 100.0
        embedding[3] = Float(glyph.outline.contours.count) / 10.0
        embedding[4] = Float(glyph.advanceWidth) / 1000.0

        return embedding
    }

    private func interpolateEmbeddings(_ a: [Float], _ b: [Float], t: Float) -> [Float] {
        guard a.count == b.count else {
            return a
        }

        return zip(a, b).map { $0 * (1 - t) + $1 * t }
    }
}
