import Foundation
import CoreGraphics
import CoreImage
import AppKit

/// Service for preprocessing images before vectorization
final class ImageProcessor {

    enum ImageProcessorError: Error, LocalizedError {
        case invalidImage
        case processingFailed
        case conversionFailed

        var errorDescription: String? {
            switch self {
            case .invalidImage:
                return "Invalid or corrupted image"
            case .processingFailed:
                return "Image processing failed"
            case .conversionFailed:
                return "Failed to convert image format"
            }
        }
    }

    struct ProcessingSettings {
        var threshold: Double = 0.5       // 0-1, controls black/white threshold
        var contrast: Double = 1.0        // Contrast enhancement
        var brightness: Double = 0.0      // Brightness adjustment
        var denoise: Bool = true          // Apply noise reduction
        var deskew: Bool = true           // Auto-correct rotation
        var invert: Bool = false          // Invert colors (white on black -> black on white)

        static let `default` = ProcessingSettings()
    }

    // MARK: - Main Processing Pipeline

    /// Process an image for vectorization
    static func process(
        image: NSImage,
        settings: ProcessingSettings = .default
    ) throws -> CGImage {
        guard let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
            throw ImageProcessorError.invalidImage
        }

        return try process(cgImage: cgImage, settings: settings)
    }

    /// Process a CGImage for vectorization
    static func process(
        cgImage: CGImage,
        settings: ProcessingSettings = .default
    ) throws -> CGImage {
        var ciImage = CIImage(cgImage: cgImage)

        // Apply processing steps in order
        if settings.denoise {
            ciImage = applyNoiseReduction(ciImage)
        }

        if settings.brightness != 0 || settings.contrast != 1.0 {
            ciImage = applyBrightnessContrast(ciImage, brightness: settings.brightness, contrast: settings.contrast)
        }

        if settings.deskew {
            ciImage = applyDeskew(ciImage)
        }

        // Convert to grayscale
        ciImage = convertToGrayscale(ciImage)

        // Apply threshold to create binary image
        ciImage = applyThreshold(ciImage, threshold: settings.threshold)

        if settings.invert {
            ciImage = applyInvert(ciImage)
        }

        // Convert back to CGImage
        let context = CIContext()
        guard let outputImage = context.createCGImage(ciImage, from: ciImage.extent) else {
            throw ImageProcessorError.conversionFailed
        }

        return outputImage
    }

    // MARK: - Processing Steps

    /// Apply noise reduction using median filter
    private static func applyNoiseReduction(_ image: CIImage) -> CIImage {
        let filter = CIFilter(name: "CIMedianFilter")!
        filter.setValue(image, forKey: kCIInputImageKey)
        return filter.outputImage ?? image
    }

    /// Adjust brightness and contrast
    private static func applyBrightnessContrast(_ image: CIImage, brightness: Double, contrast: Double) -> CIImage {
        let filter = CIFilter(name: "CIColorControls")!
        filter.setValue(image, forKey: kCIInputImageKey)
        filter.setValue(brightness, forKey: kCIInputBrightnessKey)
        filter.setValue(contrast, forKey: kCIInputContrastKey)
        return filter.outputImage ?? image
    }

    /// Auto-correct image rotation (simple implementation)
    private static func applyDeskew(_ image: CIImage) -> CIImage {
        // For now, just return the image - full deskew would require detecting text lines
        // and calculating rotation angle
        return image
    }

    /// Convert to grayscale
    private static func convertToGrayscale(_ image: CIImage) -> CIImage {
        let filter = CIFilter(name: "CIPhotoEffectMono")!
        filter.setValue(image, forKey: kCIInputImageKey)
        return filter.outputImage ?? image
    }

    /// Apply threshold to create binary (black/white) image
    private static func applyThreshold(_ image: CIImage, threshold: Double) -> CIImage {
        // Use CIColorMatrix to apply threshold
        // Pixels darker than threshold become black, lighter become white
        let filter = CIFilter(name: "CIColorMatrix")!
        filter.setValue(image, forKey: kCIInputImageKey)

        // Create a high-contrast matrix effect
        // This approximates thresholding
        let scale: CGFloat = 10.0 // High value for sharp transition
        let offset = CGFloat(-threshold * scale + 0.5)

        filter.setValue(CIVector(x: scale, y: 0, z: 0, w: 0), forKey: "inputRVector")
        filter.setValue(CIVector(x: 0, y: scale, z: 0, w: 0), forKey: "inputGVector")
        filter.setValue(CIVector(x: 0, y: 0, z: scale, w: 0), forKey: "inputBVector")
        filter.setValue(CIVector(x: 0, y: 0, z: 0, w: 1), forKey: "inputAVector")
        filter.setValue(CIVector(x: offset, y: offset, z: offset, w: 0), forKey: "inputBiasVector")

        let contrasted = filter.outputImage ?? image

        // Clamp to 0-1 range
        let clampFilter = CIFilter(name: "CIColorClamp")!
        clampFilter.setValue(contrasted, forKey: kCIInputImageKey)
        clampFilter.setValue(CIVector(x: 0, y: 0, z: 0, w: 0), forKey: "inputMinComponents")
        clampFilter.setValue(CIVector(x: 1, y: 1, z: 1, w: 1), forKey: "inputMaxComponents")

        return clampFilter.outputImage ?? contrasted
    }

    /// Invert image colors
    private static func applyInvert(_ image: CIImage) -> CIImage {
        let filter = CIFilter(name: "CIColorInvert")!
        filter.setValue(image, forKey: kCIInputImageKey)
        return filter.outputImage ?? image
    }

    // MARK: - Pixel Data Access

    /// Get raw pixel data from a CGImage (grayscale values 0-255)
    static func getPixelData(_ image: CGImage) throws -> PixelData {
        let width = image.width
        let height = image.height

        let colorSpace = CGColorSpaceCreateDeviceGray()
        var pixelData = [UInt8](repeating: 0, count: width * height)

        guard let context = CGContext(
            data: &pixelData,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.none.rawValue
        ) else {
            throw ImageProcessorError.conversionFailed
        }

        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

        return PixelData(data: pixelData, width: width, height: height)
    }

    /// Pixel data structure
    struct PixelData {
        let data: [UInt8]
        let width: Int
        let height: Int

        /// Get pixel value at coordinates (0 = black, 255 = white)
        func pixel(x: Int, y: Int) -> UInt8 {
            guard x >= 0, x < width, y >= 0, y < height else { return 255 }
            return data[y * width + x]
        }

        /// Check if pixel is black (below threshold)
        func isBlack(x: Int, y: Int, threshold: UInt8 = 128) -> Bool {
            return pixel(x: x, y: y) < threshold
        }

        /// Create binary image (true = foreground/black, false = background/white)
        func toBinary(threshold: UInt8 = 128) -> [[Bool]] {
            var result = [[Bool]](repeating: [Bool](repeating: false, count: width), count: height)
            for y in 0..<height {
                for x in 0..<width {
                    result[y][x] = isBlack(x: x, y: y, threshold: threshold)
                }
            }
            return result
        }
    }

    // MARK: - Character Detection

    /// Detect bounding boxes of separate characters in a binary image
    static func detectCharacterBounds(
        in pixelData: PixelData,
        minSize: Int = 10,
        padding: Int = 5
    ) -> [CGRect] {
        let binary = pixelData.toBinary()
        var visited = [[Bool]](repeating: [Bool](repeating: false, count: pixelData.width), count: pixelData.height)
        var bounds: [CGRect] = []

        for y in 0..<pixelData.height {
            for x in 0..<pixelData.width {
                if binary[y][x] && !visited[y][x] {
                    // Found a new connected component - flood fill to find extent
                    var minX = x, maxX = x, minY = y, maxY = y
                    var stack = [(x, y)]

                    while !stack.isEmpty {
                        let (px, py) = stack.removeLast()

                        guard px >= 0, px < pixelData.width, py >= 0, py < pixelData.height else { continue }
                        guard binary[py][px] && !visited[py][px] else { continue }

                        visited[py][px] = true
                        minX = min(minX, px)
                        maxX = max(maxX, px)
                        minY = min(minY, py)
                        maxY = max(maxY, py)

                        // Add 4-connected neighbors
                        stack.append((px - 1, py))
                        stack.append((px + 1, py))
                        stack.append((px, py - 1))
                        stack.append((px, py + 1))
                    }

                    // Check if component is large enough
                    let width = maxX - minX + 1
                    let height = maxY - minY + 1

                    if width >= minSize && height >= minSize {
                        // Add padding
                        let rect = CGRect(
                            x: max(0, minX - padding),
                            y: max(0, minY - padding),
                            width: min(pixelData.width - minX + padding, width + padding * 2),
                            height: min(pixelData.height - minY + padding, height + padding * 2)
                        )
                        bounds.append(rect)
                    }
                }
            }
        }

        // Sort by position (left to right, top to bottom)
        return bounds.sorted { a, b in
            let rowA = Int(a.midY / 100)
            let rowB = Int(b.midY / 100)
            if rowA != rowB {
                return rowA < rowB
            }
            return a.minX < b.minX
        }
    }

    // MARK: - Grid Detection

    /// Detect grid cells in a sample sheet image
    static func detectGridCells(
        in image: CGImage,
        expectedRows: Int = 6,
        expectedCols: Int = 10
    ) throws -> [[CGRect]] {
        let pixelData = try getPixelData(image)

        // Simple approach: divide image into expected grid
        let cellWidth = CGFloat(pixelData.width) / CGFloat(expectedCols)
        let cellHeight = CGFloat(pixelData.height) / CGFloat(expectedRows)

        var grid: [[CGRect]] = []

        for row in 0..<expectedRows {
            var rowCells: [CGRect] = []
            for col in 0..<expectedCols {
                let rect = CGRect(
                    x: CGFloat(col) * cellWidth,
                    y: CGFloat(row) * cellHeight,
                    width: cellWidth,
                    height: cellHeight
                )
                rowCells.append(rect)
            }
            grid.append(rowCells)
        }

        return grid
    }

    // MARK: - Extract Character Image

    /// Extract a sub-image for a detected character
    static func extractCharacter(from image: CGImage, bounds: CGRect) throws -> CGImage {
        let rect = CGRect(
            x: bounds.origin.x,
            y: CGFloat(image.height) - bounds.origin.y - bounds.height, // Flip Y
            width: bounds.width,
            height: bounds.height
        )

        guard let cropped = image.cropping(to: rect) else {
            throw ImageProcessorError.processingFailed
        }

        return cropped
    }
}
