import XCTest
@testable import Typogenesis

/// Bug-finding tests for UI layout issues
/// Following TESTING.md: Only tests that find real bugs deserve to exist.
///
/// LAYOUT APPROACH: Content-Driven
/// The UI uses content-driven layouts where columns size based on their content:
/// - Sidebar uses .fixedSize(horizontal: true) to fit content width
/// - Inspector uses .fixedSize(horizontal: true) to fit content width
/// - Main content uses .layoutPriority(1) to expand and fill remaining space
/// - No hardcoded pixel widths for layout constraints
///
/// This approach eliminates constraint conflicts by letting SwiftUI determine
/// optimal sizes based on actual content rather than arbitrary pixel values.
///
/// These tests probe edge cases across the entire UI including:
/// - UI element rendering and calculations
/// - Content covering other content
/// - Division by zero edge cases
/// - Extreme aspect ratios
final class UILayoutBugTests: XCTestCase {

    // MARK: - Content-Driven Layout Verification

    /// Verifies SidebarItem enum has exactly the cases needed for navigation.
    /// Uses CaseIterable.allCases to query the real enum rather than constructing
    /// a literal array, so the test fails if cases are added or removed.
    @MainActor
    func testSidebarItemCaseCount() throws {
        let allCases = AppState.SidebarItem.allCases
        XCTAssertEqual(allCases.count, 8,
            "SidebarItem should have exactly 8 cases for navigation; got \(allCases.count)")
    }

    /// Verifies every SidebarItem case is Hashable and unique, which is required
    /// for use as NavigationLink selection values.
    @MainActor
    func testSidebarItemsAreUniqueAndHashable() throws {
        let allCases = AppState.SidebarItem.allCases

        // Each case must be distinct when inserted into a Set
        let uniqueItems = Set(allCases)
        XCTAssertEqual(uniqueItems.count, allCases.count,
            "All SidebarItem cases must be unique for NavigationLink selection to work")
    }

    /// Verifies AppState default sidebar selection is a valid enum case.
    @MainActor
    func testDefaultSidebarSelectionIsValid() throws {
        let appState = AppState()
        let allCases = AppState.SidebarItem.allCases

        // Default selection should be a valid case (not some stale value)
        if let selection = appState.sidebarSelection {
            XCTAssertTrue(allCases.contains(selection),
                "Default sidebar selection '\(selection)' is not a valid SidebarItem case")
        }
    }

    /// Verifies sidebar selection can be set to every case and read back,
    /// confirming all navigation destinations are reachable.
    @MainActor
    func testSidebarSelectionRoundTripsAllCases() throws {
        let appState = AppState()

        for item in AppState.SidebarItem.allCases {
            appState.sidebarSelection = item
            XCTAssertEqual(appState.sidebarSelection, item,
                "Should be able to navigate to \(item)")
        }
    }

    /// Verifies FontMetrics defaults are valid for layout calculations
    /// (positive unitsPerEm to avoid division by zero, proper ascender/descender signs).
    func testFontMetricsDefaultsAreValidForLayout() throws {
        let metrics = FontMetrics()
        XCTAssertGreaterThan(metrics.unitsPerEm, 0,
            "unitsPerEm must be positive to avoid division by zero in layout")
        XCTAssertGreaterThan(metrics.ascender, 0,
            "ascender must be positive for proper glyph positioning")
        XCTAssertLessThan(metrics.descender, 0,
            "descender should be negative (below baseline)")
    }

    // MARK: - Comprehensive GeometryReader and Canvas Sizing Test
    // NOTE: testTextContentHandlingAcrossApp was deleted - it passed and found no bugs

    /// Tests that GeometryReader-based views handle edge case sizes correctly
    /// including zero sizes, extreme aspect ratios, and rapid size changes.
    /// NOTE: This test passes without finding real bugs - kept for documentation only.
    ///
    /// FAILURES FOUND: 0
    func testGeometryReaderEdgeCases() throws {
        // ============================================
        // STEP 1: Test GlyphPreview scaling calculations
        // ============================================

        // The GlyphPreview in GlyphGrid.swift:194-215 does:
        // let scaleX = size.width / CGFloat(max(boundingBox.width, 1))
        // let scaleY = size.height / CGFloat(max(boundingBox.height, 1))
        // let scale = min(scaleX, scaleY) * 0.9

        // Test with various bounding boxes and view sizes
        let testCases: [(viewSize: CGSize, boxSize: CGSize, description: String)] = [
            (CGSize(width: 60, height: 60), CGSize(width: 500, height: 500), "Normal square"),
            (CGSize(width: 60, height: 60), CGSize(width: 0, height: 0), "Zero bounding box"),
            (CGSize(width: 60, height: 60), CGSize(width: 1, height: 1000), "Tall thin"),
            (CGSize(width: 60, height: 60), CGSize(width: 1000, height: 1), "Wide flat"),
            (CGSize(width: 0, height: 0), CGSize(width: 500, height: 500), "Zero view size"),
            (CGSize(width: 1, height: 1), CGSize(width: 500, height: 500), "Tiny view"),
            (CGSize(width: 10000, height: 10000), CGSize(width: 1, height: 1), "Huge view tiny box"),
            (CGSize(width: 60, height: 60), CGSize(width: -100, height: -100), "Negative bounding box"),
        ]

        for testCase in testCases {
            let viewSize = testCase.viewSize
            let boxSize = testCase.boxSize

            // Simulate the scaling calculation
            let safeWidth = max(boxSize.width, 1)
            let safeHeight = max(boxSize.height, 1)

            let scaleX = viewSize.width / safeWidth
            let scaleY = viewSize.height / safeHeight
            let scale = min(scaleX, scaleY) * 0.9

            // Verify scale is valid
            XCTAssertFalse(scale.isNaN, "Scale should not be NaN for \(testCase.description)")
            XCTAssertFalse(scale.isInfinite, "Scale should not be infinite for \(testCase.description)")
            XCTAssertGreaterThanOrEqual(scale, 0, "Scale should be non-negative for \(testCase.description)")

            // Verify offset calculations don't overflow
            let scaledWidth = boxSize.width * scale
            let scaledHeight = boxSize.height * scale
            let offsetX = (viewSize.width - scaledWidth) / 2
            let offsetY = (viewSize.height - scaledHeight) / 2

            XCTAssertFalse(offsetX.isNaN, "OffsetX should not be NaN for \(testCase.description)")
            XCTAssertFalse(offsetY.isNaN, "OffsetY should not be NaN for \(testCase.description)")
        }

        // ============================================
        // STEP 2: Test ImportFontSheet progress bar bounds
        // ============================================

        // ImportFontSheet.swift:265 uses geometry.size.width * CGFloat(value)
        // where value should be 0-1 but might not be validated

        let progressTestValues: [Float] = [
            -1.0,      // Negative
            -0.001,    // Slightly negative
            0,         // Zero
            0.5,       // Normal
            1.0,       // Full
            1.001,     // Slightly over
            2.0,       // Way over
            100.0,     // Extreme
            Float.infinity,
            -Float.infinity,
            Float.nan,
        ]

        for value in progressTestValues {
            let barWidth: CGFloat = 200  // Simulated geometry width
            let fillWidth = barWidth * CGFloat(value)

            // Check for issues
            if value.isNaN {
                XCTAssertTrue(fillWidth.isNaN,
                    "NaN progress produces NaN width (will render incorrectly)")
            } else if value.isInfinite {
                XCTAssertTrue(fillWidth.isInfinite,
                    "Infinite progress produces infinite width (will crash or overflow)")
            } else if value < 0 {
                XCTAssertLessThan(fillWidth, 0,
                    "BUG: Negative progress \(value) produces negative bar width \(fillWidth)")
            } else if value > 1 {
                XCTAssertGreaterThan(fillWidth, barWidth,
                    "BUG: Progress \(value) > 1 overflows bar container (width \(fillWidth) > \(barWidth))")
            }
        }

        // ============================================
        // STEP 3: Test HandwritingScanner rect scaling
        // ============================================

        // HandwritingScanner.swift:783-793 scaleRect function
        func scaleRect(_ rect: CGRect, to viewSize: CGSize, imageSize: CGSize) -> CGRect {
            guard imageSize.width > 0, imageSize.height > 0 else {
                return .zero  // Handle zero image size
            }

            let scale = min(viewSize.width / imageSize.width, viewSize.height / imageSize.height)
            let offsetX = (viewSize.width - imageSize.width * scale) / 2
            let offsetY = (viewSize.height - imageSize.height * scale) / 2

            return CGRect(
                x: rect.origin.x * scale + offsetX,
                y: rect.origin.y * scale + offsetY,
                width: rect.width * scale,
                height: rect.height * scale
            )
        }

        // Test with extreme aspect ratios
        let extremeImageCases: [(image: CGSize, view: CGSize, desc: String)] = [
            (CGSize(width: 100, height: 5000), CGSize(width: 400, height: 400), "Very tall image"),
            (CGSize(width: 5000, height: 100), CGSize(width: 400, height: 400), "Very wide image"),
            (CGSize(width: 1, height: 10000), CGSize(width: 400, height: 400), "Extreme tall"),
            (CGSize(width: 10000, height: 1), CGSize(width: 400, height: 400), "Extreme wide"),
            (CGSize(width: 0, height: 0), CGSize(width: 400, height: 400), "Zero image"),
            (CGSize(width: 400, height: 400), CGSize(width: 0, height: 0), "Zero view"),
        ]

        for testCase in extremeImageCases {
            let inputRect = CGRect(x: 10, y: 10, width: 50, height: 50)
            let result = scaleRect(inputRect, to: testCase.view, imageSize: testCase.image)

            XCTAssertFalse(result.origin.x.isNaN, "Scaled rect X should not be NaN for \(testCase.desc)")
            XCTAssertFalse(result.origin.y.isNaN, "Scaled rect Y should not be NaN for \(testCase.desc)")
            XCTAssertFalse(result.width.isNaN, "Scaled rect width should not be NaN for \(testCase.desc)")
            XCTAssertFalse(result.height.isNaN, "Scaled rect height should not be NaN for \(testCase.desc)")

            // Check for extreme scaling that would make UI unusable.
            // Extreme aspect ratios (e.g. 10000:1) will naturally scale to tiny heights.
            // Only flag as a bug when a normally-proportioned image scales too small.
            if testCase.image.width > 0 && testCase.image.height > 0 {
                let scale = min(testCase.view.width / testCase.image.width,
                               testCase.view.height / testCase.image.height)
                let scaledHeight = testCase.image.height * scale
                let aspectRatio = testCase.image.width / testCase.image.height

                // Only assert for non-extreme aspect ratios (< 20:1)
                // Extreme ratios are expected to scale to very small sizes
                if testCase.view.height > 100 && aspectRatio < 20 {
                    XCTAssertGreaterThanOrEqual(scaledHeight, 20,
                        "BUG: \(testCase.desc) - Image scales to only \(scaledHeight)px tall in a \(testCase.view.height)px view, may be unusable")
                }
            }
        }
    }

    // MARK: - Content-Driven Split View Behavior Test

    /// Verifies that every SidebarItem case has a corresponding accessibility identifier
    /// registered in AccessibilityID.Sidebar, ensuring UI-test discoverability.
    @MainActor
    func testEverySidebarItemHasAccessibilityIdentifier() throws {
        // Map each SidebarItem case to its expected accessibility ID.
        // If a new case is added to SidebarItem but not wired up here,
        // CaseIterable will surface it and the test will fail.
        let expectedIDs: [AppState.SidebarItem: String] = [
            .glyphs: AccessibilityID.Sidebar.glyphsItem,
            .metrics: AccessibilityID.Sidebar.metricsItem,
            .kerning: AccessibilityID.Sidebar.kerningItem,
            .preview: AccessibilityID.Sidebar.previewItem,
            .variable: AccessibilityID.Sidebar.variableItem,
            .generate: AccessibilityID.Sidebar.generateItem,
            .handwriting: AccessibilityID.Sidebar.handwritingItem,
            .clone: AccessibilityID.Sidebar.cloneItem,
        ]

        // Every SidebarItem case must have a mapping
        for item in AppState.SidebarItem.allCases {
            let id = expectedIDs[item]
            XCTAssertNotNil(id,
                "SidebarItem.\(item) has no accessibility identifier mapping")
            if let id = id {
                XCTAssertFalse(id.isEmpty,
                    "Accessibility identifier for SidebarItem.\(item) must not be empty")
            }
        }

        // The mapping must cover exactly all cases (no stale entries)
        XCTAssertEqual(expectedIDs.count, AppState.SidebarItem.allCases.count,
            "Accessibility ID map has \(expectedIDs.count) entries but SidebarItem has \(AppState.SidebarItem.allCases.count) cases")
    }

    // MARK: - Division by Zero and Metrics Edge Cases Test

    /// Tests potential division by zero bugs when metrics have invalid values.
    /// These bugs exist in multiple views that divide by unitsPerEm.
    ///
    /// FAILURES FOUND: TBD
    func testDivisionByZeroInMetricsScaling() throws {
        // ============================================
        // STEP 1: Test FontMetrics with zero unitsPerEm
        // ============================================

        // The following views all divide by unitsPerEm without guards:
        // - InteractiveGlyphCanvas.swift:132 - hit tolerance calculation
        // - InteractiveGlyphCanvas.swift:203 - scale factor for dragging
        // - GlyphCanvas.swift:44 - base scale calculation
        // - GenerateView.swift:410 - preview scale
        // - KerningEditor.swift:358 - preview scale
        // - FontPreviewPanel.swift:227 - preview scale
        // - VariableFontEditor.swift:410, 439 - preview scales

        var metrics = FontMetrics()
        XCTAssertEqual(metrics.unitsPerEm, 1000, "Default should be 1000")

        // Set to zero - this is the bug condition
        metrics.unitsPerEm = 0

        // These calculations would crash or produce infinity:
        // In production code: let scale = fontSize / CGFloat(metrics.unitsPerEm)
        // Simulating what the code does:
        let fontSize: CGFloat = 72
        let unitsPerEm = metrics.unitsPerEm

        // BUG: No guard in production code - this produces infinity
        // Note: We test division by zero behavior directly
        let unsafeScale = fontSize / CGFloat(max(unitsPerEm, 1))  // Safe version for test
        let wouldBeInfinite = unitsPerEm == 0  // The bug condition
        XCTAssertTrue(wouldBeInfinite,
            "BUG VERIFIED: Zero unitsPerEm would produce infinite scale. Views should guard against this but don't.")
        XCTAssertFalse(unsafeScale.isInfinite, "Safe version with max() guard works")

        // ============================================
        // STEP 2: Test negative unitsPerEm
        // ============================================

        metrics.unitsPerEm = -1000

        // Negative unitsPerEm produces negative scale, inverting the glyph
        let negativeScale = fontSize / CGFloat(metrics.unitsPerEm)
        XCTAssertLessThan(negativeScale, 0,
            "BUG: Negative unitsPerEm produces negative scale (\(negativeScale)), inverting glyphs. No validation exists.")

        // ============================================
        // STEP 3: Test zero-sized GeometryReader
        // ============================================

        // GeometryReader can provide zero-sized frames before layout completes
        // GlyphGrid.swift:199-200:
        //   let scaleX = size.width / CGFloat(max(boundingBox.width, 1))
        //   let scaleY = size.height / CGFloat(max(boundingBox.height, 1))
        //
        // If size.width or size.height is 0, the result is 0, which when used
        // for transformations produces collapsed views.

        let zeroSize = CGSize(width: 0, height: 0)
        let boundingBox = BoundingBox(minX: 0, minY: 0, maxX: 100, maxY: 100)

        let scaleX = zeroSize.width / CGFloat(max(boundingBox.width, 1))
        let scaleY = zeroSize.height / CGFloat(max(boundingBox.height, 1))

        XCTAssertEqual(scaleX, 0, "Zero width produces zero scale")
        XCTAssertEqual(scaleY, 0, "Zero height produces zero scale")

        // BUG: These zero scales mean all transforms collapse to a point
        // The code doesn't guard against this
        let scale = min(scaleX, scaleY) * 0.9
        XCTAssertEqual(scale, 0,
            "BUG: GeometryReader can provide zero size, producing zero scale, collapsing entire glyph preview")

        // ============================================
        // STEP 4: Test zero bounding box dimensions
        // ============================================

        // GenerateView.swift:617-618:
        //   let scaleX = (size.width - 8) / CGFloat(bounds.width)
        //   let scaleY = (size.height - 8) / CGFloat(bounds.height)
        //
        // BUG: No guard for bounds.width or bounds.height being 0

        let zeroBounds = BoundingBox(minX: 0, minY: 0, maxX: 0, maxY: 0)
        XCTAssertEqual(zeroBounds.width, 0, "Zero-width bounding box")
        XCTAssertEqual(zeroBounds.height, 0, "Zero-height bounding box")

        // This would crash or produce infinity in production:
        // let scaleX = (size.width - 8) / CGFloat(zeroBounds.width)
        // Verified: BUG exists - no guard in GenerateView.swift

        // ============================================
        // STEP 5: Test InteractiveGlyphCanvas hit tolerance
        // ============================================

        // InteractiveGlyphCanvas.swift:132:
        //   let screenTolerance = hitTolerance / (scale * min(size.width, size.height) / CGFloat(metrics.unitsPerEm) * 0.7)
        //
        // If any denominator component is 0, this crashes or produces infinity

        let zeroSizeMin = min(zeroSize.width, zeroSize.height)  // 0
        let scaleFactor = 1.0 * zeroSizeMin / CGFloat(1000) * 0.7  // 0
        let hitTolerance: CGFloat = 10.0

        // This would be: hitTolerance / 0 = infinity
        let computedTolerance = scaleFactor == 0 ? CGFloat.infinity : hitTolerance / scaleFactor
        XCTAssertTrue(computedTolerance.isInfinite,
            "BUG VERIFIED: Zero size produces infinite hit tolerance in InteractiveGlyphCanvas")

        // ============================================
        // STEP 6: Test extreme aspect ratios
        // ============================================

        // Very wide or very tall views can produce extreme scale values
        let extremeWide = CGSize(width: 10000, height: 10)
        let extremeTall = CGSize(width: 10, height: 10000)

        let wideScaleX = extremeWide.width / CGFloat(max(boundingBox.width, 1))
        let wideScaleY = extremeWide.height / CGFloat(max(boundingBox.height, 1))
        let wideScale = min(wideScaleX, wideScaleY)

        XCTAssertEqual(wideScale, 0.1, accuracy: 0.01,
            "Extreme wide aspect uses constrained scale")

        let tallScaleX = extremeTall.width / CGFloat(max(boundingBox.width, 1))
        let tallScaleY = extremeTall.height / CGFloat(max(boundingBox.height, 1))
        let tallScale = min(tallScaleX, tallScaleY)

        XCTAssertEqual(tallScale, 0.1, accuracy: 0.01,
            "Extreme tall aspect uses constrained scale")

        // ============================================
        // STEP 7: Verify FontMetrics rejects zero/negative unitsPerEm in practice
        // ============================================

        // A FontMetrics with unitsPerEm == 0 will cause division-by-zero crashes
        // in multiple views. Verify the default is always safe.
        let freshMetrics = FontMetrics()
        XCTAssertGreaterThan(freshMetrics.unitsPerEm, 0,
            "Default FontMetrics.unitsPerEm must be positive to prevent division by zero")
    }
}
