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

    // MARK: - Production BoundingBox and GlyphOutline Edge Cases

    /// Tests that GlyphOutline.boundingBox (production code) handles edge cases:
    /// empty outlines, single-point outlines, and degenerate contours.
    /// GlyphPreview relies on boundingBox to compute scaling transforms.
    func testBoundingBoxEdgeCases() throws {
        // Empty outline should produce zero bounding box
        let emptyOutline = GlyphOutline()
        let emptyBB = emptyOutline.boundingBox
        XCTAssertEqual(emptyBB.width, 0, "Empty outline bounding box width should be 0")
        XCTAssertEqual(emptyBB.height, 0, "Empty outline bounding box height should be 0")

        // Outline with empty contour should also produce zero bounding box
        let emptyContourOutline = GlyphOutline(contours: [Contour(points: [], isClosed: true)])
        let emptyContourBB = emptyContourOutline.boundingBox
        XCTAssertEqual(emptyContourBB.width, 0, "Empty contour bounding box width should be 0")
        XCTAssertEqual(emptyContourBB.height, 0, "Empty contour bounding box height should be 0")

        // Single point outline produces zero-area bounding box
        let singlePointOutline = GlyphOutline(contours: [
            Contour(points: [
                PathPoint(position: CGPoint(x: 100, y: 200), type: .corner)
            ], isClosed: true)
        ])
        let singleBB = singlePointOutline.boundingBox
        XCTAssertEqual(singleBB.width, 0, "Single point bounding box should have zero width")
        XCTAssertEqual(singleBB.height, 0, "Single point bounding box should have zero height")

        // Normal outline should produce valid bounding box
        let normalOutline = GlyphOutline(contours: [
            Contour(points: [
                PathPoint(position: CGPoint(x: 50, y: 0), type: .corner),
                PathPoint(position: CGPoint(x: 450, y: 0), type: .corner),
                PathPoint(position: CGPoint(x: 450, y: 700), type: .corner),
                PathPoint(position: CGPoint(x: 50, y: 700), type: .corner),
            ], isClosed: true)
        ])
        let normalBB = normalOutline.boundingBox
        XCTAssertEqual(normalBB.width, 400, "Normal bounding box width")
        XCTAssertEqual(normalBB.height, 700, "Normal bounding box height")
        XCTAssertEqual(normalBB.minX, 50)
        XCTAssertEqual(normalBB.maxX, 450)
    }

    /// Tests that GlyphOutline.cgPath (production code) handles edge cases
    /// without crashing, and produces non-degenerate paths for valid outlines.
    func testCGPathEdgeCases() throws {
        // Empty outline produces a valid (empty) CGPath
        let emptyPath = GlyphOutline().cgPath
        XCTAssertTrue(emptyPath.isEmpty, "Empty outline should produce empty CGPath")

        // Normal outline produces a non-empty path with correct bounds
        let outline = GlyphOutline(contours: [
            Contour(points: [
                PathPoint(position: CGPoint(x: 0, y: 0), type: .corner),
                PathPoint(position: CGPoint(x: 100, y: 0), type: .corner),
                PathPoint(position: CGPoint(x: 100, y: 100), type: .corner),
                PathPoint(position: CGPoint(x: 0, y: 100), type: .corner),
            ], isClosed: true)
        ])
        let path = outline.cgPath
        XCTAssertFalse(path.isEmpty, "Normal outline should produce non-empty CGPath")
        let bounds = path.boundingBox
        XCTAssertFalse(bounds.isNull, "CGPath bounds should not be null")
        XCTAssertEqual(bounds.width, 100, accuracy: 0.01, "CGPath width should match outline")
        XCTAssertEqual(bounds.height, 100, accuracy: 0.01, "CGPath height should match outline")
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

    // MARK: - FontMetrics and BoundingBox Production Code Edge Cases

    /// Tests production FontMetrics properties for values used in layout calculations.
    /// Multiple views divide by unitsPerEm, so defaults must be safe.
    func testFontMetricsLayoutSafety() throws {
        // Default FontMetrics must have positive unitsPerEm to prevent division by zero
        let metrics = FontMetrics()
        XCTAssertEqual(metrics.unitsPerEm, 1000, "Default unitsPerEm should be 1000")
        XCTAssertGreaterThan(metrics.unitsPerEm, 0,
            "unitsPerEm must be positive to avoid division by zero in layout calculations")

        // FontMetrics allows zero unitsPerEm (no validation in init).
        // This documents the risk: views that divide by unitsPerEm must guard against zero.
        let zeroMetrics = FontMetrics(unitsPerEm: 0)
        XCTAssertEqual(zeroMetrics.unitsPerEm, 0,
            "FontMetrics allows zero unitsPerEm (views must guard against this)")

        // Negative unitsPerEm is also allowed but would invert glyph rendering.
        let negativeMetrics = FontMetrics(unitsPerEm: -1000)
        XCTAssertEqual(negativeMetrics.unitsPerEm, -1000,
            "FontMetrics allows negative unitsPerEm (views must guard against this)")

        // Ascender/descender relationship: ascender should be above baseline
        XCTAssertGreaterThan(metrics.ascender, metrics.baseline,
            "Ascender should be above baseline")
        XCTAssertLessThan(metrics.descender, metrics.baseline,
            "Descender should be below baseline")
    }

    /// Tests production BoundingBox computed properties (width, height, cgRect)
    /// to verify correct behavior with edge case values.
    func testBoundingBoxComputedProperties() throws {
        // Zero-area bounding box (e.g., from empty glyph outline)
        let zeroBB = BoundingBox(minX: 0, minY: 0, maxX: 0, maxY: 0)
        XCTAssertEqual(zeroBB.width, 0, "Zero bounding box width")
        XCTAssertEqual(zeroBB.height, 0, "Zero bounding box height")
        XCTAssertEqual(zeroBB.cgRect, CGRect.zero, "Zero bounding box cgRect")

        // Normal bounding box
        let normalBB = BoundingBox(minX: 50, minY: 0, maxX: 450, maxY: 700)
        XCTAssertEqual(normalBB.width, 400)
        XCTAssertEqual(normalBB.height, 700)
        XCTAssertEqual(normalBB.cgRect, CGRect(x: 50, y: 0, width: 400, height: 700))

        // Bounding box with negative coordinates (descender glyphs)
        let descenderBB = BoundingBox(minX: 50, minY: -200, maxX: 450, maxY: 500)
        XCTAssertEqual(descenderBB.width, 400)
        XCTAssertEqual(descenderBB.height, 700)
        XCTAssertEqual(descenderBB.cgRect.origin.y, -200)
    }

    /// Tests that GlyphOutline.boundingBox produces correct zero-area results
    /// for empty outlines, which downstream views use for scaling calculations.
    /// Views that divide by boundingBox.width or height must guard against zero.
    func testEmptyOutlineBoundingBoxForScaling() throws {
        // An empty outline's bounding box has zero dimensions.
        // GlyphPreview in GlyphGrid.swift uses max(boundingBox.width, 1) to guard.
        // GenerateView.swift does NOT guard, which would cause division by zero.
        let emptyOutline = GlyphOutline()
        let bb = emptyOutline.boundingBox
        XCTAssertEqual(bb.width, 0,
            "Empty outline bounding box has zero width - views must guard before dividing by this")
        XCTAssertEqual(bb.height, 0,
            "Empty outline bounding box has zero height - views must guard before dividing by this")
    }
}
