import Foundation
import CoreGraphics

// MARK: - Constants

/// Mathematical constant for bezier circle approximation.
/// Calculated as 4 * (sqrt(2) - 1) / 3 ≈ 0.5522847498.
/// When used as control point offset, produces a near-perfect circular arc.
private let kBezierCircleApproximation: CGFloat = 0.5523

/// Stroke-based glyph templates for algorithmic generation.
///
/// This system defines letterforms using parameterized strokes (stems, bowls, bars)
/// that can be modified by style parameters to create consistent, recognizable glyphs.
///
/// **Design Principles:**
/// - Characters are built from composable stroke primitives
/// - All dimensions are normalized to 0-1 coordinate space
/// - Style parameters (weight, contrast, roundness) modify stroke output
/// - Templates produce typographically informed shapes, not arbitrary geometry

// MARK: - Stroke Primitives

/// A stroke segment that can be converted to bezier curves
enum StrokeSegment: Sendable {
    /// Straight line from start to end
    case line(start: CGPoint, end: CGPoint)

    /// Quadratic curve with control point
    case quadCurve(start: CGPoint, control: CGPoint, end: CGPoint)

    /// Cubic bezier curve
    case cubicCurve(start: CGPoint, control1: CGPoint, control2: CGPoint, end: CGPoint)

    /// Circular/elliptical arc
    case arc(center: CGPoint, radius: CGSize, startAngle: CGFloat, endAngle: CGFloat, clockwise: Bool)

    var startPoint: CGPoint {
        switch self {
        case .line(let start, _): return start
        case .quadCurve(let start, _, _): return start
        case .cubicCurve(let start, _, _, _): return start
        case .arc(let center, let radius, let startAngle, _, _):
            return CGPoint(
                x: center.x + radius.width * cos(startAngle),
                y: center.y + radius.height * sin(startAngle)
            )
        }
    }

    var endPoint: CGPoint {
        switch self {
        case .line(_, let end): return end
        case .quadCurve(_, _, let end): return end
        case .cubicCurve(_, _, _, let end): return end
        case .arc(let center, let radius, _, let endAngle, _):
            return CGPoint(
                x: center.x + radius.width * cos(endAngle),
                y: center.y + radius.height * sin(endAngle)
            )
        }
    }
}

/// A complete stroke path (centerline) that will be expanded to have width
struct StrokePath: Sendable {
    var segments: [StrokeSegment]
    var isClosed: Bool

    init(segments: [StrokeSegment] = [], isClosed: Bool = false) {
        self.segments = segments
        self.isClosed = isClosed
    }

    static func line(from start: CGPoint, to end: CGPoint) -> StrokePath {
        StrokePath(segments: [.line(start: start, end: end)])
    }

    /// Create an ellipse approximated with bezier curves
    static func ellipse(center: CGPoint, radiusX: CGFloat, radiusY: CGFloat) -> StrokePath {
        // Bezier approximation of circle: control point offset = radius * kBezierCircleApproximation
        let kx = radiusX * kBezierCircleApproximation
        let ky = radiusY * kBezierCircleApproximation

        // Four quadrants as cubic beziers
        let right = CGPoint(x: center.x + radiusX, y: center.y)
        let top = CGPoint(x: center.x, y: center.y + radiusY)
        let left = CGPoint(x: center.x - radiusX, y: center.y)
        let bottom = CGPoint(x: center.x, y: center.y - radiusY)

        return StrokePath(segments: [
            .cubicCurve(
                start: right,
                control1: CGPoint(x: center.x + radiusX, y: center.y + ky),
                control2: CGPoint(x: center.x + kx, y: center.y + radiusY),
                end: top
            ),
            .cubicCurve(
                start: top,
                control1: CGPoint(x: center.x - kx, y: center.y + radiusY),
                control2: CGPoint(x: center.x - radiusX, y: center.y + ky),
                end: left
            ),
            .cubicCurve(
                start: left,
                control1: CGPoint(x: center.x - radiusX, y: center.y - ky),
                control2: CGPoint(x: center.x - kx, y: center.y - radiusY),
                end: bottom
            ),
            .cubicCurve(
                start: bottom,
                control1: CGPoint(x: center.x + kx, y: center.y - radiusY),
                control2: CGPoint(x: center.x + radiusX, y: center.y - ky),
                end: right
            )
        ], isClosed: true)
    }

}

// MARK: - Glyph Template

/// A template for a single glyph, containing stroke definitions
struct GlyphTemplate: Sendable {
    /// The character this template represents
    let character: Character

    /// Stroke paths that form the glyph (in normalized 0-1 coordinates)
    /// Width 1.0 = standard width, Height 1.0 = cap height for uppercase, x-height for lowercase
    var strokes: [StrokePath]

    /// Width of the glyph relative to em (before sidebearings)
    var normalizedWidth: CGFloat

    /// Whether this is an uppercase letter
    var isUppercase: Bool

    /// Baseline offset (for descenders like g, p, y)
    var baselineOffset: CGFloat = 0

    /// Height multiplier (1.0 = cap/x-height, >1.0 for ascenders, <0 baseline for descenders)
    var heightMultiplier: CGFloat = 1.0

    init(
        character: Character,
        strokes: [StrokePath],
        normalizedWidth: CGFloat,
        isUppercase: Bool = false,
        baselineOffset: CGFloat = 0,
        heightMultiplier: CGFloat = 1.0
    ) {
        self.character = character
        self.strokes = strokes
        self.normalizedWidth = normalizedWidth
        self.isUppercase = isUppercase
        self.baselineOffset = baselineOffset
        self.heightMultiplier = heightMultiplier
    }
}

// MARK: - Template Library

/// Library of glyph templates for ASCII characters
final class GlyphTemplateLibrary: Sendable {
    static let shared = GlyphTemplateLibrary()

    private let templates: [Character: GlyphTemplate]

    private init() {
        var t: [Character: GlyphTemplate] = [:]

        // Build all templates
        Self.buildUppercaseTemplates(&t)
        Self.buildLowercaseTemplates(&t)
        Self.buildDigitTemplates(&t)
        Self.buildPunctuationTemplates(&t)

        templates = t
    }

    func template(for character: Character) -> GlyphTemplate? {
        templates[character]
    }

    // MARK: - Uppercase Letters

    private static func buildUppercaseTemplates(_ t: inout [Character: GlyphTemplate]) {
        // A - Two diagonals meeting at apex + crossbar
        t["A"] = GlyphTemplate(
            character: "A",
            strokes: [
                // Left diagonal
                .line(from: CGPoint(x: 0, y: 0), to: CGPoint(x: 0.5, y: 1)),
                // Right diagonal
                .line(from: CGPoint(x: 0.5, y: 1), to: CGPoint(x: 1, y: 0)),
                // Crossbar
                .line(from: CGPoint(x: 0.2, y: 0.35), to: CGPoint(x: 0.8, y: 0.35))
            ],
            normalizedWidth: 0.75,
            isUppercase: true
        )

        // B - Vertical stem + two bowls
        t["B"] = GlyphTemplate(
            character: "B",
            strokes: [
                // Vertical stem
                .line(from: CGPoint(x: 0, y: 0), to: CGPoint(x: 0, y: 1)),
                // Upper bowl (smaller)
                StrokePath(segments: [
                    .line(start: CGPoint(x: 0, y: 1), end: CGPoint(x: 0.5, y: 1)),
                    .cubicCurve(
                        start: CGPoint(x: 0.5, y: 1),
                        control1: CGPoint(x: 0.85, y: 1),
                        control2: CGPoint(x: 0.85, y: 0.55),
                        end: CGPoint(x: 0.5, y: 0.55)
                    ),
                    .line(start: CGPoint(x: 0.5, y: 0.55), end: CGPoint(x: 0, y: 0.55))
                ], isClosed: true),
                // Lower bowl (larger)
                StrokePath(segments: [
                    .line(start: CGPoint(x: 0, y: 0.55), end: CGPoint(x: 0.55, y: 0.55)),
                    .cubicCurve(
                        start: CGPoint(x: 0.55, y: 0.55),
                        control1: CGPoint(x: 0.95, y: 0.55),
                        control2: CGPoint(x: 0.95, y: 0),
                        end: CGPoint(x: 0.55, y: 0)
                    ),
                    .line(start: CGPoint(x: 0.55, y: 0), end: CGPoint(x: 0, y: 0))
                ], isClosed: true)
            ],
            normalizedWidth: 0.7,
            isUppercase: true
        )

        // C - Open bowl
        t["C"] = GlyphTemplate(
            character: "C",
            strokes: [
                StrokePath(segments: [
                    .cubicCurve(
                        start: CGPoint(x: 0.9, y: 0.85),
                        control1: CGPoint(x: 0.7, y: 1.05),
                        control2: CGPoint(x: 0.3, y: 1.05),
                        end: CGPoint(x: 0.1, y: 0.75)
                    ),
                    .cubicCurve(
                        start: CGPoint(x: 0.1, y: 0.75),
                        control1: CGPoint(x: -0.1, y: 0.5),
                        control2: CGPoint(x: -0.1, y: 0.5),
                        end: CGPoint(x: 0.1, y: 0.25)
                    ),
                    .cubicCurve(
                        start: CGPoint(x: 0.1, y: 0.25),
                        control1: CGPoint(x: 0.3, y: -0.05),
                        control2: CGPoint(x: 0.7, y: -0.05),
                        end: CGPoint(x: 0.9, y: 0.15)
                    )
                ])
            ],
            normalizedWidth: 0.75,
            isUppercase: true
        )

        // D - Vertical stem + large bowl
        t["D"] = GlyphTemplate(
            character: "D",
            strokes: [
                .line(from: CGPoint(x: 0, y: 0), to: CGPoint(x: 0, y: 1)),
                StrokePath(segments: [
                    .line(start: CGPoint(x: 0, y: 1), end: CGPoint(x: 0.4, y: 1)),
                    .cubicCurve(
                        start: CGPoint(x: 0.4, y: 1),
                        control1: CGPoint(x: 1.0, y: 1),
                        control2: CGPoint(x: 1.0, y: 0),
                        end: CGPoint(x: 0.4, y: 0)
                    ),
                    .line(start: CGPoint(x: 0.4, y: 0), end: CGPoint(x: 0, y: 0))
                ], isClosed: true)
            ],
            normalizedWidth: 0.75,
            isUppercase: true
        )

        // E - Vertical stem + three horizontals
        t["E"] = GlyphTemplate(
            character: "E",
            strokes: [
                .line(from: CGPoint(x: 0, y: 0), to: CGPoint(x: 0, y: 1)),
                .line(from: CGPoint(x: 0, y: 1), to: CGPoint(x: 0.8, y: 1)),
                .line(from: CGPoint(x: 0, y: 0.5), to: CGPoint(x: 0.6, y: 0.5)),
                .line(from: CGPoint(x: 0, y: 0), to: CGPoint(x: 0.8, y: 0))
            ],
            normalizedWidth: 0.65,
            isUppercase: true
        )

        // F - Like E without bottom bar
        t["F"] = GlyphTemplate(
            character: "F",
            strokes: [
                .line(from: CGPoint(x: 0, y: 0), to: CGPoint(x: 0, y: 1)),
                .line(from: CGPoint(x: 0, y: 1), to: CGPoint(x: 0.8, y: 1)),
                .line(from: CGPoint(x: 0, y: 0.5), to: CGPoint(x: 0.6, y: 0.5))
            ],
            normalizedWidth: 0.6,
            isUppercase: true
        )

        // G - C with crossbar
        t["G"] = GlyphTemplate(
            character: "G",
            strokes: [
                StrokePath(segments: [
                    .cubicCurve(
                        start: CGPoint(x: 0.9, y: 0.85),
                        control1: CGPoint(x: 0.7, y: 1.05),
                        control2: CGPoint(x: 0.3, y: 1.05),
                        end: CGPoint(x: 0.1, y: 0.75)
                    ),
                    .cubicCurve(
                        start: CGPoint(x: 0.1, y: 0.75),
                        control1: CGPoint(x: -0.1, y: 0.5),
                        control2: CGPoint(x: -0.1, y: 0.5),
                        end: CGPoint(x: 0.1, y: 0.25)
                    ),
                    .cubicCurve(
                        start: CGPoint(x: 0.1, y: 0.25),
                        control1: CGPoint(x: 0.3, y: -0.05),
                        control2: CGPoint(x: 0.7, y: -0.05),
                        end: CGPoint(x: 0.9, y: 0.15)
                    ),
                    .line(start: CGPoint(x: 0.9, y: 0.15), end: CGPoint(x: 0.9, y: 0.45)),
                    .line(start: CGPoint(x: 0.9, y: 0.45), end: CGPoint(x: 0.5, y: 0.45))
                ])
            ],
            normalizedWidth: 0.8,
            isUppercase: true
        )

        // H - Two verticals + crossbar
        t["H"] = GlyphTemplate(
            character: "H",
            strokes: [
                .line(from: CGPoint(x: 0, y: 0), to: CGPoint(x: 0, y: 1)),
                .line(from: CGPoint(x: 1, y: 0), to: CGPoint(x: 1, y: 1)),
                .line(from: CGPoint(x: 0, y: 0.5), to: CGPoint(x: 1, y: 0.5))
            ],
            normalizedWidth: 0.75,
            isUppercase: true
        )

        // I - Single vertical (narrow)
        t["I"] = GlyphTemplate(
            character: "I",
            strokes: [
                .line(from: CGPoint(x: 0.5, y: 0), to: CGPoint(x: 0.5, y: 1))
            ],
            normalizedWidth: 0.25,
            isUppercase: true
        )

        // J - Hook shape
        t["J"] = GlyphTemplate(
            character: "J",
            strokes: [
                StrokePath(segments: [
                    .line(start: CGPoint(x: 0.7, y: 1), end: CGPoint(x: 0.7, y: 0.25)),
                    .cubicCurve(
                        start: CGPoint(x: 0.7, y: 0.25),
                        control1: CGPoint(x: 0.7, y: -0.05),
                        control2: CGPoint(x: 0.3, y: -0.05),
                        end: CGPoint(x: 0.1, y: 0.15)
                    )
                ])
            ],
            normalizedWidth: 0.55,
            isUppercase: true
        )

        // K - Vertical + two diagonals
        t["K"] = GlyphTemplate(
            character: "K",
            strokes: [
                .line(from: CGPoint(x: 0, y: 0), to: CGPoint(x: 0, y: 1)),
                .line(from: CGPoint(x: 0, y: 0.45), to: CGPoint(x: 0.9, y: 1)),
                .line(from: CGPoint(x: 0, y: 0.45), to: CGPoint(x: 0.9, y: 0))
            ],
            normalizedWidth: 0.7,
            isUppercase: true
        )

        // L - Vertical + bottom horizontal
        t["L"] = GlyphTemplate(
            character: "L",
            strokes: [
                .line(from: CGPoint(x: 0, y: 0), to: CGPoint(x: 0, y: 1)),
                .line(from: CGPoint(x: 0, y: 0), to: CGPoint(x: 0.75, y: 0))
            ],
            normalizedWidth: 0.6,
            isUppercase: true
        )

        // M - Two verticals + two diagonals meeting at center
        t["M"] = GlyphTemplate(
            character: "M",
            strokes: [
                .line(from: CGPoint(x: 0, y: 0), to: CGPoint(x: 0, y: 1)),
                .line(from: CGPoint(x: 0, y: 1), to: CGPoint(x: 0.5, y: 0.3)),
                .line(from: CGPoint(x: 0.5, y: 0.3), to: CGPoint(x: 1, y: 1)),
                .line(from: CGPoint(x: 1, y: 1), to: CGPoint(x: 1, y: 0))
            ],
            normalizedWidth: 0.9,
            isUppercase: true
        )

        // N - Two verticals + diagonal
        t["N"] = GlyphTemplate(
            character: "N",
            strokes: [
                .line(from: CGPoint(x: 0, y: 0), to: CGPoint(x: 0, y: 1)),
                .line(from: CGPoint(x: 0, y: 1), to: CGPoint(x: 1, y: 0)),
                .line(from: CGPoint(x: 1, y: 0), to: CGPoint(x: 1, y: 1))
            ],
            normalizedWidth: 0.75,
            isUppercase: true
        )

        // O - Full ellipse
        t["O"] = GlyphTemplate(
            character: "O",
            strokes: [
                .ellipse(center: CGPoint(x: 0.5, y: 0.5), radiusX: 0.5, radiusY: 0.5)
            ],
            normalizedWidth: 0.8,
            isUppercase: true
        )

        // P - Vertical + upper bowl
        t["P"] = GlyphTemplate(
            character: "P",
            strokes: [
                .line(from: CGPoint(x: 0, y: 0), to: CGPoint(x: 0, y: 1)),
                StrokePath(segments: [
                    .line(start: CGPoint(x: 0, y: 1), end: CGPoint(x: 0.5, y: 1)),
                    .cubicCurve(
                        start: CGPoint(x: 0.5, y: 1),
                        control1: CGPoint(x: 0.95, y: 1),
                        control2: CGPoint(x: 0.95, y: 0.5),
                        end: CGPoint(x: 0.5, y: 0.5)
                    ),
                    .line(start: CGPoint(x: 0.5, y: 0.5), end: CGPoint(x: 0, y: 0.5))
                ], isClosed: true)
            ],
            normalizedWidth: 0.7,
            isUppercase: true
        )

        // Q - O with tail
        t["Q"] = GlyphTemplate(
            character: "Q",
            strokes: [
                .ellipse(center: CGPoint(x: 0.5, y: 0.5), radiusX: 0.5, radiusY: 0.5),
                .line(from: CGPoint(x: 0.6, y: 0.3), to: CGPoint(x: 1.0, y: -0.1))
            ],
            normalizedWidth: 0.85,
            isUppercase: true
        )

        // R - P with diagonal leg
        t["R"] = GlyphTemplate(
            character: "R",
            strokes: [
                .line(from: CGPoint(x: 0, y: 0), to: CGPoint(x: 0, y: 1)),
                StrokePath(segments: [
                    .line(start: CGPoint(x: 0, y: 1), end: CGPoint(x: 0.5, y: 1)),
                    .cubicCurve(
                        start: CGPoint(x: 0.5, y: 1),
                        control1: CGPoint(x: 0.9, y: 1),
                        control2: CGPoint(x: 0.9, y: 0.5),
                        end: CGPoint(x: 0.5, y: 0.5)
                    ),
                    .line(start: CGPoint(x: 0.5, y: 0.5), end: CGPoint(x: 0, y: 0.5))
                ], isClosed: true),
                .line(from: CGPoint(x: 0.4, y: 0.5), to: CGPoint(x: 0.9, y: 0))
            ],
            normalizedWidth: 0.7,
            isUppercase: true
        )

        // S - Double curve
        t["S"] = GlyphTemplate(
            character: "S",
            strokes: [
                StrokePath(segments: [
                    .cubicCurve(
                        start: CGPoint(x: 0.85, y: 0.85),
                        control1: CGPoint(x: 0.6, y: 1.05),
                        control2: CGPoint(x: 0.1, y: 0.95),
                        end: CGPoint(x: 0.1, y: 0.75)
                    ),
                    .cubicCurve(
                        start: CGPoint(x: 0.1, y: 0.75),
                        control1: CGPoint(x: 0.1, y: 0.55),
                        control2: CGPoint(x: 0.9, y: 0.55),
                        end: CGPoint(x: 0.9, y: 0.3)
                    ),
                    .cubicCurve(
                        start: CGPoint(x: 0.9, y: 0.3),
                        control1: CGPoint(x: 0.9, y: 0.05),
                        control2: CGPoint(x: 0.4, y: -0.05),
                        end: CGPoint(x: 0.15, y: 0.15)
                    )
                ])
            ],
            normalizedWidth: 0.65,
            isUppercase: true
        )

        // T - Horizontal top + vertical
        t["T"] = GlyphTemplate(
            character: "T",
            strokes: [
                .line(from: CGPoint(x: 0, y: 1), to: CGPoint(x: 1, y: 1)),
                .line(from: CGPoint(x: 0.5, y: 1), to: CGPoint(x: 0.5, y: 0))
            ],
            normalizedWidth: 0.7,
            isUppercase: true
        )

        // U - Two verticals + bottom curve
        t["U"] = GlyphTemplate(
            character: "U",
            strokes: [
                StrokePath(segments: [
                    .line(start: CGPoint(x: 0, y: 1), end: CGPoint(x: 0, y: 0.3)),
                    .cubicCurve(
                        start: CGPoint(x: 0, y: 0.3),
                        control1: CGPoint(x: 0, y: -0.05),
                        control2: CGPoint(x: 1, y: -0.05),
                        end: CGPoint(x: 1, y: 0.3)
                    ),
                    .line(start: CGPoint(x: 1, y: 0.3), end: CGPoint(x: 1, y: 1))
                ])
            ],
            normalizedWidth: 0.75,
            isUppercase: true
        )

        // V - Two diagonals meeting at bottom
        t["V"] = GlyphTemplate(
            character: "V",
            strokes: [
                .line(from: CGPoint(x: 0, y: 1), to: CGPoint(x: 0.5, y: 0)),
                .line(from: CGPoint(x: 0.5, y: 0), to: CGPoint(x: 1, y: 1))
            ],
            normalizedWidth: 0.75,
            isUppercase: true
        )

        // W - Four diagonals
        t["W"] = GlyphTemplate(
            character: "W",
            strokes: [
                .line(from: CGPoint(x: 0, y: 1), to: CGPoint(x: 0.25, y: 0)),
                .line(from: CGPoint(x: 0.25, y: 0), to: CGPoint(x: 0.5, y: 0.6)),
                .line(from: CGPoint(x: 0.5, y: 0.6), to: CGPoint(x: 0.75, y: 0)),
                .line(from: CGPoint(x: 0.75, y: 0), to: CGPoint(x: 1, y: 1))
            ],
            normalizedWidth: 1.0,
            isUppercase: true
        )

        // X - Two diagonals crossing
        t["X"] = GlyphTemplate(
            character: "X",
            strokes: [
                .line(from: CGPoint(x: 0, y: 0), to: CGPoint(x: 1, y: 1)),
                .line(from: CGPoint(x: 0, y: 1), to: CGPoint(x: 1, y: 0))
            ],
            normalizedWidth: 0.7,
            isUppercase: true
        )

        // Y - Two diagonals meeting + vertical down
        t["Y"] = GlyphTemplate(
            character: "Y",
            strokes: [
                .line(from: CGPoint(x: 0, y: 1), to: CGPoint(x: 0.5, y: 0.5)),
                .line(from: CGPoint(x: 1, y: 1), to: CGPoint(x: 0.5, y: 0.5)),
                .line(from: CGPoint(x: 0.5, y: 0.5), to: CGPoint(x: 0.5, y: 0))
            ],
            normalizedWidth: 0.7,
            isUppercase: true
        )

        // Z - Horizontal + diagonal + horizontal
        t["Z"] = GlyphTemplate(
            character: "Z",
            strokes: [
                .line(from: CGPoint(x: 0, y: 1), to: CGPoint(x: 1, y: 1)),
                .line(from: CGPoint(x: 1, y: 1), to: CGPoint(x: 0, y: 0)),
                .line(from: CGPoint(x: 0, y: 0), to: CGPoint(x: 1, y: 0))
            ],
            normalizedWidth: 0.7,
            isUppercase: true
        )
    }

    // MARK: - Lowercase Letters

    private static func buildLowercaseTemplates(_ t: inout [Character: GlyphTemplate]) {
        // a - Bowl with stem on right
        t["a"] = GlyphTemplate(
            character: "a",
            strokes: [
                .ellipse(center: CGPoint(x: 0.45, y: 0.45), radiusX: 0.45, radiusY: 0.45),
                .line(from: CGPoint(x: 0.9, y: 0), to: CGPoint(x: 0.9, y: 0.9))
            ],
            normalizedWidth: 0.6,
            isUppercase: false
        )

        // b - Ascender stem + bowl on right
        t["b"] = GlyphTemplate(
            character: "b",
            strokes: [
                .line(from: CGPoint(x: 0.1, y: 0), to: CGPoint(x: 0.1, y: 1.4)),
                StrokePath(segments: [
                    .line(start: CGPoint(x: 0.1, y: 0.9), end: CGPoint(x: 0.5, y: 0.9)),
                    .cubicCurve(
                        start: CGPoint(x: 0.5, y: 0.9),
                        control1: CGPoint(x: 0.95, y: 0.9),
                        control2: CGPoint(x: 0.95, y: 0),
                        end: CGPoint(x: 0.5, y: 0)
                    ),
                    .line(start: CGPoint(x: 0.5, y: 0), end: CGPoint(x: 0.1, y: 0))
                ], isClosed: true)
            ],
            normalizedWidth: 0.6,
            isUppercase: false,
            heightMultiplier: 1.4
        )

        // c - Open bowl (like small C)
        t["c"] = GlyphTemplate(
            character: "c",
            strokes: [
                StrokePath(segments: [
                    .cubicCurve(
                        start: CGPoint(x: 0.85, y: 0.8),
                        control1: CGPoint(x: 0.6, y: 1.0),
                        control2: CGPoint(x: 0.2, y: 1.0),
                        end: CGPoint(x: 0.1, y: 0.65)
                    ),
                    .cubicCurve(
                        start: CGPoint(x: 0.1, y: 0.65),
                        control1: CGPoint(x: -0.05, y: 0.35),
                        control2: CGPoint(x: 0.2, y: 0),
                        end: CGPoint(x: 0.5, y: 0)
                    ),
                    .cubicCurve(
                        start: CGPoint(x: 0.5, y: 0),
                        control1: CGPoint(x: 0.7, y: 0),
                        control2: CGPoint(x: 0.85, y: 0.1),
                        end: CGPoint(x: 0.85, y: 0.2)
                    )
                ])
            ],
            normalizedWidth: 0.55,
            isUppercase: false
        )

        // d - Bowl + ascender stem on right
        t["d"] = GlyphTemplate(
            character: "d",
            strokes: [
                StrokePath(segments: [
                    .line(start: CGPoint(x: 0.9, y: 0), end: CGPoint(x: 0.5, y: 0)),
                    .cubicCurve(
                        start: CGPoint(x: 0.5, y: 0),
                        control1: CGPoint(x: 0.05, y: 0),
                        control2: CGPoint(x: 0.05, y: 0.9),
                        end: CGPoint(x: 0.5, y: 0.9)
                    ),
                    .line(start: CGPoint(x: 0.5, y: 0.9), end: CGPoint(x: 0.9, y: 0.9))
                ], isClosed: true),
                .line(from: CGPoint(x: 0.9, y: 0), to: CGPoint(x: 0.9, y: 1.4))
            ],
            normalizedWidth: 0.6,
            isUppercase: false,
            heightMultiplier: 1.4
        )

        // e - Bowl with crossbar and opening
        t["e"] = GlyphTemplate(
            character: "e",
            strokes: [
                StrokePath(segments: [
                    .line(start: CGPoint(x: 0.1, y: 0.45), end: CGPoint(x: 0.9, y: 0.45)),
                    .cubicCurve(
                        start: CGPoint(x: 0.9, y: 0.45),
                        control1: CGPoint(x: 0.9, y: 0.9),
                        control2: CGPoint(x: 0.5, y: 0.95),
                        end: CGPoint(x: 0.2, y: 0.8)
                    ),
                    .cubicCurve(
                        start: CGPoint(x: 0.2, y: 0.8),
                        control1: CGPoint(x: -0.05, y: 0.6),
                        control2: CGPoint(x: -0.05, y: 0.3),
                        end: CGPoint(x: 0.2, y: 0.1)
                    ),
                    .cubicCurve(
                        start: CGPoint(x: 0.2, y: 0.1),
                        control1: CGPoint(x: 0.5, y: -0.05),
                        control2: CGPoint(x: 0.85, y: 0.1),
                        end: CGPoint(x: 0.85, y: 0.2)
                    )
                ])
            ],
            normalizedWidth: 0.55,
            isUppercase: false
        )

        // f - Ascender hook + crossbar
        t["f"] = GlyphTemplate(
            character: "f",
            strokes: [
                StrokePath(segments: [
                    .line(start: CGPoint(x: 0.3, y: 0), end: CGPoint(x: 0.3, y: 1.2)),
                    .cubicCurve(
                        start: CGPoint(x: 0.3, y: 1.2),
                        control1: CGPoint(x: 0.3, y: 1.4),
                        control2: CGPoint(x: 0.6, y: 1.4),
                        end: CGPoint(x: 0.8, y: 1.3)
                    )
                ]),
                .line(from: CGPoint(x: 0, y: 0.9), to: CGPoint(x: 0.65, y: 0.9))
            ],
            normalizedWidth: 0.4,
            isUppercase: false,
            heightMultiplier: 1.4
        )

        // g - Bowl + descender
        t["g"] = GlyphTemplate(
            character: "g",
            strokes: [
                .ellipse(center: CGPoint(x: 0.45, y: 0.5), radiusX: 0.4, radiusY: 0.4),
                StrokePath(segments: [
                    .line(start: CGPoint(x: 0.85, y: 0.9), end: CGPoint(x: 0.85, y: -0.15)),
                    .cubicCurve(
                        start: CGPoint(x: 0.85, y: -0.15),
                        control1: CGPoint(x: 0.85, y: -0.45),
                        control2: CGPoint(x: 0.3, y: -0.45),
                        end: CGPoint(x: 0.15, y: -0.3)
                    )
                ])
            ],
            normalizedWidth: 0.6,
            isUppercase: false,
            baselineOffset: -0.45,
            heightMultiplier: 1.0
        )

        // h - Ascender stem + shoulder
        t["h"] = GlyphTemplate(
            character: "h",
            strokes: [
                .line(from: CGPoint(x: 0.1, y: 0), to: CGPoint(x: 0.1, y: 1.4)),
                StrokePath(segments: [
                    .line(start: CGPoint(x: 0.1, y: 0.65), end: CGPoint(x: 0.3, y: 0.65)),
                    .cubicCurve(
                        start: CGPoint(x: 0.3, y: 0.65),
                        control1: CGPoint(x: 0.7, y: 0.95),
                        control2: CGPoint(x: 0.9, y: 0.85),
                        end: CGPoint(x: 0.9, y: 0.55)
                    ),
                    .line(start: CGPoint(x: 0.9, y: 0.55), end: CGPoint(x: 0.9, y: 0))
                ])
            ],
            normalizedWidth: 0.6,
            isUppercase: false,
            heightMultiplier: 1.4
        )

        // i - Short vertical with dot
        t["i"] = GlyphTemplate(
            character: "i",
            strokes: [
                .line(from: CGPoint(x: 0.5, y: 0), to: CGPoint(x: 0.5, y: 0.9)),
                .ellipse(center: CGPoint(x: 0.5, y: 1.2), radiusX: 0.12, radiusY: 0.12)
            ],
            normalizedWidth: 0.25,
            isUppercase: false,
            heightMultiplier: 1.35
        )

        // j - Descender hook with dot
        t["j"] = GlyphTemplate(
            character: "j",
            strokes: [
                StrokePath(segments: [
                    .line(start: CGPoint(x: 0.6, y: 0.9), end: CGPoint(x: 0.6, y: -0.15)),
                    .cubicCurve(
                        start: CGPoint(x: 0.6, y: -0.15),
                        control1: CGPoint(x: 0.6, y: -0.45),
                        control2: CGPoint(x: 0.2, y: -0.45),
                        end: CGPoint(x: 0.1, y: -0.3)
                    )
                ]),
                .ellipse(center: CGPoint(x: 0.6, y: 1.2), radiusX: 0.12, radiusY: 0.12)
            ],
            normalizedWidth: 0.35,
            isUppercase: false,
            baselineOffset: -0.45,
            heightMultiplier: 1.35
        )

        // k - Ascender stem + two diagonals
        t["k"] = GlyphTemplate(
            character: "k",
            strokes: [
                .line(from: CGPoint(x: 0.1, y: 0), to: CGPoint(x: 0.1, y: 1.4)),
                .line(from: CGPoint(x: 0.1, y: 0.4), to: CGPoint(x: 0.85, y: 0.9)),
                .line(from: CGPoint(x: 0.4, y: 0.55), to: CGPoint(x: 0.85, y: 0))
            ],
            normalizedWidth: 0.55,
            isUppercase: false,
            heightMultiplier: 1.4
        )

        // l - Simple ascender stroke
        t["l"] = GlyphTemplate(
            character: "l",
            strokes: [
                .line(from: CGPoint(x: 0.5, y: 0), to: CGPoint(x: 0.5, y: 1.4))
            ],
            normalizedWidth: 0.25,
            isUppercase: false,
            heightMultiplier: 1.4
        )

        // m - Two arches
        t["m"] = GlyphTemplate(
            character: "m",
            strokes: [
                .line(from: CGPoint(x: 0.08, y: 0), to: CGPoint(x: 0.08, y: 0.9)),
                StrokePath(segments: [
                    .line(start: CGPoint(x: 0.08, y: 0.65), end: CGPoint(x: 0.2, y: 0.65)),
                    .cubicCurve(
                        start: CGPoint(x: 0.2, y: 0.65),
                        control1: CGPoint(x: 0.4, y: 0.95),
                        control2: CGPoint(x: 0.5, y: 0.85),
                        end: CGPoint(x: 0.5, y: 0.55)
                    ),
                    .line(start: CGPoint(x: 0.5, y: 0.55), end: CGPoint(x: 0.5, y: 0))
                ]),
                StrokePath(segments: [
                    .line(start: CGPoint(x: 0.5, y: 0.65), end: CGPoint(x: 0.62, y: 0.65)),
                    .cubicCurve(
                        start: CGPoint(x: 0.62, y: 0.65),
                        control1: CGPoint(x: 0.82, y: 0.95),
                        control2: CGPoint(x: 0.92, y: 0.85),
                        end: CGPoint(x: 0.92, y: 0.55)
                    ),
                    .line(start: CGPoint(x: 0.92, y: 0.55), end: CGPoint(x: 0.92, y: 0))
                ])
            ],
            normalizedWidth: 0.85,
            isUppercase: false
        )

        // n - Single arch
        t["n"] = GlyphTemplate(
            character: "n",
            strokes: [
                .line(from: CGPoint(x: 0.1, y: 0), to: CGPoint(x: 0.1, y: 0.9)),
                StrokePath(segments: [
                    .line(start: CGPoint(x: 0.1, y: 0.65), end: CGPoint(x: 0.3, y: 0.65)),
                    .cubicCurve(
                        start: CGPoint(x: 0.3, y: 0.65),
                        control1: CGPoint(x: 0.7, y: 0.95),
                        control2: CGPoint(x: 0.9, y: 0.85),
                        end: CGPoint(x: 0.9, y: 0.55)
                    ),
                    .line(start: CGPoint(x: 0.9, y: 0.55), end: CGPoint(x: 0.9, y: 0))
                ])
            ],
            normalizedWidth: 0.6,
            isUppercase: false
        )

        // o - Simple oval
        t["o"] = GlyphTemplate(
            character: "o",
            strokes: [
                .ellipse(center: CGPoint(x: 0.5, y: 0.45), radiusX: 0.45, radiusY: 0.45)
            ],
            normalizedWidth: 0.6,
            isUppercase: false
        )

        // p - Bowl + descender stem
        t["p"] = GlyphTemplate(
            character: "p",
            strokes: [
                .line(from: CGPoint(x: 0.1, y: -0.45), to: CGPoint(x: 0.1, y: 0.9)),
                StrokePath(segments: [
                    .line(start: CGPoint(x: 0.1, y: 0.9), end: CGPoint(x: 0.5, y: 0.9)),
                    .cubicCurve(
                        start: CGPoint(x: 0.5, y: 0.9),
                        control1: CGPoint(x: 0.95, y: 0.9),
                        control2: CGPoint(x: 0.95, y: 0),
                        end: CGPoint(x: 0.5, y: 0)
                    ),
                    .line(start: CGPoint(x: 0.5, y: 0), end: CGPoint(x: 0.1, y: 0))
                ], isClosed: true)
            ],
            normalizedWidth: 0.6,
            isUppercase: false,
            baselineOffset: -0.45
        )

        // q - Bowl + descender stem on right
        t["q"] = GlyphTemplate(
            character: "q",
            strokes: [
                StrokePath(segments: [
                    .line(start: CGPoint(x: 0.9, y: 0), end: CGPoint(x: 0.5, y: 0)),
                    .cubicCurve(
                        start: CGPoint(x: 0.5, y: 0),
                        control1: CGPoint(x: 0.05, y: 0),
                        control2: CGPoint(x: 0.05, y: 0.9),
                        end: CGPoint(x: 0.5, y: 0.9)
                    ),
                    .line(start: CGPoint(x: 0.5, y: 0.9), end: CGPoint(x: 0.9, y: 0.9))
                ], isClosed: true),
                .line(from: CGPoint(x: 0.9, y: -0.45), to: CGPoint(x: 0.9, y: 0.9))
            ],
            normalizedWidth: 0.6,
            isUppercase: false,
            baselineOffset: -0.45
        )

        // r - Stem + shoulder beginning
        t["r"] = GlyphTemplate(
            character: "r",
            strokes: [
                .line(from: CGPoint(x: 0.15, y: 0), to: CGPoint(x: 0.15, y: 0.9)),
                StrokePath(segments: [
                    .line(start: CGPoint(x: 0.15, y: 0.6), end: CGPoint(x: 0.3, y: 0.6)),
                    .cubicCurve(
                        start: CGPoint(x: 0.3, y: 0.6),
                        control1: CGPoint(x: 0.6, y: 0.95),
                        control2: CGPoint(x: 0.85, y: 0.85),
                        end: CGPoint(x: 0.85, y: 0.7)
                    )
                ])
            ],
            normalizedWidth: 0.45,
            isUppercase: false
        )

        // s - Small double curve
        t["s"] = GlyphTemplate(
            character: "s",
            strokes: [
                StrokePath(segments: [
                    .cubicCurve(
                        start: CGPoint(x: 0.8, y: 0.75),
                        control1: CGPoint(x: 0.55, y: 0.95),
                        control2: CGPoint(x: 0.1, y: 0.85),
                        end: CGPoint(x: 0.1, y: 0.65)
                    ),
                    .cubicCurve(
                        start: CGPoint(x: 0.1, y: 0.65),
                        control1: CGPoint(x: 0.1, y: 0.5),
                        control2: CGPoint(x: 0.85, y: 0.45),
                        end: CGPoint(x: 0.85, y: 0.25)
                    ),
                    .cubicCurve(
                        start: CGPoint(x: 0.85, y: 0.25),
                        control1: CGPoint(x: 0.85, y: 0.05),
                        control2: CGPoint(x: 0.45, y: -0.02),
                        end: CGPoint(x: 0.15, y: 0.15)
                    )
                ])
            ],
            normalizedWidth: 0.5,
            isUppercase: false
        )

        // t - Cross with stem
        t["t"] = GlyphTemplate(
            character: "t",
            strokes: [
                StrokePath(segments: [
                    .line(start: CGPoint(x: 0.35, y: 1.1), end: CGPoint(x: 0.35, y: 0.15)),
                    .cubicCurve(
                        start: CGPoint(x: 0.35, y: 0.15),
                        control1: CGPoint(x: 0.35, y: -0.02),
                        control2: CGPoint(x: 0.55, y: 0),
                        end: CGPoint(x: 0.75, y: 0.08)
                    )
                ]),
                .line(from: CGPoint(x: 0.05, y: 0.85), to: CGPoint(x: 0.7, y: 0.85))
            ],
            normalizedWidth: 0.4,
            isUppercase: false,
            heightMultiplier: 1.1
        )

        // u - Inverted n
        t["u"] = GlyphTemplate(
            character: "u",
            strokes: [
                StrokePath(segments: [
                    .line(start: CGPoint(x: 0.1, y: 0.9), end: CGPoint(x: 0.1, y: 0.35)),
                    .cubicCurve(
                        start: CGPoint(x: 0.1, y: 0.35),
                        control1: CGPoint(x: 0.1, y: 0.05),
                        control2: CGPoint(x: 0.4, y: 0),
                        end: CGPoint(x: 0.7, y: 0.15)
                    ),
                    .line(start: CGPoint(x: 0.7, y: 0.15), end: CGPoint(x: 0.9, y: 0.15))
                ]),
                .line(from: CGPoint(x: 0.9, y: 0), to: CGPoint(x: 0.9, y: 0.9))
            ],
            normalizedWidth: 0.6,
            isUppercase: false
        )

        // v - Two diagonals
        t["v"] = GlyphTemplate(
            character: "v",
            strokes: [
                .line(from: CGPoint(x: 0, y: 0.9), to: CGPoint(x: 0.5, y: 0)),
                .line(from: CGPoint(x: 0.5, y: 0), to: CGPoint(x: 1, y: 0.9))
            ],
            normalizedWidth: 0.55,
            isUppercase: false
        )

        // w - Four diagonals
        t["w"] = GlyphTemplate(
            character: "w",
            strokes: [
                .line(from: CGPoint(x: 0, y: 0.9), to: CGPoint(x: 0.2, y: 0)),
                .line(from: CGPoint(x: 0.2, y: 0), to: CGPoint(x: 0.5, y: 0.55)),
                .line(from: CGPoint(x: 0.5, y: 0.55), to: CGPoint(x: 0.8, y: 0)),
                .line(from: CGPoint(x: 0.8, y: 0), to: CGPoint(x: 1, y: 0.9))
            ],
            normalizedWidth: 0.8,
            isUppercase: false
        )

        // x - Two diagonals crossing
        t["x"] = GlyphTemplate(
            character: "x",
            strokes: [
                .line(from: CGPoint(x: 0, y: 0), to: CGPoint(x: 1, y: 0.9)),
                .line(from: CGPoint(x: 0, y: 0.9), to: CGPoint(x: 1, y: 0))
            ],
            normalizedWidth: 0.55,
            isUppercase: false
        )

        // y - v + descender
        t["y"] = GlyphTemplate(
            character: "y",
            strokes: [
                .line(from: CGPoint(x: 0, y: 0.9), to: CGPoint(x: 0.5, y: 0.1)),
                StrokePath(segments: [
                    .line(start: CGPoint(x: 1, y: 0.9), end: CGPoint(x: 0.5, y: 0.1)),
                    .cubicCurve(
                        start: CGPoint(x: 0.5, y: 0.1),
                        control1: CGPoint(x: 0.3, y: -0.25),
                        control2: CGPoint(x: 0.1, y: -0.4),
                        end: CGPoint(x: 0, y: -0.35)
                    )
                ])
            ],
            normalizedWidth: 0.55,
            isUppercase: false,
            baselineOffset: -0.4
        )

        // z - Three strokes
        t["z"] = GlyphTemplate(
            character: "z",
            strokes: [
                .line(from: CGPoint(x: 0, y: 0.9), to: CGPoint(x: 1, y: 0.9)),
                .line(from: CGPoint(x: 1, y: 0.9), to: CGPoint(x: 0, y: 0)),
                .line(from: CGPoint(x: 0, y: 0), to: CGPoint(x: 1, y: 0))
            ],
            normalizedWidth: 0.5,
            isUppercase: false
        )
    }

    // MARK: - Digits

    private static func buildDigitTemplates(_ t: inout [Character: GlyphTemplate]) {
        // 0 - Oval
        t["0"] = GlyphTemplate(
            character: "0",
            strokes: [
                .ellipse(center: CGPoint(x: 0.5, y: 0.5), radiusX: 0.45, radiusY: 0.5)
            ],
            normalizedWidth: 0.65,
            isUppercase: true
        )

        // 1 - Vertical with optional serif
        t["1"] = GlyphTemplate(
            character: "1",
            strokes: [
                .line(from: CGPoint(x: 0.5, y: 0), to: CGPoint(x: 0.5, y: 1)),
                .line(from: CGPoint(x: 0.5, y: 1), to: CGPoint(x: 0.25, y: 0.75))
            ],
            normalizedWidth: 0.4,
            isUppercase: true
        )

        // 2 - Curved top + diagonal + base
        t["2"] = GlyphTemplate(
            character: "2",
            strokes: [
                StrokePath(segments: [
                    .cubicCurve(
                        start: CGPoint(x: 0.1, y: 0.75),
                        control1: CGPoint(x: 0.1, y: 1.05),
                        control2: CGPoint(x: 0.9, y: 1.05),
                        end: CGPoint(x: 0.9, y: 0.7)
                    ),
                    .cubicCurve(
                        start: CGPoint(x: 0.9, y: 0.7),
                        control1: CGPoint(x: 0.9, y: 0.45),
                        control2: CGPoint(x: 0.1, y: 0.15),
                        end: CGPoint(x: 0.1, y: 0)
                    ),
                    .line(start: CGPoint(x: 0.1, y: 0), end: CGPoint(x: 0.9, y: 0))
                ])
            ],
            normalizedWidth: 0.6,
            isUppercase: true
        )

        // 3 - Two stacked curves
        t["3"] = GlyphTemplate(
            character: "3",
            strokes: [
                StrokePath(segments: [
                    .cubicCurve(
                        start: CGPoint(x: 0.15, y: 0.85),
                        control1: CGPoint(x: 0.35, y: 1.05),
                        control2: CGPoint(x: 0.85, y: 1.0),
                        end: CGPoint(x: 0.85, y: 0.75)
                    ),
                    .cubicCurve(
                        start: CGPoint(x: 0.85, y: 0.75),
                        control1: CGPoint(x: 0.85, y: 0.55),
                        control2: CGPoint(x: 0.55, y: 0.5),
                        end: CGPoint(x: 0.4, y: 0.5)
                    ),
                    .cubicCurve(
                        start: CGPoint(x: 0.4, y: 0.5),
                        control1: CGPoint(x: 0.6, y: 0.5),
                        control2: CGPoint(x: 0.9, y: 0.4),
                        end: CGPoint(x: 0.9, y: 0.2)
                    ),
                    .cubicCurve(
                        start: CGPoint(x: 0.9, y: 0.2),
                        control1: CGPoint(x: 0.9, y: -0.05),
                        control2: CGPoint(x: 0.4, y: -0.05),
                        end: CGPoint(x: 0.1, y: 0.15)
                    )
                ])
            ],
            normalizedWidth: 0.6,
            isUppercase: true
        )

        // 4 - Vertical + horizontal + diagonal
        t["4"] = GlyphTemplate(
            character: "4",
            strokes: [
                .line(from: CGPoint(x: 0.7, y: 0), to: CGPoint(x: 0.7, y: 1)),
                .line(from: CGPoint(x: 0.0, y: 0.35), to: CGPoint(x: 0.95, y: 0.35)),
                .line(from: CGPoint(x: 0.0, y: 0.35), to: CGPoint(x: 0.7, y: 1))
            ],
            normalizedWidth: 0.65,
            isUppercase: true
        )

        // 5 - Top bar + upper curve + lower curve
        t["5"] = GlyphTemplate(
            character: "5",
            strokes: [
                StrokePath(segments: [
                    .line(start: CGPoint(x: 0.85, y: 1), end: CGPoint(x: 0.15, y: 1)),
                    .line(start: CGPoint(x: 0.15, y: 1), end: CGPoint(x: 0.1, y: 0.55)),
                    .cubicCurve(
                        start: CGPoint(x: 0.1, y: 0.55),
                        control1: CGPoint(x: 0.3, y: 0.65),
                        control2: CGPoint(x: 0.9, y: 0.6),
                        end: CGPoint(x: 0.9, y: 0.3)
                    ),
                    .cubicCurve(
                        start: CGPoint(x: 0.9, y: 0.3),
                        control1: CGPoint(x: 0.9, y: -0.05),
                        control2: CGPoint(x: 0.35, y: -0.05),
                        end: CGPoint(x: 0.1, y: 0.15)
                    )
                ])
            ],
            normalizedWidth: 0.6,
            isUppercase: true
        )

        // 6 - Hook top + bowl
        t["6"] = GlyphTemplate(
            character: "6",
            strokes: [
                StrokePath(segments: [
                    .cubicCurve(
                        start: CGPoint(x: 0.8, y: 0.9),
                        control1: CGPoint(x: 0.5, y: 1.05),
                        control2: CGPoint(x: 0.1, y: 0.75),
                        end: CGPoint(x: 0.1, y: 0.4)
                    ),
                    .cubicCurve(
                        start: CGPoint(x: 0.1, y: 0.4),
                        control1: CGPoint(x: 0.1, y: 0.05),
                        control2: CGPoint(x: 0.35, y: -0.05),
                        end: CGPoint(x: 0.5, y: -0.02)
                    ),
                    .cubicCurve(
                        start: CGPoint(x: 0.5, y: -0.02),
                        control1: CGPoint(x: 0.85, y: -0.02),
                        control2: CGPoint(x: 0.9, y: 0.25),
                        end: CGPoint(x: 0.9, y: 0.35)
                    ),
                    .cubicCurve(
                        start: CGPoint(x: 0.9, y: 0.35),
                        control1: CGPoint(x: 0.9, y: 0.6),
                        control2: CGPoint(x: 0.6, y: 0.7),
                        end: CGPoint(x: 0.4, y: 0.6)
                    ),
                    .cubicCurve(
                        start: CGPoint(x: 0.4, y: 0.6),
                        control1: CGPoint(x: 0.15, y: 0.5),
                        control2: CGPoint(x: 0.1, y: 0.45),
                        end: CGPoint(x: 0.1, y: 0.4)
                    )
                ])
            ],
            normalizedWidth: 0.6,
            isUppercase: true
        )

        // 7 - Top bar + diagonal
        t["7"] = GlyphTemplate(
            character: "7",
            strokes: [
                .line(from: CGPoint(x: 0.05, y: 1), to: CGPoint(x: 0.95, y: 1)),
                .line(from: CGPoint(x: 0.95, y: 1), to: CGPoint(x: 0.3, y: 0))
            ],
            normalizedWidth: 0.6,
            isUppercase: true
        )

        // 8 - Two stacked bowls
        t["8"] = GlyphTemplate(
            character: "8",
            strokes: [
                // Upper bowl (smaller)
                .ellipse(center: CGPoint(x: 0.5, y: 0.73), radiusX: 0.35, radiusY: 0.27),
                // Lower bowl (larger)
                .ellipse(center: CGPoint(x: 0.5, y: 0.27), radiusX: 0.4, radiusY: 0.27)
            ],
            normalizedWidth: 0.6,
            isUppercase: true
        )

        // 9 - Inverted 6
        t["9"] = GlyphTemplate(
            character: "9",
            strokes: [
                StrokePath(segments: [
                    .cubicCurve(
                        start: CGPoint(x: 0.2, y: 0.1),
                        control1: CGPoint(x: 0.5, y: -0.05),
                        control2: CGPoint(x: 0.9, y: 0.25),
                        end: CGPoint(x: 0.9, y: 0.6)
                    ),
                    .cubicCurve(
                        start: CGPoint(x: 0.9, y: 0.6),
                        control1: CGPoint(x: 0.9, y: 0.95),
                        control2: CGPoint(x: 0.65, y: 1.05),
                        end: CGPoint(x: 0.5, y: 1.02)
                    ),
                    .cubicCurve(
                        start: CGPoint(x: 0.5, y: 1.02),
                        control1: CGPoint(x: 0.15, y: 1.02),
                        control2: CGPoint(x: 0.1, y: 0.75),
                        end: CGPoint(x: 0.1, y: 0.65)
                    ),
                    .cubicCurve(
                        start: CGPoint(x: 0.1, y: 0.65),
                        control1: CGPoint(x: 0.1, y: 0.4),
                        control2: CGPoint(x: 0.4, y: 0.3),
                        end: CGPoint(x: 0.6, y: 0.4)
                    ),
                    .cubicCurve(
                        start: CGPoint(x: 0.6, y: 0.4),
                        control1: CGPoint(x: 0.85, y: 0.5),
                        control2: CGPoint(x: 0.9, y: 0.55),
                        end: CGPoint(x: 0.9, y: 0.6)
                    )
                ])
            ],
            normalizedWidth: 0.6,
            isUppercase: true
        )
    }

    // MARK: - Punctuation

    private static func buildPunctuationTemplates(_ t: inout [Character: GlyphTemplate]) {
        // Period
        t["."] = GlyphTemplate(
            character: ".",
            strokes: [
                .ellipse(center: CGPoint(x: 0.5, y: 0.1), radiusX: 0.15, radiusY: 0.1)
            ],
            normalizedWidth: 0.25,
            isUppercase: false
        )

        // Comma
        t[","] = GlyphTemplate(
            character: ",",
            strokes: [
                StrokePath(segments: [
                    .cubicCurve(
                        start: CGPoint(x: 0.6, y: 0.15),
                        control1: CGPoint(x: 0.7, y: 0.15),
                        control2: CGPoint(x: 0.7, y: 0.05),
                        end: CGPoint(x: 0.5, y: 0)
                    ),
                    .cubicCurve(
                        start: CGPoint(x: 0.5, y: 0),
                        control1: CGPoint(x: 0.3, y: -0.1),
                        control2: CGPoint(x: 0.25, y: -0.25),
                        end: CGPoint(x: 0.2, y: -0.25)
                    )
                ])
            ],
            normalizedWidth: 0.25,
            isUppercase: false,
            baselineOffset: -0.25
        )

        // Exclamation mark
        t["!"] = GlyphTemplate(
            character: "!",
            strokes: [
                .line(from: CGPoint(x: 0.5, y: 0.35), to: CGPoint(x: 0.5, y: 1)),
                .ellipse(center: CGPoint(x: 0.5, y: 0.1), radiusX: 0.12, radiusY: 0.1)
            ],
            normalizedWidth: 0.25,
            isUppercase: true
        )

        // Question mark
        t["?"] = GlyphTemplate(
            character: "?",
            strokes: [
                StrokePath(segments: [
                    .cubicCurve(
                        start: CGPoint(x: 0.15, y: 0.8),
                        control1: CGPoint(x: 0.15, y: 1.05),
                        control2: CGPoint(x: 0.85, y: 1.05),
                        end: CGPoint(x: 0.85, y: 0.7)
                    ),
                    .cubicCurve(
                        start: CGPoint(x: 0.85, y: 0.7),
                        control1: CGPoint(x: 0.85, y: 0.45),
                        control2: CGPoint(x: 0.5, y: 0.45),
                        end: CGPoint(x: 0.5, y: 0.35)
                    )
                ]),
                .ellipse(center: CGPoint(x: 0.5, y: 0.1), radiusX: 0.1, radiusY: 0.1)
            ],
            normalizedWidth: 0.5,
            isUppercase: true
        )

        // Colon
        t[":"] = GlyphTemplate(
            character: ":",
            strokes: [
                .ellipse(center: CGPoint(x: 0.5, y: 0.65), radiusX: 0.12, radiusY: 0.1),
                .ellipse(center: CGPoint(x: 0.5, y: 0.15), radiusX: 0.12, radiusY: 0.1)
            ],
            normalizedWidth: 0.25,
            isUppercase: false
        )

        // Semicolon
        t[";"] = GlyphTemplate(
            character: ";",
            strokes: [
                .ellipse(center: CGPoint(x: 0.5, y: 0.65), radiusX: 0.12, radiusY: 0.1),
                StrokePath(segments: [
                    .cubicCurve(
                        start: CGPoint(x: 0.6, y: 0.15),
                        control1: CGPoint(x: 0.7, y: 0.15),
                        control2: CGPoint(x: 0.7, y: 0.05),
                        end: CGPoint(x: 0.5, y: 0)
                    ),
                    .cubicCurve(
                        start: CGPoint(x: 0.5, y: 0),
                        control1: CGPoint(x: 0.3, y: -0.1),
                        control2: CGPoint(x: 0.25, y: -0.25),
                        end: CGPoint(x: 0.2, y: -0.25)
                    )
                ])
            ],
            normalizedWidth: 0.25,
            isUppercase: false,
            baselineOffset: -0.25
        )

        // Apostrophe / single quote
        t["'"] = GlyphTemplate(
            character: "'",
            strokes: [
                .line(from: CGPoint(x: 0.5, y: 0.65), to: CGPoint(x: 0.5, y: 1))
            ],
            normalizedWidth: 0.2,
            isUppercase: true
        )

        // Double quote
        t["\""] = GlyphTemplate(
            character: "\"",
            strokes: [
                .line(from: CGPoint(x: 0.3, y: 0.65), to: CGPoint(x: 0.3, y: 1)),
                .line(from: CGPoint(x: 0.7, y: 0.65), to: CGPoint(x: 0.7, y: 1))
            ],
            normalizedWidth: 0.35,
            isUppercase: true
        )

        // Hyphen
        t["-"] = GlyphTemplate(
            character: "-",
            strokes: [
                .line(from: CGPoint(x: 0.1, y: 0.45), to: CGPoint(x: 0.9, y: 0.45))
            ],
            normalizedWidth: 0.4,
            isUppercase: false
        )

        // Parentheses
        t["("] = GlyphTemplate(
            character: "(",
            strokes: [
                StrokePath(segments: [
                    .cubicCurve(
                        start: CGPoint(x: 0.7, y: 1.1),
                        control1: CGPoint(x: 0.1, y: 0.8),
                        control2: CGPoint(x: 0.1, y: 0.1),
                        end: CGPoint(x: 0.7, y: -0.2)
                    )
                ])
            ],
            normalizedWidth: 0.35,
            isUppercase: true,
            baselineOffset: -0.2,
            heightMultiplier: 1.3
        )

        t[")"] = GlyphTemplate(
            character: ")",
            strokes: [
                StrokePath(segments: [
                    .cubicCurve(
                        start: CGPoint(x: 0.3, y: 1.1),
                        control1: CGPoint(x: 0.9, y: 0.8),
                        control2: CGPoint(x: 0.9, y: 0.1),
                        end: CGPoint(x: 0.3, y: -0.2)
                    )
                ])
            ],
            normalizedWidth: 0.35,
            isUppercase: true,
            baselineOffset: -0.2,
            heightMultiplier: 1.3
        )

        // Brackets
        t["["] = GlyphTemplate(
            character: "[",
            strokes: [
                .line(from: CGPoint(x: 0.3, y: -0.15), to: CGPoint(x: 0.3, y: 1.05)),
                .line(from: CGPoint(x: 0.3, y: 1.05), to: CGPoint(x: 0.75, y: 1.05)),
                .line(from: CGPoint(x: 0.3, y: -0.15), to: CGPoint(x: 0.75, y: -0.15))
            ],
            normalizedWidth: 0.35,
            isUppercase: true,
            baselineOffset: -0.15,
            heightMultiplier: 1.2
        )

        t["]"] = GlyphTemplate(
            character: "]",
            strokes: [
                .line(from: CGPoint(x: 0.7, y: -0.15), to: CGPoint(x: 0.7, y: 1.05)),
                .line(from: CGPoint(x: 0.25, y: 1.05), to: CGPoint(x: 0.7, y: 1.05)),
                .line(from: CGPoint(x: 0.25, y: -0.15), to: CGPoint(x: 0.7, y: -0.15))
            ],
            normalizedWidth: 0.35,
            isUppercase: true,
            baselineOffset: -0.15,
            heightMultiplier: 1.2
        )

        // Curly braces
        t["{"] = GlyphTemplate(
            character: "{",
            strokes: [
                StrokePath(segments: [
                    .cubicCurve(
                        start: CGPoint(x: 0.75, y: 1.05),
                        control1: CGPoint(x: 0.45, y: 1.05),
                        control2: CGPoint(x: 0.45, y: 0.8),
                        end: CGPoint(x: 0.45, y: 0.6)
                    ),
                    .cubicCurve(
                        start: CGPoint(x: 0.45, y: 0.6),
                        control1: CGPoint(x: 0.45, y: 0.5),
                        control2: CGPoint(x: 0.15, y: 0.48),
                        end: CGPoint(x: 0.15, y: 0.45)
                    ),
                    .cubicCurve(
                        start: CGPoint(x: 0.15, y: 0.45),
                        control1: CGPoint(x: 0.15, y: 0.42),
                        control2: CGPoint(x: 0.45, y: 0.4),
                        end: CGPoint(x: 0.45, y: 0.3)
                    ),
                    .cubicCurve(
                        start: CGPoint(x: 0.45, y: 0.3),
                        control1: CGPoint(x: 0.45, y: 0.1),
                        control2: CGPoint(x: 0.45, y: -0.15),
                        end: CGPoint(x: 0.75, y: -0.15)
                    )
                ])
            ],
            normalizedWidth: 0.4,
            isUppercase: true,
            baselineOffset: -0.15,
            heightMultiplier: 1.2
        )

        t["}"] = GlyphTemplate(
            character: "}",
            strokes: [
                StrokePath(segments: [
                    .cubicCurve(
                        start: CGPoint(x: 0.25, y: 1.05),
                        control1: CGPoint(x: 0.55, y: 1.05),
                        control2: CGPoint(x: 0.55, y: 0.8),
                        end: CGPoint(x: 0.55, y: 0.6)
                    ),
                    .cubicCurve(
                        start: CGPoint(x: 0.55, y: 0.6),
                        control1: CGPoint(x: 0.55, y: 0.5),
                        control2: CGPoint(x: 0.85, y: 0.48),
                        end: CGPoint(x: 0.85, y: 0.45)
                    ),
                    .cubicCurve(
                        start: CGPoint(x: 0.85, y: 0.45),
                        control1: CGPoint(x: 0.85, y: 0.42),
                        control2: CGPoint(x: 0.55, y: 0.4),
                        end: CGPoint(x: 0.55, y: 0.3)
                    ),
                    .cubicCurve(
                        start: CGPoint(x: 0.55, y: 0.3),
                        control1: CGPoint(x: 0.55, y: 0.1),
                        control2: CGPoint(x: 0.55, y: -0.15),
                        end: CGPoint(x: 0.25, y: -0.15)
                    )
                ])
            ],
            normalizedWidth: 0.4,
            isUppercase: true,
            baselineOffset: -0.15,
            heightMultiplier: 1.2
        )

        // Slash
        t["/"] = GlyphTemplate(
            character: "/",
            strokes: [
                .line(from: CGPoint(x: 0, y: -0.1), to: CGPoint(x: 1, y: 1.1))
            ],
            normalizedWidth: 0.45,
            isUppercase: true,
            baselineOffset: -0.1,
            heightMultiplier: 1.2
        )

        // Backslash
        t["\\"] = GlyphTemplate(
            character: "\\",
            strokes: [
                .line(from: CGPoint(x: 0, y: 1.1), to: CGPoint(x: 1, y: -0.1))
            ],
            normalizedWidth: 0.45,
            isUppercase: true,
            baselineOffset: -0.1,
            heightMultiplier: 1.2
        )

        // At symbol
        t["@"] = GlyphTemplate(
            character: "@",
            strokes: [
                .ellipse(center: CGPoint(x: 0.55, y: 0.45), radiusX: 0.2, radiusY: 0.2),
                StrokePath(segments: [
                    .line(start: CGPoint(x: 0.75, y: 0.25), end: CGPoint(x: 0.75, y: 0.55)),
                    .cubicCurve(
                        start: CGPoint(x: 0.75, y: 0.55),
                        control1: CGPoint(x: 0.75, y: 0.7),
                        control2: CGPoint(x: 0.85, y: 0.68),
                        end: CGPoint(x: 0.9, y: 0.55)
                    ),
                    .cubicCurve(
                        start: CGPoint(x: 0.9, y: 0.55),
                        control1: CGPoint(x: 1.0, y: 0.3),
                        control2: CGPoint(x: 0.85, y: 0),
                        end: CGPoint(x: 0.5, y: 0)
                    ),
                    .cubicCurve(
                        start: CGPoint(x: 0.5, y: 0),
                        control1: CGPoint(x: 0.0, y: 0),
                        control2: CGPoint(x: 0.0, y: 0.9),
                        end: CGPoint(x: 0.5, y: 0.9)
                    ),
                    .cubicCurve(
                        start: CGPoint(x: 0.5, y: 0.9),
                        control1: CGPoint(x: 0.85, y: 0.9),
                        control2: CGPoint(x: 1.0, y: 0.7),
                        end: CGPoint(x: 1.0, y: 0.45)
                    )
                ])
            ],
            normalizedWidth: 0.9,
            isUppercase: true
        )

        // Ampersand
        t["&"] = GlyphTemplate(
            character: "&",
            strokes: [
                StrokePath(segments: [
                    .line(start: CGPoint(x: 0.9, y: 0), end: CGPoint(x: 0.4, y: 0.5)),
                    .cubicCurve(
                        start: CGPoint(x: 0.4, y: 0.5),
                        control1: CGPoint(x: 0.1, y: 0.5),
                        control2: CGPoint(x: 0.0, y: 0.2),
                        end: CGPoint(x: 0.2, y: 0.1)
                    ),
                    .cubicCurve(
                        start: CGPoint(x: 0.2, y: 0.1),
                        control1: CGPoint(x: 0.35, y: 0),
                        control2: CGPoint(x: 0.6, y: 0.05),
                        end: CGPoint(x: 0.65, y: 0.2)
                    ),
                    .cubicCurve(
                        start: CGPoint(x: 0.65, y: 0.2),
                        control1: CGPoint(x: 0.7, y: 0.35),
                        control2: CGPoint(x: 0.55, y: 0.55),
                        end: CGPoint(x: 0.35, y: 0.55)
                    ),
                    .cubicCurve(
                        start: CGPoint(x: 0.35, y: 0.55),
                        control1: CGPoint(x: 0.1, y: 0.55),
                        control2: CGPoint(x: 0.1, y: 0.85),
                        end: CGPoint(x: 0.3, y: 0.95)
                    ),
                    .cubicCurve(
                        start: CGPoint(x: 0.3, y: 0.95),
                        control1: CGPoint(x: 0.55, y: 1.05),
                        control2: CGPoint(x: 0.75, y: 0.85),
                        end: CGPoint(x: 0.9, y: 0.6)
                    )
                ])
            ],
            normalizedWidth: 0.75,
            isUppercase: true
        )

        // Hash/pound
        t["#"] = GlyphTemplate(
            character: "#",
            strokes: [
                .line(from: CGPoint(x: 0.3, y: 0), to: CGPoint(x: 0.4, y: 1)),
                .line(from: CGPoint(x: 0.6, y: 0), to: CGPoint(x: 0.7, y: 1)),
                .line(from: CGPoint(x: 0.05, y: 0.35), to: CGPoint(x: 0.95, y: 0.35)),
                .line(from: CGPoint(x: 0.05, y: 0.65), to: CGPoint(x: 0.95, y: 0.65))
            ],
            normalizedWidth: 0.7,
            isUppercase: true
        )

        // Dollar sign
        t["$"] = GlyphTemplate(
            character: "$",
            strokes: [
                StrokePath(segments: [
                    .cubicCurve(
                        start: CGPoint(x: 0.8, y: 0.8),
                        control1: CGPoint(x: 0.55, y: 0.95),
                        control2: CGPoint(x: 0.15, y: 0.85),
                        end: CGPoint(x: 0.15, y: 0.65)
                    ),
                    .cubicCurve(
                        start: CGPoint(x: 0.15, y: 0.65),
                        control1: CGPoint(x: 0.15, y: 0.5),
                        control2: CGPoint(x: 0.85, y: 0.5),
                        end: CGPoint(x: 0.85, y: 0.3)
                    ),
                    .cubicCurve(
                        start: CGPoint(x: 0.85, y: 0.3),
                        control1: CGPoint(x: 0.85, y: 0.1),
                        control2: CGPoint(x: 0.45, y: 0.05),
                        end: CGPoint(x: 0.2, y: 0.2)
                    )
                ]),
                .line(from: CGPoint(x: 0.5, y: -0.1), to: CGPoint(x: 0.5, y: 1.1))
            ],
            normalizedWidth: 0.6,
            isUppercase: true,
            baselineOffset: -0.1,
            heightMultiplier: 1.2
        )

        // Percent
        t["%"] = GlyphTemplate(
            character: "%",
            strokes: [
                .ellipse(center: CGPoint(x: 0.25, y: 0.75), radiusX: 0.2, radiusY: 0.2),
                .ellipse(center: CGPoint(x: 0.75, y: 0.25), radiusX: 0.2, radiusY: 0.2),
                .line(from: CGPoint(x: 0.1, y: 0), to: CGPoint(x: 0.9, y: 1))
            ],
            normalizedWidth: 0.8,
            isUppercase: true
        )

        // Plus
        t["+"] = GlyphTemplate(
            character: "+",
            strokes: [
                .line(from: CGPoint(x: 0.5, y: 0.15), to: CGPoint(x: 0.5, y: 0.85)),
                .line(from: CGPoint(x: 0.15, y: 0.5), to: CGPoint(x: 0.85, y: 0.5))
            ],
            normalizedWidth: 0.6,
            isUppercase: false
        )

        // Equals
        t["="] = GlyphTemplate(
            character: "=",
            strokes: [
                .line(from: CGPoint(x: 0.1, y: 0.35), to: CGPoint(x: 0.9, y: 0.35)),
                .line(from: CGPoint(x: 0.1, y: 0.6), to: CGPoint(x: 0.9, y: 0.6))
            ],
            normalizedWidth: 0.6,
            isUppercase: false
        )

        // Asterisk
        t["*"] = GlyphTemplate(
            character: "*",
            strokes: [
                .line(from: CGPoint(x: 0.5, y: 0.5), to: CGPoint(x: 0.5, y: 1)),
                .line(from: CGPoint(x: 0.5, y: 0.75), to: CGPoint(x: 0.15, y: 0.55)),
                .line(from: CGPoint(x: 0.5, y: 0.75), to: CGPoint(x: 0.85, y: 0.55)),
                .line(from: CGPoint(x: 0.5, y: 0.75), to: CGPoint(x: 0.2, y: 0.95)),
                .line(from: CGPoint(x: 0.5, y: 0.75), to: CGPoint(x: 0.8, y: 0.95))
            ],
            normalizedWidth: 0.5,
            isUppercase: true
        )

        // Underscore
        t["_"] = GlyphTemplate(
            character: "_",
            strokes: [
                .line(from: CGPoint(x: 0, y: -0.1), to: CGPoint(x: 1, y: -0.1))
            ],
            normalizedWidth: 0.6,
            isUppercase: false,
            baselineOffset: -0.1
        )

        // Caret
        t["^"] = GlyphTemplate(
            character: "^",
            strokes: [
                .line(from: CGPoint(x: 0.1, y: 0.6), to: CGPoint(x: 0.5, y: 1)),
                .line(from: CGPoint(x: 0.5, y: 1), to: CGPoint(x: 0.9, y: 0.6))
            ],
            normalizedWidth: 0.5,
            isUppercase: true
        )

        // Tilde
        t["~"] = GlyphTemplate(
            character: "~",
            strokes: [
                StrokePath(segments: [
                    .cubicCurve(
                        start: CGPoint(x: 0.1, y: 0.45),
                        control1: CGPoint(x: 0.3, y: 0.65),
                        control2: CGPoint(x: 0.4, y: 0.65),
                        end: CGPoint(x: 0.5, y: 0.5)
                    ),
                    .cubicCurve(
                        start: CGPoint(x: 0.5, y: 0.5),
                        control1: CGPoint(x: 0.6, y: 0.35),
                        control2: CGPoint(x: 0.7, y: 0.35),
                        end: CGPoint(x: 0.9, y: 0.55)
                    )
                ])
            ],
            normalizedWidth: 0.55,
            isUppercase: false
        )

        // Backtick
        t["`"] = GlyphTemplate(
            character: "`",
            strokes: [
                .line(from: CGPoint(x: 0.3, y: 1.1), to: CGPoint(x: 0.7, y: 0.8))
            ],
            normalizedWidth: 0.3,
            isUppercase: true
        )

        // Pipe
        t["|"] = GlyphTemplate(
            character: "|",
            strokes: [
                .line(from: CGPoint(x: 0.5, y: -0.15), to: CGPoint(x: 0.5, y: 1.1))
            ],
            normalizedWidth: 0.2,
            isUppercase: true,
            baselineOffset: -0.15,
            heightMultiplier: 1.25
        )

        // Less than / Greater than
        t["<"] = GlyphTemplate(
            character: "<",
            strokes: [
                .line(from: CGPoint(x: 0.85, y: 0.8), to: CGPoint(x: 0.15, y: 0.45)),
                .line(from: CGPoint(x: 0.15, y: 0.45), to: CGPoint(x: 0.85, y: 0.1))
            ],
            normalizedWidth: 0.55,
            isUppercase: false
        )

        t[">"] = GlyphTemplate(
            character: ">",
            strokes: [
                .line(from: CGPoint(x: 0.15, y: 0.8), to: CGPoint(x: 0.85, y: 0.45)),
                .line(from: CGPoint(x: 0.85, y: 0.45), to: CGPoint(x: 0.15, y: 0.1))
            ],
            normalizedWidth: 0.55,
            isUppercase: false
        )

        // Space
        t[" "] = GlyphTemplate(
            character: " ",
            strokes: [],  // No strokes for space
            normalizedWidth: 0.3,
            isUppercase: false
        )
    }
}
