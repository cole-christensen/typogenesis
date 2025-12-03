import Foundation
import CoreGraphics

struct GlyphOutline: Codable, Sendable, Equatable {
    var contours: [Contour]

    init(contours: [Contour] = []) {
        self.contours = contours
    }

    var isEmpty: Bool {
        contours.isEmpty || contours.allSatisfy { $0.points.isEmpty }
    }

    var boundingBox: BoundingBox {
        guard !isEmpty else {
            return BoundingBox(minX: 0, minY: 0, maxX: 0, maxY: 0)
        }

        var minX = CGFloat.infinity
        var minY = CGFloat.infinity
        var maxX = -CGFloat.infinity
        var maxY = -CGFloat.infinity

        for contour in contours {
            for point in contour.points {
                minX = min(minX, point.position.x)
                minY = min(minY, point.position.y)
                maxX = max(maxX, point.position.x)
                maxY = max(maxY, point.position.y)

                if let controlIn = point.controlIn {
                    minX = min(minX, controlIn.x)
                    minY = min(minY, controlIn.y)
                    maxX = max(maxX, controlIn.x)
                    maxY = max(maxY, controlIn.y)
                }

                if let controlOut = point.controlOut {
                    minX = min(minX, controlOut.x)
                    minY = min(minY, controlOut.y)
                    maxX = max(maxX, controlOut.x)
                    maxY = max(maxY, controlOut.y)
                }
            }
        }

        return BoundingBox(
            minX: Int(minX),
            minY: Int(minY),
            maxX: Int(maxX),
            maxY: Int(maxY)
        )
    }

    func toCGPath() -> CGPath {
        let path = CGMutablePath()

        for contour in contours {
            guard !contour.points.isEmpty else { continue }

            let firstPoint = contour.points[0]
            path.move(to: firstPoint.position)

            for i in 1..<contour.points.count {
                let point = contour.points[i]
                let prevPoint = contour.points[i - 1]

                if let controlOut = prevPoint.controlOut, let controlIn = point.controlIn {
                    path.addCurve(to: point.position, control1: controlOut, control2: controlIn)
                } else if let controlOut = prevPoint.controlOut {
                    path.addQuadCurve(to: point.position, control: controlOut)
                } else if let controlIn = point.controlIn {
                    path.addQuadCurve(to: point.position, control: controlIn)
                } else {
                    path.addLine(to: point.position)
                }
            }

            if contour.isClosed {
                let lastPoint = contour.points[contour.points.count - 1]
                if let controlOut = lastPoint.controlOut, let controlIn = firstPoint.controlIn {
                    path.addCurve(to: firstPoint.position, control1: controlOut, control2: controlIn)
                } else if let controlOut = lastPoint.controlOut {
                    path.addQuadCurve(to: firstPoint.position, control: controlOut)
                } else if let controlIn = firstPoint.controlIn {
                    path.addQuadCurve(to: firstPoint.position, control: controlIn)
                } else {
                    path.addLine(to: firstPoint.position)
                }
                path.closeSubpath()
            }
        }

        return path
    }
}

struct Contour: Codable, Sendable, Equatable, Identifiable {
    let id: UUID
    var points: [PathPoint]
    var isClosed: Bool

    init(id: UUID = UUID(), points: [PathPoint] = [], isClosed: Bool = true) {
        self.id = id
        self.points = points
        self.isClosed = isClosed
    }
}

struct PathPoint: Codable, Sendable, Equatable, Identifiable {
    let id: UUID
    var position: CGPoint
    var type: PointType
    var controlIn: CGPoint?
    var controlOut: CGPoint?

    init(
        id: UUID = UUID(),
        position: CGPoint,
        type: PointType = .corner,
        controlIn: CGPoint? = nil,
        controlOut: CGPoint? = nil
    ) {
        self.id = id
        self.position = position
        self.type = type
        self.controlIn = controlIn
        self.controlOut = controlOut
    }

    enum PointType: String, Codable, Sendable {
        case corner
        case smooth
        case symmetric
    }
}

struct BoundingBox: Codable, Sendable, Equatable {
    var minX: Int
    var minY: Int
    var maxX: Int
    var maxY: Int

    var width: Int { maxX - minX }
    var height: Int { maxY - minY }

    var cgRect: CGRect {
        CGRect(x: minX, y: minY, width: width, height: height)
    }
}

extension CGPoint: @retroactive Hashable {
    public func hash(into hasher: inout Hasher) {
        hasher.combine(x)
        hasher.combine(y)
    }
}
