import { useMemo, useState } from 'react'
import DeckGL from '@deck.gl/react'
import { ScatterplotLayer, LineLayer, TextLayer, PolygonLayer } from '@deck.gl/layers'
import { OrthographicView, COORDINATE_SYSTEM } from '@deck.gl/core'
import type { PickingInfo, Color, Layer } from '@deck.gl/core'

// ============================================================================
// Types
// ============================================================================

interface Point2D {
  x: number
  y: number
}

interface MouseData {
  points: number[][]
  labels: string[]
}

interface FrameData {
  frame_number: number
  mice: Record<string, MouseData>
}

interface Metadata {
  total_frames: number
  num_mice: number
  fps?: number
  video_width?: number
  video_height?: number
  arena_width_cm?: number
  arena_height_cm?: number
  pix_per_cm?: number
  body_parts?: string[]
}

interface ViewerProps {
  frame: FrameData | null
  metadata: Metadata | null
  onHover?: (info: PickingInfo) => void
  viewState?: any
  onViewStateChange?: (params: { viewState: any }) => void
  onMouseClick?: (mouseId: string) => void
  onViewerClick?: () => void
  trackedMouseId?: string | null
}

// ============================================================================
// Color Configuration
// ============================================================================

const MOUSE_COLORS: Record<string, [number, number, number]> = {
  '0': [255, 100, 100],
  '1': [100, 100, 255],
  '2': [100, 255, 100],
  '3': [255, 255, 100],
  '4': [255, 100, 255],
  '5': [100, 255, 255],
}

const UI_TEXT_MUTED = [139, 149, 168]
const UI_TEXT_PRIMARY = [229, 231, 235]
const UI_ACCENT = [96, 165, 250]

// ============================================================================
// Geometry Utilities
// ============================================================================

function calculateCentroid(points: Point2D[]): Point2D {
  if (points.length === 0) return { x: 0, y: 0 }
  const sum = points.reduce((acc, p) => ({ x: acc.x + p.x, y: acc.y + p.y }), { x: 0, y: 0 })
  return { x: sum.x / points.length, y: sum.y / points.length }
}

function distance(p1: Point2D, p2: Point2D): number {
  const dx = p1.x - p2.x
  const dy = p1.y - p2.y
  return Math.sqrt(dx * dx + dy * dy)
}

/**
 * Transform raw pixel coordinates to centered scene coordinates
 * Raw coords are (0,0) at top-left, need to center and flip Y
 */
function toSceneCoords(points: number[][], videoWidth: number, videoHeight: number): Point2D[] {
  const centerX = videoWidth / 2
  const centerY = videoHeight / 2
  
  return points.map(([x, y]) => ({
    x: x - centerX,      // Center X
    y: -(y - centerY)    // Center Y and flip (pixel Y goes down, scene Y goes up)
  }))
}

/**
 * Split points into body and tail using statistical outlier detection
 * Based on distance from centroid with MAD (Median Absolute Deviation)
 */
function splitBodyTail(points: Point2D[], labels: string[]): {
  body: { points: Point2D[]; labels: string[] }
  tail: { points: Point2D[]; labels: string[] }
} {
  const n = points.length
  if (n === 0) {
    return {
      body: { points: [], labels: [] },
      tail: { points: [], labels: [] }
    }
  }

  // Check labels first
  const tailIndices = new Set<number>()
  labels.forEach((label, idx) => {
    if (label.toLowerCase().includes('tail')) {
      tailIndices.add(idx)
    }
  })

  // If no tail labels and we have enough points, use statistical detection
  if (tailIndices.size === 0 && n >= 4) {
    const centroid = calculateCentroid(points)
    const distances = points.map(p => distance(p, centroid))
    
    // Calculate median
    const sorted = [...distances].sort((a, b) => a - b)
    const median = sorted[Math.floor(n / 2)]
    
    // Calculate MAD (Median Absolute Deviation)
    const deviations = distances.map(d => Math.abs(d - median))
    const mad = [...deviations].sort((a, b) => a - b)[Math.floor(n / 2)]
    
    // Threshold: median + 3.2 * MAD
    const threshold = median + 3.2 * (mad || 1)
    
    distances.forEach((d, idx) => {
      if (d > threshold) tailIndices.add(idx)
    })
    
    // Limit tail points to reasonable number
    if (tailIndices.size > Math.max(2, n / 2)) {
      const sortedIndices = Array.from(tailIndices).sort((a, b) => distances[b] - distances[a])
      tailIndices.clear()
      sortedIndices.slice(0, Math.max(2, Math.floor(n / 4))).forEach(idx => tailIndices.add(idx))
    }
  }

  const body: { points: Point2D[]; labels: string[] } = { points: [], labels: [] }
  const tail: { points: Point2D[]; labels: string[] } = { points: [], labels: [] }

  points.forEach((p, idx) => {
    if (tailIndices.has(idx)) {
      tail.points.push(p)
      tail.labels.push(labels[idx] || `tail-${tail.points.length}`)
    } else {
      body.points.push(p)
      body.labels.push(labels[idx] || `bp-${body.points.length}`)
    }
  })

  return { body, tail }
}

/**
 * Order tail points into a connected sequence using greedy nearest-neighbor
 */
function orderTailSequence(basePoint: Point2D, tailPoints: Point2D[]): Point2D[] {
  if (tailPoints.length === 0) return []
  if (tailPoints.length === 1) return tailPoints

  const remaining = [...tailPoints]
  const ordered: Point2D[] = []

  // Find closest to base
  let minIdx = 0
  let minDist = Infinity
  remaining.forEach((p, idx) => {
    const d = distance(p, basePoint)
    if (d < minDist) {
      minDist = d
      minIdx = idx
    }
  })

  ordered.push(remaining[minIdx])
  remaining.splice(minIdx, 1)

  // Greedy nearest neighbor
  while (remaining.length > 0) {
    const last = ordered[ordered.length - 1]
    minIdx = 0
    minDist = Infinity
    remaining.forEach((p, idx) => {
      const d = distance(p, last)
      if (d < minDist) {
        minDist = d
        minIdx = idx
      }
    })
    ordered.push(remaining[minIdx])
    remaining.splice(minIdx, 1)
  }

  return ordered
}

/**
 * Find nose point from labels
 */
function findNosePoint(points: Point2D[], labels: string[]): Point2D | null {
  const idx = labels.findIndex(l => l.toLowerCase().includes('nose'))
  return idx >= 0 ? points[idx] : null
}

/**
 * Calculate convex hull using Andrew's monotone chain algorithm
 */
function convexHull(points: Point2D[]): Point2D[] | null {
  if (points.length < 3) return null

  // Remove duplicates
  const unique = Array.from(new Set(points.map(p => `${p.x},${p.y}`))).map(s => {
    const [x, y] = s.split(',').map(Number)
    return { x, y }
  })

  if (unique.length < 3) return null

  // Sort by x, then y
  const sorted = unique.slice().sort((a, b) => a.x === b.x ? a.y - b.y : a.x - b.x)

  const cross = (o: Point2D, a: Point2D, b: Point2D): number => {
    return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)
  }

  // Build lower hull
  const lower: Point2D[] = []
  for (const p of sorted) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) {
      lower.pop()
    }
    lower.push(p)
  }

  // Build upper hull
  const upper: Point2D[] = []
  for (let i = sorted.length - 1; i >= 0; i--) {
    const p = sorted[i]
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) {
      upper.pop()
    }
    upper.push(p)
  }

  // Remove last point of each half because it's repeated
  lower.pop()
  upper.pop()

  const hull = [...lower, ...upper]
  return hull.length >= 3 ? hull : null
}

/**
 * Calculate minimum spanning tree edges using Prim's algorithm
 */
function minimumSpanningEdges(points: Point2D[]): Array<[Point2D, Point2D]> {
  const n = points.length
  if (n <= 1) return []

  const visited = new Set<number>()
  visited.add(0)
  const edges: Array<[Point2D, Point2D]> = []

  while (visited.size < n) {
    let bestEdge: [number, number] | null = null
    let bestDist = Infinity

    for (const i of visited) {
      for (let j = 0; j < n; j++) {
        if (visited.has(j)) continue
        const d = distance(points[i], points[j])
        if (d < bestDist) {
          bestDist = d
          bestEdge = [i, j]
        }
      }
    }

    if (!bestEdge) break
    const [i, j] = bestEdge
    visited.add(j)
    edges.push([points[i], points[j]])
  }

  return edges
}

// ============================================================================
// Grid System
// ============================================================================

interface GridLines {
  positions: Float32Array
  indices: Uint32Array
}

function calculateGridLines(
  bounds: { x: number; y: number; width: number; height: number },
  viewSpan: number,
  viewportSizePx: number = 1000
): GridLines | null {
  // Calculate pixels per unit
  const pixelsPerUnit = viewportSizePx / Math.max(viewSpan, 1e-9)

  // Dynamic bin count based on zoom - INCREASED DENSITY
  const base = Math.max(viewportSizePx / 80.0, 8.0)  // Changed from 140.0 to 80.0 for denser grid
  const zoomBonus = Math.max(0, Math.log10(Math.max(pixelsPerUnit, 1.0))) * 8.0  // Increased from 5.0 to 8.0
  const binCount = Math.max(8, Math.min(32, Math.round(base + zoomBonus)))  // Increased min from 5 to 8, max from 24 to 32

  // Generate tick positions
  const xTicks = generateTicks(bounds.x, bounds.x + bounds.width, binCount)
  const yTicks = generateTicks(bounds.y, bounds.y + bounds.height, binCount)

  // Build line segments (skip edge lines)
  const positions: number[] = []
  const indices: number[] = []

  for (const x of xTicks) {
    if (Math.abs(x - bounds.x) > 1e-6 && Math.abs(x - (bounds.x + bounds.width)) > 1e-6) {
      const idx = positions.length / 3
      positions.push(x, bounds.y, -0.15)
      positions.push(x, bounds.y + bounds.height, -0.15)
      indices.push(idx, idx + 1)
    }
  }

  for (const y of yTicks) {
    if (Math.abs(y - bounds.y) > 1e-6 && Math.abs(y - (bounds.y + bounds.height)) > 1e-6) {
      const idx = positions.length / 3
      positions.push(bounds.x, y, -0.15)
      positions.push(bounds.x + bounds.width, y, -0.15)
      indices.push(idx, idx + 1)
    }
  }

  if (positions.length === 0) return null

  return {
    positions: new Float32Array(positions),
    indices: new Uint32Array(indices)
  }
}

function generateTicks(min: number, max: number, targetCount: number): number[] {
  const range = max - min
  if (range <= 0) return []

  // Find nice step size
  const roughStep = range / targetCount
  const magnitude = Math.pow(10, Math.floor(Math.log10(roughStep)))
  const normalized = roughStep / magnitude

  let step: number
  if (normalized <= 1) step = magnitude
  else if (normalized <= 2) step = 2 * magnitude
  else if (normalized <= 5) step = 5 * magnitude
  else step = 10 * magnitude

  // Generate ticks
  const ticks: number[] = []
  const start = Math.ceil(min / step) * step
  for (let val = start; val <= max + 1e-6; val += step) {
    if (val >= min - 1e-6 && val <= max + 1e-6) {
      ticks.push(val)
    }
  }

  return ticks
}

// ============================================================================
// Smart Label Positioning
// ============================================================================

interface LabelPosition {
  mouseId: string
  position: [number, number, number]
  text: string
}

/**
 * Calculate smart orbital label positions that avoid collisions and viewport edges
 * Labels orbit around their mice and push away from each other
 */
function calculateSmartLabelPositions(
  mice: Array<{ id: string; centroid: Point2D; text: string }>,
  viewState: any,
  metadata: Metadata
): LabelPosition[] {
  if (mice.length === 0) return []
  
  const labels: LabelPosition[] = []
  const labelRadius = 60 // Distance from centroid (increased from 30 to 60)
  const minSeparation = 80 // Minimum pixels between label centers (increased from 60 to 80)
  
  // Calculate viewport bounds in scene coordinates
  const zoom = viewState?.zoom || 0
  const pixelsPerUnit = Math.pow(2, zoom)
  const viewportWidth = (window.innerWidth - 280) / pixelsPerUnit
  const viewportHeight = (window.innerHeight - 64 - 64 - 40) / pixelsPerUnit
  const targetX = viewState?.target?.[0] || 0
  const targetY = viewState?.target?.[1] || 0
  
  const viewLeft = targetX - viewportWidth / 2
  const viewRight = targetX + viewportWidth / 2
  const viewBottom = targetY - viewportHeight / 2
  const viewTop = targetY + viewportHeight / 2
  
  // Video bounds
  const videoWidth = metadata.video_width || 640
  const videoHeight = metadata.video_height || 480
  const videoLeft = -videoWidth / 2
  const videoRight = videoWidth / 2
  const videoBottom = -videoHeight / 2
  const videoTop = videoHeight / 2
  
  // Try 8 positions around each mouse (cardinal + diagonal directions)
  const angles = [0, 45, 90, 135, 180, 225, 270, 315]
  
  mice.forEach(mouse => {
    let bestAngle = 0
    let bestScore = -Infinity
    
    // Try each angle and score it
    angles.forEach(angleDeg => {
      const angleRad = (angleDeg * Math.PI) / 180
      const offsetX = labelRadius * Math.cos(angleRad)
      const offsetY = labelRadius * Math.sin(angleRad)
      const labelX = mouse.centroid.x + offsetX
      const labelY = mouse.centroid.y + offsetY
      
      let score = 0
      
      // Prefer positions inside viewport
      const inViewport = labelX >= viewLeft && labelX <= viewRight && 
                        labelY >= viewBottom && labelY <= viewTop
      if (inViewport) score += 100
      
      // Prefer positions inside video bounds
      const inVideo = labelX >= videoLeft && labelX <= videoRight &&
                      labelY >= videoBottom && labelY <= videoTop
      if (inVideo) score += 50
      
      // Avoid other labels (repulsion)
      labels.forEach(otherLabel => {
        const dx = labelX - otherLabel.position[0]
        const dy = labelY - otherLabel.position[1]
        const dist = Math.sqrt(dx * dx + dy * dy)
        if (dist < minSeparation) {
          score -= (minSeparation - dist) * 2 // Penalize overlap
        }
      })
      
      // Prefer top positions (easier to read)
      if (angleDeg >= 45 && angleDeg <= 135) score += 10
      
      if (score > bestScore) {
        bestScore = score
        bestAngle = angleDeg
      }
    })
    
    // Apply best position
    const angleRad = (bestAngle * Math.PI) / 180
    const offsetX = labelRadius * Math.cos(angleRad)
    const offsetY = labelRadius * Math.sin(angleRad)
    
    labels.push({
      mouseId: mouse.id,
      position: [mouse.centroid.x + offsetX, mouse.centroid.y + offsetY, 0.2],
      text: mouse.text
    })
  })
  
  return labels
}

// ============================================================================
// Main Component
// ============================================================================

export function EnhancedViewer({ frame, metadata, onHover, viewState, onViewStateChange, onMouseClick, onViewerClick, trackedMouseId }: ViewerProps) {
  const [hoverInfo, setHoverInfo] = useState<PickingInfo | null>(null)

  // Custom hover handler that captures the info
  const handleHover = (info: PickingInfo) => {
    setHoverInfo(info)
    if (onHover) {
      onHover(info)
    }
  }

  // Convert screen coordinates to scene coordinates for hover display
  const getSceneCoordinatesFromHover = (info: PickingInfo): { x: number; y: number } | null => {
    if (!info.coordinate || !metadata || !metadata.video_width || !metadata.video_height) return null
    
    const [sceneX, sceneY] = info.coordinate
    
    // Convert from scene coordinates back to pixel coordinates
    // Scene coords are centered at origin and Y-flipped
    // So: scene_x = pixel_x - width/2, scene_y = -(pixel_y - height/2)
    // Inverse: pixel_x = scene_x + width/2, pixel_y = height/2 - scene_y
    const pixelX = sceneX + metadata.video_width / 2
    const pixelY = metadata.video_height / 2 - sceneY
    
    // Check if within video borders
    if (pixelX < 0 || pixelX > metadata.video_width || pixelY < 0 || pixelY > metadata.video_height) {
      return null
    }
    
    return { x: pixelX, y: pixelY }
  }

  const layers = useMemo(() => {
    if (!frame || !metadata) {
      console.log('⚠️ No frame or metadata')
      return []
    }

    const result: Layer[] = []

    // Calculate view span from actual viewport and zoom (critical for lighting!)
    // OrthographicView: zoom = log2(pixels_per_unit)
    // viewSpan = viewport_size / (2^zoom)
    const viewportWidth = window.innerWidth - 280  // Subtract sidebar
    const viewportHeight = window.innerHeight - 64 - 64 - 40  // Subtract top bar, timeline, controls
    const zoom = viewState?.zoom || 0
    const pixelsPerUnit = Math.pow(2, zoom)
    const viewSpanX = viewportWidth / pixelsPerUnit
    const viewSpanY = viewportHeight / pixelsPerUnit
    const viewSpan = Math.max(viewSpanX, viewSpanY)

    console.log(`🎨 Rendering with zoom=${zoom.toFixed(2)}, viewSpan=${viewSpan.toFixed(1)}, frame has ${Object.keys(frame.mice).length} mice`)

    // Get video dimensions for coordinate transformation
    const videoWidth = metadata.video_width || 640
    const videoHeight = metadata.video_height || 480

    console.log(`📏 Video dimensions: ${videoWidth}x${videoHeight}`)

    // First pass: collect all mice centroids for smart label positioning
    const miceData: Array<{ id: string; centroid: Point2D; baseColor: [number, number, number]; data: any; rawPoints: Point2D[]; labels: string[] }> = []
    
    Object.entries(frame.mice).forEach(([mouseId, mouseData]) => {
      const baseColor = MOUSE_COLORS[mouseId] || [150, 150, 150]
      const rawPoints = toSceneCoords(mouseData.points, videoWidth, videoHeight)
      const labels = mouseData.labels || []
      
      if (rawPoints.length === 0) return
      
      const centroid = calculateCentroid(rawPoints)
      miceData.push({
        id: mouseId,
        centroid,
        baseColor,
        data: mouseData,
        rawPoints,
        labels
      })
    })
    
    // Calculate smart label positions
    const smartLabels = calculateSmartLabelPositions(
      miceData.map(m => ({ id: m.id, centroid: m.centroid, text: `Mouse ${m.id}` })),
      viewState,
      metadata
    )

    // Second pass: render each mouse
    miceData.forEach((mouse) => {
      const { id: mouseId, centroid: initialCentroid, baseColor, rawPoints, labels, data: mouseData } = mouse

      // Log first point to verify transform
      if (mouseId === '0') {
        console.log(`🔍 Mouse 0 first raw point: [${mouseData.points[0][0]}, ${mouseData.points[0][1]}]`)
        console.log(`🔍 Mouse 0 first transformed: {x: ${rawPoints[0].x.toFixed(1)}, y: ${rawPoints[0].y.toFixed(1)}}`)
      }

      // Split body/tail
      const { body, tail } = splitBodyTail(rawPoints, labels)
      const centroid = body.points.length > 0 ? calculateCentroid(body.points) : initialCentroid

      // === Convex Hull (transparent filled polygon + CLICKABLE) ===
      const hull = convexHull(body.points)
      if (hull && hull.length >= 3) {
        const hullColor = baseColor.map(c => c + (255 - c) * 0.6) as [number, number, number]
        const isTracked = trackedMouseId === mouseId
        
        result.push(new PolygonLayer({
          id: `hull-${mouseId}`,
          data: [{ polygon: hull.map(p => [p.x, p.y, -0.1]), mouseId }],
          getPolygon: (d: any) => d.polygon,
          getFillColor: [...hullColor, isTracked ? 80 : 56] as Color,  // brighter when tracked
          getLineColor: isTracked ? [...baseColor, 255] as Color : [0, 0, 0, 0] as Color,
          getLineWidth: isTracked ? 3 : 0,
          filled: true,
          stroked: isTracked,
          coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
          pickable: true,
          onClick: (info: any) => {
            if (info.object && onMouseClick) {
              onMouseClick(info.object.mouseId)
            }
          }
        }))
      }

      // === MST Edges (body point connections) ===
      const mstEdges = minimumSpanningEdges(body.points)
      if (mstEdges.length > 0) {
        const edgeColor = baseColor.map(c => Math.min(255, c + (255 - c) * 0.35)) as [number, number, number]
        const segments = mstEdges.map(([start, end]) => ({
          source: [start.x, start.y, -0.08],
          target: [end.x, end.y, -0.08]
        }))
        result.push(new LineLayer({
          id: `body-edges-${mouseId}`,
          data: segments,
          getSourcePosition: (d: any) => d.source,
          getTargetPosition: (d: any) => d.target,
          getColor: [...edgeColor, 230] as Color,  // alpha=0.9*255
          getWidth: 1.5,
          widthUnits: 'pixels',
          coordinateSystem: COORDINATE_SYSTEM.CARTESIAN
        }))
      }

      // === Tail Polyline (dual layer) ===
      if (tail.points.length >= 1) {
        const orderedTail = orderTailSequence(centroid, tail.points)
        
        // Calculate centroid radius (for finding hull intersection)
        const centroidRadius = hull && hull.length > 0 
          ? Math.max(...hull.map(p => distance(p, centroid)))
          : 8  // Default radius if no hull
        
        // Find first tail point outside centroid radius (for hull connection)
        let firstOutsideIdx = 0
        for (let i = 0; i < orderedTail.length; i++) {
          if (distance(orderedTail[i], centroid) > centroidRadius) {
            firstOutsideIdx = i
            break
          }
        }
        
        console.log(`🐭 Mouse ${mouseId}: ${tail.points.length} tail points, first outside at index ${firstOutsideIdx}`)

        // Convert ALL tail points to line segments (don't skip any)
        const segments: any[] = []
        
        // Add connection from hull border to first tail point (ALWAYS)
        if (orderedTail.length > 0 && hull && hull.length >= 3) {
          // Determine which point to aim for: first outside point if exists, otherwise last tail point
          const targetPoint = firstOutsideIdx < orderedTail.length 
            ? orderedTail[firstOutsideIdx] 
            : orderedTail[orderedTail.length - 1]
          
          const direction = {
            x: targetPoint.x - centroid.x,
            y: targetPoint.y - centroid.y
          }
          const len = Math.sqrt(direction.x * direction.x + direction.y * direction.y)
          
          if (len > 0) {
            // Extend the line far beyond the hull to ensure intersection
            const farPoint = {
              x: centroid.x + (direction.x / len) * (centroidRadius * 3),
              y: centroid.y + (direction.y / len) * (centroidRadius * 3)
            }
            
            // Find intersection with hull polygon edges
            let closestIntersection: Point2D | null = null
            let closestDist = Infinity
            
            for (let i = 0; i < hull.length; i++) {
              const p1 = hull[i]
              const p2 = hull[(i + 1) % hull.length]
              
              // Line-line intersection between (centroid -> farPoint) and (p1 -> p2)
              const x1 = centroid.x, y1 = centroid.y
              const x2 = farPoint.x, y2 = farPoint.y
              const x3 = p1.x, y3 = p1.y
              const x4 = p2.x, y4 = p2.y
              
              const denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
              if (Math.abs(denom) > 1e-10) {
                const t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
                const u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
                
                if (t >= 0 && t <= 1 && u >= 0 && u <= 1) {
                  const ix = x1 + t * (x2 - x1)
                  const iy = y1 + t * (y2 - y1)
                  const dist = Math.sqrt((ix - centroid.x) ** 2 + (iy - centroid.y) ** 2)
                  
                  if (dist < closestDist) {
                    closestDist = dist
                    closestIntersection = { x: ix, y: iy }
                  }
                }
              }
            }
            
            const borderPoint = closestIntersection || {
              x: centroid.x + (direction.x / len) * centroidRadius,
              y: centroid.y + (direction.y / len) * centroidRadius
            }
            
            // Connect hull border to first tail point
            segments.push({
              source: [borderPoint.x, borderPoint.y, 0],
              target: [orderedTail[0].x, orderedTail[0].y, 0]
            })
          }
        } else if (orderedTail.length > 0) {
          // Fallback if no hull - simple radial connection
          const direction = {
            x: orderedTail[0].x - centroid.x,
            y: orderedTail[0].y - centroid.y
          }
          const len = Math.sqrt(direction.x * direction.x + direction.y * direction.y)
          const borderPoint = len > 0 ? {
            x: centroid.x + (direction.x / len) * centroidRadius,
            y: centroid.y + (direction.y / len) * centroidRadius
          } : centroid
          
          segments.push({
            source: [borderPoint.x, borderPoint.y, 0],
            target: [orderedTail[0].x, orderedTail[0].y, 0]
          })
        }
        
        // Add ALL tail point segments (through every single tail point)
        for (let i = 0; i < orderedTail.length - 1; i++) {
          segments.push({
            source: [orderedTail[i].x, orderedTail[i].y, 0],
            target: [orderedTail[i + 1].x, orderedTail[i + 1].y, 0]
          })
        }

        // Primary layer (thicker, lighter)
        const primaryColor = baseColor.map(c => Math.min(255, c + (255 - c) * 0.35)) as [number, number, number]
        result.push(new LineLayer({
          id: `tail-primary-${mouseId}`,
          data: segments,
          getSourcePosition: (d: any) => d.source,
          getTargetPosition: (d: any) => d.target,
          getColor: [...primaryColor, 153] as Color,  // alpha=0.6*255
          getWidth: 3.5,
          widthUnits: 'pixels',
          coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
          visible: true
        }))

        // Secondary layer (thinner, darker)
        result.push(new LineLayer({
          id: `tail-secondary-${mouseId}`,
          data: segments,
          getSourcePosition: (d: any) => d.source,
          getTargetPosition: (d: any) => d.target,
          getColor: [...baseColor, 115] as Color,  // alpha=0.45*255
          getWidth: 2.0,
          widthUnits: 'pixels',
          coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
          visible: true
        }))
      } else {
        console.log(`⚠️ Mouse ${mouseId}: only ${tail.points.length} tail point(s), skipping polyline`)
      }

      // === Whiskers ===
      const nosePoint = findNosePoint(rawPoints, labels)
      if (nosePoint && body.points.length > 0) {
        const dir = { x: nosePoint.x - centroid.x, y: nosePoint.y - centroid.y }
        const norm = Math.sqrt(dir.x * dir.x + dir.y * dir.y)

        if (norm > 1e-5) {
          const dirUnit = { x: dir.x / norm, y: dir.y / norm }
          const whiskerLength = Math.min(Math.max(norm * 0.85, 18), 48)

          const segments: any[] = []
          const angles = [0.25, 0.45, 0.68]
          const lengths = [0.92, 1.0, 1.08]

          angles.forEach((angle, i) => {
            const factor = lengths[i]
            // Left
            const cos = Math.cos(-angle), sin = Math.sin(-angle)
            const leftDir = { x: cos * dirUnit.x - sin * dirUnit.y, y: sin * dirUnit.x + cos * dirUnit.y }
            segments.push({
              source: [nosePoint.x, nosePoint.y, -0.1],
              target: [nosePoint.x + leftDir.x * whiskerLength * factor, nosePoint.y + leftDir.y * whiskerLength * factor, -0.1]
            })
            // Right
            const cos2 = Math.cos(angle), sin2 = Math.sin(angle)
            const rightDir = { x: cos2 * dirUnit.x - sin2 * dirUnit.y, y: sin2 * dirUnit.x + cos2 * dirUnit.y }
            segments.push({
              source: [nosePoint.x, nosePoint.y, -0.1],
              target: [nosePoint.x + rightDir.x * whiskerLength * factor, nosePoint.y + rightDir.y * whiskerLength * factor, -0.1]
            })
          })

          const whiskerColor = baseColor.map(c => Math.min(255, c + (255 - c) * 0.55)) as [number, number, number]
          result.push(new LineLayer({
            id: `whiskers-${mouseId}`,
            data: segments,
            getSourcePosition: (d: any) => d.source,
            getTargetPosition: (d: any) => d.target,
            getColor: [...whiskerColor, 153] as Color,
            getWidth: 1.2,
            widthUnits: 'pixels',
            coordinateSystem: COORDINATE_SYSTEM.CARTESIAN
          }))
        }
      }

      // === Body Points (SMALL pixels, always same size!) ===
      body.points.forEach((p, idx) => {
        result.push(new ScatterplotLayer({
          id: `body-${mouseId}-${idx}`,
          data: [{ pos: p, label: body.labels[idx], mouseId }],
          getPosition: (d: any) => [d.pos.x, d.pos.y, 0],
          getFillColor: [...baseColor, 235] as Color,
          getLineColor: [255, 255, 255, 217] as Color,
          getLineWidth: 0.8,
          getRadius: 3.5,  // SMALL pixels!
          radiusUnits: 'pixels',
          stroked: true,
          coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
          pickable: true,
          autoHighlight: true
        }))
      })

      // === Tail Points (SMALL pixels, always same size!) ===
      const tailColor = baseColor.map(c => Math.min(255, c + (255 - c) * 0.35)) as [number, number, number]
      tail.points.forEach((p, idx) => {
        result.push(new ScatterplotLayer({
          id: `tail-${mouseId}-${idx}`,
          data: [{ pos: p, label: tail.labels[idx], mouseId }],
          getPosition: (d: any) => [d.pos.x, d.pos.y, 0],
          getFillColor: [...tailColor, 230] as Color,
          getLineColor: [...baseColor, 217] as Color,
          getLineWidth: 0.6,
          getRadius: 2.5,  // SMALL pixels!
          radiusUnits: 'pixels',
          stroked: true,
          coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
          pickable: true,
          autoHighlight: true
        }))
      })

      // === Mouse Label (use smart position) ===
      const labelColor = baseColor.map(c => Math.min(255, c + (255 - c) * 0.35)) as [number, number, number]
      const smartLabel = smartLabels.find(l => l.mouseId === mouseId)
      const labelPos = smartLabel 
        ? smartLabel.position
        : [centroid.x, centroid.y + 20, 0.2]
      
      // === Label Connector Line ===
      if (smartLabel) {
        result.push(new LineLayer({
          id: `label-connector-${mouseId}`,
          data: [{
            source: [centroid.x, centroid.y, 0.1],
            target: [labelPos[0], labelPos[1], 0.1]
          }],
          getSourcePosition: (d: any) => d.source,
          getTargetPosition: (d: any) => d.target,
          getColor: [...labelColor, 120] as Color, // Semi-transparent
          getWidth: 1.5,
          coordinateSystem: COORDINATE_SYSTEM.CARTESIAN
        }))
      }
      
      result.push(new TextLayer({
        id: `label-${mouseId}`,
        data: [{ pos: labelPos, text: `Mouse ${mouseId}` }],
        getPosition: (d: any) => d.pos,
        getText: (d: any) => d.text,
        getColor: [...labelColor, 255] as Color,
        getSize: 14,
        getAngle: 0,
        getTextAnchor: 'middle',
        getAlignmentBaseline: 'center',
        coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
        fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
        fontWeight: 600
      }))
    })

    // === Video Border (always render if we have dimensions) ===
    if (metadata.video_width && metadata.video_height) {
      const w = metadata.video_width / 2
      const h = metadata.video_height / 2
      console.log(`📐 Creating video border: ${metadata.video_width}x${metadata.video_height}`)
      console.log(`   Bounds: x=[${-w}, ${w}], y=[${-h}, ${h}]`)
      
      const borderSegments = [
        { source: [-w, -h, 0], target: [w, -h, 0] },  // bottom
        { source: [w, -h, 0], target: [w, h, 0] },    // right
        { source: [w, h, 0], target: [-w, h, 0] },    // top
        { source: [-w, h, 0], target: [-w, -h, 0] }   // left
      ]
      
      console.log(`   Creating LineLayer with ${borderSegments.length} segments`)
      
      result.push(new LineLayer({
        id: 'video-border',
        data: borderSegments,
        getSourcePosition: (d: any) => d.source,
        getTargetPosition: (d: any) => d.target,
        getColor: [...UI_TEXT_MUTED, 140] as Color,
        getWidth: 2.0,
        widthUnits: 'pixels',
        coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
        pickable: false,
        visible: true  // Explicitly set visible
      }))
      
      console.log(`✅ Video border layer added`)
    } else {
      console.log('⚠️ No video dimensions in metadata:', metadata)
    }

    // === Arena Border (if available) ===
    if (metadata.arena_width_cm && metadata.arena_height_cm && metadata.pix_per_cm) {
      const w = (metadata.arena_width_cm * metadata.pix_per_cm) / 2
      const h = (metadata.arena_height_cm * metadata.pix_per_cm) / 2
      console.log(`📐 Arena border: ${metadata.arena_width_cm}x${metadata.arena_height_cm} cm @ ${metadata.pix_per_cm} px/cm, bounds: [${-w}, ${-h}] to [${w}, ${h}]`)
      
      const borderSegments = [
        { source: [-w, -h, 0], target: [w, -h, 0] },
        { source: [w, -h, 0], target: [w, h, 0] },
        { source: [w, h, 0], target: [-w, h, 0] },
        { source: [-w, h, 0], target: [-w, -h, 0] }
      ]
      
      result.push(new LineLayer({
        id: 'arena-border',
        data: borderSegments,
        getSourcePosition: (d: any) => d.source,
        getTargetPosition: (d: any) => d.target,
        getColor: [...UI_ACCENT, 180] as Color,
        getWidth: 2.5,
        widthUnits: 'pixels',
        coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
        pickable: false
      }))
    }

    // === Grid Lines ===
    const gridBounds = metadata.video_width && metadata.video_height
      ? { x: -metadata.video_width / 2, y: -metadata.video_height / 2, width: metadata.video_width, height: metadata.video_height }
      : metadata.arena_width_cm && metadata.arena_height_cm && metadata.pix_per_cm
      ? { x: -(metadata.arena_width_cm * metadata.pix_per_cm) / 2, y: -(metadata.arena_height_cm * metadata.pix_per_cm) / 2, 
          width: metadata.arena_width_cm * metadata.pix_per_cm, height: metadata.arena_height_cm * metadata.pix_per_cm }
      : null

    if (gridBounds) {
      console.log(`📊 Grid bounds:`, gridBounds)
      const grid = calculateGridLines(gridBounds, viewSpan, viewportWidth)
      
      if (grid && grid.positions.length > 0) {
        // Convert to line segments (z=0 for all grid lines)
        const segments: any[] = []
        for (let i = 0; i < grid.indices.length; i += 2) {
          const idx1 = grid.indices[i]
          const idx2 = grid.indices[i + 1]
          segments.push({
            source: [grid.positions[idx1 * 3], grid.positions[idx1 * 3 + 1], 0],
            target: [grid.positions[idx2 * 3], grid.positions[idx2 * 3 + 1], 0]
          })
        }
        
        console.log(`📊 Grid rendered: ${segments.length} lines`)
        
        result.push(new LineLayer({
          id: 'grid',
          data: segments,
          getSourcePosition: (d: any) => d.source,
          getTargetPosition: (d: any) => d.target,
          getColor: [...UI_TEXT_MUTED, 71] as Color,
          getWidth: 1.0,
          widthUnits: 'pixels',
          coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
          pickable: false,
          visible: true
        }))
      } else {
        console.log('⚠️ Grid calculation returned no lines')
      }
    } else {
      console.log('⚠️ No grid bounds calculated')
    }

    console.log(`✅ Total layers: ${result.length}`)

    // === Border Labels ===
    if (metadata.video_width && metadata.video_height) {
      const h = metadata.video_height / 2
      const centerX = 0
      const topY = h
      
      result.push(new TextLayer({
        id: 'video-border-label',
        data: [{
          text: `Video ${metadata.video_width}px × ${metadata.video_height}px`,
          position: [centerX, topY + 12, 0]
        }],
        getText: (d: any) => d.text,
        getPosition: (d: any) => d.position,
        getColor: [...UI_TEXT_PRIMARY, 224] as Color,
        getSize: 14,
        getTextAnchor: 'middle',
        getAlignmentBaseline: 'top',
        fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
        coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
        billboard: true,  // Always face camera (prevents upside-down text)
        pickable: false
      }))
    }

    if (metadata.arena_width_cm && metadata.arena_height_cm && metadata.pix_per_cm) {
      const h = (metadata.arena_height_cm * metadata.pix_per_cm) / 2
      const centerX = 0
      const bottomY = -h
      
      const widthPx = metadata.arena_width_cm * metadata.pix_per_cm
      const heightPx = metadata.arena_height_cm * metadata.pix_per_cm
      const text = `Arena ${widthPx.toFixed(0)}px × ${heightPx.toFixed(0)}px (${metadata.arena_width_cm.toFixed(1)} cm × ${metadata.arena_height_cm.toFixed(1)} cm)`
      
      result.push(new TextLayer({
        id: 'arena-border-label',
        data: [{
          text: text,
          position: [centerX, bottomY - 12, 0]
        }],
        getText: (d: any) => d.text,
        getPosition: (d: any) => d.position,
        getColor: [...UI_ACCENT, 224] as Color,  // Match blue arena border color
        getSize: 14,
        getTextAnchor: 'middle',
        getAlignmentBaseline: 'bottom',
        fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
        coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
        billboard: true,  // Always face camera (prevents upside-down text)
        pickable: false
      }))
    }

    return result
  }, [frame, metadata, viewState])  // Include viewState so lighting updates on zoom!

  // Calculate scale bar for HTML overlay (adapts to zoom, positioned in screen space)
  const scaleBarInfo = useMemo(() => {
    if (!metadata?.pix_per_cm || !viewState) return null
    
    // Get current zoom level
    const zoom = viewState.zoom || 0
    const scale = Math.pow(2, zoom)
    
    // Viewport width in pixels (assume 1000px as default, deck.gl will scale appropriately)
    const viewportWidthPx = 1000
    
    // Target scale bar: 10-20% of viewport width
    const maxScaleBarPx = viewportWidthPx * 0.15
    const minScaleBarPx = 48
    
    // Convert to scene units
    const maxScaleBarUnits = maxScaleBarPx / scale
    const minScaleBarUnits = minScaleBarPx / scale
    
    // Convert to centimeters
    const maxScaleCm = maxScaleBarUnits / metadata.pix_per_cm
    const minScaleCm = minScaleBarUnits / metadata.pix_per_cm
    
    // Select nice round value
    const selectNiceLength = (minVal: number, maxVal: number): number => {
      if (maxVal <= 0) return 1
      const exponent = Math.floor(Math.log10(maxVal))
      for (let exp = exponent; exp >= exponent - 3; exp--) {
        const base = Math.pow(10, exp)
        for (const mult of [5, 2, 1]) {
          const candidate = mult * base
          if (candidate <= maxVal && candidate >= minVal) {
            return candidate
          }
        }
      }
      return Math.max(minVal, 1)
    }
    
    const scaleCm = selectNiceLength(minScaleCm, maxScaleCm)
    const scaleSceneUnits = scaleCm * metadata.pix_per_cm
    const scaleScreenPx = scaleSceneUnits * scale
    
    const label = `${scaleSceneUnits.toFixed(0)} px  (≈ ${scaleCm.toFixed(scaleCm >= 10 ? 0 : 1)} cm)`
    
    return {
      widthPx: scaleScreenPx,
      label
    }
  }, [metadata, viewState])

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%' }}>
      <DeckGL
        layers={layers}
        views={[new OrthographicView({ controller: true })]}
        viewState={viewState}
        onViewStateChange={onViewStateChange}
        onHover={handleHover}
        onClick={(info: any) => {
          // If not clicking on an object (clicking background), call onViewerClick
          if (!info.object && onViewerClick) {
            onViewerClick()
          }
        }}
        controller={true}
        getCursor={() => 'crosshair'}
      />
      
      {/* Hover tooltip (next to cursor) */}
      {hoverInfo && hoverInfo.x !== undefined && hoverInfo.y !== undefined && (() => {
        const sceneCoords = getSceneCoordinatesFromHover(hoverInfo)
        if (!sceneCoords) return null
        
        // If hovering over a mouse point, show detailed info
        if (hoverInfo.object && hoverInfo.object.label) {
          return (
            <div style={{
              position: 'absolute',
              left: `${hoverInfo.x + 15}px`,
              top: `${hoverInfo.y + 15}px`,
              pointerEvents: 'none',
              backgroundColor: 'rgba(0, 0, 0, 0.85)',
              color: '#fff',
              padding: '8px 12px',
              borderRadius: '4px',
              fontSize: '12px',
              fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
              boxShadow: '0 2px 8px rgba(0,0,0,0.4)',
              zIndex: 1000,
              whiteSpace: 'nowrap'
            }}>
              <div><strong>Mouse:</strong> {hoverInfo.object.mouseId}</div>
              <div><strong>Part:</strong> {hoverInfo.object.label}</div>
              <div><strong>X:</strong> {sceneCoords.x.toFixed(1)} px</div>
              <div><strong>Y:</strong> {sceneCoords.y.toFixed(1)} px</div>
            </div>
          )
        }
        
        // Otherwise show generic coordinates
        return (
          <div style={{
            position: 'absolute',
            left: `${hoverInfo.x + 15}px`,
            top: `${hoverInfo.y + 15}px`,
            pointerEvents: 'none',
            backgroundColor: 'rgba(0, 0, 0, 0.75)',
            color: '#fff',
            padding: '6px 10px',
            borderRadius: '4px',
            fontSize: '11px',
            fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
            boxShadow: '0 2px 8px rgba(0,0,0,0.3)',
            zIndex: 1000,
            whiteSpace: 'nowrap'
          }}>
            <div><strong>X:</strong> {sceneCoords.x.toFixed(1)} px</div>
            <div><strong>Y:</strong> {sceneCoords.y.toFixed(1)} px</div>
          </div>
        )
      })()}
      
      {/* Scale bar overlay (bottom-right corner, above playback controls) */}
      {scaleBarInfo && (
        <div style={{
          position: 'absolute',
          bottom: '84px', // Above 64px playback controls + 20px margin
          right: '20px',
          pointerEvents: 'none',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'flex-end',
          gap: '8px',
          transition: 'all 0.4s cubic-bezier(0.4, 0, 0.2, 1)' // Smooth animation for size changes
        }}>
          {/* Scale bar label */}
          <div style={{
            color: 'rgba(229, 231, 235, 0.9)',
            fontSize: '11px',
            fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
            fontWeight: '600',
            letterSpacing: '0.3px',
            textTransform: 'uppercase',
            background: 'rgba(20, 20, 35, 0.85)',
            backdropFilter: 'blur(12px)',
            padding: '6px 10px',
            borderRadius: '6px',
            border: '1px solid rgba(96, 165, 250, 0.3)',
            boxShadow: '0 4px 12px rgba(0, 0, 0, 0.4)',
            transition: 'all 0.4s cubic-bezier(0.4, 0, 0.2, 1)' // Match parent transition
          }}>
            {scaleBarInfo.label}
          </div>
          
          {/* Scale bar visual */}
          <div style={{
            position: 'relative',
            height: '12px',
            display: 'flex',
            alignItems: 'center',
            transition: 'all 0.4s cubic-bezier(0.4, 0, 0.2, 1)' // Smooth width changes
          }}>
            {/* Main bar */}
            <div style={{
              width: `${scaleBarInfo.widthPx}px`,
              height: '3px',
              background: 'linear-gradient(90deg, rgba(96, 165, 250, 0.8) 0%, rgba(96, 165, 250, 1) 100%)',
              borderRadius: '1.5px',
              boxShadow: '0 0 8px rgba(96, 165, 250, 0.5), 0 2px 4px rgba(0, 0, 0, 0.3)',
              transition: 'all 0.4s cubic-bezier(0.4, 0, 0.2, 1)' // Smooth width changes
            }} />
            
            {/* Left tick */}
            <div style={{
              position: 'absolute',
              left: '0',
              bottom: '0',
              width: '2px',
              height: '12px',
              background: 'rgba(96, 165, 250, 1)',
              borderRadius: '1px',
              boxShadow: '0 0 6px rgba(96, 165, 250, 0.6)'
            }} />
            
            {/* Right tick */}
            <div style={{
              position: 'absolute',
              right: '0',
              bottom: '0',
              width: '2px',
              height: '12px',
              background: 'rgba(96, 165, 250, 1)',
              borderRadius: '1px',
              boxShadow: '0 0 6px rgba(96, 165, 250, 0.6)',
              transition: 'all 0.4s cubic-bezier(0.4, 0, 0.2, 1)' // Position changes with width
            }} />
          </div>
        </div>
      )}
    </div>
  )
}
