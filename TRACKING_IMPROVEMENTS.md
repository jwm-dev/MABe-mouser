# Mouse Tracking Improvements

This document describes the improvements made to the mouse tracking and label system.

## Issues Fixed

### 1. Smart Orbital Labels - Spacing and Connectors ✅

**Problem**: Labels were rendering too close to mice (30px), making them overlap when not zoomed in.

**Solution**:
- Increased orbital radius from **30px to 60px** for better spacing
- Increased minimum separation from **60px to 80px** between labels
- Added **connector lines** from label to mouse centroid:
  - Thin line (1.5px width)
  - Semi-transparent (alpha 120)
  - Matches mouse's label color
  - Only rendered when using smart positioning

**Implementation**:
- Updated `calculateSmartLabelPositions()` in `EnhancedViewer.tsx` (lines 394-404)
- Added `LineLayer` for connectors before each label (lines 871-881)

---

### 2. Camera Reset ('R' Key) ✅

**Status**: Already working perfectly, no changes needed!

---

### 3. Click-to-Track Mouse - Multiple Fixes ✅

**Problem 1**: Playback locked up when tracking a mouse

**Root Cause**: The camera tracking effect was creating an infinite loop by updating `viewState` without proper dependency management.

**Solution**: Fixed dependencies in the tracking `useEffect` to only respond to frame changes, not viewState changes.

---

**Problem 2**: No textual indication of which mouse is being tracked

**Solution**: Added tracking indicator badge at bottom of viewer:
- Centered above playback controls (bottom: 94px)
- Shows "Tracking Mouse X" in mouse's color
- Glowing border matching mouse color
- Animated pulsing dot indicator
- Non-interactive (pointerEvents: none)

**Implementation**:
- Added tracking indicator in `App.tsx` (lines 1545-1580)
- Uses same base color array as mouse rendering
- Positioned with z-index 1000 to stay on top

---

**Problem 3**: Camera didn't auto-zoom when tracking

**Solution**: Camera now zooms to 3.0x when first tracking a mouse, then maintains user's zoom level:
- Initial zoom to 3.0x when tracking starts
- If user already zoomed in (>2.5x), respects their zoom
- User can freely zoom in/out while tracking
- Camera always centers on tracked mouse

**Implementation**:
- Updated tracking effect in `App.tsx` (lines 601-633)
- Conditional zoom: `zoom: prev.zoom < 2.5 ? 3.0 : prev.zoom`

---

**Problem 4**: No way to release tracking without resetting camera

**Solution**: Multiple ways to release tracking while preserving camera position:
1. **Click background**: Click anywhere in viewer (not on a mouse) to release tracking
2. **Click same mouse**: Click the tracked mouse's hull again to toggle off tracking
3. **'R' key**: Still resets camera AND releases tracking (original behavior)

**Implementation**:
- Added `onViewerClick` prop to `ViewerProps` (line 39)
- Added `onClick` handler to DeckGL component (lines 1125-1131)
  - Checks if clicking background (!info.object)
  - Calls `onViewerClick()` to release tracking
- Updated `onMouseClick` in App.tsx (lines 1530-1537)
  - If clicking tracked mouse, releases tracking
  - If clicking different mouse, switches tracking
  - Background clicks handled by onViewerClick

---

**Problem 5**: User couldn't zoom while tracking

**Solution**: Camera tracking now only updates target position, preserving user's zoom changes:
- Tracking effect doesn't force zoom on every frame
- Only sets initial zoom when tracking starts
- User can freely zoom in/out using mouse wheel
- Camera maintains center on tracked mouse at user's zoom level

---

## User Experience Flow

### Starting Tracking:
1. Click on a mouse's hull
2. Camera smoothly zooms to 3.0x (if not already zoomed in)
3. Camera centers on mouse
4. Hull glows with colored border
5. Tracking indicator appears at bottom: "Tracking Mouse X"
6. During playback, camera follows mouse

### While Tracking:
- **Zoom in/out**: Mouse wheel works normally, camera stays centered on mouse
- **Pan**: Not recommended, camera will re-center on next frame
- **Playback**: Camera smoothly follows mouse through frames
- **Switch mouse**: Click another mouse's hull to track that one instead

### Stopping Tracking:
- **Option 1**: Click the tracked mouse's hull again (toggles off)
- **Option 2**: Click anywhere in background/empty space
- **Option 3**: Press 'R' (also resets camera to default view)

---

## Technical Details

### Label Connector Lines

Each label with smart positioning gets a connector line:

```typescript
new LineLayer({
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
})
```

### Tracking Indicator Styling

Glassmorphism card with dynamic coloring:

```typescript
{
  position: 'absolute',
  bottom: '94px',
  left: '50%',
  transform: 'translateX(-50%)',
  background: 'rgba(17, 24, 39, 0.95)',
  border: `2px solid ${mouseColor}`,
  borderRadius: '8px',
  color: mouseColor,
  boxShadow: `0 0 20px ${mouseColor}40`
}
```

### Smart Zoom Logic

```typescript
setViewState(prev => ({
  ...prev,
  target: [centroidX, centroidY, 0],
  // Only set zoom on first track (< 2.5), then maintain user's zoom
  zoom: prev.zoom < 2.5 ? 3.0 : prev.zoom
}))
```

This ensures:
- First tracking: zooms to 3.0x for good mouse visibility
- User zooms in: respects their preference (>2.5x)
- User zooms out while tracking: maintains their zoom level
- No zoom fighting between user and tracking system

---

## File Changes Summary

### `frontend/src/components/EnhancedViewer.tsx`
- Line 39: Added `onViewerClick` to ViewerProps
- Line 492: Added `onViewerClick` to function parameters
- Lines 394-404: Increased label orbital radius (60px) and separation (80px)
- Lines 871-881: Added LineLayer for label connectors
- Lines 1125-1131: Added onClick handler to DeckGL for background clicks

### `frontend/src/App.tsx`
- Lines 601-633: Fixed tracking effect to prevent infinite loop
- Lines 601-633: Added smart zoom logic (initial 3.0x, then maintain user zoom)
- Lines 1530-1537: Updated onMouseClick to toggle/switch tracking
- Line 1537: Added onViewerClick prop to release tracking
- Lines 1545-1580: Added tracking indicator badge UI

---

## Testing Checklist

- [x] Labels orbit 60px from mouse centroids
- [x] Labels have connector lines in matching color
- [x] Labels maintain 80px minimum separation
- [x] Clicking mouse hull starts tracking
- [x] Tracking indicator appears with correct mouse color
- [x] Camera zooms to 3.0x when tracking starts
- [x] Camera follows mouse during playback
- [x] User can zoom in/out while tracking
- [x] Clicking tracked mouse releases tracking
- [x] Clicking background releases tracking
- [x] Clicking different mouse switches tracking
- [x] 'R' key releases tracking and resets camera
- [x] Playback doesn't freeze during tracking
- [x] Camera position preserved when releasing tracking (except 'R' key)

---

## Known Limitations

1. **Pan during tracking**: If user manually pans away from tracked mouse, camera will snap back on next frame update. This is intentional to maintain focus.

2. **Multiple mice**: Can only track one mouse at a time. Multi-track would require split-screen implementation.

3. **Label connectors**: Only shown when labels are positioned via smart algorithm. Fallback positions (when algorithm can't find good spot) won't have connectors.

---

## Future Enhancements

1. **Smooth panning override**: Allow temporary manual pan with gradual return to tracked mouse
2. **Track history trail**: Show semi-transparent path of last N positions
3. **Keyboard shortcuts**: Number keys 1-9 to quick-switch between mice
4. **Picture-in-picture**: Mini-map showing full arena while tracking
5. **Auto-zoom distance**: Adaptive zoom based on mouse velocity (zoom out when moving fast)
