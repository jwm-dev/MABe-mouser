# Advanced Interaction Features

This document describes three major new features added to the mouse behavior viewer.

## 1. Smart Orbital Labels 🏷️

**Feature**: Mouse labels now intelligently position themselves to avoid collisions and stay within the viewport.

**How It Works**:
- Each label tries 8 candidate positions around its mouse (cardinal + diagonal directions: 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°)
- Labels orbit at 30px radius from the mouse centroid
- Positions are scored based on:
  - **+100 points**: Label center is within viewport bounds
  - **+50 points**: Label center is within video bounds
  - **+10 points**: Label is in top half (preferred for readability)
  - **-2x overlap distance**: Penalty for colliding with other labels (minimum 60px separation)
- The highest-scoring position is selected for each label
- Uses two-pass rendering: collect all mice → calculate positions → render with smart positions

**Implementation**:
- `calculateSmartLabelPositions()` function in `EnhancedViewer.tsx` (lines 394-482)
- Refactored rendering to two-pass system (lines 520-580)
- Labels use `TextLayer` with center alignment for smooth orbiting

---

## 2. Camera Reset Hotkey ⌨️

**Feature**: Press **R** to instantly reset the camera to the default view.

**How It Works**:
- Pressing 'R' resets the view to:
  - **Zoom**: 1.0 (default zoom level)
  - **Target**: [0, 0, 0] (center of coordinate system)
- Also stops any active mouse tracking (sets `trackedMouseId` to null)
- Smoothly transitions to default position

**Implementation**:
- Keyboard handler in `App.tsx` (lines 358-400)
- Integrated with existing keyboard shortcut system
- Works both when viewer has focus and during playback

**Usage**:
```
R - Reset camera to default position and stop tracking
```

---

## 3. Click-to-Track Mouse 🖱️

**Feature**: Click on any mouse's hull to focus the camera on it and track it during playback.

**How It Works**:
- Mouse hulls are now clickable (deck.gl `pickable` polygons)
- Clicking a hull sets that mouse as the tracked target
- Camera automatically follows the tracked mouse's centroid during playback
- Tracked mouse has visual feedback:
  - Brighter hull fill (alpha 80 vs 56)
  - Colored border outline (3px stroke in mouse's base color)
- Press 'R' to stop tracking and reset camera

**Implementation**:
- **EnhancedViewer.tsx**:
  - Hull `PolygonLayer` now has `pickable: true` and `onClick` handler (lines 593-617)
  - Visual indicator for tracked mouse (brighter fill + colored stroke)
  - Accepts `onMouseClick` callback and `trackedMouseId` prop
- **App.tsx**:
  - `trackedMouseId` state to track which mouse is being followed (line 41)
  - Camera tracking effect that updates `viewState.target` (lines 597-634)
  - Passes `onMouseClick` and `trackedMouseId` to EnhancedViewer (lines 1520-1527)

**User Experience**:
1. Click on a mouse hull in the 3D viewer
2. Camera focuses on that mouse (hull glows with colored border)
3. During playback, camera smoothly follows the mouse
4. Press 'R' to stop tracking and reset view
5. Click another mouse to switch tracking

---

## Technical Details

### Smart Label Algorithm Complexity
- **Time**: O(n²) where n = number of mice
  - For each mouse: tries 8 positions
  - For each position: checks overlap with all other labels
  - Total: 8n × n = O(n²)
- **Space**: O(n) for storing label positions
- **Optimization**: Early exit when perfect score (no overlaps, in viewport)

### Camera Tracking Performance
- Uses React `useEffect` with dependencies: `[trackedMouseId, currentFrame, frames, metadata, getFrame]`
- Only recalculates when tracked mouse changes or frame updates
- Smoothly interpolates to new position (deck.gl transition)

### Hull Interaction
- Hulls rendered as convex polygons at z = -0.1 (below mice)
- Click detection via deck.gl's picking system
- `mouseId` stored in polygon data for identification

---

## File Changes Summary

### `frontend/src/components/EnhancedViewer.tsx`
- Added `LabelPosition` interface (line 384)
- Added `calculateSmartLabelPositions()` function (lines 394-482)
- Refactored to two-pass rendering (lines 520-580)
- Made hulls clickable with visual feedback (lines 593-617)
- Updated label TextLayer to use smart positions (lines 861-881)
- Extended `ViewerProps` with `onMouseClick` and `trackedMouseId` (lines 37-40)

### `frontend/src/App.tsx`
- Added `trackedMouseId` state (line 41)
- Added 'R' key reset handler (lines 358-400)
- Added camera tracking effect (lines 597-634)
- Passed click handler and tracked ID to EnhancedViewer (lines 1520-1527)

---

## Testing Checklist

- [ ] Smart labels avoid collisions between mice
- [ ] Labels stay within viewport when zooming/panning
- [ ] Labels prefer positions in top half and inside video bounds
- [ ] Pressing 'R' resets camera and stops tracking
- [ ] Clicking mouse hull starts tracking and shows visual indicator
- [ ] Camera follows tracked mouse during playback
- [ ] Tracked mouse has brighter fill and colored border
- [ ] Clicking another mouse switches tracking
- [ ] Pressing 'R' during tracking stops following

---

## Future Enhancements

Potential improvements to consider:
1. **Label fade**: Fade out labels when zoomed far out
2. **Smart zoom**: Auto-zoom when tracking to keep mouse at consistent size
3. **Multi-track**: Option to track multiple mice simultaneously with split screen
4. **Label animation**: Smooth transitions when labels reposition
5. **Keyboard shortcuts**: Number keys to jump to specific mice (1-9)
6. **Track history**: Show trail of tracked mouse's path
