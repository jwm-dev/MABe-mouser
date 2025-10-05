import { useState, useEffect, useCallback, useRef } from 'react'
import { useProgressiveLoading } from './hooks/useProgressiveLoading'
import { EnhancedViewer } from './components/EnhancedViewer'

import './App.css'

// Types
interface FileInfo {
  name: string
  lab: string
  path: string
  size_bytes: number
}

function App() {
  const [labFiles, setLabFiles] = useState<Record<string, FileInfo[]>>({})
  const [selectedLab, setSelectedLab] = useState<string>('')
  const [selectedFile, setSelectedFile] = useState<FileInfo | null>(null)
  const [currentFrame, setCurrentFrame] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [timelineHovered, setTimelineHovered] = useState(false)
  const [timelineMouseX, setTimelineMouseX] = useState<number | null>(null)
  const [isDragging, setIsDragging] = useState(false)
  const [hoveredBehavior, setHoveredBehavior] = useState<string | null>(null)
  const [forceRedraw, setForceRedraw] = useState(0)
  const [framesLoaded, setFramesLoaded] = useState(0) // Track when frames finish loading
  const [sidebarVisible, setSidebarVisible] = useState(false) // Control sidebar visibility
  const [showIntro, setShowIntro] = useState(true) // Control intro animation
  
  // Skip intro on any user interaction
  useEffect(() => {
    const skipIntro = () => setShowIntro(false)
    
    window.addEventListener('keydown', skipIntro)
    window.addEventListener('mousedown', skipIntro)
    window.addEventListener('touchstart', skipIntro)
    
    // Auto-hide intro after 6 seconds (full cycle: cat -> mouse -> cat)
    const timer = setTimeout(() => setShowIntro(false), 6000)
    
    return () => {
      window.removeEventListener('keydown', skipIntro)
      window.removeEventListener('mousedown', skipIntro)
      window.removeEventListener('touchstart', skipIntro)
      clearTimeout(timer)
    }
  }, [])
  
  // Track global mouse position for sidebar hover trigger
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      // Auto-show sidebar when mouse is near left edge (within 50px)
      // But NOT if:
      // - Currently scrubbing
      // - Near bottom of screen (timeline/playback controls area - bottom 200px)
      const isNearBottom = e.clientY > window.innerHeight - 200
      if (e.clientX < 50 && !sidebarVisible && !isDragging && !isNearBottom) {
        setSidebarVisible(true)
      }
    }
    
    window.addEventListener('mousemove', handleMouseMove)
    return () => window.removeEventListener('mousemove', handleMouseMove)
  }, [sidebarVisible, isDragging])
  
  // Use progressive loading hook
  const { 
    frames, 
    metadata, 
    loading, 
    progress, 
    error
  } = useProgressiveLoading({
    lab: selectedFile?.lab || '',
    filename: selectedFile?.name || '',
    chunkSize: 100,
    onProgress: (loaded, total) => {
      if (loaded % 1000 === 0) {
        console.log(`📦 Progressive load: ${loaded}/${total} frames (${((loaded/total)*100).toFixed(1)}%)`)
      }
    }
  })
  
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const framesRef = useRef<any[]>([]) // Keep latest frames reference for canvas drawing

  // Update frames ref whenever frames change
  useEffect(() => {
    if (frames && frames.length > 0) {
      framesRef.current = frames
    }
  }, [frames, frames?.length])

  // Load file list on mount
  useEffect(() => {
    fetch('/api/files')
      .then(res => res.json())
      .then(data => {
        setLabFiles(data.labs || {})
        // Auto-select first lab
        const firstLab = Object.keys(data.labs || {})[0]
        if (firstLab) {
          setSelectedLab(firstLab)
        }
      })
      .catch(err => console.error('Failed to load file list:', err))
  }, [])

  // Load file data when selected (just update state, hook will auto-trigger)
  const loadFile = useCallback(async (file: FileInfo) => {
    setCurrentFrame(0)
    setPlaying(false) // Stop playback when switching files
    console.log(`🚀 Starting progressive load: ${file.name}`)
  }, [])

  // Playback animation (FPS-based timing from metadata)
  useEffect(() => {
    if (!playing || !frames || frames.length === 0 || !metadata) return

    // Use actual FPS from metadata, default to 30
    const fps = metadata.fps || 30
    const frameInterval = 1000 / fps

    console.log(`▶️ Playing at ${fps} FPS (${frameInterval.toFixed(1)}ms per frame)`)

    const interval = setInterval(() => {
      setCurrentFrame(prev => {
        if (prev >= frames.length - 1) {
          setPlaying(false)
          return prev
        }
        return prev + 1
      })
    }, frameInterval)

    return () => clearInterval(interval)
  }, [playing, frames, metadata])

  // Update browser tab title
  useEffect(() => {
    if (playing && frames && frames.length > 0 && metadata) {
      // Show playback timer when playing
      const fps = metadata.fps || 30
      const currentTime = currentFrame / fps
      const totalTime = frames.length / fps
      
      const formatTime = (seconds: number) => {
        const mins = Math.floor(seconds / 60)
        const secs = Math.floor(seconds % 60)
        return `${mins}:${secs.toString().padStart(2, '0')}`
      }
      
      document.title = `${formatTime(currentTime)} / ${formatTime(totalTime)} - MABe Mouser`
    } else if (selectedFile) {
      // Show filename when file is loaded but not playing
      document.title = `${selectedFile.name} - MABe Mouser`
    } else {
      // Default title
      document.title = 'MABe Mouser'
    }
  }, [playing, currentFrame, frames, metadata, selectedFile])

  // Keyboard shortcuts
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      // Play/Pause: Space or K (YouTube-style)
      if (e.key === ' ' || e.key === 'k' || e.key === 'K') {
        e.preventDefault()
        setPlaying(p => !p)
      } 
      // Previous frame: Left Arrow or J (YouTube-style)
      else if (e.key === 'ArrowLeft' || e.key === 'j' || e.key === 'J') {
        e.preventDefault()
        setCurrentFrame(f => Math.max(0, f - 1))
      } 
      // Next frame: Right Arrow or L (YouTube-style)
      else if (e.key === 'ArrowRight' || e.key === 'l' || e.key === 'L') {
        e.preventDefault()
        setCurrentFrame(f => Math.min((frames?.length || 1) - 1, f + 1))
      }
    }

    window.addEventListener('keydown', handleKey)
    return () => window.removeEventListener('keydown', handleKey)
  }, [frames])

  // Draw timeline
  useEffect(() => {
    if (!canvasRef.current || !metadata) return
    
    // Use framesRef.current to get latest frames data (avoid stale closure)
    const currentFrames = framesRef.current
    if (!currentFrames || currentFrames.length === 0) return

    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')!
    const rect = canvas.getBoundingClientRect()
    canvas.width = rect.width * window.devicePixelRatio
    canvas.height = rect.height * window.devicePixelRatio
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio)

    // Clear with dark background
    ctx.fillStyle = 'rgba(15, 15, 30, 0.8)'
    ctx.fillRect(0, 0, rect.width, rect.height)

    // Use metadata.total_frames for accurate progress calculation
    const totalFrames = metadata.total_frames || currentFrames.length
    const loadedProgress = currentFrames.length / totalFrames
    const frameProgress = currentFrame / totalFrames
    
    // When expanded (hovered), draw frame preview filmstrip in the loaded region
    // Use timelineHovered state only - don't check height (it transitions slowly)
    if (timelineHovered) {
      // Only draw thumbnails if height is actually expanded enough
      // This prevents weird rendering during the transition
      if (rect.height > 20) {
        // Create smooth filmstrip of thumbnails
        const thumbHeight = rect.height
        const aspectRatio = 4 / 3 // Approximate aspect ratio for mouse tracking data visualization
        const thumbWidth = thumbHeight * aspectRatio
        const loadedWidth = rect.width * loadedProgress
        const numThumbs = Math.max(1, Math.ceil(loadedWidth / thumbWidth))
        
        // Draw subtle background for thumbnail area (no inset)
        ctx.fillStyle = 'rgba(15, 15, 25, 0.95)'
        ctx.fillRect(0, 0, loadedWidth, thumbHeight)
        
        // Draw continuous filmstrip across the loaded region (NO GAPS)
        for (let i = 0; i < numThumbs; i++) {
          // Calculate which frame this thumbnail represents
          // Map thumbnail position to actual loaded frame index
          const thumbStartX = i * thumbWidth
          const thumbEndX = Math.min((i + 1) * thumbWidth, loadedWidth)
          const thumbCenterX = (thumbStartX + thumbEndX) / 2
          
          // Get the frame at this position in the LOADED frames
          const thumbFrameIndex = Math.floor((thumbCenterX / loadedWidth) * currentFrames.length)
          const clampedIndex = Math.max(0, Math.min(thumbFrameIndex, currentFrames.length - 1))
          
          const x = thumbStartX
          const currentThumbWidth = Math.min(thumbWidth, loadedWidth - x)
          
          if (currentThumbWidth > 2 && clampedIndex < currentFrames.length) {
            const frame = currentFrames[clampedIndex]
            
            // NO background inset - draw edge to edge for smooth filmstrip
            ctx.fillStyle = 'rgba(25, 25, 40, 0.95)'
            ctx.fillRect(x, 0, currentThumbWidth, thumbHeight)
            
            // Render simplified visualization of the frame data
            // Draw a subtle representation based on frame data presence
            if (frame && frame.mice) {
              // Collect all points from all mice in this frame
              const allPoints: number[][] = []
              Object.values(frame.mice).forEach((mouse: any) => {
                if (mouse.points && mouse.points.length > 0) {
                  allPoints.push(...mouse.points)
                }
              })
              
              if (allPoints.length > 0) {
                // Calculate bounds for this frame's data to create a unique thumbnail
                let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity
                
                allPoints.forEach((pt) => {
                  if (pt[0] < minX) minX = pt[0]
                  if (pt[0] > maxX) maxX = pt[0]
                  if (pt[1] < minY) minY = pt[1]
                  if (pt[1] > maxY) maxY = pt[1]
                })
                
                const dataWidth = maxX - minX
                const dataHeight = maxY - minY
                const padding = 4  // Reduced padding
                const scale = Math.min(
                  (currentThumbWidth - padding * 2) / (dataWidth || 1),
                  (thumbHeight - padding * 2) / (dataHeight || 1)
                )
                
                const offsetX = x + (currentThumbWidth - (dataWidth * scale)) / 2
                const offsetY = (thumbHeight - (dataHeight * scale)) / 2
                
                // Draw points as a mini visualization with varying colors
                const baseHue = (clampedIndex * 137.5) % 360 // Golden angle for color distribution
                
                allPoints.slice(0, 30).forEach((pt, idx) => {
                  const px = offsetX + (pt[0] - minX) * scale
                  const py = offsetY + (pt[1] - minY) * scale
                  
                  if (px >= x && px <= x + currentThumbWidth && py >= 0 && py <= thumbHeight) {
                    // Color gradient based on point index - more vibrant
                    const hue = (baseHue + idx * 10) % 360
                    ctx.fillStyle = `hsla(${hue}, 75%, 68%, 0.9)`
                    ctx.beginPath()
                    ctx.arc(px, py, 1.5, 0, Math.PI * 2)
                    ctx.fill()
                  }
                })
              } else {
                // No points - show smooth gradient
                const gradient = ctx.createLinearGradient(x, 0, x, thumbHeight)
                const hue = (clampedIndex * 30) % 360
                gradient.addColorStop(0, `hsla(${hue}, 55%, 42%, 0.6)`)
                gradient.addColorStop(1, `hsla(${(hue + 60) % 360}, 55%, 32%, 0.6)`)
                ctx.fillStyle = gradient
                ctx.fillRect(x, 0, currentThumbWidth, thumbHeight)
              }
            } else {
              // Fallback: smooth gradient pattern
              const gradient = ctx.createLinearGradient(x, 0, x, thumbHeight)
              const hue = (clampedIndex * 30) % 360
              gradient.addColorStop(0, `hsla(${hue}, 55%, 42%, 0.6)`)
              gradient.addColorStop(1, `hsla(${(hue + 60) % 360}, 55%, 32%, 0.6)`)
              ctx.fillStyle = gradient
              ctx.fillRect(x, 0, currentThumbWidth, thumbHeight)
            }
            
            // NO separators between thumbnails for smooth filmstrip look
          }
        }
      }
      
      // NO border around filmstrip for cleaner look
      
      // NO gray loading bar when expanded - thumbnails show the loaded region
    } else {
      // When collapsed, show clean progress bars only (no thumbnails)
      // Gray loading bar shows buffered/loaded content
      ctx.fillStyle = 'rgba(255, 255, 255, 0.2)'
      ctx.fillRect(0, 0, rect.width * loadedProgress, rect.height)
    }
    
    // Draw playback progress with gradient
    const gradient = ctx.createLinearGradient(0, 0, rect.width * frameProgress, 0)
    if (timelineHovered) {
      // When expanded, more transparent overlay so thumbnails show through
      gradient.addColorStop(0, 'rgba(99, 102, 241, 0.35)')
      gradient.addColorStop(1, 'rgba(139, 92, 246, 0.35)')
    } else {
      // When collapsed, solid progress bar
      gradient.addColorStop(0, 'rgba(99, 102, 241, 0.8)')
      gradient.addColorStop(1, 'rgba(139, 92, 246, 0.8)')
    }
    ctx.fillStyle = gradient
    ctx.fillRect(0, 0, rect.width * frameProgress, rect.height)

    // Draw current position marker
    const markerX = (currentFrame / totalFrames) * rect.width
    
    if (timelineHovered) {
      // Expanded: vertical line with glow
      ctx.shadowBlur = 15
      ctx.shadowColor = 'rgba(167, 139, 250, 0.8)'
      ctx.strokeStyle = '#a78bfa'
      ctx.lineWidth = 3
      ctx.beginPath()
      ctx.moveTo(markerX, 0)
      ctx.lineTo(markerX, rect.height)
      ctx.stroke()
      ctx.shadowBlur = 0
    } else {
      // Collapsed: circular marker (dot)
      ctx.shadowBlur = 8
      ctx.shadowColor = 'rgba(167, 139, 250, 0.8)'
      ctx.fillStyle = '#a78bfa'
      ctx.beginPath()
      ctx.arc(markerX, rect.height / 2, 4, 0, Math.PI * 2)
      ctx.fill()
      ctx.shadowBlur = 0
    }
  }, [currentFrame, frames, frames?.length, metadata, timelineHovered, loading, progress, forceRedraw])

  // Force continuous redraws during canvas transition
  useEffect(() => {
    if (!timelineHovered || !canvasRef.current) return
    
    let animationFrame: number
    const startTime = Date.now()
    const transitionDuration = 250 // Match CSS transition duration
    
    const animate = () => {
      const elapsed = Date.now() - startTime
      if (elapsed < transitionDuration) {
        setForceRedraw(prev => prev + 1) // Trigger redraw
        animationFrame = requestAnimationFrame(animate)
      } else {
        setForceRedraw(prev => prev + 1) // Final redraw when settled
      }
    }
    
    animationFrame = requestAnimationFrame(animate)
    
    return () => {
      if (animationFrame) cancelAnimationFrame(animationFrame)
    }
  }, [timelineHovered])

  // Track when frames finish loading
  useEffect(() => {
    if (!loading && frames && frames.length > 0 && metadata?.total_frames) {
      // Check if we've loaded all frames
      if (frames.length >= metadata.total_frames) {
        setFramesLoaded(prev => prev + 1)
      }
    }
  }, [loading, frames?.length, metadata?.total_frames])
  
  // Force redraw when all frames are loaded
  // Force redraw when all frames are loaded
  useEffect(() => {
    if (framesLoaded > 0) {
      setForceRedraw(prev => prev + 1)
    }
  }, [framesLoaded, frames?.length])
  // Handle global mouseup and mousemove for drag scrubbing
  useEffect(() => {
    const handleGlobalMouseUp = () => {
      setIsDragging(false)
      setTimelineHovered(false) // Collapse timeline after scrubbing ends
      setTimelineMouseX(null)
      // Restore cursor completely
      document.body.style.cursor = ''
      if (canvasRef.current) {
        canvasRef.current.style.cursor = 'pointer'
      }
    }
    
    const handleGlobalMouseMove = (e: MouseEvent) => {
      if (isDragging && canvasRef.current && metadata) {
        const rect = canvasRef.current.getBoundingClientRect()
        const x = e.clientX - rect.left
        
        // Keep timeline expanded and hide cursor during scrubbing
        setTimelineHovered(true)
        document.body.style.cursor = 'none'
        
        // Allow scrubbing even if mouse is outside canvas bounds
        const totalFrames = metadata.total_frames || (frames?.length || 0)
        const clampedX = Math.max(0, Math.min(x, rect.width))
        const frame = Math.floor((clampedX / rect.width) * totalFrames)
        setCurrentFrame(Math.max(0, Math.min(frame, totalFrames - 1)))
      }
    }
    
    if (isDragging) {
      // Hide cursor globally when scrubbing starts
      document.body.style.cursor = 'none'
      
      window.addEventListener('mouseup', handleGlobalMouseUp)
      window.addEventListener('mousemove', handleGlobalMouseMove)
      return () => {
        window.removeEventListener('mouseup', handleGlobalMouseUp)
        window.removeEventListener('mousemove', handleGlobalMouseMove)
        // Ensure cursor is restored when cleaning up
        document.body.style.cursor = ''
      }
    }
  }, [isDragging, metadata, frames])

  // Initialize camera based on metadata
  const [viewState, setViewState] = useState({
    target: [0, 0, 0],
    zoom: -1
  })

  // Calculate active behaviors for current frame
  const activeBehaviors = useCallback(() => {
    if (!metadata?.annotations || !frames || frames.length === 0) return []
    
    const frameNumber = frames[currentFrame]?.frame_number || 0
    
    return metadata.annotations.filter((annotation: any) => 
      frameNumber >= annotation.start_frame && frameNumber <= annotation.stop_frame
    )
  }, [metadata, frames, currentFrame])

  // Update camera when metadata loads
  useEffect(() => {
    if (!metadata) return
    
    // Calculate appropriate zoom to fit content
    let width = 640  // default
    let height = 480 // default
    
    if (metadata.video_width && metadata.video_height) {
      width = metadata.video_width
      height = metadata.video_height
    } else if (metadata.arena_width_cm && metadata.arena_height_cm && metadata.pix_per_cm) {
      width = metadata.arena_width_cm * metadata.pix_per_cm
      height = metadata.arena_height_cm * metadata.pix_per_cm
    }
    
    // Fit to viewport (assuming 1000x800 viewport)
    const viewportWidth = 1000
    const viewportHeight = 800
    const scaleX = viewportWidth / width
    const scaleY = viewportHeight / height
    const scale = Math.min(scaleX, scaleY) * 0.9  // 90% to add margins
    
    // deck.gl zoom is log2(scale)
    const zoom = Math.log2(scale)
    
    setViewState({
      target: [0, 0, 0],  // Center on origin
      zoom
    })
  }, [metadata])

  return (
    <>
      {/* Global cursor hiding during scrubbing */}
      {isDragging && (
        <style>{`
          * { cursor: none !important; }
        `}</style>
      )}
      
      {/* Intro Animation */}
      {showIntro && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          width: '100vw',
          height: '100vh',
          background: 'linear-gradient(135deg, #0a0a15 0%, #1a1a2e 100%)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 10000,
          animation: 'introFadeOut 0.5s ease-out 5.5s forwards'
        }}>
          {/* Ripple effects */}
          <div style={{
            position: 'absolute',
            width: '400px',
            height: '400px',
            borderRadius: '50%',
            border: '2px solid rgba(102, 126, 234, 0.3)',
            animation: 'ripple 3s ease-out infinite'
          }} />
          <div style={{
            position: 'absolute',
            width: '400px',
            height: '400px',
            borderRadius: '50%',
            border: '2px solid rgba(118, 75, 162, 0.3)',
            animation: 'ripple 3s ease-out 0.5s infinite'
          }} />
          <div style={{
            position: 'absolute',
            width: '400px',
            height: '400px',
            borderRadius: '50%',
            border: '2px solid rgba(139, 92, 246, 0.2)',
            animation: 'ripple 3s ease-out 1s infinite'
          }} />
          
          {/* Energy pulse rings - first morph */}
          <div style={{
            position: 'absolute',
            width: '200px',
            height: '200px',
            borderRadius: '50%',
            border: '3px solid rgba(102, 126, 234, 0.8)',
            animation: 'energyPulse 0.8s ease-out 2s',
            opacity: 0
          }} />
          <div style={{
            position: 'absolute',
            width: '200px',
            height: '200px',
            borderRadius: '50%',
            border: '3px solid rgba(118, 75, 162, 0.8)',
            animation: 'energyPulse 0.8s ease-out 2.1s',
            opacity: 0
          }} />
          
          {/* Energy pulse rings - second morph */}
          <div style={{
            position: 'absolute',
            width: '200px',
            height: '200px',
            borderRadius: '50%',
            border: '3px solid rgba(102, 126, 234, 0.8)',
            animation: 'energyPulse 0.8s ease-out 4s',
            opacity: 0
          }} />
          <div style={{
            position: 'absolute',
            width: '200px',
            height: '200px',
            borderRadius: '50%',
            border: '3px solid rgba(118, 75, 162, 0.8)',
            animation: 'energyPulse 0.8s ease-out 4.1s',
            opacity: 0
          }} />
          
          {/* Sparkle particles - first morph */}
          <div style={{
            position: 'absolute',
            width: '400px',
            height: '400px',
            pointerEvents: 'none'
          }}>
            {[...Array(20)].map((_, i) => {
              const angle = (i / 20) * 2 * Math.PI
              const distance = 60 + Math.random() * 80
              const tx = Math.cos(angle) * distance
              const ty = Math.sin(angle) * distance
              const delay = 1.8 + (i * 0.02)
              
              return (
                <div
                  key={`sparkle1-${i}`}
                  style={{
                    position: 'absolute',
                    left: '50%',
                    top: '50%',
                    width: '4px',
                    height: '4px',
                    background: i % 3 === 0 ? '#667eea' : i % 3 === 1 ? '#764ba2' : '#8b5cf6',
                    borderRadius: '50%',
                    animation: `sparkle 0.8s ease-out ${delay}s`,
                    opacity: 0,
                    boxShadow: `0 0 8px ${i % 3 === 0 ? '#667eea' : i % 3 === 1 ? '#764ba2' : '#8b5cf6'}`,
                    // @ts-ignore - CSS custom properties
                    '--tx': `${tx}px`,
                    '--ty': `${ty}px`
                  }}
                />
              )
            })}
          </div>
          
          {/* Sparkle particles - second morph */}
          <div style={{
            position: 'absolute',
            width: '400px',
            height: '400px',
            pointerEvents: 'none'
          }}>
            {[...Array(20)].map((_, i) => {
              const angle = (i / 20) * 2 * Math.PI + Math.PI / 20 // Offset from first set
              const distance = 60 + Math.random() * 80
              const tx = Math.cos(angle) * distance
              const ty = Math.sin(angle) * distance
              const delay = 3.8 + (i * 0.02)
              
              return (
                <div
                  key={`sparkle2-${i}`}
                  style={{
                    position: 'absolute',
                    left: '50%',
                    top: '50%',
                    width: '4px',
                    height: '4px',
                    background: i % 3 === 0 ? '#667eea' : i % 3 === 1 ? '#764ba2' : '#8b5cf6',
                    borderRadius: '50%',
                    animation: `sparkle 0.8s ease-out ${delay}s`,
                    opacity: 0,
                    boxShadow: `0 0 8px ${i % 3 === 0 ? '#667eea' : i % 3 === 1 ? '#764ba2' : '#8b5cf6'}`,
                    // @ts-ignore - CSS custom properties
                    '--tx': `${tx}px`,
                    '--ty': `${ty}px`
                  }}
                />
              )
            })}
          </div>
          
          {/* Main logo container */}
          <div style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            gap: '40px', // Increased from 20px to prevent clipping
            animation: 'introExpand 1.2s ease-out',
            position: 'relative',
            zIndex: 1
          }}>
            {/* Animated emoji that morphs from cat to mouse */}
            <div style={{
              position: 'relative',
              fontSize: '120px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              minHeight: '140px', // Reserve space to prevent clipping
              minWidth: '140px'
            }}>
              {/* Cat emoji - initial display, fades out, fades back in */}
              <span style={{
                position: 'absolute',
                background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                filter: 'drop-shadow(0 0 20px rgba(102, 126, 234, 0.6))',
                animation: 'introGlow 3s ease-in-out infinite, fadeOut 0.6s ease-out 1.7s forwards, fadeIn 0.6s ease-out 4.2s forwards'
              }}>
                🐱
              </span>
              
              {/* Mouse emoji - fades in at middle, fades out */}
              <span style={{
                position: 'absolute',
                background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                filter: 'drop-shadow(0 0 20px rgba(118, 75, 162, 0.6))',
                opacity: 0,
                animation: 'introGlow 3s ease-in-out infinite, fadeIn 0.6s ease-out 2.2s forwards, fadeOut 0.6s ease-out 3.8s forwards'
              }}>
                🐭
              </span>
            </div>
            
            <div style={{
              fontSize: '48px',
              fontWeight: '700',
              background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              letterSpacing: '-1px',
              animation: 'introExpand 1s ease-out 0.3s backwards'
            }}>
              MABe Mouser
            </div>
            <div style={{
              fontSize: '14px',
              color: 'rgba(229, 231, 235, 0.5)',
              textTransform: 'uppercase',
              letterSpacing: '3px',
              fontWeight: '500',
              animation: 'fadeIn 0.8s ease-out 0.7s backwards'
            }}>
              Mouse Behavior Analysis
            </div>
          </div>
          
          {/* Skip hint */}
          <div style={{
            position: 'absolute',
            bottom: '40px',
            fontSize: '12px',
            color: 'rgba(229, 231, 235, 0.3)',
            animation: 'fadeIn 0.5s ease-out 1.5s backwards'
          }}>
            Click or press any key to skip
          </div>
        </div>
      )}
      
      <div style={{ 
        display: 'flex', 
        height: '100vh', 
        width: '100vw',
        background: 'rgba(15, 15, 30, 1)', // Match viewer background - solid dark color
        fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
        position: 'relative',
        overflow: 'hidden'
      }}>
      {/* Sidebar - slides in from left */}
      <div 
        onMouseLeave={() => setSidebarVisible(false)}
        style={{
        position: 'fixed',
        left: sidebarVisible ? '0' : '-280px',
        top: '0',
        height: '100vh',
        width: '280px',
        background: 'rgba(20, 20, 35, 0.6)',
        backdropFilter: 'blur(20px)',
        borderRight: sidebarVisible ? 'none' : '1px solid rgba(255, 255, 255, 0.1)', // Hide border when expanded
        padding: '20px',
        overflowY: 'auto',
        boxShadow: '2px 0 20px rgba(0, 0, 0, 0.3)',
        transition: 'left 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
        zIndex: 999
      }}>
        <h1 style={{ 
          fontSize: '26px', 
          fontWeight: '700', 
          marginBottom: '24px',
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
          letterSpacing: '-0.5px',
          display: 'flex',
          alignItems: 'center',
          gap: '10px'
        }}>
          <span style={{ fontSize: '28px' }}>🐱</span>
          <span>MABe Mouser</span>
        </h1>
        
        {selectedFile && (
          <div style={{
            padding: '14px 16px',
            background: loading 
              ? 'linear-gradient(135deg, rgba(59, 130, 246, 0.15) 0%, rgba(96, 165, 250, 0.15) 100%)'
              : 'linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.15) 100%)',
            backdropFilter: 'blur(10px)',
            border: '1px solid ' + (loading ? 'rgba(59, 130, 246, 0.3)' : 'rgba(16, 185, 129, 0.3)'),
            borderRadius: '12px',
            marginBottom: '20px',
            fontSize: '14px',
            color: '#e5e7eb',
            boxShadow: '0 4px 15px rgba(0, 0, 0, 0.2)'
          }}>
            {loading ? (
              <>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <div style={{
                    width: '12px',
                    height: '12px',
                    borderRadius: '50%',
                    border: '2px solid rgba(59, 130, 246, 0.3)',
                    borderTopColor: '#3b82f6',
                    animation: 'spin 1s linear infinite'
                  }} />
                  <span>Loading {progress.toFixed(0)}%</span>
                </div>
                <div style={{
                  width: '100%',
                  height: '6px',
                  background: 'rgba(0, 0, 0, 0.3)',
                  borderRadius: '3px',
                  overflow: 'hidden',
                  marginTop: '10px'
                }}>
                  <div style={{
                    width: `${progress}%`,
                    height: '100%',
                    background: 'linear-gradient(90deg, #3b82f6 0%, #60a5fa 100%)',
                    borderRadius: '3px',
                    transition: 'width 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
                    boxShadow: '0 0 10px rgba(59, 130, 246, 0.5)'
                  }} />
                </div>
              </>
            ) : metadata ? (
              <>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <div style={{
                    width: '8px',
                    height: '8px',
                    borderRadius: '50%',
                    background: '#10b981',
                    boxShadow: '0 0 8px rgba(16, 185, 129, 0.6)'
                  }} />
                  <span style={{ fontWeight: '500' }}>Ready</span>
                </div>
                <div style={{ 
                  fontSize: '12px', 
                  marginTop: '6px', 
                  color: 'rgba(229, 231, 235, 0.7)',
                  fontWeight: '400'
                }}>
                  {metadata.total_frames.toLocaleString()} frames • {metadata.num_mice} mice
                </div>
              </>
            ) : null}
            {error && (
              <div style={{ 
                marginTop: '10px', 
                color: '#fca5a5',
                fontSize: '13px',
                display: 'flex',
                alignItems: 'center',
                gap: '6px'
              }}>
                <span>⚠</span>
                <span>{error}</span>
              </div>
            )}
          </div>
        )}

        <h2 style={{ 
          fontSize: '12px', 
          fontWeight: '600', 
          marginBottom: '10px', 
          color: 'rgba(229, 231, 235, 0.5)',
          textTransform: 'uppercase',
          letterSpacing: '1px'
        }}>
          Lab
        </h2>
        
        <select
          value={selectedLab}
          onChange={(e) => setSelectedLab(e.target.value)}
          style={{
            width: '100%',
            padding: '10px 12px',
            marginBottom: '20px',
            borderRadius: '8px',
            border: '1px solid rgba(167, 139, 250, 0.3)',
            background: 'linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%)',
            backdropFilter: 'blur(12px)',
            color: '#e5e7eb',
            fontSize: '14px',
            cursor: 'pointer',
            transition: 'all 0.2s ease',
            boxShadow: '0 2px 8px rgba(0, 0, 0, 0.2)'
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.background = 'linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%)'
            e.currentTarget.style.borderColor = 'rgba(167, 139, 250, 0.5)'
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.background = 'linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%)'
            e.currentTarget.style.borderColor = 'rgba(167, 139, 250, 0.3)'
          }}
        >
          <option value="" style={{ background: 'rgba(20, 20, 35, 0.95)', color: '#e5e7eb' }}>Select a lab...</option>
          {Object.keys(labFiles).sort().map(lab => (
            <option key={lab} value={lab} style={{ background: 'rgba(20, 20, 35, 0.95)', color: '#e5e7eb' }}>{lab}</option>
          ))}
        </select>

        {selectedLab && (
          <>
            <h2 style={{ 
              fontSize: '12px', 
              fontWeight: '600', 
              marginBottom: '10px', 
              color: 'rgba(229, 231, 235, 0.5)',
              textTransform: 'uppercase',
              letterSpacing: '1px'
            }}>
              Files ({labFiles[selectedLab]?.length || 0})
            </h2>
            
            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
              {(labFiles[selectedLab] || []).map((file) => (
                <button
                  key={file.path}
                  onClick={() => {
                    setSelectedFile(file)
                    loadFile(file)
                  }}
                  style={{
                    width: '100%',
                    textAlign: 'left',
                    padding: '12px 14px',
                    borderRadius: '10px',
                    border: selectedFile?.path === file.path 
                      ? '1px solid rgba(99, 102, 241, 0.4)' 
                      : '1px solid transparent',
                    background: selectedFile?.path === file.path 
                      ? 'linear-gradient(135deg, rgba(99, 102, 241, 0.2) 0%, rgba(139, 92, 246, 0.2) 100%)' 
                      : 'rgba(255, 255, 255, 0.03)',
                    backdropFilter: 'blur(10px)',
                    color: '#e5e7eb',
                    cursor: 'pointer',
                    transition: 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
                    boxShadow: selectedFile?.path === file.path 
                      ? '0 4px 12px rgba(99, 102, 241, 0.2)' 
                      : '0 2px 8px rgba(0, 0, 0, 0.1)'
                  }}
                  onMouseEnter={(e) => {
                    if (selectedFile?.path !== file.path) {
                      e.currentTarget.style.background = 'rgba(255, 255, 255, 0.08)'
                      e.currentTarget.style.transform = 'translateX(2px)'
                    }
                  }}
                  onMouseLeave={(e) => {
                    if (selectedFile?.path !== file.path) {
                      e.currentTarget.style.background = 'rgba(255, 255, 255, 0.03)'
                      e.currentTarget.style.transform = 'translateX(0)'
                    }
                  }}
                >
                  <div style={{ 
                    fontWeight: '500', 
                    fontSize: '14px', 
                    marginBottom: '4px',
                    color: '#f3f4f6'
                  }}>
                    {file.name.replace('.parquet', '')}
                  </div>
                  <div style={{ 
                    fontSize: '11px', 
                    color: 'rgba(229, 231, 235, 0.5)',
                    fontWeight: '400'
                  }}>
                    {(file.size_bytes / 1024 / 1024).toFixed(1)} MB
                  </div>
                </button>
              ))}
            </div>
          </>
        )}
      </div>

      {/* Main viewer - shifts right when sidebar visible */}
      <div style={{ 
        flex: 1, 
        display: 'flex', 
        flexDirection: 'column', 
        position: 'relative',
        transform: sidebarVisible ? 'translateX(280px)' : 'translateX(0)', // Use transform instead of margin
        transition: 'transform 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
        willChange: 'transform'
      }}>
        {/* Top bar */}
        <div style={{
          background: 'rgba(15, 15, 30, 0.95)', // Match viewer background
          backdropFilter: 'blur(20px)',
          borderBottom: 'none', // Remove border, we'll add it separately
          padding: '12px 20px',
          paddingLeft: sidebarVisible ? '20px' : '90px', // Make room for logo when visible
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          boxShadow: '0 2px 10px rgba(0, 0, 0, 0.3)',
          position: 'relative',
          overflow: 'visible',
          zIndex: 100,
          minHeight: '60px', // Ensure consistent height
          transition: 'padding-left 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
          clipPath: sidebarVisible 
            ? 'none' 
            : 'polygon(70px 0, 100% 0, 100% 100%, 70px 100%, 70px calc(100% + 20px), 0 calc(100% + 20px), 0 0, 70px 0)'
        }}>
          {/* Background layer for logo - viewer background color */}
          <div style={{
            position: 'absolute',
            left: '0',
            top: '0',
            height: '80px', // Same as logo
            width: '70px',
            background: 'rgba(15, 15, 30, 1)', // Solid viewer background color
            borderBottomRightRadius: '28px',
            zIndex: 9,
            opacity: sidebarVisible ? 0 : 1,
            transition: 'opacity 0.3s cubic-bezier(0.4, 0, 0.2, 1)'
          }} />
          
          {/* Logo button - integrated into top bar, extends below bar */}
          <div
            onMouseEnter={() => setSidebarVisible(true)}
            style={{
              position: 'absolute',
              left: '0',
              top: '0',
              height: '80px', // Fixed height: 60px bar + 20px extension
              width: '70px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              cursor: 'pointer',
              background: 'linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%)', // Glassy purple
              backdropFilter: 'blur(10px)',
              borderRight: '1px solid rgba(255, 255, 255, 0.15)',
              borderBottomRightRadius: '28px', // Larger radius for the extended bottom
              transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
              opacity: sidebarVisible ? 0 : 1,
              pointerEvents: sidebarVisible ? 'none' : 'auto',
              transform: sidebarVisible ? 'scale(0.9) translateX(-10px)' : 'scale(1) translateX(0)',
              zIndex: 10
            }}
            onMouseOver={(e) => {
              if (!sidebarVisible) {
                e.currentTarget.style.background = 'linear-gradient(135deg, rgba(102, 126, 234, 0.25) 0%, rgba(118, 75, 162, 0.25) 100%)'
                e.currentTarget.style.borderRight = '1px solid rgba(255, 255, 255, 0.25)'
              }
            }}
            onMouseOut={(e) => {
              if (!sidebarVisible) {
                e.currentTarget.style.background = 'linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%)'
                e.currentTarget.style.borderRight = '1px solid rgba(255, 255, 255, 0.15)'
              }
            }}
          >
            <span style={{ 
              fontSize: '32px',
              background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              filter: 'drop-shadow(0 2px 4px rgba(102, 126, 234, 0.3))'
            }}>🐱</span>
          </div>
          
          {/* Top bar bottom border - hidden when sidebar expanded */}
          <div style={{
            position: 'absolute',
            left: '0',
            right: '0',
            bottom: '-1px',
            height: '1px',
            background: 'rgba(255, 255, 255, 0.1)',
            zIndex: 5,
            opacity: sidebarVisible ? 0 : 1,
            transition: 'opacity 0.3s cubic-bezier(0.4, 0, 0.2, 1)'
          }} />
          
          {/* Centered title when no file is loaded */}
          {!selectedFile && (
            <div style={{
              position: 'absolute',
              left: '50%',
              top: '50%',
              transform: 'translate(-50%, -50%)',
              fontSize: '26px',
              fontWeight: '700',
              background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              letterSpacing: '-0.5px',
              display: 'flex',
              alignItems: 'center',
              gap: '10px',
              opacity: sidebarVisible ? 0 : 1,
              transition: 'opacity 0.3s ease-out',
              pointerEvents: sidebarVisible ? 'none' : 'auto'
            }}>
              <span style={{ fontSize: '28px' }}>🐱</span>
              <span>MABe Mouser</span>
            </div>
          )}
          
          <div style={{ 
            fontSize: '13px',
            color: 'rgba(229, 231, 235, 0.7)',
            fontWeight: '400',
            marginLeft: sidebarVisible ? '0' : '0', // No extra margin needed, padding handles it
            transition: 'margin-left 0.3s cubic-bezier(0.4, 0, 0.2, 1)'
          }}>
            {selectedFile && (
              <>
                <span style={{ color: 'rgba(167, 139, 250, 0.8)' }}>{selectedFile.lab}</span>
                {' / '}
                <span style={{ color: '#e5e7eb' }}>{selectedFile.name}</span>
              </>
            )}
          </div>
          
          <div style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
            {frames && frames.length > 0 && (
              <>
                <div style={{ 
                  fontSize: '13px', 
                  color: 'rgba(229, 231, 235, 0.7)',
                  fontFamily: 'monospace'
                }}>
                  <span style={{ color: '#a78bfa' }}>Frame</span> {currentFrame + 1} / {frames.length}
                  {metadata && metadata.fps && (
                    <span style={{ marginLeft: '12px', color: 'rgba(96, 165, 250, 0.8)' }}>
                      {metadata.fps} FPS
                    </span>
                  )}
                  {loading && metadata && frames.length < metadata.total_frames && (
                    <span style={{ color: 'rgba(96, 165, 250, 0.7)', marginLeft: '12px' }}>
                      ({((frames.length / metadata.total_frames) * 100).toFixed(0)}%)
                    </span>
                  )}
                </div>
              </>
            )}
          </div>
        </div>

        {/* Viewer */}
        <div style={{ 
          position: 'absolute',
          top: '60px', // Height of top bar
          left: '0',
          right: '0',
          bottom: '0',
          background: 'linear-gradient(135deg, #0a0a15 0%, #15151f 100%)'
        }}>
          {/* Active Behaviors Panel - positioned absolutely over viewer */}
          {metadata?.has_annotations && frames && frames.length > 0 && !sidebarVisible && (
            <div 
              style={{
                position: 'absolute',
                top: 0,
                left: '90px', // Start after logo
                right: 0,
                padding: '16px 20px',
                zIndex: 100,
                pointerEvents: 'none',
                transition: 'opacity 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
                opacity: sidebarVisible ? 0 : 1
              }}>
              <div style={{
                fontSize: '11px',
                fontWeight: '600',
                color: 'rgba(229, 231, 235, 0.9)',
                textTransform: 'uppercase',
                letterSpacing: '1px',
                marginBottom: '10px',
                background: 'rgba(20, 20, 35, 0.9)', // Match top bar background
                backdropFilter: 'blur(12px)',
                padding: '6px 12px',
                borderRadius: '6px',
                display: 'inline-block',
                border: '1px solid rgba(255, 255, 255, 0.1)',
                pointerEvents: 'auto',
                float: 'right' // Align to the right
              }}>
                Active Behaviors
              </div>
              <div style={{ clear: 'both' }} /> {/* Clear float */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', pointerEvents: 'auto', alignItems: 'flex-end' }}>
                {activeBehaviors().length > 0 ? (
                  activeBehaviors().map((behavior, idx) => {
                    // Color code by action type
                    const actionColors: Record<string, string> = {
                      'chase': 'rgba(239, 68, 68, 0.8)',    // red
                      'attack': 'rgba(239, 68, 68, 0.9)',   // red (stronger)
                      'avoid': 'rgba(59, 130, 246, 0.8)',   // blue
                      'escape': 'rgba(59, 130, 246, 0.9)',  // blue (stronger)
                      'mount': 'rgba(168, 85, 247, 0.8)',   // purple
                      'groom': 'rgba(16, 185, 129, 0.8)',   // green
                      'sniff': 'rgba(251, 191, 36, 0.8)'    // yellow
                    }
                    
                    const color = actionColors[behavior.action] || 'rgba(167, 139, 250, 0.8)'
                    
                    return (
                      <div
                        key={idx}
                        style={{
                          display: 'flex',
                          alignItems: 'center',
                          gap: '10px',
                          padding: '8px 12px',
                          background: 'rgba(15, 15, 30, 0.75)',
                          border: `1px solid ${color}`,
                          borderRadius: '6px',
                          fontSize: '13px',
                          backdropFilter: 'blur(12px)',
                          animation: 'popIn 0.25s ease-out'
                        }}
                      >
                        <div
                          style={{
                            width: '6px',
                            height: '6px',
                            borderRadius: '50%',
                            background: color,
                            boxShadow: `0 0 8px ${color}`,
                            flexShrink: 0
                          }}
                        />
                        <span style={{ 
                          color: '#e5e7eb',
                          fontWeight: '600',
                          textTransform: 'capitalize'
                        }}>
                          {behavior.action}
                        </span>
                        <span style={{ color: 'rgba(229, 231, 235, 0.6)' }}>
                          Mouse {behavior.agent_id} → Mouse {behavior.target_id}
                        </span>
                        <span style={{ 
                          marginLeft: 'auto',
                          fontSize: '11px',
                          color: 'rgba(229, 231, 235, 0.5)',
                          fontFamily: 'monospace'
                        }}>
                          {behavior.start_frame}-{behavior.stop_frame}
                        </span>
                      </div>
                    )
                  })
                ) : (
                  <div style={{
                    padding: '12px',
                    color: 'rgba(229, 231, 235, 0.5)',
                    fontSize: '12px',
                    fontStyle: 'italic',
                    textAlign: 'center',
                    background: 'rgba(15, 15, 30, 0.7)',
                    borderRadius: '6px',
                    backdropFilter: 'blur(12px)',
                    border: '1px solid rgba(255, 255, 255, 0.1)',
                    animation: 'fadeIn 0.3s ease-out'
                  }}>
                    No active behaviors in this frame
                  </div>
                )}
              </div>
            </div>
          )}
          
          {/* Show viewer as soon as we have frames, even while still loading */}
          {frames && frames.length > 0 && (
            <div style={{
              position: 'absolute',
              top: '0',
              left: '0',
              right: '0',
              bottom: '0',
              contain: 'strict', // Strict containment
              contentVisibility: 'auto',
              transform: 'translateZ(0)',
              willChange: 'contents'
            }}>
              <EnhancedViewer
                frame={frames[currentFrame]}
                metadata={metadata}
                viewState={viewState}
                onViewStateChange={({ viewState }: any) => setViewState(viewState)}
              />
            </div>
          )}

          {/* Only show loading overlay if no frames yet */}
          {loading && (!frames || frames.length === 0) && (
            <div style={{
              position: 'absolute',
              top: '50%',
              left: '50%',
              transform: 'translate(-50%, -50%)',
              textAlign: 'center',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center'
            }}>
              <div style={{
                width: '48px',
                height: '48px',
                border: '4px solid rgba(99, 102, 241, 0.2)',
                borderTopColor: '#6366f1',
                borderRadius: '50%',
                animation: 'spin 1s linear infinite',
                marginBottom: '16px'
              }} />
              <div style={{ fontSize: '16px', color: 'rgba(229, 231, 235, 0.7)' }}>
                Loading first frames...
              </div>
            </div>
          )}

          {!loading && (!frames || frames.length === 0) && (
            <>
              <div style={{
                position: 'absolute',
                top: '50%',
                left: '50%',
                transform: 'translate(-50%, -50%)',
                textAlign: 'center',
                color: 'rgba(229, 231, 235, 0.5)'
              }}>
                <div style={{ fontSize: '72px', marginBottom: '20px', opacity: 0.3 }}>🐭</div>
                <div style={{ fontSize: '20px', marginBottom: '10px', fontWeight: '500' }}>
                  Select a file to begin
                </div>
                <div style={{ fontSize: '14px', opacity: 0.7 }}>
                  Interactive mouse behavior visualization and analysis
                </div>
              </div>
              
              {/* Hand-drawn arrow pointing to logo */}
              <div style={{
                position: 'absolute',
                top: '20px',
                left: '100px',
                animation: 'bounce 2s ease-in-out infinite',
                opacity: sidebarVisible ? 0 : 1,
                transition: 'opacity 0.3s ease-out',
                pointerEvents: 'none'
              }}>
                {/* Curved arrow SVG with hand-drawn look */}
                <svg width="150" height="100" viewBox="0 0 150 100" style={{ 
                  filter: 'drop-shadow(0 2px 4px rgba(102, 126, 234, 0.3))',
                  overflow: 'visible'
                }}>
                  {/* Roughen filter for hand-drawn effect */}
                  <defs>
                    <filter id="roughen">
                      <feTurbulence type="fractalNoise" baseFrequency="0.05" numOctaves="2" result="noise" />
                      <feDisplacementMap in="SourceGraphic" in2="noise" scale="1" />
                    </filter>
                  </defs>
                  
                  {/* Curved path pointing up-left to logo - more curvy with S-curve */}
                  <path
                    d="M 80 80 Q 65 60, 50 45 Q 40 30, 18 15"
                    stroke="rgba(102, 126, 234, 0.7)"
                    strokeWidth="3"
                    fill="none"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    style={{
                      strokeDasharray: '3, 5',
                      filter: 'url(#roughen)'
                    }}
                  />
                  {/* Arrowhead pointing up-left */}
                  <path
                    d="M 18 15 L 28 15 L 20 25 Z"
                    fill="rgba(102, 126, 234, 0.7)"
                    stroke="rgba(102, 126, 234, 0.7)"
                    strokeWidth="2"
                    strokeLinejoin="round"
                  />
                </svg>
                
                {/* Hand-drawn text label */}
                <div style={{
                  position: 'absolute',
                  top: '80px',
                  left: '60px',
                  fontSize: '13px',
                  fontWeight: '500',
                  color: 'rgba(102, 126, 234, 0.9)',
                  background: 'rgba(15, 15, 30, 0.95)',
                  padding: '7px 13px',
                  borderRadius: '12px',
                  border: '2px solid rgba(102, 126, 234, 0.5)',
                  whiteSpace: 'nowrap',
                  fontStyle: 'italic',
                  transform: 'rotate(-2deg)',
                  boxShadow: '0 3px 12px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(102, 126, 234, 0.2)',
                  letterSpacing: '0.3px',
                  textShadow: '0 1px 2px rgba(0, 0, 0, 0.3)',
                  borderStyle: 'dashed',
                  borderWidth: '2px'
                }}>
                  Hover here to open files
                </div>
              </div>
            </>
          )}
        </div>

        {/* Timeline */}
        {frames && frames.length > 0 && metadata && (
          <>
            {/* Hover detection area - larger than visual timeline for better UX */}
            <div style={{
              position: 'absolute',
              bottom: '64px',
              left: '0',
              right: '0',
              height: '80px', // Larger than expanded timeline (72px) for easier triggering
              zIndex: 1000,
              pointerEvents: 'auto',
              cursor: isDragging ? 'none' : 'pointer'
            }}
            onMouseEnter={() => {
              setTimelineHovered(true)
            }}
            onMouseMove={(e) => {
              // Update mouse position for tracking line
              if (canvasRef.current) {
                const rect = canvasRef.current.getBoundingClientRect()
                const x = e.clientX - rect.left
                setTimelineMouseX(x)
                
                // Scrub while dragging (YouTube-style)
                if (isDragging) {
                  const totalFrames = metadata.total_frames || frames.length
                  const frame = Math.floor((x / rect.width) * totalFrames)
                  setCurrentFrame(Math.max(0, Math.min(frame, totalFrames - 1)))
                }
              }
            }}
            onMouseDown={(e) => {
              if (canvasRef.current) {
                setIsDragging(true)
                const rect = canvasRef.current.getBoundingClientRect()
                const x = e.clientX - rect.left
                const totalFrames = metadata.total_frames || frames.length
                const frame = Math.floor((x / rect.width) * totalFrames)
                setCurrentFrame(Math.max(0, Math.min(frame, totalFrames - 1)))
                
                // Pause playback when starting to drag
                if (playing) setPlaying(false)
              }
            }}
            onMouseLeave={() => {
              // Collapse timeline when mouse leaves detection area (unless scrubbing)
              if (!isDragging) {
                setTimelineHovered(false)
                setTimelineMouseX(null)
              }
            }}
            />
            
            <div style={{
              position: 'absolute',
              bottom: '64px',
              left: '0',
              right: '0',
              zIndex: 10,
              pointerEvents: 'none' // Don't interfere with hover detection area
            }}
            >
            <canvas
              ref={canvasRef}
              style={{
                width: '100%',
                height: timelineHovered ? '72px' : '8px',
                cursor: 'pointer',
                borderTop: '1px solid rgba(255, 255, 255, 0.1)',
                background: 'rgba(20, 20, 35, 0.6)',
                backdropFilter: 'blur(10px)',
                transition: 'all 0.25s cubic-bezier(0.4, 0, 0.2, 1)',
                display: 'block',
                willChange: 'height',
                pointerEvents: 'none' // Events handled by hover detection area
              }}
            />
            
            {/* Mouse position preview - shows when hovering over expanded timeline */}
            {timelineHovered && timelineMouseX !== null && metadata && (
              <div style={{
                position: 'absolute',
                left: isDragging 
                  ? `${((currentFrame / (metadata.total_frames || frames.length)) * (canvasRef.current?.getBoundingClientRect().width || 0))}px`
                  : `${timelineMouseX}px`, // During scrubbing, lock to playback position
                top: '0',
                bottom: '0',
                pointerEvents: 'none',
                zIndex: 100
              }}>
                {/* Vertical line tracker */}
                <div style={{
                  position: 'absolute',
                  left: '0',
                  top: '0',
                  bottom: '0',
                  width: '2px',
                  background: 'rgba(255, 255, 255, 0.6)',
                  boxShadow: '0 0 8px rgba(255, 255, 255, 0.4)'
                }} />
                
                {/* Time preview label */}
                <div style={{
                  position: 'absolute',
                  left: '50%',
                  top: '-32px',
                  transform: 'translateX(-50%)',
                  background: 'rgba(20, 20, 35, 0.95)',
                  backdropFilter: 'blur(10px)',
                  padding: '4px 10px',
                  borderRadius: '6px',
                  fontSize: '12px',
                  fontWeight: '600',
                  color: '#e5e7eb',
                  whiteSpace: 'nowrap',
                  border: '1px solid rgba(255, 255, 255, 0.1)',
                  boxShadow: '0 4px 12px rgba(0, 0, 0, 0.4)'
                }}>
                  {(() => {
                    const canvasRect = canvasRef.current?.getBoundingClientRect()
                    if (!canvasRect) return ''
                    
                    const totalFrames = metadata.total_frames || frames.length
                    
                    // During scrubbing, show current frame time; during hover, show hover position
                    const displayFrame = isDragging ? currentFrame : Math.floor((timelineMouseX / canvasRect.width) * totalFrames)
                    
                    if (metadata.fps) {
                      const displayTimeSec = displayFrame / metadata.fps
                      const mins = Math.floor(displayTimeSec / 60)
                      const secs = Math.floor(displayTimeSec % 60)
                      const ms = Math.floor((displayTimeSec % 1) * 100)
                      return `${mins}:${secs.toString().padStart(2, '0')}.${ms.toString().padStart(2, '0')}`
                    } else {
                      return `Frame ${displayFrame + 1}`
                    }
                  })()}
                </div>
              </div>
            )}
            
            {/* Behavior markers - start and end points for VALID behaviors only */}
            {timelineHovered && metadata?.annotations && metadata.annotations.length > 0 && (
              <>
                {metadata.annotations.map((annotation: any, idx: number) => {
                  const canvasRect = canvasRef.current?.getBoundingClientRect()
                  if (!canvasRect) return null
                  
                  // Only show markers for valid behavior types (matching Active Behaviors panel)
                  const validBehaviors = ['chase', 'attack', 'avoid', 'escape', 'mount', 'groom', 'sniff']
                  const actionLower = annotation.action.toLowerCase()
                  if (!validBehaviors.includes(actionLower)) return null
                  
                  const totalFrames = metadata.total_frames || frames.length
                  const startX = (annotation.start_frame / totalFrames) * canvasRect.width
                  const endX = (annotation.stop_frame / totalFrames) * canvasRect.width
                  
                  // Behavior colors (matching Active Behaviors panel exactly)
                  const behaviorColors: Record<string, string> = {
                    'chase': '#ef4444',      // red
                    'attack': '#ef4444',     // red
                    'avoid': '#3b82f6',      // blue
                    'escape': '#3b82f6',     // blue
                    'mount': '#a855f7',      // purple
                    'groom': '#10b981',      // green
                    'sniff': '#fbbf24'       // yellow
                  }
                  
                  const color = behaviorColors[actionLower] || '#6b7280'
                  const isHovered = hoveredBehavior === `${annotation.action}-${idx}`
                  
                  return (
                    <div key={`behavior-${idx}`}>
                      {/* Start marker - short visible line with full-height hover zone */}
                      <div
                        onMouseEnter={() => setHoveredBehavior(`${annotation.action}-${idx}`)}
                        onMouseLeave={() => setHoveredBehavior(null)}
                        style={{
                          position: 'absolute',
                          left: `${startX - 6}px`, // Center on position with wider hover area
                          top: '-17px', // Start from above timeline (adjusted to not overlap)
                          height: '17px', // Triangle (6px) + spacing (2px) + line (9px)
                          width: '12px', // Wider for easier hovering
                          pointerEvents: 'auto',
                          cursor: 'pointer',
                          zIndex: 50,
                          display: 'flex',
                          flexDirection: 'column',
                          justifyContent: 'flex-start',
                          alignItems: 'center'
                        }}
                      >
                        {/* Start triangle marker at top */}
                        <div style={{
                          width: '0',
                          height: '0',
                          borderLeft: '4px solid transparent',
                          borderRight: '4px solid transparent',
                          borderTop: `5px solid ${color}`,
                          filter: isHovered ? `drop-shadow(0 0 4px ${color})` : 'none',
                          transition: 'filter 0.2s ease',
                          marginBottom: '2px',
                          transform: 'translateZ(0)', // Force GPU rendering for crisp edges
                          WebkitTransform: 'translateZ(0)'
                        }} />
                        
                        {/* Visible line - extends from triangle to timeline */}
                        <div style={{
                          width: '2px',
                          height: '10px', // Fixed height to reach timeline exactly
                          background: color,
                          opacity: isHovered ? 1 : 0.6,
                          boxShadow: isHovered ? `0 0 8px ${color}` : `0 0 3px ${color}`,
                          transition: 'all 0.2s ease',
                          transform: 'translateZ(0)', // Force GPU rendering for crisp edges
                          WebkitTransform: 'translateZ(0)'
                        }} />
                      </div>
                      
                      {/* End marker - short visible line with full-height hover zone */}
                      <div
                        onMouseEnter={() => setHoveredBehavior(`${annotation.action}-${idx}`)}
                        onMouseLeave={() => setHoveredBehavior(null)}
                        style={{
                          position: 'absolute',
                          left: `${endX - 6}px`, // Center on position with wider hover area
                          top: '-17px', // Start from above timeline (adjusted to not overlap)
                          height: '17px', // Triangle (6px) + spacing (2px) + line (9px)
                          width: '12px', // Wider for easier hovering
                          pointerEvents: 'auto',
                          cursor: 'pointer',
                          zIndex: 50,
                          display: 'flex',
                          flexDirection: 'column',
                          justifyContent: 'flex-start',
                          alignItems: 'center'
                        }}
                      >
                        {/* End inverted triangle marker at top */}
                        <div style={{
                          width: '0',
                          height: '0',
                          borderLeft: '4px solid transparent',
                          borderRight: '4px solid transparent',
                          borderTop: `5px solid ${color}`,
                          transform: 'rotate(180deg) translateZ(0)', // Force GPU rendering for crisp edges
                          WebkitTransform: 'rotate(180deg) translateZ(0)',
                          filter: isHovered ? `drop-shadow(0 0 4px ${color})` : 'none',
                          transition: 'filter 0.2s ease',
                          marginBottom: '2px'
                        }} />
                        
                        {/* Visible line - extends from triangle to timeline */}
                        <div style={{
                          width: '2px',
                          height: '10px', // Fixed height to reach timeline exactly
                          background: color,
                          opacity: isHovered ? 1 : 0.6,
                          boxShadow: isHovered ? `0 0 8px ${color}` : `0 0 3px ${color}`,
                          transition: 'all 0.2s ease',
                          transform: 'translateZ(0)', // Force GPU rendering for crisp edges
                          WebkitTransform: 'translateZ(0)'
                        }} />
                      </div>
                    </div>
                  )
                })}
              </>
            )}
          </div>
          </>
        )}

        {/* Playback Controls */}
        {frames && frames.length > 0 && (
          <div style={{
            position: 'absolute',
            bottom: '0',
            left: '0',
            right: '0',
            zIndex: 10,
            background: 'rgba(20, 20, 35, 0.8)',
            backdropFilter: 'blur(20px)',
            padding: '10px 20px',
            borderTop: '1px solid rgba(255, 255, 255, 0.1)',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center'
          }}>
            {/* Play/Pause Button - Left Side */}
            <button
              onClick={() => setPlaying(!playing)}
              style={{
                padding: '0',
                background: 'transparent',
                color: playing ? '#ef4444' : '#a78bfa',
                border: 'none',
                cursor: 'pointer',
                fontWeight: '500',
                fontSize: '28px',
                width: '44px',
                height: '44px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                transition: 'color 0.2s ease, opacity 0.2s ease, transform 0.2s ease',
                opacity: 0.9,
                transform: 'scale(1)'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.opacity = '1'
                e.currentTarget.style.transform = 'scale(1.15)'
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.opacity = '0.9'
                e.currentTarget.style.transform = 'scale(1)'
              }}
              onMouseDown={(e) => {
                e.currentTarget.style.transform = 'scale(0.9)'
              }}
              onMouseUp={(e) => {
                e.currentTarget.style.transform = 'scale(1.15)'
              }}
              title={playing ? 'Pause (K or Space)' : 'Play (K or Space)'}
            >
              {playing ? '⏸' : '▶'}
            </button>

            {/* Timer Display - Center (or behavior name when hovering marker) */}
            <div style={{
              position: 'absolute',
              left: '50%',
              transform: 'translateX(-50%)',
              fontSize: '13px',
              fontWeight: '600',
              color: '#e5e7eb',
              fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto',
              textShadow: '0 2px 4px rgba(0, 0, 0, 0.5)',
              userSelect: 'none'
            }}>
              {(() => {
                // Show behavior name when tracking line intersects a behavior range
                if ((timelineHovered || isDragging) && metadata?.annotations) {
                  const validBehaviors = ['chase', 'attack', 'avoid', 'escape', 'mount', 'groom', 'sniff']
                  const canvasRect = canvasRef.current?.getBoundingClientRect()
                  if (!canvasRect) return null
                  
                  const totalFrames = metadata.total_frames || frames.length
                  const trackingFrame = isDragging 
                    ? currentFrame 
                    : timelineMouseX !== null 
                      ? Math.floor((timelineMouseX / canvasRect.width) * totalFrames)
                      : null
                  
                  if (trackingFrame !== null) {
                    const intersectedBehavior = metadata.annotations.find((annotation: any) => {
                      const actionLower = annotation.action.toLowerCase()
                      if (!validBehaviors.includes(actionLower)) return false
                      return trackingFrame >= annotation.start_frame && trackingFrame <= annotation.stop_frame
                    })
                    
                    if (intersectedBehavior) {
                      const behaviorColors: Record<string, string> = {
                        'chase': '#ef4444',
                        'attack': '#ef4444',
                        'avoid': '#3b82f6',
                        'escape': '#3b82f6',
                        'mount': '#a855f7',
                        'groom': '#10b981',
                        'sniff': '#fbbf24'
                      }
                      const actionLower = intersectedBehavior.action.toLowerCase()
                      const color = behaviorColors[actionLower] || '#6b7280'
                      
                      return (
                        <span style={{ color, textTransform: 'capitalize' }}>
                          {intersectedBehavior.action}
                        </span>
                      )
                    }
                  }
                }
                
                // Show time/frame normally
                return metadata && metadata.fps ? (() => {
                  const totalFrames = metadata.total_frames || frames.length
                  const currentTimeSec = currentFrame / metadata.fps
                  const totalTimeSec = totalFrames / metadata.fps
                  const formatTime = (sec: number) => {
                    const mins = Math.floor(sec / 60)
                    const secs = Math.floor(sec % 60)
                    return `${mins}:${secs.toString().padStart(2, '0')}`
                  }
                  return `${formatTime(currentTimeSec)} / ${formatTime(totalTimeSec)}`
                })() : `Frame ${currentFrame + 1}/${(metadata?.total_frames || frames.length)}`
              })()}
            </div>

            {/* Frame Navigation Buttons - Right Side */}
            <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
              {/* Previous Frame Button */}
              <button
                onClick={() => setCurrentFrame(f => Math.max(0, f - 1))}
                disabled={currentFrame === 0}
                style={{
                  padding: '0',
                  background: 'transparent',
                  color: currentFrame === 0 ? 'rgba(167, 139, 250, 0.3)' : '#a78bfa',
                  border: 'none',
                  cursor: currentFrame === 0 ? 'not-allowed' : 'pointer',
                  fontWeight: '500',
                  fontSize: '24px',
                  width: '40px',
                  height: '40px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  transition: 'color 0.2s ease, opacity 0.2s ease, transform 0.2s ease',
                  opacity: currentFrame === 0 ? 0.3 : 0.9,
                  transform: 'scale(1)'
                }}
                onMouseEnter={(e) => {
                  if (currentFrame > 0) {
                    e.currentTarget.style.opacity = '1'
                    e.currentTarget.style.transform = 'scale(1.15)'
                  }
                }}
                onMouseLeave={(e) => {
                  if (currentFrame > 0) {
                    e.currentTarget.style.opacity = '0.9'
                    e.currentTarget.style.transform = 'scale(1)'
                  }
                }}
                onMouseDown={(e) => {
                  if (currentFrame > 0) {
                    e.currentTarget.style.transform = 'scale(0.9)'
                  }
                }}
                onMouseUp={(e) => {
                  if (currentFrame > 0) {
                    e.currentTarget.style.transform = 'scale(1.15)'
                  }
                }}
                title="Previous Frame (J or ←)"
              >
                ⏮
              </button>

              {/* Next Frame Button */}
              <button
                onClick={() => setCurrentFrame(f => Math.min((metadata?.total_frames || frames.length) - 1, f + 1))}
                disabled={currentFrame >= (metadata?.total_frames || frames.length) - 1}
                style={{
                  padding: '0',
                  background: 'transparent',
                  color: currentFrame >= (metadata?.total_frames || frames.length) - 1 ? 'rgba(167, 139, 250, 0.3)' : '#a78bfa',
                  border: 'none',
                  cursor: currentFrame >= (metadata?.total_frames || frames.length) - 1 ? 'not-allowed' : 'pointer',
                  fontWeight: '500',
                  fontSize: '24px',
                  width: '40px',
                  height: '40px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  transition: 'color 0.2s ease, opacity 0.2s ease, transform 0.2s ease',
                  opacity: currentFrame >= (metadata?.total_frames || frames.length) - 1 ? 0.3 : 0.9,
                  transform: 'scale(1)'
                }}
                onMouseEnter={(e) => {
                  if (currentFrame < (metadata?.total_frames || frames.length) - 1) {
                    e.currentTarget.style.opacity = '1'
                    e.currentTarget.style.transform = 'scale(1.15)'
                  }
                }}
                onMouseLeave={(e) => {
                  if (currentFrame < (metadata?.total_frames || frames.length) - 1) {
                    e.currentTarget.style.opacity = '0.9'
                    e.currentTarget.style.transform = 'scale(1)'
                  }
                }}
                onMouseDown={(e) => {
                  if (currentFrame < (metadata?.total_frames || frames.length) - 1) {
                    e.currentTarget.style.transform = 'scale(0.9)'
                  }
                }}
                onMouseUp={(e) => {
                  if (currentFrame < (metadata?.total_frames || frames.length) - 1) {
                    e.currentTarget.style.transform = 'scale(1.15)'
                  }
                }}
                title="Next Frame (L or →)"
              >
                ⏭
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
    </>
  )
}

export default App
