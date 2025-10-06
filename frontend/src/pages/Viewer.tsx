import { useState, useEffect, useCallback, useRef } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import { EnhancedViewer } from '../components/EnhancedViewer'
import { useSmartLoading } from '../hooks/useSmartLoading'
import '../App.css'

// Mouse colors - must match EnhancedViewer.tsx MOUSE_COLORS
const MOUSE_COLORS_HEX = [
  '#ff6464', // Mouse 0: red
  '#6464ff', // Mouse 1: blue
  '#64ff64', // Mouse 2: green
  '#ffff64', // Mouse 3: yellow
  '#ff64ff', // Mouse 4: magenta
  '#64ffff', // Mouse 5: cyan
]

// Types
interface FileInfo {
  name: string
  lab: string
  path: string
  size_bytes: number
}

function Viewer() {
  const navigate = useNavigate()
  const location = useLocation()
  
  // Check if we're auto-loading from analytics IMMEDIATELY on mount
  const state = location.state as { loadSession?: { video_id: string; lab_id: string } } | null
  const shouldAutoLoad = !!state?.loadSession
  
  const [labFiles, setLabFiles] = useState<Record<string, FileInfo[]>>({})
  const [selectedLab, setSelectedLab] = useState<string>('')
  const [selectedFile, setSelectedFile] = useState<FileInfo | null>(null)
  const [isAutoLoading, setIsAutoLoading] = useState(shouldAutoLoad) // Set immediately if we have loadSession state
  const [currentFrame, setCurrentFrame] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [timelineHovered, setTimelineHovered] = useState(false)
  const [timelineMouseX, setTimelineMouseX] = useState<number | null>(null)
  const [isDragging, setIsDragging] = useState(false)
  const [hoveredBehavior, setHoveredBehavior] = useState<string | null>(null)
  const [previewFrame, setPreviewFrame] = useState<any | null>(null) // Frame data for popup preview
  const [previewFrameNumber, setPreviewFrameNumber] = useState<number | null>(null) // Track which frame we're previewing
  const [lastValidFrame, setLastValidFrame] = useState<any | null>(null) // Keep last valid frame to show while loading
  const [sidebarVisible, setSidebarVisible] = useState(false) // Control sidebar visibility
  const [playStateBeforeScrub, setPlayStateBeforeScrub] = useState<boolean | null>(null) // Track play state before scrubbing
  const [labDropdownOpen, setLabDropdownOpen] = useState(false) // Custom dropdown state
  const [trackedMouseId, setTrackedMouseId] = useState<string | null>(null) // Track which mouse to follow
  const [menuButtonHovered, setMenuButtonHovered] = useState(false) // Track menu button hover state
  const [recentFrames, setRecentFrames] = useState<any[]>([]) // For tail smoothing ghost trails
  const lastTrackedPositionRef = useRef<{ x: number; y: number } | null>(null) // Prevent unnecessary viewState updates
  
  // Refs for keyboard repeat state (avoid closure issues)
  const repeatTimerRef = useRef<number | null>(null)
  const accelerationTimerRef = useRef<number | null>(null)
  const isRepeatingRef = useRef(false)
  const currentSpeedRef = useRef(300)
  const advanceFrameRef = useRef<((direction: 'next' | 'prev') => void) | null>(null)
  
  // Throttle seek calls to prevent spamming backend during rapid scrubbing
  const lastSeekTimeRef = useRef<number>(0)
  const pendingSeekRef = useRef<number | null>(null)
  const seekThrottleMs = 200 // Only allow one seek every 200ms
  const isSeekingRef = useRef<boolean>(false) // Track if we're currently loading from a seek
  
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
  
  // Use smart loading hook with on-demand seeking
  const {
    frames,
    getFrame,
    fetchFrame,
    metadata,
    loading,
    loadedRanges,
    progress, 
    error,
    seekToFrame,
    isFrameLoaded
  } = useSmartLoading({
    lab: selectedFile?.lab || '',
    filename: selectedFile?.name || '',
    chunkSize: 100,
    seekChunkSize: 50,
    onProgress: (loaded, total) => {
      if (loaded % 1000 === 0) {
        console.log(`📦 Smart load: ${loaded}/${total} frames (${((loaded/total)*100).toFixed(1)}%)`)
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
        
        // Check if we came from analytics with a specific session to load
        const state = location.state as { loadSession?: { video_id: string; lab_id: string } } | null
        if (state?.loadSession) {
          setIsAutoLoading(true) // Mark that we're auto-loading
          const { video_id, lab_id } = state.loadSession
          console.log(`📂 Auto-loading session from analytics: ${video_id} from ${lab_id}`)
          
          // Find the matching file in the lab files
          const labData = data.labs?.[lab_id]
          if (labData) {
            // Match by filename without extension (video_id might be number or string)
            const matchingFile = labData.find((f: FileInfo) => {
              const fileBaseName = f.name.replace(/\.parquet$/, '')
              return fileBaseName === String(video_id)
            })
            if (matchingFile) {
              console.log(`✅ Setting lab to: ${lab_id}`)
              console.log(`✅ Setting file to: ${matchingFile.name}`)
              setSelectedLab(lab_id)
              setSelectedFile(matchingFile)
              loadFile(matchingFile)
              setIsAutoLoading(false) // Auto-load complete
              console.log(`✅ Auto-loaded: ${matchingFile.name} from ${lab_id}`)
            } else {
              setIsAutoLoading(false)
              console.warn(`⚠️ File ${video_id}.parquet not found in lab ${lab_id}`)
              console.log('Available files:', labData.map((f: FileInfo) => f.name).slice(0, 5))
            }
          } else {
            setIsAutoLoading(false)
            console.warn(`⚠️ Lab ${lab_id} not found in file list`)
            console.log('Available labs:', Object.keys(data.labs || {}))
          }
          
          // Clear the navigation state so it doesn't reload on refresh
          navigate(location.pathname, { replace: true, state: {} })
        } else {
          // Only auto-select first lab if NOT auto-loading from analytics
          const firstLab = Object.keys(data.labs || {})[0]
          if (firstLab) {
            setSelectedLab(firstLab)
          }
        }
      })
      .catch(err => console.error('Failed to load file list:', err))
  }, []) // Empty dependency array - only run once on mount

  // Load file data when selected (just update state, hook will auto-trigger)
  const loadFile = useCallback(async (file: FileInfo) => {
    setCurrentFrame(0)
    setPlaying(false) // Stop playback when switching files
    console.log(`🚀 Starting progressive load: ${file.name}`)
  }, [])

  // Throttled seek to prevent backend spam during rapid scrubbing
  const throttledSeek = useCallback((frameNumber: number) => {
    const now = Date.now()
    const timeSinceLastSeek = now - lastSeekTimeRef.current

    if (timeSinceLastSeek >= seekThrottleMs) {
      // Enough time has passed, seek immediately
      console.log(`🎯 Immediate seek to frame ${frameNumber}`)
      lastSeekTimeRef.current = now
      pendingSeekRef.current = null
      isSeekingRef.current = true
      seekToFrame(frameNumber)
    } else {
      // Too soon, store as pending and it will be executed on drag end
      console.log(`⏱️ Throttled - storing seek to frame ${frameNumber} as pending`)
      pendingSeekRef.current = frameNumber
    }
  }, [seekToFrame, seekThrottleMs])

  // Execute pending seek when dragging ends
  useEffect(() => {
    if (!isDragging && pendingSeekRef.current !== null) {
      console.log(`🎯 Executing pending seek to frame ${pendingSeekRef.current}`)
      seekToFrame(pendingSeekRef.current)
      pendingSeekRef.current = null
      lastSeekTimeRef.current = Date.now()
    }
  }, [isDragging, seekToFrame])

  // Fetch preview frame on hover (YouTube-style: fetch unloaded frames for preview)
  useEffect(() => {
    if (!timelineHovered || timelineMouseX === null || !metadata || isDragging) {
      setPreviewFrame(null)
      setPreviewFrameNumber(null)
      return
    }

    const rect = canvasRef.current?.getBoundingClientRect()
    if (!rect) return

    const totalFrames = metadata.total_frames || frames.length
    const hoverFrameNumber = Math.floor((timelineMouseX / rect.width) * totalFrames)
    
    // Clamp to valid range
    const clampedFrameNumber = Math.max(0, Math.min(hoverFrameNumber, totalFrames - 1))
    
    // If we're already previewing this frame, don't refetch
    if (clampedFrameNumber === previewFrameNumber) return

    // Check if frame is already loaded in memory
    const existingFrame = getFrame(clampedFrameNumber)
    if (existingFrame) {
      setPreviewFrame(existingFrame)
      setPreviewFrameNumber(clampedFrameNumber)
      return
    }

    // Frame not loaded - fetch it on demand for preview
    console.log(`🔍 Fetching preview frame ${clampedFrameNumber}`)
    setPreviewFrameNumber(clampedFrameNumber)
    setPreviewFrame(null) // Clear old preview while fetching
    
    fetchFrame(clampedFrameNumber)
      .then(frame => {
        if (frame) {
          console.log(`✅ Preview frame ${clampedFrameNumber} loaded`)
          setPreviewFrame(frame)
        } else {
          console.warn(`⚠️ Preview frame ${clampedFrameNumber} returned null`)
        }
      })
      .catch(err => {
        console.error(`❌ Failed to fetch preview frame ${clampedFrameNumber}:`, err)
      })
  }, [timelineHovered, timelineMouseX, metadata, isDragging, frames.length, previewFrameNumber, getFrame, fetchFrame])

  // Keep track of last valid frame to show while seeking/loading
  useEffect(() => {
    const currentFrameData = getFrame(currentFrame)
    if (currentFrameData) {
      setLastValidFrame(currentFrameData)
      // Also update recent frames buffer for tail smoothing
      setRecentFrames(prev => {
        const updated = [...prev, currentFrameData]
        // Keep only last 5 frames for ghost trail (3 ghosts + current + 1 extra)
        return updated.slice(-5)
      })
    }
  }, [currentFrame, getFrame])

  // Clear recent frames when switching files
  useEffect(() => {
    if (selectedFile) {
      setRecentFrames([])
    }
  }, [selectedFile])

  // Playback animation (FPS-based timing from metadata)
  useEffect(() => {
    if (!playing || !frames || frames.length === 0 || !metadata) return

    // Use actual FPS from metadata, default to 30
    const fps = metadata.fps || 30
    const frameInterval = 1000 / fps

    console.log(`▶️ Playing at ${fps} FPS (${frameInterval.toFixed(1)}ms per frame)`)

    const interval = setInterval(() => {
      setCurrentFrame(prev => {
        const nextFrame = prev + 1
        // Check if we've reached the end of the video
        if (nextFrame >= metadata.total_frames) {
          setPlaying(false)
          return prev
        }
        
        // If next frame isn't loaded yet, seek to it (but only if not already seeking)
        if (!isFrameLoaded(nextFrame)) {
          if (!isSeekingRef.current) {
            console.log(`⏭️ Next frame ${nextFrame} not loaded, seeking...`)
            isSeekingRef.current = true
            seekToFrame(nextFrame)
          } else {
            console.log(`⏸️ Waiting for seek to complete before advancing...`)
            return prev // Stay on current frame while seeking
          }
        } else {
          // Frame is loaded, clear seeking flag
          isSeekingRef.current = false
        }
        
        return nextFrame
      })
    }, frameInterval)

    return () => clearInterval(interval)
  }, [playing, frames, metadata, isFrameLoaded, seekToFrame])

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

  // Frame advance function with stable reference
  const advanceFrame = useCallback((direction: 'next' | 'prev') => {
    setCurrentFrame(f => {
      if (!metadata) return f
      const newFrame = direction === 'next'
        ? Math.min(metadata.total_frames - 1, f + 1)
        : Math.max(0, f - 1)
      
      // If frame isn't loaded, seek to it
      if (!isFrameLoaded(newFrame)) {
        seekToFrame(newFrame)
      }
      return newFrame
    })
  }, [metadata, isFrameLoaded, seekToFrame])

  // Keep ref updated
  useEffect(() => {
    advanceFrameRef.current = advanceFrame
  }, [advanceFrame])

  // Keyboard shortcuts - global, work from anywhere on the page
  // With acceleration for frame navigation when keys are held
  useEffect(() => {
    const startRepeating = (direction: 'next' | 'prev') => {
      if (isRepeatingRef.current) {
        console.log('⚠️ Already repeating, ignoring startRepeating call')
        return
      }
      
      console.log(`▶️ Starting repeat in ${direction} direction`)
      isRepeatingRef.current = true
      currentSpeedRef.current = 300 // Start slow
      
      // First advance
      advanceFrameRef.current?.(direction)
      
      // Set up repeating with current speed
      const repeat = () => {
        advanceFrameRef.current?.(direction)
        repeatTimerRef.current = window.setTimeout(repeat, currentSpeedRef.current)
      }
      
      repeatTimerRef.current = window.setTimeout(repeat, currentSpeedRef.current)
      
      // Gradually accelerate (every 500ms, reduce delay by 30ms, min 50ms)
      const accelerate = () => {
        if (currentSpeedRef.current > 50) {
          currentSpeedRef.current = Math.max(50, currentSpeedRef.current - 30)
          console.log(`⚡ Accelerated to ${currentSpeedRef.current}ms per frame`)
        }
        accelerationTimerRef.current = window.setTimeout(accelerate, 500)
      }
      
      accelerationTimerRef.current = window.setTimeout(accelerate, 500)
    }
    
    const stopRepeating = () => {
      if (!isRepeatingRef.current) return
      
      console.log('⏹️ Stopping repeat')
      isRepeatingRef.current = false
      if (repeatTimerRef.current) {
        clearTimeout(repeatTimerRef.current)
        repeatTimerRef.current = null
      }
      if (accelerationTimerRef.current) {
        clearTimeout(accelerationTimerRef.current)
        accelerationTimerRef.current = null
      }
      currentSpeedRef.current = 300
    }
    
    const handleKeyDown = (e: KeyboardEvent) => {
      if (!metadata) return
      
      // Skip shortcuts only when typing in text fields/textareas
      const target = e.target as HTMLElement
      if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA') {
        return
      }
      
      // Play/Pause: Space or K (YouTube-style)
      if (e.key === ' ' || e.key === 'k' || e.key === 'K') {
        e.preventDefault()
        e.stopPropagation()
        setPlaying(p => !p)
      } 
      // Reset camera: R
      else if (e.key === 'r' || e.key === 'R') {
        e.preventDefault()
        e.stopPropagation()
        setTrackedMouseId(null) // Stop tracking
        lastTrackedPositionRef.current = null // Reset position tracking
        // Reset to default view
        if (metadata) {
          let width = 640
          let height = 480
          
          if (metadata.video_width && metadata.video_height) {
            width = metadata.video_width
            height = metadata.video_height
          } else if (metadata.arena_width_cm && metadata.arena_height_cm && metadata.pix_per_cm) {
            width = metadata.arena_width_cm * metadata.pix_per_cm
            height = metadata.arena_height_cm * metadata.pix_per_cm
          }
          
          const viewportWidth = 1000
          const viewportHeight = 800
          const scaleX = viewportWidth / width
          const scaleY = viewportHeight / height
          const scale = Math.min(scaleX, scaleY) * 0.9
          const zoom = Math.log2(scale)
          
          setViewState({
            target: [0, 0, 0],
            zoom
          })
        }
      }
      // Previous frame: Left Arrow or J (YouTube-style) - with acceleration on hold
      else if (e.key === 'ArrowLeft' || e.key === 'j' || e.key === 'J') {
        e.preventDefault()
        e.stopPropagation()
        if (!e.repeat) {
          // First keypress
          startRepeating('prev')
        }
      } 
      // Next frame: Right Arrow or L (YouTube-style) - with acceleration on hold
      else if (e.key === 'ArrowRight' || e.key === 'l' || e.key === 'L') {
        e.preventDefault()
        e.stopPropagation()
        if (!e.repeat) {
          // First keypress
          startRepeating('next')
        }
      }
    }
    
    const handleKeyUp = (e: KeyboardEvent) => {
      // Stop repeating when key is released
      if (e.key === 'ArrowLeft' || e.key === 'j' || e.key === 'J' ||
          e.key === 'ArrowRight' || e.key === 'l' || e.key === 'L') {
        stopRepeating()
      }
    }

    // Use capture phase to intercept before other handlers (like buttons or deck.gl)
    window.addEventListener('keydown', handleKeyDown, true)
    window.addEventListener('keyup', handleKeyUp, true)
    
    return () => {
      stopRepeating()
      window.removeEventListener('keydown', handleKeyDown, true)
      window.removeEventListener('keyup', handleKeyUp, true)
    }
  }, [metadata]) // Only depend on metadata, not advanceFrame

  // Draw timeline (YouTube-style: simple bar with continuous buffer indicator)
  useEffect(() => {
    if (!canvasRef.current || !metadata) return
    
    const currentFrames = framesRef.current
    if (!currentFrames || currentFrames.length === 0) return

    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')!
    const rect = canvas.getBoundingClientRect()
    canvas.width = rect.width * window.devicePixelRatio
    canvas.height = rect.height * window.devicePixelRatio
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio)

    // Clear with dark background
    ctx.fillStyle = 'rgba(15, 15, 30, 0.95)'
    ctx.fillRect(0, 0, rect.width, rect.height)

    const totalFrames = metadata.total_frames

    // YouTube-style: Draw single continuous loaded buffer (gray bar)
    // Find the continuous loaded range from the first loaded frame
    if (loadedRanges.length > 0) {
      // YouTube loads continuously, so we just show one bar from start to end of first range
      const loadedRange = loadedRanges[0]
      const startX = (loadedRange.start / totalFrames) * rect.width
      const endX = (loadedRange.end / totalFrames) * rect.width
      
      ctx.fillStyle = 'rgba(255, 255, 255, 0.3)'
      ctx.fillRect(startX, 0, endX - startX, rect.height)
    }
    
    // Draw playback progress (purple bar matching UI theme)
    const frameProgress = currentFrame / totalFrames
    const gradient = ctx.createLinearGradient(0, 0, rect.width * frameProgress, 0)
    gradient.addColorStop(0, 'rgba(167, 139, 250, 0.9)')  // Purple theme
    gradient.addColorStop(1, 'rgba(139, 92, 246, 0.9)')
    ctx.fillStyle = gradient
    ctx.fillRect(0, 0, rect.width * frameProgress, rect.height)

    // Draw current position marker (circle)
    const markerX = (currentFrame / totalFrames) * rect.width
    ctx.shadowBlur = 8
    ctx.shadowColor = 'rgba(167, 139, 250, 0.8)'
    ctx.fillStyle = '#a78bfa'
    ctx.beginPath()
    ctx.arc(markerX, rect.height / 2, 6, 0, Math.PI * 2)
    ctx.fill()
    ctx.shadowBlur = 0
  }, [currentFrame, frames, metadata, loadedRanges, getFrame])

  // Handle global mouseup and mousemove for drag scrubbing
  useEffect(() => {
    const handleGlobalMouseUp = () => {
      setIsDragging(false)
      setTimelineHovered(false) // Collapse timeline after scrubbing ends
      setTimelineMouseX(null)
      
      // Restore play state after scrubbing
      if (playStateBeforeScrub !== null) {
        setPlaying(playStateBeforeScrub)
        setPlayStateBeforeScrub(null)
      }
      
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
        const clampedFrame = Math.max(0, Math.min(frame, totalFrames - 1))
        setCurrentFrame(clampedFrame)
        
        // If frame isn't loaded, trigger seek to load it (YouTube-style)
        if (!isFrameLoaded(clampedFrame)) {
          console.log(`🎯 Scrubbed to unloaded frame ${clampedFrame}, seeking...`)
          seekToFrame(clampedFrame)
        }
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
  }, [isDragging, metadata, frames, isFrameLoaded, seekToFrame, playStateBeforeScrub])

  // Initialize camera based on metadata
  const [viewState, setViewState] = useState({
    target: [0, 0, 0],
    zoom: -1
  })

  // Calculate active behaviors for current frame
  const activeBehaviors = useCallback(() => {
    if (!metadata?.annotations || !frames || frames.length === 0) return []
    
    // currentFrame is now the actual frame number, not an index
    const frameNumber = currentFrame
    
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

  // Track mouse camera follow
  useEffect(() => {
    if (!trackedMouseId || !frames || frames.length === 0 || !metadata) return
    
    const frame = getFrame(currentFrame)
    if (!frame || !frame.mice || !frame.mice[trackedMouseId]) return
    
    // Calculate mouse centroid
    const mouseData = frame.mice[trackedMouseId]
    const videoWidth = metadata.video_width || 640
    const videoHeight = metadata.video_height || 480
    
    // Transform to scene coordinates
    const centerX = videoWidth / 2
    const centerY = videoHeight / 2
    
    let sumX = 0, sumY = 0, count = 0
    mouseData.points.forEach(([x, y]: number[]) => {
      sumX += x - centerX
      sumY += -(y - centerY)
      count++
    })
    
    if (count > 0) {
      const centroidX = sumX / count
      const centroidY = sumY / count
      
      // Only update viewState if position changed significantly (> 2 pixels)
      const lastPos = lastTrackedPositionRef.current
      const positionChanged = !lastPos || 
        Math.abs(centroidX - lastPos.x) > 2 || 
        Math.abs(centroidY - lastPos.y) > 2
      
      if (positionChanged) {
        lastTrackedPositionRef.current = { x: centroidX, y: centroidY }
        
        setViewState(prev => ({
          ...prev,
          target: [centroidX, centroidY, 0],
          // Only set zoom on first track (less aggressive: 2.0x instead of 3.0x)
          zoom: prev.zoom < 1.5 ? 2.0 : prev.zoom
        }))
      }
    }
  }, [trackedMouseId, currentFrame, frames, metadata, getFrame])

  return (
    <>
      {/* Global cursor hiding during scrubbing */}
      {isDragging && (
        <style>{`
          * { cursor: none !important; }
        `}</style>
      )}
      
      <div style={{ 
        display: 'flex', 
        height: '100vh', 
        width: '100vw',
        background: 'rgba(15, 15, 30, 1)', // Match viewer background - solid dark color
        fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
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
          fontWeight: '800', 
          fontFamily: "'Poppins', sans-serif",
          marginBottom: '24px',
          letterSpacing: '-0.5px',
          display: 'flex',
          alignItems: 'center',
          gap: '10px',
          userSelect: 'none',
          WebkitUserSelect: 'none',
          cursor: 'pointer',
          transition: 'opacity 0.2s ease',
          opacity: 1
        }}
        onClick={() => navigate('/')}
        onMouseEnter={(e) => e.currentTarget.style.opacity = '0.7'}
        onMouseLeave={(e) => e.currentTarget.style.opacity = '1'}
        >
          {/* Cat emoji with negative space features */}
          <span style={{
            position: 'relative',
            fontSize: '28px',
            background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            filter: 'drop-shadow(0 0 10px rgba(102, 126, 234, 0.4))',
            userSelect: 'none',
            WebkitUserSelect: 'none'
          }}>
            🐱
            {/* Negative space features */}
            <span style={{
              position: 'absolute',
              top: '38%',
              left: '32%',
              width: '3px',
              height: '4px',
              background: 'rgba(20, 20, 35, 1)',
              borderRadius: '50% 50% 50% 50% / 60% 60% 40% 40%',
              transform: 'rotate(-5deg)',
              pointerEvents: 'none'
            }}></span>
            <span style={{
              position: 'absolute',
              top: '38%',
              right: '32%',
              width: '3px',
              height: '4px',
              background: 'rgba(20, 20, 35, 1)',
              borderRadius: '50% 50% 50% 50% / 60% 60% 40% 40%',
              transform: 'rotate(5deg)',
              pointerEvents: 'none'
            }}></span>
            <span style={{
              position: 'absolute',
              top: '54%',
              left: '12%',
              width: '7px',
              height: '1px',
              background: 'rgba(20, 20, 35, 1)',
              transform: 'rotate(-12deg)',
              pointerEvents: 'none'
            }}></span>
            <span style={{
              position: 'absolute',
              top: '60%',
              left: '8%',
              width: '8px',
              height: '1px',
              background: 'rgba(20, 20, 35, 1)',
              transform: 'rotate(-4deg)',
              pointerEvents: 'none'
            }}></span>
            <span style={{
              position: 'absolute',
              top: '54%',
              right: '12%',
              width: '7px',
              height: '1px',
              background: 'rgba(20, 20, 35, 1)',
              transform: 'rotate(12deg)',
              pointerEvents: 'none'
            }}></span>
            <span style={{
              position: 'absolute',
              top: '60%',
              right: '8%',
              width: '8px',
              height: '1px',
              background: 'rgba(20, 20, 35, 1)',
              transform: 'rotate(4deg)',
              pointerEvents: 'none'
            }}></span>
            <span style={{
              position: 'absolute',
              top: '58%',
              left: '50%',
              transform: 'translateX(-50%)',
              width: '0',
              height: '0',
              borderLeft: '2px solid transparent',
              borderRight: '2px solid transparent',
              borderTop: '2.5px solid rgba(20, 20, 35, 1)',
              pointerEvents: 'none'
            }}></span>
          </span>
          
          <span style={{
            background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            textShadow: '0 0 30px rgba(102, 126, 234, 0.4), 0 0 60px rgba(118, 75, 162, 0.2)',
            filter: 'drop-shadow(0 2px 8px rgba(102, 126, 234, 0.3))',
            WebkitTextStroke: '0.5px rgba(102, 126, 234, 0.15)'
          }}>MABe Mouser</span>
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
        
        {/* Custom Dropdown for Lab Selection */}
        <div style={{ position: 'relative', marginBottom: '20px' }}>
          {/* Dropdown Button */}
          <button
            onClick={() => setLabDropdownOpen(!labDropdownOpen)}
            onBlur={() => setTimeout(() => setLabDropdownOpen(false), 150)}
            style={{
              width: '100%',
              padding: '12px 14px',
              borderRadius: '10px',
              border: '1px solid rgba(167, 139, 250, 0.4)',
              background: 'linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%)',
              backdropFilter: 'blur(12px)',
              color: selectedLab ? '#f3f4f6' : 'rgba(243, 244, 246, 0.5)',
              fontSize: '14px',
              fontWeight: '500',
              cursor: 'pointer',
              transition: 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
              boxShadow: labDropdownOpen 
                ? '0 0 0 3px rgba(167, 139, 250, 0.2), 0 4px 16px rgba(102, 126, 234, 0.3)'
                : '0 2px 12px rgba(0, 0, 0, 0.3)',
              outline: 'none',
              textAlign: 'left',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between'
            }}
            onMouseEnter={(e) => {
              if (!labDropdownOpen) {
                e.currentTarget.style.background = 'linear-gradient(135deg, rgba(102, 126, 234, 0.25) 0%, rgba(118, 75, 162, 0.25) 100%)'
                e.currentTarget.style.borderColor = 'rgba(167, 139, 250, 0.6)'
                e.currentTarget.style.boxShadow = '0 4px 16px rgba(102, 126, 234, 0.3)'
              }
            }}
            onMouseLeave={(e) => {
              if (!labDropdownOpen) {
                e.currentTarget.style.background = 'linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%)'
                e.currentTarget.style.borderColor = 'rgba(167, 139, 250, 0.4)'
                e.currentTarget.style.boxShadow = '0 2px 12px rgba(0, 0, 0, 0.3)'
              }
            }}
          >
            <span>{selectedLab || 'Select a lab...'}</span>
            <svg 
              width="12" 
              height="8" 
              viewBox="0 0 12 8" 
              fill="none" 
              style={{
                transition: 'transform 0.2s ease',
                transform: labDropdownOpen ? 'rotate(180deg)' : 'rotate(0deg)'
              }}
            >
              <path 
                d="M1 1L6 6L11 1" 
                stroke="#a78bfa" 
                strokeWidth="2" 
                strokeLinecap="round" 
                strokeLinejoin="round"
              />
            </svg>
          </button>
          
          {/* Dropdown List */}
          {labDropdownOpen && (
            <div style={{
              position: 'absolute',
              top: 'calc(100% + 4px)',
              left: '0',
              right: '0',
              maxHeight: '240px',
              overflowY: 'auto',
              background: 'rgba(30, 30, 50, 0.95)',
              backdropFilter: 'blur(20px)',
              borderRadius: '10px',
              border: '1px solid rgba(167, 139, 250, 0.4)',
              boxShadow: '0 8px 32px rgba(0, 0, 0, 0.5)',
              zIndex: 1000,
              padding: '4px'
            }}>
              {Object.keys(labFiles).sort().map(lab => (
                <button
                  key={lab}
                  onClick={() => {
                    setSelectedLab(lab)
                    setLabDropdownOpen(false)
                  }}
                  style={{
                    width: '100%',
                    padding: '10px 12px',
                    background: selectedLab === lab 
                      ? 'rgba(167, 139, 250, 0.2)' 
                      : 'transparent',
                    border: 'none',
                    borderRadius: '6px',
                    color: '#f3f4f6',
                    fontSize: '14px',
                    fontWeight: selectedLab === lab ? '600' : '500',
                    cursor: 'pointer',
                    transition: 'all 0.15s ease',
                    textAlign: 'left',
                    marginBottom: '2px'
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.background = 'rgba(167, 139, 250, 0.15)'
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.background = selectedLab === lab 
                      ? 'rgba(167, 139, 250, 0.2)' 
                      : 'transparent'
                  }}
                >
                  {lab}
                </button>
              ))}
            </div>
          )}
        </div>

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
            onMouseEnter={() => {
              setSidebarVisible(true)
              setMenuButtonHovered(true)
            }}
            onMouseLeave={() => setMenuButtonHovered(false)}
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
              background: menuButtonHovered 
                ? 'linear-gradient(135deg, rgba(80, 90, 160, 0.22) 0%, rgba(90, 70, 120, 0.22) 100%)'
                : 'linear-gradient(135deg, rgba(80, 90, 160, 0.12) 0%, rgba(90, 70, 120, 0.12) 100%)',
              backdropFilter: 'blur(10px)',
              borderRight: menuButtonHovered 
                ? '1px solid rgba(255, 255, 255, 0.25)'
                : '1px solid rgba(255, 255, 255, 0.15)',
              borderBottomRightRadius: '28px', // Larger radius for the extended bottom
              transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
              opacity: sidebarVisible ? 0 : 1,
              pointerEvents: sidebarVisible ? 'none' : 'auto',
              transform: sidebarVisible ? 'scale(0.9) translateX(-10px)' : 'scale(1) translateX(0)',
              zIndex: 10
            }}
          >
            {/* Cat emoji with negative space features */}
            <span style={{
              position: 'relative',
              fontSize: '32px',
              background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              filter: 'drop-shadow(0 2px 4px rgba(102, 126, 234, 0.3))'
            }}>
              🐱
              {/* Negative space features */}
              <span style={{
                position: 'absolute',
                top: '38%',
                left: '32%',
                width: '3px',
                height: '5px',
                background: 'rgba(15, 15, 30, 1)',
                borderRadius: '50% 50% 50% 50% / 60% 60% 40% 40%',
                transform: 'rotate(-5deg)',
                pointerEvents: 'none'
              }}></span>
              <span style={{
                position: 'absolute',
                top: '38%',
                right: '32%',
                width: '3px',
                height: '5px',
                background: 'rgba(15, 15, 30, 1)',
                borderRadius: '50% 50% 50% 50% / 60% 60% 40% 40%',
                transform: 'rotate(5deg)',
                pointerEvents: 'none'
              }}></span>
              <span style={{
                position: 'absolute',
                top: '54%',
                left: '12%',
                width: '8px',
                height: '1px',
                background: 'rgba(15, 15, 30, 1)',
                transform: 'rotate(-12deg)',
                pointerEvents: 'none'
              }}></span>
              <span style={{
                position: 'absolute',
                top: '60%',
                left: '8%',
                width: '9px',
                height: '1px',
                background: 'rgba(15, 15, 30, 1)',
                transform: 'rotate(-4deg)',
                pointerEvents: 'none'
              }}></span>
              <span style={{
                position: 'absolute',
                top: '54%',
                right: '12%',
                width: '8px',
                height: '1px',
                background: 'rgba(15, 15, 30, 1)',
                transform: 'rotate(12deg)',
                pointerEvents: 'none'
              }}></span>
              <span style={{
                position: 'absolute',
                top: '60%',
                right: '8%',
                width: '9px',
                height: '1px',
                background: 'rgba(15, 15, 30, 1)',
                transform: 'rotate(4deg)',
                pointerEvents: 'none'
              }}></span>
              <span style={{
                position: 'absolute',
                top: '58%',
                left: '50%',
                transform: 'translateX(-50%)',
                width: '0',
                height: '0',
                borderLeft: '2.5px solid transparent',
                borderRight: '2.5px solid transparent',
                borderTop: '3px solid rgba(15, 15, 30, 1)',
                pointerEvents: 'none'
              }}></span>
            </span>
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
          
          {/* Centered title when no file is loaded - positioned relative to main viewer, not top bar */}
          {!selectedFile && (
            <div style={{
              position: 'fixed',
              left: sidebarVisible ? 'calc(50% + 140px)' : '50%',
              top: '30px', // Vertically centered in the top bar (60px / 2 = 30px)
              transform: 'translate(-50%, -50%)',
              fontSize: '26px',
              fontWeight: '800',
              fontFamily: "'Poppins', sans-serif",
              background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              letterSpacing: '-0.5px',
              opacity: sidebarVisible ? 0 : 1,
              transition: 'all 0.3s ease-out',
              pointerEvents: sidebarVisible ? 'none' : 'auto',
              zIndex: 101,
              textShadow: '0 0 30px rgba(102, 126, 234, 0.4), 0 0 60px rgba(118, 75, 162, 0.2)',
              filter: 'drop-shadow(0 2px 8px rgba(102, 126, 234, 0.3))',
              WebkitTextStroke: '0.5px rgba(102, 126, 234, 0.15)',
              userSelect: 'none',
              WebkitUserSelect: 'none'
            }}>
              MABe Mouser
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
                  <span style={{ color: '#a78bfa' }}>Frame</span> {currentFrame} / {metadata?.total_frames || frames.length}
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
                frame={getFrame(currentFrame) || lastValidFrame || null}
                metadata={metadata}
                viewState={viewState}
                onViewStateChange={({ viewState }: any) => setViewState(viewState)}
                onMouseClick={(mouseId: string) => {
                  // If clicking the same mouse or background, release tracking
                  if (trackedMouseId === mouseId || !mouseId) {
                    setTrackedMouseId(null)
                    lastTrackedPositionRef.current = null
                  } else {
                    setTrackedMouseId(mouseId)
                    lastTrackedPositionRef.current = null // Reset position when switching mice
                  }
                }}
                onViewerClick={() => {
                  setTrackedMouseId(null)
                  lastTrackedPositionRef.current = null
                }}
                trackedMouseId={trackedMouseId}
                recentFrames={recentFrames}
                tailGhostFrames={3}
              />
              
              {/* Tracking Indicator */}
              {trackedMouseId !== null && frames && frames.length > 0 && (() => {
                const frame = getFrame(currentFrame)
                if (!frame || !frame.mice || !frame.mice[trackedMouseId]) return null
                
                // Get mouse color
                const mouseIndex = parseInt(trackedMouseId)
                const baseColors: [number, number, number][] = [
                  [255, 99, 71],   // tomato
                  [135, 206, 250], // sky blue
                  [144, 238, 144], // light green
                  [255, 215, 0],   // gold
                  [221, 160, 221], // plum
                  [255, 165, 0],   // orange
                  [173, 216, 230], // light blue
                  [255, 182, 193]  // light pink
                ]
                const baseColor = baseColors[mouseIndex % baseColors.length]
                const colorStr = `rgb(${baseColor[0]}, ${baseColor[1]}, ${baseColor[2]})`
                
                return (
                  <div style={{
                    position: 'absolute',
                    bottom: '94px',
                    left: '50%',
                    transform: 'translateX(-50%)',
                    padding: '8px 16px',
                    background: 'rgba(17, 24, 39, 0.95)',
                    border: `2px solid ${colorStr}`,
                    borderRadius: '8px',
                    color: colorStr,
                    fontFamily: "'Inter', sans-serif",
                    fontWeight: 600,
                    fontSize: '14px',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px',
                    pointerEvents: 'none',
                    zIndex: 1000,
                    boxShadow: `0 0 20px ${colorStr}40`
                  }}>
                    <div style={{
                      width: '8px',
                      height: '8px',
                      borderRadius: '50%',
                      backgroundColor: colorStr,
                      boxShadow: `0 0 8px ${colorStr}`
                    }} />
                    Tracking Mouse {trackedMouseId}
                  </div>
                )
              })()}
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
                {isAutoLoading ? (
                  <>
                    <div className="spinner" style={{ margin: '0 auto 20px', width: '48px', height: '48px' }} />
                    <div style={{ fontSize: '20px', marginBottom: '10px', fontWeight: '500' }}>
                      Loading session...
                    </div>
                    <div style={{ fontSize: '14px', opacity: 0.7 }}>
                      Preparing visualization from analytics
                    </div>
                  </>
                ) : (
                  <>
                    <div style={{ fontSize: '72px', marginBottom: '20px', opacity: 0.3 }}>🐭</div>
                    <div style={{ fontSize: '20px', marginBottom: '10px', fontWeight: '500' }}>
                      Select a file to begin
                    </div>
                    <div style={{ fontSize: '14px', opacity: 0.7 }}>
                      Interactive mouse behavior visualization and analysis
                    </div>
                  </>
                )}
              </div>
              
              {/* Hand-drawn arrow pointing to logo - hide while auto-loading */}
              {!isAutoLoading && (
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
              )}
            </>
          )}
        </div>

        {/* Timeline - YouTube Style: Always small with popup preview */}
        {frames && frames.length > 0 && metadata && (
          <>
            {/* Hover detection area */}
            <div style={{
              position: 'absolute',
              bottom: '64px',
              left: '0',
              right: '0',
              height: '30px', // Small hover area above timeline
              zIndex: 1000,
              pointerEvents: 'auto',
              cursor: isDragging ? 'none' : 'pointer'
            }}
            onMouseEnter={() => {
              setTimelineHovered(true)
            }}
            onMouseMove={(e) => {
              if (canvasRef.current) {
                const rect = canvasRef.current.getBoundingClientRect()
                const x = e.clientX - rect.left
                setTimelineMouseX(x)
                
                // Scrub while dragging
                if (isDragging) {
                  const totalFrames = metadata.total_frames
                  const frame = Math.floor((x / rect.width) * totalFrames)
                  const clampedFrame = Math.max(0, Math.min(frame, totalFrames - 1))
                  setCurrentFrame(clampedFrame)
                  
                  // Use throttled seek to prevent spamming backend during rapid scrubbing
                  if (!isFrameLoaded(clampedFrame)) {
                    throttledSeek(clampedFrame)
                  }
                }
              }
            }}
            onMouseDown={(e) => {
              if (canvasRef.current) {
                // Save play state before scrubbing starts
                setPlayStateBeforeScrub(playing)
                
                setIsDragging(true)
                const rect = canvasRef.current.getBoundingClientRect()
                const x = e.clientX - rect.left
                const totalFrames = metadata.total_frames
                const frame = Math.floor((x / rect.width) * totalFrames)
                const clampedFrame = Math.max(0, Math.min(frame, totalFrames - 1))
                setCurrentFrame(clampedFrame)
                
                // If frame isn't loaded, trigger IMMEDIATE seek (bypass throttle)
                if (!isFrameLoaded(clampedFrame)) {
                  console.log(`🎯 Clicked unloaded frame ${clampedFrame}, seeking immediately...`)
                  // Clear pending seeks and throttle state
                  pendingSeekRef.current = null
                  lastSeekTimeRef.current = Date.now()
                  isSeekingRef.current = true
                  // Immediate seek
                  seekToFrame(clampedFrame)
                }
                
                // Pause playback while scrubbing
                setPlaying(false)
              }
            }}
            onMouseLeave={() => {
              if (!isDragging) {
                setTimelineHovered(false)
                setTimelineMouseX(null)
              }
            }}
            />
            
            {/* Timeline canvas - always small (8px) */}
            <canvas
              ref={canvasRef}
              style={{
                position: 'absolute',
                bottom: '64px',
                left: '0',
                right: '0',
                width: '100%',
                height: '8px', // Always small like YouTube
                cursor: 'pointer',
                borderTop: '0.5px solid rgba(255, 255, 255, 0.1)',
                background: 'rgba(20, 20, 35, 0.8)',
                zIndex: 10,
                display: 'block',
                pointerEvents: 'none' // Events handled by hover detection area
              }}
            />
            
            {/* YouTube-style popup frame preview */}
            {timelineHovered && timelineMouseX !== null && metadata && !isDragging && previewFrameNumber !== null && (
              (() => {
                const rect = canvasRef.current?.getBoundingClientRect()
                if (!rect) return null
                
                // Calculate position - center popup on mouse, but clamp to screen edges
                const popupWidth = 200
                const popupHeight = 150
                let popupX = timelineMouseX - popupWidth / 2
                
                // Clamp to screen bounds
                popupX = Math.max(10, Math.min(popupX, rect.width - popupWidth - 10))
                
                return (
                  <div style={{
                    position: 'absolute',
                    left: `${popupX}px`,
                    bottom: '80px', // Above timeline
                    width: `${popupWidth}px`,
                    height: `${popupHeight}px`,
                    background: 'rgba(20, 20, 35, 0.95)',
                    backdropFilter: 'blur(20px)',
                    border: '0.5px solid rgba(255, 255, 255, 0.2)',
                    borderRadius: '8px',
                    zIndex: 2000,
                    pointerEvents: 'none',
                    boxShadow: '0 8px 32px rgba(0, 0, 0, 0.6)',
                    padding: '8px',
                    animation: 'fadeIn 0.1s ease-out',
                    userSelect: 'none',
                    WebkitUserSelect: 'none',
                    MozUserSelect: 'none',
                    msUserSelect: 'none'
                  }}>
                    {/* Frame preview */}
                    <div style={{
                      width: '100%',
                      height: 'calc(100% - 30px)',
                      background: 'rgba(30, 30, 45, 0.8)',
                      borderRadius: '4px',
                      overflow: 'hidden',
                      position: 'relative',
                      userSelect: 'none'
                    }}>
                      {previewFrame && metadata ? (() => {
                        // Render a mini visualization of the frame
                        const mice = previewFrame.mice || {}
                        const mouseIds = Object.keys(mice)
                        
                        // Calculate bounds to fit all points
                        let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity
                        mouseIds.forEach(mouseId => {
                          const mouseData = mice[mouseId]
                          mouseData.points.forEach((point: number[]) => {
                            minX = Math.min(minX, point[0])
                            minY = Math.min(minY, point[1])
                            maxX = Math.max(maxX, point[0])
                            maxY = Math.max(maxY, point[1])
                          })
                        })
                        
                        const dataWidth = maxX - minX
                        const dataHeight = maxY - minY
                        const padding = 10
                        const viewWidth = popupWidth - 16 // Account for popup padding
                        const viewHeight = popupHeight - 46 // Account for time label
                        
                        const scale = Math.min(
                          (viewWidth - padding * 2) / dataWidth,
                          (viewHeight - padding * 2) / dataHeight
                        )
                        
                        const offsetX = (viewWidth - dataWidth * scale) / 2 - minX * scale
                        // Flip Y coordinate to match viewer orientation
                        const offsetY = (viewHeight - dataHeight * scale) / 2 + maxY * scale
                        
                        return (
                          <svg
                            width="100%"
                            height="100%"
                            viewBox={`0 0 ${viewWidth} ${viewHeight}`}
                            style={{ display: 'block' }}
                          >
                            {mouseIds.map((mouseId) => {
                              const mouseData = mice[mouseId]
                              // Use same colors as EnhancedViewer - map by mouseId, not index
                              const color = MOUSE_COLORS_HEX[parseInt(mouseId) % MOUSE_COLORS_HEX.length]
                              
                              return (
                                <g key={mouseId}>
                                  {/* Draw lines connecting points */}
                                  {mouseData.points.map((point: number[], i: number) => {
                                    if (i === 0) return null
                                    const prev = mouseData.points[i - 1]
                                    return (
                                      <line
                                        key={i}
                                        x1={prev[0] * scale + offsetX}
                                        y1={-prev[1] * scale + offsetY}
                                        x2={point[0] * scale + offsetX}
                                        y2={-point[1] * scale + offsetY}
                                        stroke={color}
                                        strokeWidth="1"
                                        opacity="0.4"
                                      />
                                    )
                                  })}
                                  
                                  {/* Draw points */}
                                  {mouseData.points.map((point: number[], i: number) => (
                                    <circle
                                      key={i}
                                      cx={point[0] * scale + offsetX}
                                      cy={-point[1] * scale + offsetY}
                                      r="1.5"
                                      fill={color}
                                      opacity="0.8"
                                    />
                                  ))}
                                </g>
                              )
                            })}
                          </svg>
                        )
                      })() : (
                        <div style={{
                          width: '100%',
                          height: '100%',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          fontSize: '10px',
                          color: 'rgba(255, 255, 255, 0.4)'
                        }}>
                          Loading...
                        </div>
                      )}
                    </div>
                    
                    {/* Time label */}
                    <div style={{
                      marginTop: '6px',
                      textAlign: 'center',
                      fontSize: '11px',
                      color: '#fff',
                      fontFamily: 'monospace'
                    }}>
                      {metadata.fps ? (() => {
                        const seconds = previewFrameNumber / metadata.fps
                        const mins = Math.floor(seconds / 60)
                        const secs = Math.floor(seconds % 60)
                        return `${mins}:${secs.toString().padStart(2, '0')}`
                      })() : `Frame ${previewFrameNumber}`}
                    </div>
                    
                    {/* Arrow pointing down to timeline */}
                    <div style={{
                      position: 'absolute',
                      left: '50%',
                      bottom: '-8px',
                      transform: 'translateX(-50%)',
                      width: '0',
                      height: '0',
                      borderLeft: '8px solid transparent',
                      borderRight: '8px solid transparent',
                      borderTop: '8px solid rgba(20, 20, 35, 0.95)'
                    }} />
                  </div>
                )
              })()
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

            {/* Timer Display - Center */}
            <div style={{
              position: 'absolute',
              left: '50%',
              transform: 'translateX(-50%)',
              fontSize: '13px',
              fontWeight: '600',
              color: '#e5e7eb',
              fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
              textShadow: '0 2px 4px rgba(0, 0, 0, 0.5)',
              userSelect: 'none'
            }}>
              {metadata && metadata.fps ? (() => {
                const totalFrames = metadata.total_frames || frames.length
                const currentTimeSec = currentFrame / metadata.fps
                const totalTimeSec = totalFrames / metadata.fps
                const formatTime = (sec: number) => {
                  const mins = Math.floor(sec / 60)
                  const secs = Math.floor(sec % 60)
                  return `${mins}:${secs.toString().padStart(2, '0')}`
                }
                return `${formatTime(currentTimeSec)} / ${formatTime(totalTimeSec)}`
              })() : `Frame ${currentFrame + 1}/${(metadata?.total_frames || frames.length)}`}
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

export default Viewer

