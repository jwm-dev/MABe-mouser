import { useState, useEffect, useCallback, useRef } from 'react'

interface FrameData {
  frame_number: number
  mice: Record<string, any>
}

interface FileMetadata {
  total_frames: number
  min_frame: number
  max_frame: number
  num_mice: number
  fps?: number
  video_width?: number
  video_height?: number
  arena_width_cm?: number
  arena_height_cm?: number
  pix_per_cm?: number
  body_parts?: string[]
  has_annotations?: boolean
  annotations?: Array<{
    agent_id: number
    target_id: number
    action: string
    start_frame: number
    stop_frame: number
  }>
}

interface LoadedRange {
  start: number
  end: number
}

interface UseSmartLoadingOptions {
  lab: string
  filename: string
  chunkSize?: number
  seekChunkSize?: number
  onProgress?: (loaded: number, total: number) => void
}

/**
 * Smart loading hook that supports:
 * - Initial sequential loading
 * - On-demand seeking to arbitrary positions (YouTube-style)
 * - Background filling of gaps
 * - Tracking of loaded ranges for timeline visualization
 */
export function useSmartLoading({
  lab,
  filename,
  chunkSize = 100,
  seekChunkSize = 50,
  onProgress
}: UseSmartLoadingOptions) {
  const [framesMap, setFramesMap] = useState<Map<number, FrameData>>(new Map())
  const [metadata, setMetadata] = useState<FileMetadata | null>(null)
  const [loading, setLoading] = useState(false)
  const [loadedRanges, setLoadedRanges] = useState<LoadedRange[]>([])
  const [progress, setProgress] = useState(0)
  const [error, setError] = useState<string | null>(null)
  
  // Single event source - YouTube style (one stream at a time)
  const eventSourceRef = useRef<EventSource | null>(null)
  const framesMapRef = useRef<Map<number, FrameData>>(new Map())
  const metadataRef = useRef<FileMetadata | null>(null)
  const loadedRangesRef = useRef<LoadedRange[]>([])
  const currentLoadStartRef = useRef<number | null>(null)
  const lastSeekTargetRef = useRef<number | null>(null) // Track last seek to avoid redundant seeks

  // Keep refs in sync
  useEffect(() => {
    framesMapRef.current = framesMap
  }, [framesMap])

  useEffect(() => {
    metadataRef.current = metadata
  }, [metadata])

  useEffect(() => {
    loadedRangesRef.current = loadedRanges
  }, [loadedRanges])

  // Merge and optimize loaded ranges
  const mergeRanges = useCallback((ranges: LoadedRange[]): LoadedRange[] => {
    if (ranges.length === 0) return []
    
    const sorted = [...ranges].sort((a, b) => a.start - b.start)
    const merged: LoadedRange[] = [sorted[0]]
    
    for (let i = 1; i < sorted.length; i++) {
      const current = sorted[i]
      const last = merged[merged.length - 1]
      
      // Merge if overlapping or adjacent
      if (current.start <= last.end + 1) {
        last.end = Math.max(last.end, current.end)
      } else {
        merged.push(current)
      }
    }
    
    return merged
  }, [])

  // Add frames to the map and update ranges
  const addFrames = useCallback((frames: FrameData[]) => {
    if (frames.length === 0) return
    
    const meta = metadataRef.current
    if (!meta) {
      // Metadata not yet available - store frames with actual frame numbers temporarily
      // They will be re-indexed when metadata arrives
      console.log(`⏳ Storing ${frames.length} frames temporarily (metadata pending)`)
      setFramesMap(prev => {
        const newMap = new Map(prev)
        frames.forEach(frame => {
          newMap.set(frame.frame_number, frame)
        })
        return newMap
      })
      return
    }
    
    setFramesMap(prev => {
      const newMap = new Map(prev)
      frames.forEach(frame => {
        // Convert actual parquet frame number to 0-indexed display frame
        const displayFrame = frame.frame_number - meta.min_frame
        // Store frame with 0-indexed key but keep original frame_number
        newMap.set(displayFrame, frame)
      })
      return newMap
    })

    // Update loaded ranges using 0-indexed frames
    const displayFrames = frames.map(f => f.frame_number - meta.min_frame)
    const minFrame = Math.min(...displayFrames)
    const maxFrame = Math.max(...displayFrames)
    
    setLoadedRanges(prev => mergeRanges([...prev, { start: minFrame, end: maxFrame }]))
  }, [mergeRanges])

  // Check if a frame is loaded
  const isFrameLoaded = useCallback((frameNumber: number): boolean => {
    return framesMapRef.current.has(frameNumber)
  }, [])

  // Load frames from a specific start position (YouTube-style: single continuous stream)
  const loadFromPosition = useCallback((startFrame: number) => {
    if (!lab || !filename) return
    
    // Cancel any existing stream (YouTube behavior: only one stream at a time)
    if (eventSourceRef.current) {
      console.log('� Canceling existing stream')
      eventSourceRef.current.close()
      eventSourceRef.current = null
    }
    
    // Track where we're loading from (using display frame)
    currentLoadStartRef.current = startFrame
    
    const meta = metadataRef.current
    let actualFrame = startFrame
    
    // If metadata is available, convert to actual parquet frame
    if (meta) {
      actualFrame = startFrame + meta.min_frame
      console.log(`📡 Starting continuous load from display frame ${startFrame} (actual frame ${actualFrame})`)
    } else {
      // First load - metadata will come in the stream
      console.log(`📡 Starting initial load from frame ${startFrame} (metadata will be received)`)
    }
    
    setLoading(true)
    
    const url = `/api/files/${lab}/${filename}/stream?start_frame=${actualFrame}&chunk_size=${chunkSize}`
    const eventSource = new EventSource(url)
    eventSourceRef.current = eventSource

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        
        if (data.error) {
          console.error('❌ Server error:', data.error)
          setError(data.error)
          setLoading(false)
          eventSource.close()
          eventSourceRef.current = null
          return
        }

        // Handle metadata
        if (data.metadata) {
          console.log(`📊 Metadata received: ${data.metadata.total_frames} frames (${data.metadata.min_frame}-${data.metadata.max_frame})`)
          setMetadata(data.metadata)
          metadataRef.current = data.metadata
          
          // If we have frames that were stored with actual frame numbers, re-index them
          if (framesMapRef.current.size > 0) {
            console.log(`🔄 Re-indexing ${framesMapRef.current.size} frames with metadata`)
            const existingFrames = Array.from(framesMapRef.current.values())
            
            // Re-add with proper 0-indexed keys
            const newMap = new Map<number, FrameData>()
            existingFrames.forEach(frame => {
              const displayFrame = frame.frame_number - data.metadata.min_frame
              newMap.set(displayFrame, frame)
            })
            setFramesMap(newMap)
            framesMapRef.current = newMap
            
            // Update loaded ranges
            const displayFrames = existingFrames.map(f => f.frame_number - data.metadata.min_frame)
            const minFrame = Math.min(...displayFrames)
            const maxFrame = Math.max(...displayFrames)
            setLoadedRanges([{ start: minFrame, end: maxFrame }])
            loadedRangesRef.current = [{ start: minFrame, end: maxFrame }]
          }
          return
        }

        // Handle frames
        if (data.frames && data.frames.length > 0) {
          addFrames(data.frames)
          
          // Update progress
          if (metadataRef.current) {
            const loadedCount = framesMapRef.current.size
            const progressPct = (loadedCount / metadataRef.current.total_frames) * 100
            setProgress(progressPct)
            
            if (onProgress) {
              onProgress(loadedCount, metadataRef.current.total_frames)
            }
          }
          
          // If complete, we're done (YouTube: just finish, no gap filling)
          if (data.complete) {
            console.log(`✅ Stream complete from frame ${startFrame}`)
            setLoading(false)
            eventSource.close()
            eventSourceRef.current = null
            currentLoadStartRef.current = null
          }
        }
      } catch (err) {
        console.error('❌ Failed to parse SSE data:', err)
        setError('Failed to parse data from server')
        setLoading(false)
        eventSource.close()
        eventSourceRef.current = null
      }
    }

    eventSource.onerror = (err) => {
      console.error('❌ SSE error:', err)
      setLoading(false)
      eventSource.close()
      eventSourceRef.current = null
      currentLoadStartRef.current = null
    }
  }, [lab, filename, chunkSize, addFrames, onProgress])

  // Seek to a specific frame (YouTube behavior: cancel current stream, start from new position)
  const seekToFrame = useCallback((frameNumber: number) => {
    if (!metadataRef.current) return
    
    // Clamp to valid range
    const clampedFrame = Math.max(0, Math.min(frameNumber, metadataRef.current.total_frames - 1))
    
    // Calculate seek start position
    const seekStart = Math.max(0, clampedFrame - Math.floor(seekChunkSize / 4))
    
    // Avoid redundant seeks - if we're already seeking to this position, skip
    if (lastSeekTargetRef.current === seekStart) {
      return
    }
    
    console.log(`🎯 Seeking to frame ${clampedFrame} (loading from ${seekStart})`)
    lastSeekTargetRef.current = seekStart
    
    // ALWAYS cancel current stream and start from new position (YouTube-style)
    loadFromPosition(seekStart)
  }, [seekChunkSize, loadFromPosition])

  // Convert frames map to sorted array (for rendering)
  const getSortedFrames = useCallback((): FrameData[] => {
    return Array.from(framesMap.values()).sort((a, b) => a.frame_number - b.frame_number)
  }, [framesMap])

  // Get frame by number
  const getFrame = useCallback((frameNumber: number): FrameData | undefined => {
    return framesMap.get(frameNumber)
  }, [framesMap])

  // Fetch a single frame on demand (for preview hover)
  const fetchFrame = useCallback(async (frameNumber: number): Promise<FrameData | null> => {
    const meta = metadataRef.current
    if (!meta) {
      console.warn('⚠️ Cannot fetch frame without metadata')
      return null
    }
    
    // If already loaded, return immediately
    const existingFrame = framesMap.get(frameNumber)
    if (existingFrame) {
      console.log(`📦 Display frame ${frameNumber} already in cache`)
      return existingFrame
    }

    // Convert 0-indexed display frame to actual parquet frame number
    const actualFrame = frameNumber + meta.min_frame
    console.log(`🌐 Fetching display frame ${frameNumber} (actual frame ${actualFrame}) from server...`)
    
    try {
      const url = `/api/files/${lab}/${filename}/frame/${actualFrame}`
      console.log(`🔗 Request URL: ${url}`)
      
      const response = await fetch(url)
      
      if (!response.ok) {
        console.error(`❌ Server returned ${response.status} for frame ${actualFrame}`)
        return null
      }
      
      const data = await response.json()
      console.log(`✅ Frame ${actualFrame} fetched successfully:`, data)
      
      // Add to framesMap for future use (using display frame as key)
      setFramesMap(prev => {
        const next = new Map(prev)
        next.set(frameNumber, data)
        return next
      })
      
      return data
    } catch (err) {
      console.error(`❌ Network error fetching frame ${frameNumber}:`, err)
      return null
    }
  }, [lab, filename, framesMap])

  // Initial load on mount (and cleanup on file change)
  useEffect(() => {
    // CRITICAL: Close any existing connection when file changes
    if (eventSourceRef.current) {
      console.log('🛑 File changed - canceling existing stream')
      eventSourceRef.current.close()
      eventSourceRef.current = null
    }

    if (lab && filename) {
      console.log(`🚀 Initial load: ${lab}/${filename}`)
      // Clear all state
      setFramesMap(new Map())
      setLoadedRanges([])
      setProgress(0)
      setError(null)
      framesMapRef.current = new Map()
      loadedRangesRef.current = []
      currentLoadStartRef.current = null
      lastSeekTargetRef.current = null // Reset seek tracking
      
      // Start loading from beginning (YouTube: continuous forward)
      loadFromPosition(0)
    }

    // Cleanup on unmount or file change
    return () => {
      if (eventSourceRef.current) {
        console.log('🧹 Cleanup - closing stream')
        eventSourceRef.current.close()
        eventSourceRef.current = null
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [lab, filename])

  return {
    frames: getSortedFrames(),
    framesMap,
    getFrame,
    fetchFrame,
    metadata,
    loading,
    loadedRanges,
    progress,
    error,
    seekToFrame,
    isFrameLoaded
  }
}
