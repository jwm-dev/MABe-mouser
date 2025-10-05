import { useState, useEffect, useCallback, useRef } from 'react'

interface UseProgressiveLoadingOptions {
  lab: string
  filename: string
  chunkSize?: number
  onProgress?: (loaded: number, total: number) => void
}

interface FrameData {
  frame_number: number
  mice: Record<string, any>
}

interface FileData {
  frames: FrameData[]
  metadata: {
    total_frames: number
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
}

/**
 * Hook for progressive file loading via Server-Sent Events
 * Loads frames in chunks for instant first render
 */
export function useProgressiveLoading({
  lab,
  filename,
  chunkSize = 100,
  onProgress
}: UseProgressiveLoadingOptions) {
  const [frames, setFrames] = useState<FrameData[]>([])
  const [metadata, setMetadata] = useState<FileData['metadata'] | null>(null)
  const [loading, setLoading] = useState(false)
  const [progress, setProgress] = useState(0)
  const [error, setError] = useState<string | null>(null)
  
  const eventSourceRef = useRef<EventSource | null>(null)
  const framesCountRef = useRef(0)
  const metadataRef = useRef<FileData['metadata'] | null>(null)
  const timeoutRef = useRef<number | null>(null)

  const load = useCallback(() => {
    if (!lab || !filename) return

    console.log(`📡 Starting SSE stream: /api/files/${lab}/${filename}/stream?chunk_size=${chunkSize}`)

    setLoading(true)
    setError(null)
    setFrames([])
    setMetadata(null)
    setProgress(0)
    framesCountRef.current = 0
    metadataRef.current = null

    // Clear any existing timeout
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current)
    }

    // Set a timeout to detect stuck loading (30 seconds)
    timeoutRef.current = setTimeout(() => {
      console.error('⏱️ Loading timeout - no data received in 30s')
      setError('Loading timeout - no response from server')
      setLoading(false)
      if (eventSourceRef.current) {
        eventSourceRef.current.close()
      }
    }, 30000)

    // Close any existing connection
    if (eventSourceRef.current) {
      eventSourceRef.current.close()
    }

    // Create new Server-Sent Events connection
    const url = `/api/files/${lab}/${filename}/stream?chunk_size=${chunkSize}`
    console.log(`🔌 Opening EventSource: ${url}`)
    const eventSource = new EventSource(url)
    eventSourceRef.current = eventSource

    eventSource.onopen = () => {
      console.log('✅ EventSource connection opened')
    }

    eventSource.onmessage = (event) => {
      // Clear timeout - we're receiving data
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current)
        timeoutRef.current = null
      }

      try {
        const data = JSON.parse(event.data)
        console.log('📨 Received SSE message:', Object.keys(data))

        if (data.error) {
          console.error('❌ Server error:', data.error)
          setError(data.error)
          setLoading(false)
          eventSource.close()
          return
        }

        // Handle metadata message
        if (data.metadata) {
          console.log(`📊 Metadata received: ${data.metadata.total_frames} frames, ${data.metadata.num_mice} mice`)
          setMetadata(data.metadata)
          metadataRef.current = data.metadata
          return
        }

        // Handle frames message
        if (data.frames) {
          const numFrames = data.frames.length
          console.log(`📦 Received ${numFrames} frames (total: ${framesCountRef.current + numFrames})`)
          
          setFrames(prev => [...prev, ...data.frames])
          framesCountRef.current += numFrames

          // Calculate progress using metadata ref
          if (metadataRef.current) {
            const progressPct = (framesCountRef.current / metadataRef.current.total_frames) * 100
            setProgress(progressPct)

            if (onProgress) {
              onProgress(framesCountRef.current, metadataRef.current.total_frames)
            }
          }

          // Check if we're done
          if (data.complete) {
            console.log(`✅ Loading complete: ${framesCountRef.current} frames`)
            setLoading(false)
            setProgress(100)
            eventSource.close()
          }
        }
      } catch (err) {
        console.error('❌ Failed to parse SSE data:', err, event.data)
        setError('Failed to parse data from server')
        setLoading(false)
        eventSource.close()
      }
    }

    eventSource.onerror = (err) => {
      console.error('❌ SSE error:', err)
      
      // Check if we got any data before the error
      if (framesCountRef.current === 0) {
        setError('Connection error - no data received')
      } else {
        console.log(`⚠️  Connection closed after ${framesCountRef.current} frames`)
      }
      
      setLoading(false)
      eventSource.close()
    }
  }, [lab, filename, chunkSize, onProgress])

  // Auto-trigger load when lab or filename changes
  useEffect(() => {
    // Cleanup previous stream first
    if (eventSourceRef.current) {
      console.log('🔄 Canceling previous stream...')
      eventSourceRef.current.close()
      eventSourceRef.current = null
    }

    // Then start new load
    if (lab && filename) {
      console.log(`🚀 Auto-loading: ${lab}/${filename}`)
      load()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [lab, filename])  // Don't include load - it would cause infinite loop

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (eventSourceRef.current) {
        console.log('🧹 Cleanup on unmount')
        eventSourceRef.current.close()
        eventSourceRef.current = null
      }
    }
  }, [])

  return {
    frames,
    metadata,
    loading,
    progress,
    error,
    load
  }
}
