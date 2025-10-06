import { useState, useEffect, useMemo, memo } from 'react'
import { useNavigate } from 'react-router-dom'
import { DeckGL } from '@deck.gl/react'
import { HexagonLayer } from '@deck.gl/aggregation-layers'
import { OrthographicView } from '@deck.gl/core'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'

type AnalyticsView = 'overview' | 'heatmap' | 'activity' | 'social' | 'temporal' | 'comparison' | 'features'

// Tooltip component for hover explanations
function InfoTooltip({ text }: { text: string }) {
  const [show, setShow] = useState(false)
  
  return (
    <div
      style={{ position: 'relative', display: 'inline-block', marginLeft: '8px' }}
      onMouseEnter={() => setShow(true)}
      onMouseLeave={() => setShow(false)}
    >
      <span style={{
        fontSize: '14px',
        color: 'rgba(102, 126, 234, 0.7)',
        cursor: 'help',
        border: '1px solid rgba(102, 126, 234, 0.5)',
        borderRadius: '50%',
        width: '18px',
        height: '18px',
        display: 'inline-flex',
        alignItems: 'center',
        justifyContent: 'center',
        fontWeight: '600',
        transition: 'all 0.2s ease'
      }}>
        ?
      </span>
      {show && (
        <div style={{
          position: 'absolute',
          bottom: '100%',
          left: '50%',
          transform: 'translateX(-50%) translateY(-8px)',
          background: 'rgba(20, 20, 35, 0.98)',
          border: '1px solid rgba(102, 126, 234, 0.5)',
          borderRadius: '8px',
          padding: '12px 16px',
          fontSize: '13px',
          color: '#e5e7eb',
          whiteSpace: 'nowrap',
          zIndex: 1000,
          boxShadow: '0 8px 32px rgba(0, 0, 0, 0.4)',
          animation: 'fadeIn 0.2s ease',
          pointerEvents: 'none'
        }}>
          {text}
          <div style={{
            position: 'absolute',
            top: '100%',
            left: '50%',
            transform: 'translateX(-50%)',
            width: 0,
            height: 0,
            borderLeft: '6px solid transparent',
            borderRight: '6px solid transparent',
            borderTop: '6px solid rgba(20, 20, 35, 0.98)'
          }} />
        </div>
      )}
    </div>
  )
}


export default function Analytics() {
  const navigate = useNavigate()
  const [activeView, setActiveView] = useState<AnalyticsView>('overview')

  const toolbarItems = [
    { id: 'overview' as AnalyticsView, icon: '📊', label: 'Overview', description: 'Dataset Summary' },
    { id: 'heatmap' as AnalyticsView, icon: '🗺️', label: 'Heatmap', description: 'Spatial Analysis' },
    { id: 'activity' as AnalyticsView, icon: '📈', label: 'Activity', description: 'Temporal Patterns' },
    { id: 'social' as AnalyticsView, icon: '🐭', label: 'Social', description: 'Interaction Graph' },
    { id: 'temporal' as AnalyticsView, icon: '⏰', label: 'Temporal', description: 'Time Series' },
    { id: 'comparison' as AnalyticsView, icon: '⚖️', label: 'Compare', description: 'Multi-Session' },
    { id: 'features' as AnalyticsView, icon: '🔧', label: 'Features', description: 'ML Engineering' }
  ]

  return (
    <div style={{
      display: 'flex',
      height: '100vh',
      width: '100vw',
      background: 'linear-gradient(135deg, #0a0a15 0%, #1a1a2e 100%)',
      fontFamily: "'Poppins', sans-serif",
      overflow: 'hidden'
    }}>
      {/* Vertical Logographic Toolbar */}
      <div style={{
        width: '80px',
        background: 'linear-gradient(180deg, rgba(20, 20, 35, 0.95) 0%, rgba(15, 15, 30, 0.95) 100%)',
        backdropFilter: 'blur(20px)',
        borderRight: '1px solid rgba(102, 126, 234, 0.2)',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        padding: '20px 0',
        gap: '8px',
        boxShadow: '2px 0 20px rgba(0, 0, 0, 0.3)',
        zIndex: 1000
      }}>
        {/* Logo/Home Button */}
        <div
          onClick={() => navigate('/')}
          style={{
            fontSize: '32px',
            cursor: 'pointer',
            marginBottom: '20px',
            padding: '8px',
            borderRadius: '12px',
            transition: 'all 0.2s ease',
            background: 'rgba(102, 126, 234, 0.1)',
            border: '1px solid rgba(102, 126, 234, 0.3)',
            userSelect: 'none',
            WebkitUserSelect: 'none'
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.background = 'rgba(102, 126, 234, 0.2)'
            e.currentTarget.style.transform = 'scale(1.05)'
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.background = 'rgba(102, 126, 234, 0.1)'
            e.currentTarget.style.transform = 'scale(1)'
          }}
          title="Back to Dashboard"
        >
          🐱
        </div>

        {/* Toolbar Items */}
        {toolbarItems.map((item) => (
          <div
            key={item.id}
            onClick={() => setActiveView(item.id)}
            style={{
              width: '56px',
              height: '56px',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: '24px',
              cursor: 'pointer',
              borderRadius: '12px',
              transition: 'all 0.2s ease',
              background: activeView === item.id 
                ? 'linear-gradient(135deg, rgba(102, 126, 234, 0.3) 0%, rgba(118, 75, 162, 0.3) 100%)'
                : 'rgba(255, 255, 255, 0.05)',
              border: activeView === item.id
                ? '1px solid rgba(102, 126, 234, 0.5)'
                : '1px solid rgba(255, 255, 255, 0.1)',
              boxShadow: activeView === item.id
                ? '0 4px 15px rgba(102, 126, 234, 0.3), inset 0 0 20px rgba(102, 126, 234, 0.1)'
                : 'none',
              userSelect: 'none',
              WebkitUserSelect: 'none',
              position: 'relative'
            }}
            onMouseEnter={(e) => {
              if (activeView !== item.id) {
                e.currentTarget.style.background = 'rgba(102, 126, 234, 0.15)'
                e.currentTarget.style.borderColor = 'rgba(102, 126, 234, 0.3)'
              }
              e.currentTarget.style.transform = 'scale(1.05)'
            }}
            onMouseLeave={(e) => {
              if (activeView !== item.id) {
                e.currentTarget.style.background = 'rgba(255, 255, 255, 0.05)'
                e.currentTarget.style.borderColor = 'rgba(255, 255, 255, 0.1)'
              }
              e.currentTarget.style.transform = 'scale(1)'
            }}
            title={`${item.label}: ${item.description}`}
          >
            <span style={{ filter: 'drop-shadow(0 2px 4px rgba(0, 0, 0, 0.3))' }}>
              {item.icon}
            </span>
          </div>
        ))}
      </div>

      {/* Main Content Area */}
      <div style={{
        flex: 1,
        display: 'flex',
        flexDirection: 'column',
        overflow: 'hidden'
      }}>
        {/* Header */}
        <div style={{
          padding: '20px 32px',
          background: 'linear-gradient(135deg, rgba(20, 20, 35, 0.8) 0%, rgba(15, 15, 30, 0.8) 100%)',
          backdropFilter: 'blur(10px)',
          borderBottom: '1px solid rgba(102, 126, 234, 0.2)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between'
        }}>
          <div>
            <h1 style={{
              fontSize: '28px',
              fontWeight: '700',
              background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              margin: 0,
              marginBottom: '4px',
              letterSpacing: '-0.5px'
            }}>
              {toolbarItems.find(item => item.id === activeView)?.label}
            </h1>
            <p style={{
              fontSize: '14px',
              color: 'rgba(229, 231, 235, 0.6)',
              margin: 0
            }}>
              {toolbarItems.find(item => item.id === activeView)?.description}
            </p>
          </div>
          
          <div style={{
            display: 'flex',
            gap: '12px',
            alignItems: 'center'
          }}>
            {/* Action Buttons */}
            <button style={{
              padding: '10px 20px',
              background: 'linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(5, 150, 105, 0.2) 100%)',
              border: '1px solid rgba(16, 185, 129, 0.4)',
              borderRadius: '8px',
              color: '#10b981',
              fontSize: '14px',
              fontWeight: '500',
              cursor: 'pointer',
              transition: 'all 0.2s ease',
              fontFamily: "'Poppins', sans-serif"
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.background = 'linear-gradient(135deg, rgba(16, 185, 129, 0.3) 0%, rgba(5, 150, 105, 0.3) 100%)'
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.background = 'linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(5, 150, 105, 0.2) 100%)'
            }}
            >
              📥 Export Data
            </button>
            
            <button style={{
              padding: '10px 20px',
              background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.2) 0%, rgba(37, 99, 235, 0.2) 100%)',
              border: '1px solid rgba(59, 130, 246, 0.4)',
              borderRadius: '8px',
              color: '#3b82f6',
              fontSize: '14px',
              fontWeight: '500',
              cursor: 'pointer',
              transition: 'all 0.2s ease',
              fontFamily: "'Poppins', sans-serif"
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.background = 'linear-gradient(135deg, rgba(59, 130, 246, 0.3) 0%, rgba(37, 99, 235, 0.3) 100%)'
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.background = 'linear-gradient(135deg, rgba(59, 130, 246, 0.2) 0%, rgba(37, 99, 235, 0.2) 100%)'
            }}
            >
              ⚙️ Settings
            </button>
          </div>
        </div>

        {/* Content Panel */}
        <div style={{
          flex: 1,
          overflow: 'auto',
          padding: '24px',
          background: 'rgba(10, 10, 20, 0.5)'
        }}>
          {activeView === 'overview' && <OverviewPanel />}
          {activeView === 'heatmap' && <HeatmapPanel />}
          {activeView === 'activity' && <ActivityPanel />}
          {activeView === 'social' && <SocialPanel />}
          {activeView === 'temporal' && <TemporalPanel />}
          {activeView === 'comparison' && <ComparisonPanel />}
          {activeView === 'features' && <FeaturesPanel />}
        </div>
      </div>
    </div>
  )
}

// Panels implementation
function OverviewPanel() {
  const [stats, setStats] = useState({
    total_sessions: 0,
    total_frames: 0,
    avg_duration_seconds: 0,
    mice_tracked: 0,
    mice_untracked: 0,
    mice_total: 0,
    loading: true
  })

  useEffect(() => {
    fetch('http://localhost:8000/api/analytics/overview')
      .then(res => res.json())
      .then(data => setStats({ ...data, loading: false }))
      .catch(err => {
        console.error('Error fetching overview:', err)
        setStats(prev => ({ ...prev, loading: false }))
      })
  }, [])

  const statCards = [
    { icon: '📁', label: 'Total Sessions', value: stats.total_sessions.toLocaleString(), color: '#667eea', tooltip: 'Number of unique recording sessions across all labs' },
    { icon: '🎞️', label: 'Total Frames', value: stats.total_frames.toLocaleString(), color: '#764ba2', tooltip: 'Total video frames captured across all sessions' },
    { icon: '⏱️', label: 'Avg Duration', value: `${stats.avg_duration_seconds.toFixed(1)}s`, color: '#3b82f6', tooltip: 'Average session length calculated from actual video metadata' },
    { icon: '🐭', label: 'Mice Tracked', value: stats.mice_tracked.toLocaleString(), color: '#10b981', tooltip: 'Unique mice with individual IDs for tracking' },
    { icon: '👻', label: 'Mice Untracked', value: stats.mice_untracked.toLocaleString(), color: '#f59e0b', tooltip: 'Mice present in videos but without individual tracking IDs' },
    { icon: '🎯', label: 'Total Mice', value: stats.mice_total.toLocaleString(), color: '#8b5cf6', tooltip: 'Total unique mice in the dataset (tracked + untracked)' }
  ]

  return (
    <div style={{
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
      gap: '20px',
      maxWidth: '1400px',
      margin: '0 auto'
    }}>
      {/* Stats Cards */}
      {statCards.map((stat, idx) => (
        <div key={idx} style={{
          background: 'linear-gradient(135deg, rgba(20, 20, 35, 0.9) 0%, rgba(15, 15, 30, 0.9) 100%)',
          backdropFilter: 'blur(10px)',
          border: '1px solid rgba(102, 126, 234, 0.2)',
          borderRadius: '16px',
          padding: '24px',
          display: 'flex',
          alignItems: 'center',
          gap: '16px',
          boxShadow: '0 8px 32px rgba(0, 0, 0, 0.2)',
          transition: 'all 0.2s ease',
          position: 'relative'
        }}
        onMouseEnter={(e) => {
          e.currentTarget.style.transform = 'translateY(-4px)'
          e.currentTarget.style.boxShadow = '0 12px 40px rgba(102, 126, 234, 0.3)'
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.transform = 'translateY(0)'
          e.currentTarget.style.boxShadow = '0 8px 32px rgba(0, 0, 0, 0.2)'
        }}
        >
          {stats.loading && (
            <div style={{
              position: 'absolute',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              background: 'rgba(10, 10, 20, 0.7)',
              backdropFilter: 'blur(4px)',
              borderRadius: '16px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center'
            }}>
              <div style={{
                width: '24px',
                height: '24px',
                border: '3px solid rgba(102, 126, 234, 0.3)',
                borderTop: '3px solid #667eea',
                borderRadius: '50%',
                animation: 'spin 1s linear infinite'
              }} />
            </div>
          )}
          <div style={{
            fontSize: '40px',
            filter: `drop-shadow(0 4px 8px ${stat.color}40)`
          }}>
            {stat.icon}
          </div>
          <div style={{ flex: 1 }}>
            <div style={{
              fontSize: '32px',
              fontWeight: '700',
              color: stat.color,
              marginBottom: '4px'
            }}>
              {stat.value}
            </div>
            <div style={{
              fontSize: '14px',
              color: 'rgba(229, 231, 235, 0.6)',
              display: 'flex',
              alignItems: 'center'
            }}>
              {stat.label}
              <InfoTooltip text={stat.tooltip} />
            </div>
          </div>
        </div>
      ))}

      {/* Dataset Browser */}
      <div style={{
        gridColumn: '1 / -1',
        background: 'linear-gradient(135deg, rgba(20, 20, 35, 0.9) 0%, rgba(15, 15, 30, 0.9) 100%)',
        backdropFilter: 'blur(10px)',
        border: '1px solid rgba(102, 126, 234, 0.2)',
        borderRadius: '16px',
        padding: '24px',
        marginTop: '20px',
        boxShadow: '0 8px 32px rgba(0, 0, 0, 0.2)'
      }}>
        <h2 style={{
          fontSize: '20px',
          fontWeight: '600',
          color: '#e5e7eb',
          marginBottom: '16px',
          display: 'flex',
          alignItems: 'center',
          gap: '8px'
        }}>
          📂 Dataset Browser
          <InfoTooltip text="Browse all recording sessions with detailed metadata" />
        </h2>
        
        {stats.loading ? (
          <div style={{
            color: 'rgba(229, 231, 235, 0.5)',
            fontSize: '14px',
            textAlign: 'center',
            padding: '40px'
          }}>
            Loading dataset browser...
          </div>
        ) : (
          <>
            <DatasetBrowserTable />
            <DatasetStatistics />
          </>
        )}
      </div>
    </div>
  )
}

// Mouse metadata popup component
const MouseMetadataPopup = ({ mice, isVisible, position }: { 
  mice: any[]
  isVisible: boolean
  position: { x: number; y: number }
}) => {
  if (!isVisible || !mice || mice.length === 0) return null

  return (
    <div style={{
      position: 'fixed',
      left: `${position.x}px`,
      top: `${position.y}px`,
      zIndex: 10000,
      background: 'linear-gradient(135deg, rgba(30, 30, 50, 0.98) 0%, rgba(20, 20, 35, 0.98) 100%)',
      backdropFilter: 'blur(20px)',
      border: '1px solid rgba(167, 139, 250, 0.4)',
      borderRadius: '16px',
      padding: '16px',
      boxShadow: '0 12px 40px rgba(0, 0, 0, 0.5), 0 0 0 1px rgba(167, 139, 250, 0.2)',
      minWidth: '280px',
      maxWidth: '400px',
      animation: 'popupFadeIn 0.2s cubic-bezier(0.16, 1, 0.3, 1)',
      transformOrigin: 'top left',
      pointerEvents: 'none'
    }}>
      <style>{`
        @keyframes popupFadeIn {
          from {
            opacity: 0;
            transform: scale(0.95) translateY(-10px);
          }
          to {
            opacity: 1;
            transform: scale(1) translateY(0);
          }
        }
      `}</style>
      
      <div style={{
        fontSize: '11px',
        fontWeight: '700',
        color: 'rgba(167, 139, 250, 0.9)',
        textTransform: 'uppercase',
        letterSpacing: '1px',
        marginBottom: '12px',
        paddingBottom: '8px',
        borderBottom: '1px solid rgba(167, 139, 250, 0.2)'
      }}>
        Mouse Details
      </div>
      
      <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
        {mice.map((mouse, idx) => (
          <div key={idx} style={{
            padding: '12px',
            background: 'rgba(102, 126, 234, 0.1)',
            borderRadius: '10px',
            border: '1px solid rgba(102, 126, 234, 0.2)',
            transition: 'all 0.2s ease'
          }}>
            <div style={{
              fontSize: '13px',
              fontWeight: '700',
              color: '#a5b4fc',
              marginBottom: '8px',
              display: 'flex',
              alignItems: 'center',
              gap: '6px'
            }}>
              <span>🐭</span>
              <span>Mouse #{mouse.id}</span>
            </div>
            
            <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
              {mouse.strain && (
                <div style={{ fontSize: '11px', color: 'rgba(229, 231, 235, 0.8)' }}>
                  <span style={{ color: 'rgba(167, 139, 250, 0.8)', fontWeight: '600' }}>Strain:</span>{' '}
                  {mouse.strain}
                </div>
              )}
              {mouse.color && (
                <div style={{ fontSize: '11px', color: 'rgba(229, 231, 235, 0.8)' }}>
                  <span style={{ color: 'rgba(167, 139, 250, 0.8)', fontWeight: '600' }}>Color:</span>{' '}
                  {mouse.color}
                </div>
              )}
              {mouse.sex && (
                <div style={{ fontSize: '11px', color: 'rgba(229, 231, 235, 0.8)' }}>
                  <span style={{ color: 'rgba(167, 139, 250, 0.8)', fontWeight: '600' }}>Sex:</span>{' '}
                  {mouse.sex}
                </div>
              )}
              {mouse.age && (
                <div style={{ fontSize: '11px', color: 'rgba(229, 231, 235, 0.8)' }}>
                  <span style={{ color: 'rgba(167, 139, 250, 0.8)', fontWeight: '600' }}>Age:</span>{' '}
                  {mouse.age}
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

// Memoized row component for better performance
const SessionRow = memo(({ 
  session, 
  onClick 
}: { 
  session: any
  onClick: (session: any) => void
}) => {
  const [isHovered, setIsHovered] = useState(false)
  const [mousePopupVisible, setMousePopupVisible] = useState(false)
  const [mousePopupPosition, setMousePopupPosition] = useState({ x: 0, y: 0 })
  
  const handleMouseEnter = (e: React.MouseEvent) => {
    if (session.mice && session.mice.length > 0) {
      const rect = e.currentTarget.getBoundingClientRect()
      setMousePopupPosition({ 
        x: rect.right + 10,
        y: rect.top
      })
      setMousePopupVisible(true)
    }
  }
  
  return (
    <>
      <MouseMetadataPopup 
        mice={session.mice || []} 
        isVisible={mousePopupVisible} 
        position={mousePopupPosition} 
      />
      
      <tr 
        onClick={() => onClick(session)}
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
        style={{
          borderBottom: '1px solid rgba(102, 126, 234, 0.1)',
          transition: 'all 0.15s ease',
          cursor: 'pointer',
          background: isHovered ? 'rgba(102, 126, 234, 0.15)' : 'transparent',
          transform: isHovered ? 'scale(1.002)' : 'scale(1)'
        }}
    >
      <td style={{ padding: '12px', color: '#e5e7eb', fontFamily: 'monospace' }}>{session.video_id}</td>
      <td style={{ padding: '12px', color: '#a5b4fc' }}>{session.lab_id}</td>
      <td style={{ padding: '12px', textAlign: 'right', color: '#e5e7eb' }}>
        {session.duration != null 
          ? `${Math.floor(session.duration / 60)}:${String(Math.floor(session.duration % 60)).padStart(2, '0')}`
          : <span style={{ color: 'rgba(229, 231, 235, 0.3)', fontStyle: 'italic' }}>N/A</span>
        }
      </td>
      <td style={{ padding: '12px', textAlign: 'right', color: '#e5e7eb' }}>
        {session.fps != null 
          ? session.fps
          : <span style={{ color: 'rgba(229, 231, 235, 0.3)', fontStyle: 'italic' }}>N/A</span>
        }
      </td>
      <td style={{ padding: '12px', textAlign: 'center' }}>
        <span 
          onMouseEnter={handleMouseEnter}
          onMouseLeave={() => setMousePopupVisible(false)}
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: '4px',
            background: 'rgba(102, 126, 234, 0.2)',
            color: '#a5b4fc',
            padding: '4px 12px',
            borderRadius: '12px',
            fontSize: '12px',
            fontWeight: '600',
            whiteSpace: 'nowrap',
            cursor: session.mice && session.mice.length > 0 ? 'help' : 'default',
            transition: 'all 0.2s ease'
          }}
        >
          {session.mice_count} 🐭
        </span>
      </td>
      <td style={{ padding: '12px', color: '#e5e7eb', fontSize: '12px' }}>
        {session.arena_width != null && session.arena_height != null
          ? `${session.arena_width}×${session.arena_height}cm${session.arena_shape ? ` (${session.arena_shape})` : ''}`
          : <span style={{ color: 'rgba(229, 231, 235, 0.3)', fontStyle: 'italic' }}>N/A</span>
        }
      </td>
      <td style={{ padding: '12px', color: '#10b981', fontSize: '12px' }}>
        {session.tracking_method || <span style={{ color: 'rgba(229, 231, 235, 0.3)', fontStyle: 'italic' }}>N/A</span>}
      </td>
    </tr>
    </>
  )
})

function DatasetBrowserTable() {
  const navigate = useNavigate()
  const [allSessions, setAllSessions] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const [sortBy, setSortBy] = useState<string>('video_id')
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('asc')
  const [filters, setFilters] = useState({
    lab: '',
    minDuration: '',
    maxDuration: '',
    minMice: '',
    maxMice: '',
    minFps: '',
    maxFps: '',
    minArenaWidth: '',
    maxArenaWidth: '',
    minArenaHeight: '',
    maxArenaHeight: '',
    arenaShape: '',
    trackingMethod: ''
  })
  const [showFilters, setShowFilters] = useState(false)

  // Extract unique values for dropdown filters
  const uniqueValues = useMemo(() => {
    if (allSessions.length === 0) return { labs: [], arenaShapes: [], trackingMethods: [] }
    
    const labs = [...new Set(allSessions.map(s => s.lab_id).filter(Boolean))].sort()
    const arenaShapes = [...new Set(allSessions.map(s => s.arena_shape).filter(Boolean))].sort()
    const trackingMethods = [...new Set(allSessions.map(s => s.tracking_method).filter(Boolean))].sort()
    
    return { labs, arenaShapes, trackingMethods }
  }, [allSessions])

  // Load all sessions once
  useEffect(() => {
    setLoading(true)
    fetch('http://localhost:8000/api/analytics/browser')
      .then(res => res.json())
      .then(data => {
        setAllSessions(data.sessions)
        setLoading(false)
      })
      .catch(err => {
        console.error('Error fetching browser data:', err)
        setLoading(false)
      })
  }, [])

  // Apply filtering and sorting with useMemo for better performance
  const filteredSessions = useMemo(() => {
    if (allSessions.length === 0) return []

    const startTime = performance.now()
    let filtered = allSessions // Don't spread - avoid copying 8k+ items

    // Apply filters (skip null values in comparisons)
    // Exact match for dropdown selections
    if (filters.lab) {
      filtered = filtered.filter(s => s.lab_id === filters.lab)
    }
    if (filters.minDuration) {
      const minDur = parseFloat(filters.minDuration)
      filtered = filtered.filter(s => s.duration != null && s.duration >= minDur)
    }
    if (filters.maxDuration) {
      const maxDur = parseFloat(filters.maxDuration)
      filtered = filtered.filter(s => s.duration != null && s.duration <= maxDur)
    }
    if (filters.minMice) {
      const minM = parseInt(filters.minMice)
      filtered = filtered.filter(s => s.mice_count >= minM)
    }
    if (filters.maxMice) {
      const maxM = parseInt(filters.maxMice)
      filtered = filtered.filter(s => s.mice_count <= maxM)
    }
    if (filters.minFps) {
      const minF = parseFloat(filters.minFps)
      filtered = filtered.filter(s => s.fps != null && s.fps >= minF)
    }
    if (filters.maxFps) {
      const maxF = parseFloat(filters.maxFps)
      filtered = filtered.filter(s => s.fps != null && s.fps <= maxF)
    }
    if (filters.minArenaWidth) {
      const minW = parseFloat(filters.minArenaWidth)
      filtered = filtered.filter(s => s.arena_width != null && s.arena_width >= minW)
    }
    if (filters.maxArenaWidth) {
      const maxW = parseFloat(filters.maxArenaWidth)
      filtered = filtered.filter(s => s.arena_width != null && s.arena_width <= maxW)
    }
    if (filters.minArenaHeight) {
      const minH = parseFloat(filters.minArenaHeight)
      filtered = filtered.filter(s => s.arena_height != null && s.arena_height >= minH)
    }
    if (filters.maxArenaHeight) {
      const maxH = parseFloat(filters.maxArenaHeight)
      filtered = filtered.filter(s => s.arena_height != null && s.arena_height <= maxH)
    }
    if (filters.arenaShape) {
      filtered = filtered.filter(s => s.arena_shape === filters.arenaShape)
    }
    if (filters.trackingMethod) {
      filtered = filtered.filter(s => s.tracking_method === filters.trackingMethod)
    }

    // Apply sorting (handle null values by putting them at the end)
    // Create new array only when sorting
    const sorted = [...filtered]
    sorted.sort((a, b) => {
      let aVal = a[sortBy]
      let bVal = b[sortBy]
      
      // Put null values at the end
      if (aVal == null && bVal == null) return 0
      if (aVal == null) return 1
      if (bVal == null) return -1
      
      // Handle numeric values
      if (sortBy === 'duration' || sortBy === 'fps' || sortBy === 'mice_count' || 
          sortBy === 'arena_width' || sortBy === 'arena_height') {
        aVal = parseFloat(aVal)
        bVal = parseFloat(bVal)
      }
      
      if (aVal < bVal) return sortOrder === 'asc' ? -1 : 1
      if (aVal > bVal) return sortOrder === 'asc' ? 1 : -1
      return 0
    })

    const endTime = performance.now()
    console.log(`🔍 Filtered ${allSessions.length} → ${sorted.length} sessions in ${(endTime - startTime).toFixed(1)}ms`)

    return sorted
  }, [allSessions, sortBy, sortOrder, filters])

  // Handle session click - navigate to viewer
  const handleSessionClick = (session: any) => {
    // Navigate to viewer page with session info for auto-loading
    console.log('Opening session in viewer:', session.video_id, 'from lab:', session.lab_id)
    navigate('/viewer', { 
      state: { 
        loadSession: {
          video_id: session.video_id,
          lab_id: session.lab_id
        }
      } 
    })
  }

  // Handle column header click for sorting
  const handleSort = (column: string) => {
    if (sortBy === column) {
      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc')
    } else {
      setSortBy(column)
      setSortOrder('asc')
    }
  }

  // Reset filters and sorting
  const handleReset = () => {
    setSortBy('video_id')
    setSortOrder('asc')
    setFilters({
      lab: '',
      minDuration: '',
      maxDuration: '',
      minMice: '',
      maxMice: '',
      minFps: '',
      maxFps: '',
      minArenaWidth: '',
      maxArenaWidth: '',
      minArenaHeight: '',
      maxArenaHeight: '',
      arenaShape: '',
      trackingMethod: ''
    })
  }

  const SortIcon = ({ column }: { column: string }) => {
    if (sortBy !== column) return <span style={{ opacity: 0.3 }}>↕</span>
    return <span>{sortOrder === 'asc' ? '↑' : '↓'}</span>
  }

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: '40px', color: 'rgba(229, 231, 235, 0.5)' }}>
        <div className="spinner" style={{ margin: '0 auto 16px' }} />
        Loading complete dataset ({allSessions.length} sessions loaded so far)...
      </div>
    )
  }

  return (
    <div>
      {/* Filter Controls */}
      <div style={{ marginBottom: '16px', display: 'flex', gap: '12px', alignItems: 'center', flexWrap: 'wrap' }}>
        <button
          onClick={() => setShowFilters(!showFilters)}
          style={{
            padding: '8px 16px',
            background: showFilters ? 'rgba(102, 126, 234, 0.3)' : 'rgba(102, 126, 234, 0.15)',
            border: '1px solid rgba(102, 126, 234, 0.3)',
            borderRadius: '8px',
            color: '#e5e7eb',
            cursor: 'pointer',
            fontSize: '13px',
            fontWeight: '500',
            transition: 'all 0.2s ease',
            display: 'flex',
            alignItems: 'center',
            gap: '6px'
          }}
        >
          🔍 {showFilters ? 'Hide Filters' : 'Show Filters'}
        </button>

        <button
          onClick={handleReset}
          style={{
            padding: '8px 16px',
            background: 'rgba(239, 68, 68, 0.15)',
            border: '1px solid rgba(239, 68, 68, 0.3)',
            borderRadius: '8px',
            color: '#fca5a5',
            cursor: 'pointer',
            fontSize: '13px',
            fontWeight: '500',
            transition: 'all 0.2s ease',
            display: 'flex',
            alignItems: 'center',
            gap: '6px'
          }}
        >
          🔄 Reset
        </button>

        <div style={{ flex: 1 }} />

        <div style={{ color: 'rgba(229, 231, 235, 0.6)', fontSize: '13px' }}>
          Showing {filteredSessions.length.toLocaleString()} of {allSessions.length.toLocaleString()} sessions
        </div>
      </div>

      {/* Filter Panel */}
      {showFilters && (
        <div style={{
          background: 'rgba(102, 126, 234, 0.05)',
          border: '1px solid rgba(102, 126, 234, 0.2)',
          borderRadius: '12px',
          padding: '16px',
          marginBottom: '16px',
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
          gap: '12px',
          animation: 'fadeIn 0.3s ease'
        }}>
          <div>
            <label style={{ color: '#a5b4fc', fontSize: '12px', marginBottom: '4px', display: 'block' }}>
              Lab ID
            </label>
            <select
              value={filters.lab}
              onChange={(e) => setFilters({ ...filters, lab: e.target.value })}
              style={{
                width: '100%',
                padding: '8px 12px',
                background: 'rgba(15, 15, 30, 0.8)',
                border: '1px solid rgba(102, 126, 234, 0.3)',
                borderRadius: '6px',
                color: '#e5e7eb',
                fontSize: '13px',
                cursor: 'pointer'
              }}
            >
              <option value="">All labs</option>
              {uniqueValues.labs.map(lab => (
                <option key={lab} value={lab}>{lab}</option>
              ))}
            </select>
          </div>

          <div>
            <label style={{ color: '#a5b4fc', fontSize: '12px', marginBottom: '4px', display: 'block' }}>
              Min Duration (sec)
            </label>
            <input
              type="number"
              value={filters.minDuration}
              onChange={(e) => setFilters({ ...filters, minDuration: e.target.value })}
              placeholder="Min..."
              style={{
                width: '100%',
                padding: '8px 12px',
                background: 'rgba(15, 15, 30, 0.8)',
                border: '1px solid rgba(102, 126, 234, 0.3)',
                borderRadius: '6px',
                color: '#e5e7eb',
                fontSize: '13px'
              }}
            />
          </div>

          <div>
            <label style={{ color: '#a5b4fc', fontSize: '12px', marginBottom: '4px', display: 'block' }}>
              Max Duration (sec)
            </label>
            <input
              type="number"
              value={filters.maxDuration}
              onChange={(e) => setFilters({ ...filters, maxDuration: e.target.value })}
              placeholder="Max..."
              style={{
                width: '100%',
                padding: '8px 12px',
                background: 'rgba(15, 15, 30, 0.8)',
                border: '1px solid rgba(102, 126, 234, 0.3)',
                borderRadius: '6px',
                color: '#e5e7eb',
                fontSize: '13px'
              }}
            />
          </div>

          <div>
            <label style={{ color: '#a5b4fc', fontSize: '12px', marginBottom: '4px', display: 'block' }}>
              Min Mice
            </label>
            <input
              type="number"
              value={filters.minMice}
              onChange={(e) => setFilters({ ...filters, minMice: e.target.value })}
              placeholder="Min..."
              min="1"
              max="4"
              style={{
                width: '100%',
                padding: '8px 12px',
                background: 'rgba(15, 15, 30, 0.8)',
                border: '1px solid rgba(102, 126, 234, 0.3)',
                borderRadius: '6px',
                color: '#e5e7eb',
                fontSize: '13px'
              }}
            />
          </div>

          <div>
            <label style={{ color: '#a5b4fc', fontSize: '12px', marginBottom: '4px', display: 'block' }}>
              Max Mice
            </label>
            <input
              type="number"
              value={filters.maxMice}
              onChange={(e) => setFilters({ ...filters, maxMice: e.target.value })}
              placeholder="Max..."
              min="1"
              max="4"
              style={{
                width: '100%',
                padding: '8px 12px',
                background: 'rgba(15, 15, 30, 0.8)',
                border: '1px solid rgba(102, 126, 234, 0.3)',
                borderRadius: '6px',
                color: '#e5e7eb',
                fontSize: '13px'
              }}
            />
          </div>

          <div>
            <label style={{ color: '#a5b4fc', fontSize: '12px', marginBottom: '4px', display: 'block' }}>
              Min FPS
            </label>
            <input
              type="number"
              value={filters.minFps}
              onChange={(e) => setFilters({ ...filters, minFps: e.target.value })}
              placeholder="10-120"
              min="10"
              max="120"
              style={{
                width: '100%',
                padding: '8px 12px',
                background: 'rgba(15, 15, 30, 0.8)',
                border: '1px solid rgba(102, 126, 234, 0.3)',
                borderRadius: '6px',
                color: '#e5e7eb',
                fontSize: '13px'
              }}
            />
          </div>

          <div>
            <label style={{ color: '#a5b4fc', fontSize: '12px', marginBottom: '4px', display: 'block' }}>
              Max FPS
            </label>
            <input
              type="number"
              value={filters.maxFps}
              onChange={(e) => setFilters({ ...filters, maxFps: e.target.value })}
              placeholder="10-120"
              min="10"
              max="120"
              style={{
                width: '100%',
                padding: '8px 12px',
                background: 'rgba(15, 15, 30, 0.8)',
                border: '1px solid rgba(102, 126, 234, 0.3)',
                borderRadius: '6px',
                color: '#e5e7eb',
                fontSize: '13px'
              }}
            />
          </div>

          <div>
            <label style={{ color: '#a5b4fc', fontSize: '12px', marginBottom: '4px', display: 'block' }}>
              Min Arena Width (cm)
            </label>
            <input
              type="number"
              value={filters.minArenaWidth}
              onChange={(e) => setFilters({ ...filters, minArenaWidth: e.target.value })}
              placeholder="12.7-120"
              style={{
                width: '100%',
                padding: '8px 12px',
                background: 'rgba(15, 15, 30, 0.8)',
                border: '1px solid rgba(102, 126, 234, 0.3)',
                borderRadius: '6px',
                color: '#e5e7eb',
                fontSize: '13px'
              }}
            />
          </div>

          <div>
            <label style={{ color: '#a5b4fc', fontSize: '12px', marginBottom: '4px', display: 'block' }}>
              Max Arena Width (cm)
            </label>
            <input
              type="number"
              value={filters.maxArenaWidth}
              onChange={(e) => setFilters({ ...filters, maxArenaWidth: e.target.value })}
              placeholder="12.7-120"
              style={{
                width: '100%',
                padding: '8px 12px',
                background: 'rgba(15, 15, 30, 0.8)',
                border: '1px solid rgba(102, 126, 234, 0.3)',
                borderRadius: '6px',
                color: '#e5e7eb',
                fontSize: '13px'
              }}
            />
          </div>

          <div>
            <label style={{ color: '#a5b4fc', fontSize: '12px', marginBottom: '4px', display: 'block' }}>
              Min Arena Height (cm)
            </label>
            <input
              type="number"
              value={filters.minArenaHeight}
              onChange={(e) => setFilters({ ...filters, minArenaHeight: e.target.value })}
              placeholder="12-120"
              style={{
                width: '100%',
                padding: '8px 12px',
                background: 'rgba(15, 15, 30, 0.8)',
                border: '1px solid rgba(102, 126, 234, 0.3)',
                borderRadius: '6px',
                color: '#e5e7eb',
                fontSize: '13px'
              }}
            />
          </div>

          <div>
            <label style={{ color: '#a5b4fc', fontSize: '12px', marginBottom: '4px', display: 'block' }}>
              Max Arena Height (cm)
            </label>
            <input
              type="number"
              value={filters.maxArenaHeight}
              onChange={(e) => setFilters({ ...filters, maxArenaHeight: e.target.value })}
              placeholder="12-120"
              style={{
                width: '100%',
                padding: '8px 12px',
                background: 'rgba(15, 15, 30, 0.8)',
                border: '1px solid rgba(102, 126, 234, 0.3)',
                borderRadius: '6px',
                color: '#e5e7eb',
                fontSize: '13px'
              }}
            />
          </div>

          <div>
            <label style={{ color: '#a5b4fc', fontSize: '12px', marginBottom: '4px', display: 'block' }}>
              Arena Shape
            </label>
            <select
              value={filters.arenaShape}
              onChange={(e) => setFilters({ ...filters, arenaShape: e.target.value })}
              style={{
                width: '100%',
                padding: '8px 12px',
                background: 'rgba(15, 15, 30, 0.8)',
                border: '1px solid rgba(102, 126, 234, 0.3)',
                borderRadius: '6px',
                color: '#e5e7eb',
                fontSize: '13px',
                cursor: 'pointer'
              }}
            >
              <option value="">All shapes</option>
              {uniqueValues.arenaShapes.map(shape => (
                <option key={shape} value={shape}>{shape}</option>
              ))}
            </select>
          </div>

          <div>
            <label style={{ color: '#a5b4fc', fontSize: '12px', marginBottom: '4px', display: 'block' }}>
              Tracking Method
            </label>
            <select
              value={filters.trackingMethod}
              onChange={(e) => setFilters({ ...filters, trackingMethod: e.target.value })}
              style={{
                width: '100%',
                padding: '8px 12px',
                background: 'rgba(15, 15, 30, 0.8)',
                border: '1px solid rgba(102, 126, 234, 0.3)',
                borderRadius: '6px',
                color: '#e5e7eb',
                fontSize: '13px',
                cursor: 'pointer'
              }}
            >
              <option value="">All methods</option>
              {uniqueValues.trackingMethods.map(method => (
                <option key={method} value={method}>{method}</option>
              ))}
            </select>
          </div>
        </div>
      )}

      {/* Table Container with Fixed Height and Scrolling */}
      <div style={{
        maxHeight: '400px',
        overflowY: 'auto',
        overflowX: 'auto',
        border: '1px solid rgba(102, 126, 234, 0.2)',
        borderRadius: '12px',
        background: 'rgba(10, 10, 20, 0.3)'
      }}>
        <table style={{
          width: '100%',
          borderCollapse: 'collapse',
          fontSize: '13px'
        }}>
          <thead style={{ 
            position: 'sticky', 
            top: 0, 
            background: 'rgba(20, 20, 35, 0.98)',
            backdropFilter: 'blur(10px)',
            zIndex: 10,
            boxShadow: '0 2px 8px rgba(0, 0, 0, 0.3)'
          }}>
            <tr style={{ borderBottom: '2px solid rgba(102, 126, 234, 0.4)' }}>
              <th 
                onClick={() => handleSort('video_id')}
                style={{ 
                  padding: '14px 12px', 
                  textAlign: 'left', 
                  color: '#667eea', 
                  fontWeight: '600',
                  cursor: 'pointer',
                  userSelect: 'none',
                  transition: 'background 0.2s ease',
                  whiteSpace: 'nowrap'
                }}
                onMouseEnter={(e) => e.currentTarget.style.background = 'rgba(102, 126, 234, 0.15)'}
                onMouseLeave={(e) => e.currentTarget.style.background = 'transparent'}
              >
                Video ID <SortIcon column="video_id" />
              </th>
              <th 
                onClick={() => handleSort('lab_id')}
                style={{ 
                  padding: '14px 12px', 
                  textAlign: 'left', 
                  color: '#667eea', 
                  fontWeight: '600',
                  cursor: 'pointer',
                  userSelect: 'none',
                  transition: 'background 0.2s ease',
                  whiteSpace: 'nowrap'
                }}
                onMouseEnter={(e) => e.currentTarget.style.background = 'rgba(102, 126, 234, 0.15)'}
                onMouseLeave={(e) => e.currentTarget.style.background = 'transparent'}
              >
                Lab <SortIcon column="lab_id" />
              </th>
              <th 
                onClick={() => handleSort('duration')}
                style={{ 
                  padding: '14px 12px', 
                  textAlign: 'right', 
                  color: '#667eea', 
                  fontWeight: '600',
                  cursor: 'pointer',
                  userSelect: 'none',
                  transition: 'background 0.2s ease',
                  whiteSpace: 'nowrap'
                }}
                onMouseEnter={(e) => e.currentTarget.style.background = 'rgba(102, 126, 234, 0.15)'}
                onMouseLeave={(e) => e.currentTarget.style.background = 'transparent'}
              >
                Duration <SortIcon column="duration" />
              </th>
              <th 
                onClick={() => handleSort('fps')}
                style={{ 
                  padding: '14px 12px', 
                  textAlign: 'right', 
                  color: '#667eea', 
                  fontWeight: '600',
                  cursor: 'pointer',
                  userSelect: 'none',
                  transition: 'background 0.2s ease',
                  whiteSpace: 'nowrap'
                }}
                onMouseEnter={(e) => e.currentTarget.style.background = 'rgba(102, 126, 234, 0.15)'}
                onMouseLeave={(e) => e.currentTarget.style.background = 'transparent'}
              >
                FPS <SortIcon column="fps" />
              </th>
              <th 
                onClick={() => handleSort('mice_count')}
                style={{ 
                  padding: '14px 12px', 
                  textAlign: 'center', 
                  color: '#667eea', 
                  fontWeight: '600',
                  cursor: 'pointer',
                  userSelect: 'none',
                  transition: 'background 0.2s ease',
                  whiteSpace: 'nowrap'
                }}
                onMouseEnter={(e) => e.currentTarget.style.background = 'rgba(102, 126, 234, 0.15)'}
                onMouseLeave={(e) => e.currentTarget.style.background = 'transparent'}
              >
                Mice <SortIcon column="mice_count" />
              </th>
              <th style={{ padding: '14px 12px', textAlign: 'left', color: '#667eea', fontWeight: '600', whiteSpace: 'nowrap' }}>Arena</th>
              <th style={{ padding: '14px 12px', textAlign: 'left', color: '#667eea', fontWeight: '600', whiteSpace: 'nowrap' }}>Tracking</th>
            </tr>
          </thead>
          <tbody>
            {filteredSessions.map((session, idx) => (
              <SessionRow
                key={`${session.video_id}-${idx}`}
                session={session}
                onClick={handleSessionClick}
              />
            ))}
          </tbody>
        </table>
      </div>

      {/* Summary */}
      {filteredSessions.length === 0 && (
        <div style={{
          textAlign: 'center',
          padding: '30px',
          color: 'rgba(229, 231, 235, 0.5)',
          fontSize: '14px'
        }}>
          No sessions match the current filters
        </div>
      )}
    </div>
  )
}

function DatasetStatistics() {
  const [allSessions, setAllSessions] = useState<any[]>([])
  const [loading, setLoading] = useState(true)

  // Load all sessions (same data as browser)
  useEffect(() => {
    fetch('http://localhost:8000/api/analytics/browser')
      .then(res => res.json())
      .then(data => {
        setAllSessions(data.sessions)
        setLoading(false)
      })
      .catch(err => {
        console.error('Error fetching stats data:', err)
        setLoading(false)
      })
  }, [])

  const uniqueValues = useMemo(() => {
    if (allSessions.length === 0) return { labs: [], arenaShapes: [], trackingMethods: [] }
    
    const labs = [...new Set(allSessions.map(s => s.lab_id).filter(Boolean))].sort()
    const arenaShapes = [...new Set(allSessions.map(s => s.arena_shape).filter(Boolean))].sort()
    const trackingMethods = [...new Set(allSessions.map(s => s.tracking_method).filter(Boolean))].sort()
    
    return { labs, arenaShapes, trackingMethods }
  }, [allSessions])

  const datasetStats = useMemo(() => {
    if (allSessions.length === 0) return null

    // Collect all non-null values for calculations
    const durations = allSessions.map(s => s.duration).filter((d): d is number => d != null)
    const fpsValues = allSessions.map(s => s.fps).filter((f): f is number => f != null)
    const miceCountValues = allSessions.map(s => s.mice_count)
    const arenaWidths = allSessions.map(s => s.arena_width).filter((w): w is number => w != null)
    const arenaHeights = allSessions.map(s => s.arena_height).filter((h): h is number => h != null)

    const calcStats = (values: number[]) => {
      if (values.length === 0) return { min: 0, max: 0, mean: 0, median: 0 }
      const sorted = [...values].sort((a, b) => a - b)
      const sum = values.reduce((a, b) => a + b, 0)
      const mean = sum / values.length
      const median = sorted.length % 2 === 0
        ? (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2
        : sorted[Math.floor(sorted.length / 2)]
      return {
        min: sorted[0],
        max: sorted[sorted.length - 1],
        mean,
        median
      }
    }

    return {
      totalSessions: allSessions.length,
      uniqueLabs: uniqueValues.labs.length,
      uniqueArenaShapes: uniqueValues.arenaShapes.length,
      uniqueTrackingMethods: uniqueValues.trackingMethods.length,
      duration: calcStats(durations),
      fps: calcStats(fpsValues),
      miceCount: calcStats(miceCountValues),
      arenaWidth: calcStats(arenaWidths),
      arenaHeight: calcStats(arenaHeights),
      sessionsWithDuration: durations.length,
      sessionsWithFps: fpsValues.length,
      sessionsWithArenaDims: arenaWidths.length
    }
  }, [allSessions, uniqueValues])

  if (loading) {
    return (
      <div style={{
        marginTop: '30px',
        padding: '20px',
        textAlign: 'center',
        color: 'rgba(229, 231, 235, 0.5)',
        fontSize: '14px'
      }}>
        Loading statistics...
      </div>
    )
  }

  if (!datasetStats || allSessions.length === 0) {
    return null
  }

  const formatStat = (value: number, decimals: number = 1) => {
    return value.toFixed(decimals)
  }

  const StatCard = ({ 
    title, 
    value, 
    color, 
    icon 
  }: { 
    title: string
    value: string | number
    color: string
    icon: string
  }) => (
    <div style={{
      background: 'linear-gradient(135deg, rgba(30, 30, 50, 0.6) 0%, rgba(20, 20, 35, 0.6) 100%)',
      backdropFilter: 'blur(10px)',
      border: `1px solid ${color}40`,
      borderRadius: '12px',
      padding: '16px 20px',
      display: 'flex',
      flexDirection: 'column',
      gap: '8px',
      transition: 'all 0.2s ease',
      cursor: 'default'
    }}
    onMouseEnter={(e) => {
      e.currentTarget.style.borderColor = `${color}80`
      e.currentTarget.style.transform = 'translateY(-2px)'
      e.currentTarget.style.boxShadow = `0 4px 20px ${color}20`
    }}
    onMouseLeave={(e) => {
      e.currentTarget.style.borderColor = `${color}40`
      e.currentTarget.style.transform = 'translateY(0)'
      e.currentTarget.style.boxShadow = 'none'
    }}
    >
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: '8px',
        fontSize: '12px',
        fontWeight: '600',
        color: 'rgba(229, 231, 235, 0.6)',
        textTransform: 'uppercase',
        letterSpacing: '0.5px'
      }}>
        <span>{icon}</span>
        <span>{title}</span>
      </div>
      <div style={{
        fontSize: '24px',
        fontWeight: '700',
        color,
        fontFamily: "'Poppins', sans-serif"
      }}>
        {value}
      </div>
    </div>
  )

  const RangeCard = ({
    title,
    stats,
    unit,
    color,
    icon
  }: {
    title: string
    stats: { min: number, max: number, mean: number, median: number }
    unit: string
    color: string
    icon: string
  }) => (
    <div style={{
      background: 'linear-gradient(135deg, rgba(30, 30, 50, 0.6) 0%, rgba(20, 20, 35, 0.6) 100%)',
      backdropFilter: 'blur(10px)',
      border: `1px solid ${color}40`,
      borderRadius: '12px',
      padding: '16px 20px',
      transition: 'all 0.2s ease'
    }}
    onMouseEnter={(e) => {
      e.currentTarget.style.borderColor = `${color}80`
      e.currentTarget.style.transform = 'translateY(-2px)'
      e.currentTarget.style.boxShadow = `0 4px 20px ${color}20`
    }}
    onMouseLeave={(e) => {
      e.currentTarget.style.borderColor = `${color}40`
      e.currentTarget.style.transform = 'translateY(0)'
      e.currentTarget.style.boxShadow = 'none'
    }}
    >
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: '8px',
        fontSize: '12px',
        fontWeight: '600',
        color: 'rgba(229, 231, 235, 0.6)',
        textTransform: 'uppercase',
        letterSpacing: '0.5px',
        marginBottom: '12px'
      }}>
        <span>{icon}</span>
        <span>{title}</span>
      </div>
      <div style={{
        display: 'grid',
        gridTemplateColumns: '1fr 1fr',
        gap: '8px',
        fontSize: '13px'
      }}>
        <div>
          <div style={{ color: 'rgba(229, 231, 235, 0.5)', fontSize: '11px', marginBottom: '4px' }}>Min</div>
          <div style={{ color, fontWeight: '600', fontSize: '16px' }}>{formatStat(stats.min, 1)}{unit}</div>
        </div>
        <div>
          <div style={{ color: 'rgba(229, 231, 235, 0.5)', fontSize: '11px', marginBottom: '4px' }}>Max</div>
          <div style={{ color, fontWeight: '600', fontSize: '16px' }}>{formatStat(stats.max, 1)}{unit}</div>
        </div>
        <div>
          <div style={{ color: 'rgba(229, 231, 235, 0.5)', fontSize: '11px', marginBottom: '4px' }}>Mean</div>
          <div style={{ color, fontWeight: '600', fontSize: '16px' }}>{formatStat(stats.mean, 1)}{unit}</div>
        </div>
        <div>
          <div style={{ color: 'rgba(229, 231, 235, 0.5)', fontSize: '11px', marginBottom: '4px' }}>Median</div>
          <div style={{ color, fontWeight: '600', fontSize: '16px' }}>{formatStat(stats.median, 1)}{unit}</div>
        </div>
      </div>
    </div>
  )

  return (
    <div style={{
      marginTop: '30px',
      padding: '0 20px 20px'
    }}>
      <h3 style={{
        fontSize: '18px',
        fontWeight: '700',
        color: '#667eea',
        marginBottom: '20px',
        display: 'flex',
        alignItems: 'center',
        gap: '8px'
      }}>
        📊 Dataset Statistics
      </h3>

      {/* Unique Values Section */}
      <div style={{
        marginBottom: '24px'
      }}>
        <h4 style={{
          fontSize: '14px',
          fontWeight: '600',
          color: 'rgba(229, 231, 235, 0.7)',
          marginBottom: '12px',
          textTransform: 'uppercase',
          letterSpacing: '0.5px'
        }}>
          Unique Values
        </h4>
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
          gap: '12px'
        }}>
          <StatCard title="Total Sessions" value={datasetStats.totalSessions.toLocaleString()} color="#667eea" icon="📁" />
          <StatCard title="Unique Labs" value={datasetStats.uniqueLabs} color="#10b981" icon="🧪" />
          <StatCard title="Arena Shapes" value={datasetStats.uniqueArenaShapes} color="#f59e0b" icon="⬛" />
          <StatCard title="Tracking Methods" value={datasetStats.uniqueTrackingMethods} color="#ec4899" icon="📡" />
        </div>
      </div>

      {/* Statistical Ranges Section */}
      <div>
        <h4 style={{
          fontSize: '14px',
          fontWeight: '600',
          color: 'rgba(229, 231, 235, 0.7)',
          marginBottom: '12px',
          textTransform: 'uppercase',
          letterSpacing: '0.5px'
        }}>
          Metric Distributions
        </h4>
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))',
          gap: '12px'
        }}>
          <RangeCard 
            title="Duration" 
            stats={datasetStats.duration} 
            unit="s" 
            color="#60a5fa" 
            icon="⏱️"
          />
          <RangeCard 
            title="FPS" 
            stats={datasetStats.fps} 
            unit="" 
            color="#34d399" 
            icon="🎞️"
          />
          <RangeCard 
            title="Mice Count" 
            stats={datasetStats.miceCount} 
            unit=" 🐭" 
            color="#a78bfa" 
            icon="🐭"
          />
          <RangeCard 
            title="Arena Width" 
            stats={datasetStats.arenaWidth} 
            unit="cm" 
            color="#fbbf24" 
            icon="↔️"
          />
          <RangeCard 
            title="Arena Height" 
            stats={datasetStats.arenaHeight} 
            unit="cm" 
            color="#fb923c" 
            icon="↕️"
          />
        </div>
      </div>

      {/* Data Completeness */}
      <div style={{
        marginTop: '24px',
        padding: '16px',
        background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%)',
        border: '1px solid rgba(99, 102, 241, 0.3)',
        borderRadius: '12px'
      }}>
        <h4 style={{
          fontSize: '13px',
          fontWeight: '600',
          color: '#a5b4fc',
          marginBottom: '10px',
          textTransform: 'uppercase',
          letterSpacing: '0.5px'
        }}>
          📈 Data Completeness
        </h4>
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
          gap: '12px',
          fontSize: '13px',
          color: '#e5e7eb'
        }}>
          <div>
            <span style={{ color: 'rgba(229, 231, 235, 0.6)' }}>With Duration:</span>{' '}
            <span style={{ fontWeight: '600', color: '#60a5fa' }}>
              {datasetStats.sessionsWithDuration} ({((datasetStats.sessionsWithDuration / datasetStats.totalSessions) * 100).toFixed(1)}%)
            </span>
          </div>
          <div>
            <span style={{ color: 'rgba(229, 231, 235, 0.6)' }}>With FPS:</span>{' '}
            <span style={{ fontWeight: '600', color: '#34d399' }}>
              {datasetStats.sessionsWithFps} ({((datasetStats.sessionsWithFps / datasetStats.totalSessions) * 100).toFixed(1)}%)
            </span>
          </div>
          <div>
            <span style={{ color: 'rgba(229, 231, 235, 0.6)' }}>With Arena Dims:</span>{' '}
            <span style={{ fontWeight: '600', color: '#fbbf24' }}>
              {datasetStats.sessionsWithArenaDims} ({((datasetStats.sessionsWithArenaDims / datasetStats.totalSessions) * 100).toFixed(1)}%)
            </span>
          </div>
        </div>
      </div>

      {/* Unique Labs List (Collapsible) */}
      <details style={{
        marginTop: '20px',
        padding: '16px',
        background: 'linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.1) 100%)',
        border: '1px solid rgba(16, 185, 129, 0.3)',
        borderRadius: '12px',
        cursor: 'pointer'
      }}>
        <summary style={{
          fontSize: '13px',
          fontWeight: '600',
          color: '#10b981',
          textTransform: 'uppercase',
          letterSpacing: '0.5px',
          userSelect: 'none'
        }}>
          🧪 All Labs ({uniqueValues.labs.length})
        </summary>
        <div style={{
          marginTop: '12px',
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))',
          gap: '8px',
          fontSize: '12px'
        }}>
          {uniqueValues.labs.map((lab: string) => (
            <div key={lab} style={{
              padding: '8px 12px',
              background: 'rgba(16, 185, 129, 0.15)',
              borderRadius: '6px',
              color: '#6ee7b7',
              fontWeight: '500'
            }}>
              {lab}
            </div>
          ))}
        </div>
      </details>
    </div>
  )
}

function HeatmapPanel() {
  // State for all heatmap data and filtering
  const [allPoints, setAllPoints] = useState<Array<{
    x: number
    y: number
    video_id: string
    lab_id: string
    mice_count: number
    arena_shape: string
    tracking_method: string
  }>>([])
  const [loading, setLoading] = useState(true)
  const [totalPoints, setTotalPoints] = useState(0)
  
  // Filter states (matching DatasetBrowser)
  const [selectedLabs, setSelectedLabs] = useState<string[]>([])
  const [selectedArenaShapes, setSelectedArenaShapes] = useState<string[]>([])
  const [selectedMiceCounts, setSelectedMiceCounts] = useState<string[]>([])
  const [selectedTrackingMethods, setSelectedTrackingMethods] = useState<string[]>([])
  
  // Unique values for filters
  const [uniqueLabs, setUniqueLabs] = useState<string[]>([])
  const [uniqueArenaShapes, setUniqueArenaShapes] = useState<string[]>([])
  const [uniqueMiceCounts, setUniqueMiceCounts] = useState<number[]>([])
  const [uniqueTrackingMethods, setUniqueTrackingMethods] = useState<string[]>([])

  // Load all heatmap data once
  useEffect(() => {
    fetch('http://localhost:8000/api/analytics/heatmap?target_points=500000')
      .then(res => res.json())
      .then(data => {
        setAllPoints(data.points)
        setTotalPoints(data.total_points)
        
        // Extract unique values for filters
        const labs = new Set<string>()
        const arenaShapes = new Set<string>()
        const miceCounts = new Set<number>()
        const trackingMethods = new Set<string>()
        
        data.points.forEach((point: any) => {
          labs.add(point.lab_id)
          arenaShapes.add(point.arena_shape)
          miceCounts.add(point.mice_count)
          trackingMethods.add(point.tracking_method)
        })
        
        setUniqueLabs(Array.from(labs).sort())
        setUniqueArenaShapes(Array.from(arenaShapes).sort())
        setUniqueMiceCounts(Array.from(miceCounts).sort((a, b) => a - b))
        setUniqueTrackingMethods(Array.from(trackingMethods).sort())
        
        setLoading(false)
      })
      .catch(err => {
        console.error('Error fetching heatmap:', err)
        setLoading(false)
      })
  }, [])

  // Client-side filtering with useMemo
  const filteredPoints = useMemo(() => {
    if (!allPoints.length) return []
    
    return allPoints.filter(point => {
      if (selectedLabs.length > 0 && !selectedLabs.includes(point.lab_id)) return false
      if (selectedArenaShapes.length > 0 && !selectedArenaShapes.includes(point.arena_shape)) return false
      if (selectedMiceCounts.length > 0 && !selectedMiceCounts.includes(String(point.mice_count))) return false
      if (selectedTrackingMethods.length > 0 && !selectedTrackingMethods.includes(point.tracking_method)) return false
      return true
    })
  }, [allPoints, selectedLabs, selectedArenaShapes, selectedMiceCounts, selectedTrackingMethods])

  // Transform to deck.gl format
  const deckglData = useMemo(() => {
    return filteredPoints.map(point => ({
      position: [point.x, point.y],
      weight: 1.0
    }))
  }, [filteredPoints])

  // HexagonLayer for better performance with large datasets
  const layers = [
    new HexagonLayer({
      id: 'hexagon-layer',
      data: deckglData,
      getPosition: (d: any) => d.position,
      getColorWeight: (d: any) => d.weight,
      getElevationWeight: (d: any) => d.weight,
      elevationScale: 0.02,  // Much smaller for 0-1 coordinate space
      extruded: true,
      radius: 0.02,  // Smaller radius for normalized coordinates
      coverage: 0.9,
      upperPercentile: 100,
      colorRange: [
        [139, 92, 246, 100],  // Purple theme
        [147, 100, 250, 140],
        [156, 108, 254, 180],
        [165, 116, 255, 220],
        [174, 124, 255, 255],
        [183, 132, 255, 255]
      ]
    })
  ]

  // Reset all filters
  const resetFilters = () => {
    setSelectedLabs([])
    setSelectedArenaShapes([])
    setSelectedMiceCounts([])
    setSelectedTrackingMethods([])
  }

  const hasFilters = selectedLabs.length > 0 || selectedArenaShapes.length > 0 || 
                     selectedMiceCounts.length > 0 || selectedTrackingMethods.length > 0

  return (
    <div style={{
      width: '100%',
      height: 'calc(100vh - 200px)',
      display: 'flex',
      flexDirection: 'column',
      gap: '16px'
    }}>
      {/* Header */}
      <div style={{
        background: 'linear-gradient(135deg, rgba(20, 20, 35, 0.9) 0%, rgba(15, 15, 30, 0.9) 100%)',
        backdropFilter: 'blur(10px)',
        border: '1px solid rgba(139, 92, 246, 0.3)',
        borderRadius: '16px',
        padding: '20px 24px',
        display: 'flex',
        alignItems: 'center',
        gap: '12px'
      }}>
        <div style={{ fontSize: '32px' }}>🗺️</div>
        <div style={{ flex: 1 }}>
          <h2 style={{ color: 'rgba(139, 92, 246, 1)', fontSize: '20px', margin: 0, marginBottom: '4px' }}>
            Spatial Heatmap
            {loading && <span style={{ fontSize: '14px', marginLeft: '12px', color: 'rgba(229, 231, 235, 0.5)' }}>Loading...</span>}
          </h2>
          <p style={{ color: 'rgba(229, 231, 235, 0.6)', fontSize: '13px', margin: 0 }}>
            {loading 
              ? 'Fetching normalized position data from entire dataset...'
              : `Showing ${filteredPoints.length.toLocaleString()} of ${totalPoints.toLocaleString()} positions ${hasFilters ? '(filtered)' : ''}`
            }
          </p>
        </div>
        {hasFilters && (
          <button
            onClick={resetFilters}
            style={{
              padding: '8px 16px',
              background: 'rgba(139, 92, 246, 0.2)',
              border: '1px solid rgba(139, 92, 246, 0.4)',
              borderRadius: '8px',
              color: 'rgba(139, 92, 246, 1)',
              fontSize: '13px',
              cursor: 'pointer',
              transition: 'all 0.2s ease'
            }}
            onMouseEnter={e => {
              e.currentTarget.style.background = 'rgba(139, 92, 246, 0.3)'
              e.currentTarget.style.borderColor = 'rgba(139, 92, 246, 0.6)'
            }}
            onMouseLeave={e => {
              e.currentTarget.style.background = 'rgba(139, 92, 246, 0.2)'
              e.currentTarget.style.borderColor = 'rgba(139, 92, 246, 0.4)'
            }}
          >
            Reset Filters
          </button>
        )}
      </div>

      {/* Filter Controls */}
      <div style={{
        background: 'linear-gradient(135deg, rgba(20, 20, 35, 0.9) 0%, rgba(15, 15, 30, 0.9) 100%)',
        backdropFilter: 'blur(10px)',
        border: '1px solid rgba(139, 92, 246, 0.2)',
        borderRadius: '16px',
        padding: '20px 24px',
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
        gap: '16px'
      }}>
        {/* Lab Filter */}
        <div>
          <label style={{ color: 'rgba(229, 231, 235, 0.8)', fontSize: '13px', display: 'block', marginBottom: '8px' }}>
            Lab ID
          </label>
          <select
            multiple
            value={selectedLabs}
            onChange={e => setSelectedLabs(Array.from(e.target.selectedOptions, option => option.value))}
            style={{
              width: '100%',
              padding: '8px 12px',
              background: 'rgba(30, 30, 45, 0.8)',
              border: '1px solid rgba(139, 92, 246, 0.3)',
              borderRadius: '8px',
              color: '#e5e7eb',
              fontSize: '13px',
              minHeight: '80px'
            }}
          >
            {uniqueLabs.map(lab => (
              <option key={lab} value={lab}>{lab}</option>
            ))}
          </select>
        </div>

        {/* Arena Shape Filter */}
        <div>
          <label style={{ color: 'rgba(229, 231, 235, 0.8)', fontSize: '13px', display: 'block', marginBottom: '8px' }}>
            Arena Shape
          </label>
          <select
            multiple
            value={selectedArenaShapes}
            onChange={e => setSelectedArenaShapes(Array.from(e.target.selectedOptions, option => option.value))}
            style={{
              width: '100%',
              padding: '8px 12px',
              background: 'rgba(30, 30, 45, 0.8)',
              border: '1px solid rgba(139, 92, 246, 0.3)',
              borderRadius: '8px',
              color: '#e5e7eb',
              fontSize: '13px',
              minHeight: '80px'
            }}
          >
            {uniqueArenaShapes.map(shape => (
              <option key={shape} value={shape}>{shape}</option>
            ))}
          </select>
        </div>

        {/* Mice Count Filter */}
        <div>
          <label style={{ color: 'rgba(229, 231, 235, 0.8)', fontSize: '13px', display: 'block', marginBottom: '8px' }}>
            Mice Count
          </label>
          <select
            multiple
            value={selectedMiceCounts}
            onChange={e => setSelectedMiceCounts(Array.from(e.target.selectedOptions, option => option.value))}
            style={{
              width: '100%',
              padding: '8px 12px',
              background: 'rgba(30, 30, 45, 0.8)',
              border: '1px solid rgba(139, 92, 246, 0.3)',
              borderRadius: '8px',
              color: '#e5e7eb',
              fontSize: '13px',
              minHeight: '80px'
            }}
          >
            {uniqueMiceCounts.map(count => (
              <option key={count} value={String(count)}>{count} {count === 1 ? 'mouse' : 'mice'}</option>
            ))}
          </select>
        </div>

        {/* Tracking Method Filter */}
        <div>
          <label style={{ color: 'rgba(229, 231, 235, 0.8)', fontSize: '13px', display: 'block', marginBottom: '8px' }}>
            Tracking Method
          </label>
          <select
            multiple
            value={selectedTrackingMethods}
            onChange={e => setSelectedTrackingMethods(Array.from(e.target.selectedOptions, option => option.value))}
            style={{
              width: '100%',
              padding: '8px 12px',
              background: 'rgba(30, 30, 45, 0.8)',
              border: '1px solid rgba(139, 92, 246, 0.3)',
              borderRadius: '8px',
              color: '#e5e7eb',
              fontSize: '13px',
              minHeight: '80px'
            }}
          >
            {uniqueTrackingMethods.map(method => (
              <option key={method} value={method}>{method}</option>
            ))}
          </select>
        </div>
      </div>

      {/* Heatmap Visualization */}
      <div style={{
        flex: 1,
        background: 'linear-gradient(135deg, rgba(20, 20, 35, 0.9) 0%, rgba(15, 15, 30, 0.9) 100%)',
        backdropFilter: 'blur(10px)',
        border: '1px solid rgba(139, 92, 246, 0.2)',
        borderRadius: '16px',
        overflow: 'hidden',
        position: 'relative'
      }}>
        {loading ? (
          <div style={{
            position: 'absolute',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            textAlign: 'center'
          }}>
            <div style={{
              width: '60px',
              height: '60px',
              border: '4px solid rgba(139, 92, 246, 0.2)',
              borderTop: '4px solid rgba(139, 92, 246, 1)',
              borderRadius: '50%',
              animation: 'spin 1s linear infinite',
              margin: '0 auto 16px'
            }} />
            <p style={{ color: 'rgba(229, 231, 235, 0.6)', fontSize: '14px' }}>
              Loading spatial heatmap data...
            </p>
          </div>
        ) : (
          <DeckGL
            views={[new OrthographicView({ id: 'ortho' })]}
            initialViewState={{
              ortho: {
                target: [0.5, 0.5, 0],
                zoom: 0
              }
            }}
            controller={true}
            layers={layers}
            style={{ position: 'absolute', width: '100%', height: '100%' }}
          />
        )}
      </div>
    </div>
  )
}

function ActivityPanel() {
  // State for all activity data and filtering
  const [allActivityData, setAllActivityData] = useState<Array<{
    normalized_time: number
    distance: number
    velocity: number
    lab_id: string
    video_id: string
    mice_count: number
    arena_shape: string
    tracking_method: string
  }>>([])
  const [loading, setLoading] = useState(true)
  const [totalPoints, setTotalPoints] = useState(0)
  const [sessionsProcessed, setSessionsProcessed] = useState(0)
  const [bins, setBins] = useState(100)
  
  // Filter states (matching heatmap)
  const [selectedLabs, setSelectedLabs] = useState<string[]>([])
  const [selectedArenaShapes, setSelectedArenaShapes] = useState<string[]>([])
  const [selectedMiceCounts, setSelectedMiceCounts] = useState<string[]>([])
  const [selectedTrackingMethods, setSelectedTrackingMethods] = useState<string[]>([])
  
  // Unique values for filters
  const [uniqueLabs, setUniqueLabs] = useState<string[]>([])
  const [uniqueArenaShapes, setUniqueArenaShapes] = useState<string[]>([])
  const [uniqueMiceCounts, setUniqueMiceCounts] = useState<number[]>([])
  const [uniqueTrackingMethods, setUniqueTrackingMethods] = useState<string[]>([])

  // Load all activity data once
  useEffect(() => {
    fetch('http://localhost:8000/api/analytics/activity?target_sessions=100&bins=100')
      .then(res => res.json())
      .then(data => {
        setAllActivityData(data.activity_data)
        setTotalPoints(data.total_points)
        setSessionsProcessed(data.sessions_processed)
        setBins(data.bins)
        
        // Extract unique values from response
        if (data.unique_values) {
          setUniqueLabs(data.unique_values.labs || [])
          setUniqueArenaShapes(data.unique_values.arena_shapes || [])
          setUniqueMiceCounts(data.unique_values.mice_counts || [1, 2, 3, 4])
          setUniqueTrackingMethods(data.unique_values.tracking_methods || [])
        }
        
        setLoading(false)
      })
      .catch(err => {
        console.error('Error fetching activity:', err)
        setLoading(false)
      })
  }, [])

  // Client-side filtering with useMemo
  const filteredData = useMemo(() => {
    if (!allActivityData.length) return []
    
    return allActivityData.filter(point => {
      if (selectedLabs.length > 0 && !selectedLabs.includes(point.lab_id)) return false
      if (selectedArenaShapes.length > 0 && !selectedArenaShapes.includes(point.arena_shape)) return false
      if (selectedMiceCounts.length > 0 && !selectedMiceCounts.includes(String(point.mice_count))) return false
      if (selectedTrackingMethods.length > 0 && !selectedTrackingMethods.includes(point.tracking_method)) return false
      return true
    })
  }, [allActivityData, selectedLabs, selectedArenaShapes, selectedMiceCounts, selectedTrackingMethods])

  // Aggregate filtered data by time bins for charting
  const chartData = useMemo(() => {
    if (!filteredData.length) return []
    
    // Group by normalized time and calculate averages
    const binMap = new Map<number, { distances: number[], velocities: number[] }>()
    
    filteredData.forEach(point => {
      const binKey = Math.round(point.normalized_time * (bins - 1)) / (bins - 1)
      if (!binMap.has(binKey)) {
        binMap.set(binKey, { distances: [], velocities: [] })
      }
      const bin = binMap.get(binKey)!
      bin.distances.push(point.distance)
      bin.velocities.push(point.velocity)
    })
    
    // Convert to array and calculate averages
    return Array.from(binMap.entries())
      .sort((a, b) => a[0] - b[0])
      .map(([time, data]) => ({
        time: `${Math.round(time * 100)}%`,
        normalized_time: time,
        distance: data.distances.reduce((a, b) => a + b, 0) / data.distances.length,
        velocity: data.velocities.reduce((a, b) => a + b, 0) / data.velocities.length,
        count: data.distances.length
      }))
  }, [filteredData, bins])

  // Reset all filters
  const resetFilters = () => {
    setSelectedLabs([])
    setSelectedArenaShapes([])
    setSelectedMiceCounts([])
    setSelectedTrackingMethods([])
  }

  const hasFilters = selectedLabs.length > 0 || selectedArenaShapes.length > 0 || 
                     selectedMiceCounts.length > 0 || selectedTrackingMethods.length > 0

  return (
    <div style={{
      width: '100%',
      height: 'calc(100vh - 200px)',
      display: 'flex',
      flexDirection: 'column',
      gap: '16px'
    }}>
      {/* Header */}
      <div style={{
        background: 'linear-gradient(135deg, rgba(20, 20, 35, 0.9) 0%, rgba(15, 15, 30, 0.9) 100%)',
        backdropFilter: 'blur(10px)',
        border: '1px solid rgba(139, 92, 246, 0.3)',
        borderRadius: '16px',
        padding: '20px 24px',
        display: 'flex',
        alignItems: 'center',
        gap: '12px'
      }}>
        <div style={{ fontSize: '32px' }}>📈</div>
        <div style={{ flex: 1 }}>
          <h2 style={{ color: 'rgba(139, 92, 246, 1)', fontSize: '20px', margin: 0, marginBottom: '4px' }}>
            Activity Timeline
            {loading && <span style={{ fontSize: '14px', marginLeft: '12px', color: 'rgba(229, 231, 235, 0.5)' }}>Loading...</span>}
          </h2>
          <p style={{ color: 'rgba(229, 231, 235, 0.6)', fontSize: '13px', margin: 0 }}>
            {loading 
              ? 'Analyzing movement patterns across entire dataset...'
              : `Showing ${filteredData.length.toLocaleString()} of ${totalPoints.toLocaleString()} data points from ${sessionsProcessed} sessions ${hasFilters ? '(filtered)' : ''}`
            }
          </p>
        </div>
        {hasFilters && (
          <button
            onClick={resetFilters}
            style={{
              padding: '8px 16px',
              background: 'rgba(139, 92, 246, 0.2)',
              border: '1px solid rgba(139, 92, 246, 0.4)',
              borderRadius: '8px',
              color: 'rgba(139, 92, 246, 1)',
              fontSize: '13px',
              cursor: 'pointer',
              transition: 'all 0.2s ease'
            }}
            onMouseEnter={e => {
              e.currentTarget.style.background = 'rgba(139, 92, 246, 0.3)'
              e.currentTarget.style.borderColor = 'rgba(139, 92, 246, 0.6)'
            }}
            onMouseLeave={e => {
              e.currentTarget.style.background = 'rgba(139, 92, 246, 0.2)'
              e.currentTarget.style.borderColor = 'rgba(139, 92, 246, 0.4)'
            }}
          >
            Reset Filters
          </button>
        )}
      </div>

      {/* Filter Controls */}
      <div style={{
        background: 'linear-gradient(135deg, rgba(20, 20, 35, 0.9) 0%, rgba(15, 15, 30, 0.9) 100%)',
        backdropFilter: 'blur(10px)',
        border: '1px solid rgba(139, 92, 246, 0.2)',
        borderRadius: '16px',
        padding: '20px 24px',
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
        gap: '16px'
      }}>
        {/* Lab Filter */}
        <div>
          <label style={{ color: 'rgba(229, 231, 235, 0.8)', fontSize: '13px', display: 'block', marginBottom: '8px' }}>
            Lab ID
          </label>
          <select
            multiple
            value={selectedLabs}
            onChange={e => setSelectedLabs(Array.from(e.target.selectedOptions, option => option.value))}
            style={{
              width: '100%',
              padding: '8px 12px',
              background: 'rgba(30, 30, 45, 0.8)',
              border: '1px solid rgba(139, 92, 246, 0.3)',
              borderRadius: '8px',
              color: '#e5e7eb',
              fontSize: '13px',
              minHeight: '80px'
            }}
          >
            {uniqueLabs.map(lab => (
              <option key={lab} value={lab}>{lab}</option>
            ))}
          </select>
        </div>

        {/* Arena Shape Filter */}
        <div>
          <label style={{ color: 'rgba(229, 231, 235, 0.8)', fontSize: '13px', display: 'block', marginBottom: '8px' }}>
            Arena Shape
          </label>
          <select
            multiple
            value={selectedArenaShapes}
            onChange={e => setSelectedArenaShapes(Array.from(e.target.selectedOptions, option => option.value))}
            style={{
              width: '100%',
              padding: '8px 12px',
              background: 'rgba(30, 30, 45, 0.8)',
              border: '1px solid rgba(139, 92, 246, 0.3)',
              borderRadius: '8px',
              color: '#e5e7eb',
              fontSize: '13px',
              minHeight: '80px'
            }}
          >
            {uniqueArenaShapes.map(shape => (
              <option key={shape} value={shape}>{shape}</option>
            ))}
          </select>
        </div>

        {/* Mice Count Filter */}
        <div>
          <label style={{ color: 'rgba(229, 231, 235, 0.8)', fontSize: '13px', display: 'block', marginBottom: '8px' }}>
            Mice Count
          </label>
          <select
            multiple
            value={selectedMiceCounts}
            onChange={e => setSelectedMiceCounts(Array.from(e.target.selectedOptions, option => option.value))}
            style={{
              width: '100%',
              padding: '8px 12px',
              background: 'rgba(30, 30, 45, 0.8)',
              border: '1px solid rgba(139, 92, 246, 0.3)',
              borderRadius: '8px',
              color: '#e5e7eb',
              fontSize: '13px',
              minHeight: '80px'
            }}
          >
            {uniqueMiceCounts.map(count => (
              <option key={count} value={String(count)}>{count} {count === 1 ? 'mouse' : 'mice'}</option>
            ))}
          </select>
        </div>

        {/* Tracking Method Filter */}
        <div>
          <label style={{ color: 'rgba(229, 231, 235, 0.8)', fontSize: '13px', display: 'block', marginBottom: '8px' }}>
            Tracking Method
          </label>
          <select
            multiple
            value={selectedTrackingMethods}
            onChange={e => setSelectedTrackingMethods(Array.from(e.target.selectedOptions, option => option.value))}
            style={{
              width: '100%',
              padding: '8px 12px',
              background: 'rgba(30, 30, 45, 0.8)',
              border: '1px solid rgba(139, 92, 246, 0.3)',
              borderRadius: '8px',
              color: '#e5e7eb',
              fontSize: '13px',
              minHeight: '80px'
            }}
          >
            {uniqueTrackingMethods.map(method => (
              <option key={method} value={method}>{method}</option>
            ))}
          </select>
        </div>
      </div>

      {/* Charts */}
      <div style={{
        flex: 1,
        background: 'linear-gradient(135deg, rgba(20, 20, 35, 0.9) 0%, rgba(15, 15, 30, 0.9) 100%)',
        backdropFilter: 'blur(10px)',
        border: '1px solid rgba(139, 92, 246, 0.2)',
        borderRadius: '16px',
        padding: '20px',
        display: 'flex',
        flexDirection: 'column',
        minHeight: 0
      }}>
        {loading ? (
          <div style={{
            flex: 1,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            flexDirection: 'column'
          }}>
            <div style={{
              width: '60px',
              height: '60px',
              border: '4px solid rgba(139, 92, 246, 0.2)',
              borderTop: '4px solid rgba(139, 92, 246, 1)',
              borderRadius: '50%',
              animation: 'spin 1s linear infinite',
              marginBottom: '16px'
            }} />
            <p style={{ color: 'rgba(229, 231, 235, 0.6)', fontSize: '14px' }}>
              Analyzing activity patterns...
            </p>
          </div>
        ) : (
          <>
            <h3 style={{ color: '#e5e7eb', fontSize: '16px', margin: '0 0 16px 0' }}>
              Movement Metrics Over Normalized Time
              <span style={{ fontSize: '13px', color: 'rgba(229, 231, 235, 0.5)', marginLeft: '12px' }}>
                (0% = session start, 100% = session end)
              </span>
            </h3>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(139, 92, 246, 0.2)" />
                <XAxis 
                  dataKey="time" 
                  stroke="#9ca3af" 
                  tick={{ fill: '#9ca3af', fontSize: 12 }}
                  label={{ value: 'Session Progress (%)', position: 'insideBottom', offset: -5, fill: '#9ca3af' }}
                />
                <YAxis 
                  stroke="#9ca3af" 
                  tick={{ fill: '#9ca3af', fontSize: 12 }}
                  label={{ value: 'Movement', angle: -90, position: 'insideLeft', fill: '#9ca3af' }}
                />
                <Tooltip 
                  contentStyle={{ 
                    background: 'rgba(20, 20, 35, 0.95)', 
                    border: '1px solid rgba(139, 92, 246, 0.3)',
                    borderRadius: '8px',
                    color: '#e5e7eb'
                  }}
                  formatter={(value: any, name: string) => {
                    if (name === 'Data Points') return value
                    return typeof value === 'number' ? value.toFixed(2) : value
                  }}
                />
                <Legend wrapperStyle={{ color: '#e5e7eb' }} />
                <Line 
                  type="monotone" 
                  dataKey="distance" 
                  stroke="rgba(139, 92, 246, 1)" 
                  strokeWidth={2} 
                  dot={false} 
                  name="Avg Distance (px)" 
                />
                <Line 
                  type="monotone" 
                  dataKey="velocity" 
                  stroke="rgba(16, 185, 129, 1)" 
                  strokeWidth={2} 
                  dot={false} 
                  name="Avg Velocity (px/s)" 
                />
                <Line 
                  type="monotone" 
                  dataKey="count" 
                  stroke="rgba(59, 130, 246, 0.5)" 
                  strokeWidth={1} 
                  dot={false} 
                  name="Data Points" 
                  strokeDasharray="5 5"
                />
              </LineChart>
            </ResponsiveContainer>
          </>
        )}
      </div>
    </div>
  )
}

function SocialPanel() {
  const [networkData, setNetworkData] = useState<{nodes: Array<any>, edges: Array<any>}>({nodes: [], edges: []})
  const [stats, setStats] = useState({ total_interactions: 0, avg_duration: 0, peak_time: '', social_index: 0 })
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetch('http://localhost:8000/api/analytics/social')
      .then(res => res.json())
      .then(data => {
        // Layout nodes in a circle
        const nodeCount = data.nodes.length
        const radius = 180
        const centerX = 300
        const centerY = 250
        
        const nodesWithPositions = data.nodes.map((node: any, i: number) => {
          const angle = (i / nodeCount) * 2 * Math.PI
          return {
            ...node,
            x: centerX + radius * Math.cos(angle),
            y: centerY + radius * Math.sin(angle)
          }
        })

        // Convert edges to use node indices
        const edgesWithIndices = data.edges.map((edge: any) => ({
          from: data.nodes.findIndex((n: any) => n.id === edge.from),
          to: data.nodes.findIndex((n: any) => n.id === edge.to),
          weight: edge.weight
        }))

        setNetworkData({ 
          nodes: nodesWithPositions, 
          edges: edgesWithIndices 
        })
        setStats(data.stats)
        setLoading(false)
      })
      .catch(err => {
        console.error('Error fetching social network:', err)
        setLoading(false)
      })
  }, [])

  const interactionStats = [
    { metric: 'Total Interactions', value: stats.total_interactions.toString(), color: '#3b82f6' },
    { metric: 'Avg Duration', value: `${stats.avg_duration.toFixed(1)}s`, color: '#10b981' },
    { metric: 'Peak Time', value: stats.peak_time || 'N/A', color: '#f59e0b' },
    { metric: 'Social Index', value: stats.social_index.toFixed(2), color: '#ec4899' }
  ]

  return (
    <div style={{
      width: '100%',
      height: 'calc(100vh - 200px)',
      display: 'flex',
      flexDirection: 'column',
      gap: '16px'
    }}>
      {/* Header */}
      <div style={{
        background: 'linear-gradient(135deg, rgba(20, 20, 35, 0.9) 0%, rgba(15, 15, 30, 0.9) 100%)',
        backdropFilter: 'blur(10px)',
        border: '1px solid rgba(102, 126, 234, 0.2)',
        borderRadius: '16px',
        padding: '20px 24px',
        display: 'flex',
        alignItems: 'center',
        gap: '12px'
      }}>
        <div style={{ fontSize: '32px' }}>🐭</div>
        <div style={{ flex: 1 }}>
          <h2 style={{ color: '#3b82f6', fontSize: '20px', margin: 0, marginBottom: '4px' }}>
            Social Interaction Network
            {loading && <span style={{ fontSize: '14px', marginLeft: '12px', color: 'rgba(229, 231, 235, 0.5)' }}>Loading...</span>}
          </h2>
          <p style={{ color: 'rgba(229, 231, 235, 0.6)', fontSize: '13px', margin: 0 }}>
            {loading 
              ? 'Building social network from multi-mouse sessions...'
              : `${networkData.nodes.length} mice with ${networkData.edges.length} interaction patterns`
            }
          </p>
        </div>
      </div>

      {/* Network Visualization */}
      <div style={{
        flex: 1,
        display: 'grid',
        gridTemplateColumns: '3fr 1fr',
        gap: '16px',
        minHeight: 0
      }}>
        {/* Network Graph */}
        <div style={{
          background: 'linear-gradient(135deg, rgba(20, 20, 35, 0.9) 0%, rgba(15, 15, 30, 0.9) 100%)',
          backdropFilter: 'blur(10px)',
          border: '1px solid rgba(102, 126, 234, 0.2)',
          borderRadius: '16px',
          padding: '20px',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden'
        }}>
          <h3 style={{ color: '#e5e7eb', fontSize: '16px', margin: '0 0 16px 0' }}>Interaction Graph</h3>
          <div style={{ flex: 1, position: 'relative' }}>
            <svg width="100%" height="100%" viewBox="0 0 700 500" style={{ overflow: 'visible' }}>
              {/* Draw edges first */}
              {networkData.edges.map((edge, i) => {
                const from = networkData.nodes.find(n => n.id === edge.from)!
                const to = networkData.nodes.find(n => n.id === edge.to)!
                return (
                  <line
                    key={i}
                    x1={from.x}
                    y1={from.y}
                    x2={to.x}
                    y2={to.y}
                    stroke={`rgba(102, 126, 234, ${edge.weight})`}
                    strokeWidth={edge.weight * 3}
                    strokeLinecap="round"
                  />
                )
              })}
              {/* Draw nodes */}
              {networkData.nodes.map((node) => (
                <g key={node.id}>
                  <circle
                    cx={node.x}
                    cy={node.y}
                    r={20 + node.interactions}
                    fill="rgba(59, 130, 246, 0.3)"
                    stroke="#3b82f6"
                    strokeWidth="2"
                    style={{ cursor: 'pointer', transition: 'all 0.2s ease' }}
                    onMouseEnter={(e) => {
                      e.currentTarget.setAttribute('fill', 'rgba(59, 130, 246, 0.6)')
                      e.currentTarget.setAttribute('r', String(25 + node.interactions))
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.setAttribute('fill', 'rgba(59, 130, 246, 0.3)')
                      e.currentTarget.setAttribute('r', String(20 + node.interactions))
                    }}
                  />
                  <text
                    x={node.x}
                    y={node.y - 30 - node.interactions}
                    textAnchor="middle"
                    fill="#e5e7eb"
                    fontSize="12"
                    fontWeight="500"
                  >
                    {node.label}
                  </text>
                  <text
                    x={node.x}
                    y={node.y + 4}
                    textAnchor="middle"
                    fill="#e5e7eb"
                    fontSize="14"
                    fontWeight="700"
                  >
                    {node.interactions}
                  </text>
                </g>
              ))}
            </svg>
          </div>
        </div>

        {/* Stats Panel */}
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          gap: '16px'
        }}>
          {interactionStats.map((stat) => (
            <div
              key={stat.metric}
              style={{
                background: 'linear-gradient(135deg, rgba(20, 20, 35, 0.9) 0%, rgba(15, 15, 30, 0.9) 100%)',
                backdropFilter: 'blur(10px)',
                border: '1px solid rgba(102, 126, 234, 0.2)',
                borderRadius: '16px',
                padding: '20px',
                display: 'flex',
                flexDirection: 'column',
                gap: '8px'
              }}
            >
              <div style={{ color: 'rgba(229, 231, 235, 0.7)', fontSize: '13px' }}>{stat.metric}</div>
              <div style={{ color: stat.color, fontSize: '28px', fontWeight: '700' }}>{stat.value}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

function TemporalPanel() {
  return (
    <div style={{
      background: 'linear-gradient(135deg, rgba(20, 20, 35, 0.9) 0%, rgba(15, 15, 30, 0.9) 100%)',
      backdropFilter: 'blur(10px)',
      border: '1px solid rgba(102, 126, 234, 0.2)',
      borderRadius: '16px',
      padding: '24px',
      maxWidth: '1400px',
      margin: '0 auto',
      height: 'calc(100vh - 200px)',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      gap: '16px'
    }}>
      <div style={{ fontSize: '64px' }}>⏰</div>
      <h2 style={{ color: '#10b981', fontSize: '24px', margin: 0 }}>Temporal Patterns</h2>
      <p style={{ color: 'rgba(229, 231, 235, 0.6)', fontSize: '14px', margin: 0 }}>
        Discover circadian rhythms and behavior transitions
      </p>
      <div style={{ 
        marginTop: '20px',
        padding: '12px 24px',
        background: 'rgba(16, 185, 129, 0.1)',
        border: '1px solid rgba(16, 185, 129, 0.3)',
        borderRadius: '8px',
        color: 'rgba(229, 231, 235, 0.8)',
        fontSize: '14px'
      }}>
        Coming soon: Behavior transition Sankey diagrams
      </div>
    </div>
  )
}

function ComparisonPanel() {
  return (
    <div style={{
      background: 'linear-gradient(135deg, rgba(20, 20, 35, 0.9) 0%, rgba(15, 15, 30, 0.9) 100%)',
      backdropFilter: 'blur(10px)',
      border: '1px solid rgba(102, 126, 234, 0.2)',
      borderRadius: '16px',
      padding: '24px',
      maxWidth: '1400px',
      margin: '0 auto',
      height: 'calc(100vh - 200px)',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      gap: '16px'
    }}>
      <div style={{ fontSize: '64px' }}>⚖️</div>
      <h2 style={{ color: '#f59e0b', fontSize: '24px', margin: 0 }}>Multi-Session Comparison</h2>
      <p style={{ color: 'rgba(229, 231, 235, 0.6)', fontSize: '14px', margin: 0 }}>
        Compare behaviors across different sessions and labs
      </p>
      <div style={{ 
        marginTop: '20px',
        padding: '12px 24px',
        background: 'rgba(245, 158, 11, 0.1)',
        border: '1px solid rgba(245, 158, 11, 0.3)',
        borderRadius: '8px',
        color: 'rgba(229, 231, 235, 0.8)',
        fontSize: '14px'
      }}>
        Coming soon: Side-by-side session comparison
      </div>
    </div>
  )
}

function FeaturesPanel() {
  return (
    <div style={{
      background: 'linear-gradient(135deg, rgba(20, 20, 35, 0.9) 0%, rgba(15, 15, 30, 0.9) 100%)',
      backdropFilter: 'blur(10px)',
      border: '1px solid rgba(102, 126, 234, 0.2)',
      borderRadius: '16px',
      padding: '24px',
      maxWidth: '1400px',
      margin: '0 auto',
      height: 'calc(100vh - 200px)',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      gap: '16px'
    }}>
      <div style={{ fontSize: '64px' }}>🔧</div>
      <h2 style={{ color: '#ec4899', fontSize: '24px', margin: 0 }}>Feature Engineering</h2>
      <p style={{ color: 'rgba(229, 231, 235, 0.6)', fontSize: '14px', margin: 0 }}>
        Preview ML features and correlation analysis
      </p>
      <div style={{ 
        marginTop: '20px',
        padding: '12px 24px',
        background: 'rgba(236, 72, 153, 0.1)',
        border: '1px solid rgba(236, 72, 153, 0.3)',
        borderRadius: '8px',
        color: 'rgba(229, 231, 235, 0.8)',
        fontSize: '14px'
      }}>
        Coming soon: Feature importance and correlation matrix
      </div>
    </div>
  )
}
