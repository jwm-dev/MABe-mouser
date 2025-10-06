import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import '../App.css'

function Dashboard() {
  const [showIntro, setShowIntro] = useState(true)
  const navigate = useNavigate()

  // Skip intro on any user interaction
  useEffect(() => {
    const skipIntro = () => {
      setShowIntro(false)
    }
    
    window.addEventListener('keydown', skipIntro)
    window.addEventListener('click', skipIntro)
    
    // Auto-skip after 5 seconds
    const timer = setTimeout(skipIntro, 5000)
    
    return () => {
      window.removeEventListener('keydown', skipIntro)
      window.removeEventListener('click', skipIntro)
      clearTimeout(timer)
    }
  }, [])

  return (
    <div style={{
      width: '100vw',
      height: '100vh',
      background: 'linear-gradient(135deg, #0f0f1e 0%, #1a1a2e 50%, #16213e 100%)',
      overflow: 'hidden',
      position: 'relative'
    }}>
      {/* Intro Animation */}
      {showIntro && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          width: '100%',
          height: '100%',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          background: 'linear-gradient(135deg, #0f0f1e 0%, #1a1a2e 50%, #16213e 100%)',
          zIndex: 9999,
          animation: 'fadeOut 0.5s ease-out 4.5s forwards'
        }}>
          <div style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            gap: '32px',
            animation: 'fadeIn 0.8s ease-out',
            zIndex: 1
          }}>
            {/* Animated emoji that morphs from cat to mouse */}
            <div style={{
              position: 'relative',
              fontSize: '120px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              minHeight: '140px',
              minWidth: '140px'
            }}>
              {/* Cat emoji - initial display, fades out, fades back in */}
              <span style={{
                position: 'absolute',
                background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                filter: 'drop-shadow(0 0 20px rgba(102, 126, 234, 0.6))',
                animation: 'introGlow 3s ease-in-out infinite, fadeOut 0.6s ease-out 1.7s forwards, fadeIn 0.6s ease-out 4.2s forwards',
                userSelect: 'none',
                WebkitUserSelect: 'none'
              }}>
                🐱
                {/* Negative space features on cat */}
                <span style={{
                  position: 'absolute',
                  top: '38%',
                  left: '32%',
                  width: '10px',
                  height: '14px',
                  background: 'rgba(15, 15, 30, 1)',
                  borderRadius: '50% 50% 50% 50% / 60% 60% 40% 40%',
                  transform: 'rotate(-5deg)',
                  pointerEvents: 'none'
                }}></span>
                <span style={{
                  position: 'absolute',
                  top: '38%',
                  right: '32%',
                  width: '10px',
                  height: '14px',
                  background: 'rgba(15, 15, 30, 1)',
                  borderRadius: '50% 50% 50% 50% / 60% 60% 40% 40%',
                  transform: 'rotate(5deg)',
                  pointerEvents: 'none'
                }}></span>
                <span style={{
                  position: 'absolute',
                  top: '54%',
                  left: '12%',
                  width: '26px',
                  height: '2px',
                  background: 'rgba(15, 15, 30, 1)',
                  transform: 'rotate(-12deg)',
                  pointerEvents: 'none'
                }}></span>
                <span style={{
                  position: 'absolute',
                  top: '60%',
                  left: '8%',
                  width: '30px',
                  height: '2px',
                  background: 'rgba(15, 15, 30, 1)',
                  transform: 'rotate(-4deg)',
                  pointerEvents: 'none'
                }}></span>
                <span style={{
                  position: 'absolute',
                  top: '54%',
                  right: '12%',
                  width: '26px',
                  height: '2px',
                  background: 'rgba(15, 15, 30, 1)',
                  transform: 'rotate(12deg)',
                  pointerEvents: 'none'
                }}></span>
                <span style={{
                  position: 'absolute',
                  top: '60%',
                  right: '8%',
                  width: '30px',
                  height: '2px',
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
                  borderLeft: '6px solid transparent',
                  borderRight: '6px solid transparent',
                  borderTop: '8px solid rgba(15, 15, 30, 1)',
                  pointerEvents: 'none'
                }}></span>
              </span>
              
              {/* Mouse emoji - fades in at middle, fades out */}
              <span style={{
                position: 'absolute',
                background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                filter: 'drop-shadow(0 0 20px rgba(118, 75, 162, 0.6))',
                opacity: 0,
                animation: 'introGlow 3s ease-in-out infinite, fadeIn 0.6s ease-out 2.2s forwards, fadeOut 0.6s ease-out 3.8s forwards',
                userSelect: 'none',
                WebkitUserSelect: 'none'
              }}>
                🐭
                {/* Negative space features on mouse */}
                <span style={{
                  position: 'absolute',
                  top: '42%',
                  left: '34%',
                  width: '8px',
                  height: '8px',
                  background: 'rgba(15, 15, 30, 1)',
                  borderRadius: '50%',
                  pointerEvents: 'none'
                }}></span>
                <span style={{
                  position: 'absolute',
                  top: '42%',
                  right: '34%',
                  width: '8px',
                  height: '8px',
                  background: 'rgba(15, 15, 30, 1)',
                  borderRadius: '50%',
                  pointerEvents: 'none'
                }}></span>
                <span style={{
                  position: 'absolute',
                  top: '58%',
                  left: '14%',
                  width: '24px',
                  height: '2px',
                  background: 'rgba(15, 15, 30, 1)',
                  transform: 'rotate(-8deg)',
                  pointerEvents: 'none'
                }}></span>
                <span style={{
                  position: 'absolute',
                  top: '63%',
                  left: '12%',
                  width: '26px',
                  height: '2px',
                  background: 'rgba(15, 15, 30, 1)',
                  transform: 'rotate(-2deg)',
                  pointerEvents: 'none'
                }}></span>
                <span style={{
                  position: 'absolute',
                  top: '58%',
                  right: '14%',
                  width: '24px',
                  height: '2px',
                  background: 'rgba(15, 15, 30, 1)',
                  transform: 'rotate(8deg)',
                  pointerEvents: 'none'
                }}></span>
                <span style={{
                  position: 'absolute',
                  top: '63%',
                  right: '12%',
                  width: '26px',
                  height: '2px',
                  background: 'rgba(15, 15, 30, 1)',
                  transform: 'rotate(2deg)',
                  pointerEvents: 'none'
                }}></span>
                <span style={{
                  position: 'absolute',
                  top: '62%',
                  left: '50%',
                  transform: 'translateX(-50%)',
                  width: '6px',
                  height: '6px',
                  background: 'rgba(15, 15, 30, 1)',
                  borderRadius: '50% 50% 50% 0',
                  pointerEvents: 'none'
                }}></span>
              </span>
            </div>
            
            <div style={{
              fontSize: '48px',
              fontWeight: '800',
              fontFamily: "'Poppins', sans-serif",
              background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              letterSpacing: '-1px',
              animation: 'introExpand 1s ease-out 0.3s backwards',
              position: 'relative',
              textShadow: '0 0 40px rgba(102, 126, 234, 0.5), 0 0 80px rgba(118, 75, 162, 0.3)',
              filter: 'drop-shadow(0 4px 12px rgba(102, 126, 234, 0.4))',
              WebkitTextStroke: '1px rgba(102, 126, 234, 0.1)',
              userSelect: 'none',
              WebkitUserSelect: 'none'
            }}>
              MABe Mouser
            </div>
          </div>
        </div>
      )}

      {/* Dashboard Content */}
      {!showIntro && (
        <div style={{
          width: '100%',
          height: '100%',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          padding: '40px',
          animation: 'fadeIn 0.5s ease-out'
        }}>
          {/* Header */}
          <div style={{
            marginBottom: '60px',
            textAlign: 'center'
          }}>
            <h1 style={{
              fontSize: '56px',
              fontWeight: '800',
              fontFamily: "'Poppins', sans-serif",
              background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              letterSpacing: '-1.5px',
              marginBottom: '16px',
              textShadow: '0 0 40px rgba(102, 126, 234, 0.5), 0 0 80px rgba(118, 75, 162, 0.3)',
              filter: 'drop-shadow(0 4px 12px rgba(102, 126, 234, 0.4))',
              WebkitTextStroke: '1px rgba(102, 126, 234, 0.1)',
              userSelect: 'none',
              WebkitUserSelect: 'none'
            }}>
              MABe Mouser Dashboard
            </h1>
            <p style={{
              fontSize: '18px',
              color: 'rgba(229, 231, 235, 0.7)',
              fontWeight: '400',
              maxWidth: '600px',
              margin: '0 auto'
            }}>
              Advanced mouse behavior analysis and visualization toolkit
            </p>
          </div>

          {/* Cards Grid */}
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))',
            gap: '24px',
            maxWidth: '1200px',
            width: '100%'
          }}>
            {/* Viewer Card */}
            <div
              onClick={() => navigate('/viewer')}
              onMouseEnter={(e) => {
                e.currentTarget.style.transform = 'translateY(-8px)'
                e.currentTarget.style.boxShadow = '0 12px 32px rgba(102, 126, 234, 0.3), 0 0 0 2px rgba(102, 126, 234, 0.4)'
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.transform = 'translateY(0)'
                e.currentTarget.style.boxShadow = '0 8px 24px rgba(0, 0, 0, 0.3)'
              }}
              style={{
                background: 'linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%)',
                backdropFilter: 'blur(20px)',
                border: '1px solid rgba(102, 126, 234, 0.3)',
                borderRadius: '16px',
                padding: '32px',
                cursor: 'pointer',
                transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
                boxShadow: '0 8px 24px rgba(0, 0, 0, 0.3)',
                position: 'relative',
                overflow: 'hidden'
              }}
            >
              {/* Glow effect */}
              <div style={{
                position: 'absolute',
                top: 0,
                left: 0,
                width: '100%',
                height: '100%',
                background: 'radial-gradient(circle at 50% 0%, rgba(102, 126, 234, 0.1) 0%, transparent 60%)',
                pointerEvents: 'none'
              }} />
              
              <div style={{ position: 'relative', zIndex: 1 }}>
                <div style={{
                  fontSize: '48px',
                  marginBottom: '16px'
                }}>
                  🎬
                </div>
                <h2 style={{
                  fontSize: '24px',
                  fontWeight: '700',
                  fontFamily: "'Poppins', sans-serif",
                  color: '#e5e7eb',
                  marginBottom: '12px',
                  letterSpacing: '-0.5px'
                }}>
                  Behavior Viewer
                </h2>
                <p style={{
                  fontSize: '15px',
                  color: 'rgba(229, 231, 235, 0.6)',
                  lineHeight: '1.6',
                  marginBottom: '0'
                }}>
                  Visualize mouse tracking data with interactive 3D rendering, behavior annotations, and frame-by-frame playback.
                </p>
                
                <div style={{
                  marginTop: '20px',
                  fontSize: '14px',
                  color: 'rgba(102, 126, 234, 0.9)',
                  fontWeight: '600',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px'
                }}>
                  <span>Launch Viewer</span>
                  <span style={{ fontSize: '18px' }}>→</span>
                </div>
              </div>
            </div>

            {/* Placeholder cards for future features */}
            <div 
              onClick={() => navigate('/analytics')}
              style={{
              background: 'linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%)',
              backdropFilter: 'blur(10px)',
              border: '1px solid rgba(102, 126, 234, 0.3)',
              borderRadius: '16px',
              padding: '32px',
              cursor: 'pointer',
              transition: 'all 0.3s ease',
              position: 'relative',
              overflow: 'hidden'
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.transform = 'translateY(-8px)'
              e.currentTarget.style.boxShadow = '0 20px 60px rgba(102, 126, 234, 0.3)'
              e.currentTarget.style.borderColor = 'rgba(102, 126, 234, 0.5)'
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.transform = 'translateY(0)'
              e.currentTarget.style.boxShadow = 'none'
              e.currentTarget.style.borderColor = 'rgba(102, 126, 234, 0.3)'
            }}
            >
              <div style={{ position: 'relative', zIndex: 1 }}>
                <div style={{
                  fontSize: '48px',
                  marginBottom: '16px'
                }}>
                  📊
                </div>
                <h2 style={{
                  fontSize: '24px',
                  fontWeight: '700',
                  fontFamily: "'Poppins', sans-serif",
                  color: '#e5e7eb',
                  marginBottom: '12px',
                  letterSpacing: '-0.5px'
                }}>
                  Analytics
                </h2>
                <p style={{
                  fontSize: '15px',
                  color: 'rgba(229, 231, 235, 0.8)',
                  lineHeight: '1.6',
                  marginBottom: '20px'
                }}>
                  Explore datasets with spatial heatmaps, activity timelines, social networks, and ML feature engineering tools.
                </p>
                
                {/* Launch button */}
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  fontSize: '14px',
                  fontWeight: '600',
                  color: '#667eea',
                  marginTop: '16px',
                  transition: 'gap 0.2s ease'
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.gap = '12px'
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.gap = '8px'
                }}
                >
                  Launch Analytics
                  <span style={{
                    fontSize: '16px',
                    transition: 'transform 0.2s ease'
                  }}>→</span>
                </div>
              </div>
            </div>

            <div style={{
              background: 'rgba(30, 30, 45, 0.4)',
              backdropFilter: 'blur(10px)',
              border: '1px solid rgba(255, 255, 255, 0.1)',
              borderRadius: '16px',
              padding: '32px',
              opacity: 0.5,
              position: 'relative',
              overflow: 'hidden'
            }}>
              <div style={{ position: 'relative', zIndex: 1 }}>
                <div style={{
                  fontSize: '48px',
                  marginBottom: '16px',
                  opacity: 0.5
                }}>
                  🤖
                </div>
                <h2 style={{
                  fontSize: '24px',
                  fontWeight: '700',
                  fontFamily: "'Poppins', sans-serif",
                  color: '#8b8b9f',
                  marginBottom: '12px',
                  letterSpacing: '-0.5px'
                }}>
                  ML Models
                </h2>
                <p style={{
                  fontSize: '15px',
                  color: 'rgba(139, 139, 159, 0.7)',
                  lineHeight: '1.6',
                  marginBottom: '0'
                }}>
                  Coming soon: Train and deploy behavior classification models.
                </p>
              </div>
            </div>
          </div>

          {/* Footer hint */}
          <div style={{
            marginTop: '60px',
            fontSize: '14px',
            color: 'rgba(229, 231, 235, 0.4)',
            textAlign: 'center'
          }}>
            More features coming soon...
          </div>
        </div>
      )}
    </div>
  )
}

export default Dashboard
