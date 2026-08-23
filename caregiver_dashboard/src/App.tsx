import { useState, useEffect, useRef } from 'react';
import './index.css';

interface VisionContext {
  mode: string;
  torch_on: boolean;
  detected_objects: string[];
  closest_obstacle: string | null;
  wall_ahead: boolean;
}

interface LogEntry {
  id: number;
  time: string;
  type: 'speech' | 'alert' | 'system';
  message: string;
}

interface Location {
  lat: number;
  lng: number;
  city: string;
  region: string;
  country: string;
}

// ──────────── SVG Icons ────────────
const MapPinIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z" />
    <circle cx="12" cy="10" r="3" />
  </svg>
);
const EyeIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
    <circle cx="12" cy="12" r="3" />
  </svg>
);
const VolumeIcon = ({ color = '#3b82f6' }: { color?: string }) => (
  <svg viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5" />
    <path d="M19.07 4.93a10 10 0 0 1 0 14.14" />
    <path d="M15.54 8.46a5 5 0 0 1 0 7.07" />
  </svg>
);
const AlertIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="#dc2626" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
    <line x1="12" y1="9" x2="12" y2="13" />
    <line x1="12" y1="17" x2="12.01" y2="17" />
  </svg>
);
const InfoIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="#6b7280" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="10" />
    <line x1="12" y1="8" x2="12" y2="12" />
    <line x1="12" y1="16" x2="12.01" y2="16" />
  </svg>
);
const ListIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <line x1="8" y1="6" x2="21" y2="6" /><line x1="8" y1="12" x2="21" y2="12" />
    <line x1="8" y1="18" x2="21" y2="18" /><line x1="3" y1="6" x2="3.01" y2="6" />
    <line x1="3" y1="12" x2="3.01" y2="12" /><line x1="3" y1="18" x2="3.01" y2="18" />
  </svg>
);

const API = 'http://localhost:8000';
const WS  = 'ws://localhost:8000/ws';

// ──────────── Main App ────────────
function App() {
  const [vision, setVision]       = useState<VisionContext | null>(null);
  const [logs, setLogs]           = useState<LogEntry[]>([]);
  const [location, setLocation]   = useState<Location | null>(null);
  const [wsState, setWsState]     = useState<'connecting' | 'connected' | 'disconnected'>('connecting');
  const wsRef                     = useRef<WebSocket | null>(null);
  const prevWallRef               = useRef(false);
  const reconnectTimer            = useRef<ReturnType<typeof setTimeout> | null>(null);

  const addLog = (type: LogEntry['type'], message: string) => {
    setLogs(prev => [
      { id: Date.now() + Math.random(), time: new Date().toLocaleTimeString(), type, message },
      ...prev,
    ].slice(0, 80));
  };

  // Fetch location once (called when WS first connects so server is guaranteed up)
  const fetchLocation = () => {
    fetch(`${API}/location`)
      .then(r => r.json())
      .then(setLocation)
      .catch(() => {});
  };

  // WebSocket with auto-reconnect
  const connect = () => {
    if (wsRef.current) {
      wsRef.current.onclose = null;
      wsRef.current.close();
    }
    setWsState('connecting');
    const ws = new WebSocket(WS);
    wsRef.current = ws;

    ws.onopen = () => {
      setWsState('connected');
      addLog('system', '✅ Dashboard connected to SURDAS');
      fetchLocation();
    };

    ws.onclose = () => {
      setWsState('disconnected');
      // Retry every 3 seconds until SURDAS starts
      reconnectTimer.current = setTimeout(connect, 3000);
    };

    ws.onerror = () => {
      ws.close();
    };

    ws.onmessage = (event) => {
      try {
        const parsed = JSON.parse(event.data);
        if (parsed.type === 'vision') {
          setVision(parsed.data as VisionContext);
        } else if (parsed.type === 'speech') {
          addLog('speech', parsed.data.text);
        }
      } catch (_) {}
    };
  };

  useEffect(() => {
    connect();
    return () => {
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
      if (wsRef.current) { wsRef.current.onclose = null; wsRef.current.close(); }
    };
  }, []);

  // Log wall detection transitions
  useEffect(() => {
    if (vision?.wall_ahead && !prevWallRef.current) {
      addLog('alert', '🧱 Wall or barrier detected ahead!');
    }
    prevWallRef.current = vision?.wall_ahead ?? false;
  }, [vision?.wall_ahead]);

  const badgeStyle = wsState === 'connected'
    ? { background: '#ecfdf5', color: '#065f46' }
    : wsState === 'connecting'
    ? { background: '#fffbeb', color: '#92400e' }
    : { background: '#fef2f2', color: '#991b1b' };

  const dotColor = wsState === 'connected' ? '#10b981' : wsState === 'connecting' ? '#f59e0b' : '#dc2626';

  const statusLabel = wsState === 'connected'
    ? '🟢 System Online'
    : wsState === 'connecting'
    ? '🟡 Connecting…'
    : '🔴 Waiting for SURDAS…';

  return (
    <div className="app">
      {/* ── Header ── */}
      <header className="header">
        <div className="header-left">
          <svg className="header-icon" viewBox="0 0 24 24" fill="none" stroke="#2563eb" strokeWidth="2">
            <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
          </svg>
          <h1>SURDAS Caregiver Dashboard</h1>
        </div>
        <div className="status-badge" style={badgeStyle}>
          <span className="status-dot" style={{ background: dotColor }} />
          {statusLabel}
        </div>
      </header>

      {wsState !== 'connected' && (
        <div className="offline-banner">
          <InfoIcon />
          <span>Start <code>python3 test_ai_pipeline.py</code> or <code>python3 surdas_brain.py</code> — the dashboard will connect automatically.</span>
        </div>
      )}

      {/* ── Grid ── */}
      <div className="grid">

        {/* ── Panel 1: Vision State ── */}
        <div className="card">
          <div className="card-title"><EyeIcon /> Current Vision State</div>

          <div className="stat-row">
            <span className="stat-label">Mode</span>
            <span className="stat-value blue">{vision?.mode ?? '—'}</span>
          </div>
          <div className="stat-row">
            <span className="stat-label">Flashlight</span>
            <span className={`stat-value ${vision?.torch_on ? 'yellow' : 'gray'}`}>
              {vision ? (vision.torch_on ? '🔦 ON' : 'OFF') : '—'}
            </span>
          </div>
          <div className="stat-obstacle">
            <span className="stat-label">Closest Obstacle</span>
            <span className="stat-value">{vision?.closest_obstacle ?? 'Path is clear'}</span>
          </div>
          <div className={`wall-banner ${vision?.wall_ahead ? 'danger' : 'safe'}`}>
            <span>Wall Detection</span>
            <span>{vision?.wall_ahead ? '🧱 WARNING' : '✅ CLEAR'}</span>
          </div>

          {vision && vision.detected_objects.length > 0 && (
            <>
              <p className="stat-label" style={{ marginTop: 14, marginBottom: 8 }}>Detected Objects</p>
              <div className="object-tags">
                {[...new Set(vision.detected_objects)].map((obj, i) => (
                  <span key={i} className="object-tag">{obj}</span>
                ))}
              </div>
            </>
          )}
        </div>

        {/* ── Panel 2: Location ── */}
        <div className="card">
          <div className="card-title"><MapPinIcon /> Live Location (WiFi IP)</div>

          {location ? (
            <>
              <div className="location-info">
                <div className="location-city">📍 {location.city}{location.region ? `, ${location.region}` : ''}</div>
                <div className="location-coords">
                  {location.country && <span style={{ marginRight: 8 }}>🌐 {location.country}</span>}
                  Lat: {location.lat.toFixed(4)}, Lng: {location.lng.toFixed(4)}
                </div>
              </div>
              {location.lat !== 0 && (
                <iframe
                  className="map-frame"
                  title="User location map"
                  src={`https://www.openstreetmap.org/export/embed.html?bbox=${location.lng - 0.012}%2C${location.lat - 0.012}%2C${location.lng + 0.012}%2C${location.lat + 0.012}&layer=mapnik&marker=${location.lat}%2C${location.lng}`}
                />
              )}
            </>
          ) : (
            <p className="loading-text">
              {wsState === 'connected' ? 'Fetching location…' : 'Connect SURDAS to load location'}
            </p>
          )}
        </div>

        {/* ── Panel 3: Activity Logs ── */}
        <div className="card logs-container">
          <div className="card-title" style={{ justifyContent: 'space-between' }}>
            <span style={{ display: 'flex', alignItems: 'center', gap: 8 }}><ListIcon /> Activity Logs</span>
            {logs.length > 0 && (
              <button className="clear-btn" onClick={() => setLogs([])}>Clear</button>
            )}
          </div>

          <div className="logs-scroll">
            {logs.length === 0 ? (
              <p className="no-logs">No activity yet — start SURDAS to see events here.</p>
            ) : (
              logs.map(log => (
                <div key={log.id} className={`log-entry ${log.type}`}>
                  <div className="log-meta">
                    {log.type === 'speech' && <VolumeIcon />}
                    {log.type === 'alert'  && <AlertIcon />}
                    {log.type === 'system' && <InfoIcon />}
                    <span className="log-time">{log.time}</span>
                  </div>
                  <p className={`log-text ${log.type}`}>{log.message}</p>
                </div>
              ))
            )}
          </div>
        </div>

      </div>
    </div>
  );
}

export default App;
