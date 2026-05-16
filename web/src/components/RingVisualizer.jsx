import React, { useEffect, useState } from 'react';

export function RingVisualizer({ state, onToggle }) {
  const { status } = state;
  const [particles, setParticles] = useState([]);

  // Generate particles when listening
  useEffect(() => {
    let interval;
    if (status === 'listening') {
      interval = setInterval(() => {
        const id = Math.random().toString(36).substr(2, 9);
        const angle = Math.random() * 2 * Math.PI;
        const r0 = 30 + Math.random() * 30;
        const r1 = 120 + Math.random() * 80;
        const sz = 3 + Math.random() * 5;
        const dx0 = Math.cos(angle) * r0;
        const dy0 = Math.sin(angle) * r0;
        const dx1 = Math.cos(angle) * r1;
        const dy1 = Math.sin(angle) * r1;
        
        const duration = 1.2 + Math.random() * 1.2;
        const delay = Math.random() * 0.4;
        
        const particle = {
          id,
          style: {
            width: `${sz}px`,
            height: `${sz}px`,
            left: '170px',
            top: '170px',
            '--dx0': `${dx0}px`,
            '--dy0': `${dy0}px`,
            '--dx1': `${dx1}px`,
            '--dy1': `${dy1}px`,
            animationDuration: `${duration}s`,
            animationDelay: `${delay}s`
          }
        };
        
        setParticles(prev => [...prev, particle]);
        
        // Cleanup after animation completes (max ~3s)
        setTimeout(() => {
          setParticles(prev => prev.filter(p => p.id !== id));
        }, 3000);
      }, 140);
    } else {
      setParticles([]);
    }
    
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [status]);

  return (
      <div className="relative flex items-center justify-center w-full h-[400px]" data-state={status}>
        {/* Ambient Glow */}
        <div className="ambient-glow"></div>

        {/* Center Orb / Core */}
        <div 
          className="core z-20" 
          title="Click to start / stop"
          onClick={onToggle}
        >
          {status === 'idle' && '🤖'}
          {status === 'listening' && '🎤'}
          {status === 'thinking' && '⚙️'}
          {status === 'speaking' && '🔊'}
          {status === 'error' && '⚠️'}
        </div>

        {/* Ring Wrapper precisely centered behind orb */}
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 z-10 ring-wrap">
          <div className="ring ring-1"></div>
          <div className="ring ring-2"></div>
          <div className="ring ring-3"></div>
          <div className="ring ring-4"></div>
          <div className="ring ring-5"></div>
        </div>
        
        {/* Particles */}
        <div className="absolute inset-0 pointer-events-none z-30 overflow-hidden">
          {particles.map(p => (
            <div key={p.id} className="particle" style={p.style}></div>
          ))}
        </div>
      </div>
  );
}
