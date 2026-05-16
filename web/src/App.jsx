import React, { useState, useEffect } from 'react';
import { Sidebar } from './components/Sidebar';
import { RingVisualizer } from './components/RingVisualizer';
import { ChatPanel } from './components/ChatPanel';
import { useSocket } from './hooks/useSocket';
import { LoginScreen } from './components/LoginScreen';
import { AdminPanel } from './components/AdminPanel';
import { Menu, X, MessageSquare, Shield } from 'lucide-react';

export default function App() {
  const { state, systemData, startVoice, stopVoice, sendText, greetUser } = useSocket();
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [user, setUser] = useState(null);
  const [isAdminView, setIsAdminView] = useState(false);
  
  // Responsive sidebar states
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [isChatOpen, setIsChatOpen] = useState(false);

  // Auto-login check
  useEffect(() => {
    try {
      const saved = localStorage.getItem('fleaa_user_data');
      if (saved) {
        setUser(JSON.parse(saved));
        setIsLoggedIn(true);
      }
    } catch (e) {
      console.error('Failed to parse saved user data', e);
    }
  }, []);

  const handleToggleVoice = () => {
    if (state.status === 'idle') {
      startVoice(user?.name || 'User');
    } else {
      stopVoice();
    }
  };

  const handleLoginSuccess = (userData, isReturning) => {
    setUser(userData);
    setIsLoggedIn(true);
    // Emit the greeting event using the full name
    greetUser(userData.name, isReturning);
  };

  const handleLogout = () => {
    setIsLoggedIn(false);
    // Note: We deliberately do NOT delete localStorage here,
    // so the returning user's name remains saved for the next login.
  };

  if (!isLoggedIn) {
    return <LoginScreen onLoginSuccess={handleLoginSuccess} />;
  }

  if (isAdminView) {
    return <AdminPanel onBack={() => setIsAdminView(false)} />;
  }

  return (
    <div className="flex h-screen overflow-hidden bg-bg relative">
      
      {/* Mobile Header */}
      <div className="lg:hidden absolute top-0 left-0 right-0 h-16 border-b border-white/10 bg-black/40 backdrop-blur-xl z-50 flex items-center justify-between px-6">
        <button 
          onClick={() => setIsSidebarOpen(true)}
          className="p-2 text-white/70 hover:text-white"
        >
          <Menu size={20} />
        </button>
        <h1 className="text-[10px] font-bold tracking-[4px] uppercase text-white/40">
          F L E E A
        </h1>
        <button 
          onClick={() => setIsChatOpen(true)}
          className="p-2 text-white/70 hover:text-white"
        >
          <MessageSquare size={20} />
        </button>
      </div>

      {/* Left Sidebar Overlay (Mobile) */}
      {isSidebarOpen && (
        <div 
          className="lg:hidden fixed inset-0 bg-black/60 backdrop-blur-sm z-[60]"
          onClick={() => setIsSidebarOpen(false)}
        />
      )}
      
      {/* Left Sidebar */}
      <div className={`
        fixed lg:relative inset-y-0 left-0 z-[70] 
        w-80 sidebar-transition
        ${isSidebarOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}
      `}>
        <Sidebar 
          systemData={systemData} 
          user={user} 
          onLogout={handleLogout} 
          onAdmin={() => setIsAdminView(true)}
          onClose={() => setIsSidebarOpen(false)}
        />
      </div>
      
      {/* Middle: 3D Visualizer & Voice Controls */}
      <div className="flex-1 flex flex-col items-center justify-center relative min-w-0">
        
        {/* Desktop Header/Brand */}
        <div className="hidden lg:block absolute top-8 left-0 right-0 text-center z-20">
          <h1 className="text-[13px] font-semibold tracking-[5px] uppercase text-white/30">
            ⬡ &nbsp; F L E E A
          </h1>
        </div>

        {/* 3D Visualizer */}
        <div className="scale-75 md:scale-100 transition-transform">
          <RingVisualizer state={state} onToggle={handleToggleVoice} />
        </div>

        {/* Status / Buttons */}
        <div className="absolute bottom-12 left-0 right-0 flex flex-col items-center gap-6 z-20">
          
          <div className="flex items-center gap-2 px-4 py-1.5 rounded-full bg-white/5 border border-white/10 backdrop-blur-md">
             <div className="w-2 h-2 rounded-full animate-pulse" 
                   style={{ backgroundColor: `var(--color-${state.status})` }}></div>
             <span className="text-xs uppercase tracking-widest font-medium" 
                   style={{ color: `var(--color-${state.status})` }}>
               {state.status === 'idle' ? 'Ready' : state.status}
             </span>
          </div>

          <div className="flex gap-4">
            <button 
              onClick={() => startVoice(user?.name || 'User')}
              className="px-6 py-2.5 rounded-full border border-blue-500/50 bg-blue-500/20 text-blue-400 hover:bg-blue-500/30 text-xs md:text-sm font-semibold transition-all backdrop-blur-md"
            >
              ▶ START
            </button>
            <button 
              onClick={stopVoice}
              className="px-6 py-2.5 rounded-full border border-white/10 bg-white/5 text-white/70 hover:bg-white/10 hover:text-white text-xs md:text-sm font-semibold transition-all backdrop-blur-md"
            >
              ■ STOP
            </button>
          </div>
        </div>
      </div>

      {/* Right Sidebar Overlay (Mobile) */}
      {isChatOpen && (
        <div 
          className="lg:hidden fixed inset-0 bg-black/60 backdrop-blur-sm z-[60]"
          onClick={() => setIsChatOpen(false)}
        />
      )}

      {/* Right Sidebar (Text Chat) */}
      <div className={`
        fixed lg:relative inset-y-0 right-0 z-[70]
        w-[90%] md:w-[450px] lg:w-[450px] shrink-0
        sidebar-transition
        ${isChatOpen ? 'translate-x-0 shadow-[-10px_0_30px_rgba(0,0,0,0.8)]' : 'translate-x-full lg:translate-x-0 shadow-none lg:shadow-[-10px_0_30px_rgba(0,0,0,0.5)]'}
        border-l border-white/10
      `}>
        <ChatPanel 
          state={state} 
          onSendText={sendText} 
          onClose={() => setIsChatOpen(false)}
        />
      </div>

    </div>
  );
}
