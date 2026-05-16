import React, { useState } from 'react';
import { User, Database, Cpu, Activity, LogOut, Edit2, Check, Shield, X } from 'lucide-react';

export function Sidebar({ systemData, user, onLogout, onAdmin, onClose }) {
  const [isEditingName, setIsEditingName] = useState(false);
  const [editNameValue, setEditNameValue] = useState(user?.name || '');

  if (!systemData) {
    return (
      <div className="w-full border-r border-white/10 bg-white/5 backdrop-blur-xl p-6 flex flex-col h-full animate-pulse">
        <div className="h-6 bg-white/10 rounded w-1/2 mb-8"></div>
        <div className="space-y-4">
          <div className="h-24 bg-white/5 rounded-xl border border-white/5"></div>
          <div className="h-24 bg-white/5 rounded-xl border border-white/5"></div>
        </div>
      </div>
    );
  }

  const { profile, memory_stats, status } = systemData;

  const handleSaveName = async () => {
    if (!editNameValue.trim()) return;
    try {
      const newName = editNameValue.trim();
      
      const res = await fetch('/api/auth/profile', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          userId: user.userId,
          name: newName,
          email: user.email // keep same email
        })
      });

      if (!res.ok) {
        throw new Error('Failed to update profile on server');
      }

      // Update local session
      const userData = { ...user, name: newName };
      localStorage.setItem('fleaa_user_data', JSON.stringify(userData));
      
      setIsEditingName(false);
      window.location.reload(); // Refresh to propagate changes
    } catch (e) {
      console.error(e);
      alert('Error updating profile: ' + e.message);
    }
  };

  return (
    <div className="w-full border-r border-white/10 bg-[#0a0f1a] lg:bg-[#0a0f1a]/80 backdrop-blur-xl p-6 flex flex-col h-full overflow-y-auto">
      
      {/* Mobile Close Button */}
      <button 
        onClick={onClose}
        className="lg:hidden absolute top-4 right-4 p-2 text-white/40 hover:text-white"
      >
        <X size={20} />
      </button>

      {/* Authenticated User Header */}
      <div className="mb-8 flex items-center gap-4 group relative">
        <div className="w-12 h-12 rounded-full bg-blue-500/20 text-blue-400 border border-blue-500/30 flex items-center justify-center text-lg font-semibold uppercase shrink-0">
          {user?.name?.substring(0, 2) || 'FL'}
        </div>
        <div className="flex-1 min-w-0">
          {isEditingName ? (
            <div className="flex items-center gap-2">
              <input 
                type="text" 
                value={editNameValue}
                onChange={(e) => setEditNameValue(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleSaveName()}
                className="w-full bg-black/40 border border-blue-500/50 rounded px-2 py-1 text-sm text-white focus:outline-none"
                autoFocus
              />
              <button onClick={handleSaveName} className="text-green-400 hover:text-green-300">
                <Check size={16} />
              </button>
            </div>
          ) : (
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-semibold tracking-wider text-white truncate pr-2">
                {user?.name || 'Authorized User'}
              </h2>
              <button 
                onClick={() => {
                  setEditNameValue(user?.name || '');
                  setIsEditingName(true);
                }}
                className="opacity-0 group-hover:opacity-100 text-white/40 hover:text-white/80 transition-opacity"
              >
                <Edit2 size={14} />
              </button>
            </div>
          )}
          <p className="text-[10px] text-blue-400/80 uppercase tracking-widest mt-0.5">
            Level 5 Access
          </p>
        </div>
      </div>

      <div className="space-y-6 flex-1">
        
        {/* Profile Panel */}
        <div className="bg-white/5 border border-white/10 rounded-xl p-4">
          <h3 className="text-xs uppercase tracking-wider text-white/40 mb-3 flex items-center gap-2">
            <User size={14} /> Neural Profile
          </h3>
          <div className="space-y-3">
            <div>
              <div className="text-[10px] text-white/30 uppercase">Email</div>
              <div className="text-sm font-medium">{user?.email || 'Unknown'}</div>
            </div>
            {profile?.preferences && Object.keys(profile.preferences).length > 0 && (
              <div>
                <div className="text-[10px] text-white/30 uppercase mb-1">Preferences</div>
                <div className="flex flex-wrap gap-1.5">
                  {Object.entries(profile.preferences).map(([k, v]) => (
                    <span key={k} className="px-2 py-0.5 bg-blue-500/20 text-blue-300 rounded text-xs">
                      {k}: {v}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Memory Panel */}
        <div className="bg-white/5 border border-white/10 rounded-xl p-4">
          <h3 className="text-xs uppercase tracking-wider text-white/40 mb-3 flex items-center gap-2">
            <Database size={14} /> Memory Subsystem
          </h3>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <div className="text-[10px] text-white/30 uppercase">Vector Docs</div>
              <div className="text-xl font-light text-purple-300">
                {memory_stats?.total_entries || 0}
              </div>
            </div>
            <div>
              <div className="text-[10px] text-white/30 uppercase">Short Term</div>
              <div className="text-xl font-light text-cyan-300">
                {memory_stats?.short_term_count || 0}
              </div>
            </div>
          </div>
        </div>

        {/* System Status Panel */}
        <div className="bg-white/5 border border-white/10 rounded-xl p-4">
          <h3 className="text-xs uppercase tracking-wider text-white/40 mb-3 flex items-center gap-2">
            <Cpu size={14} /> Engine Status
          </h3>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between items-center">
              <span className="text-white/50">STT (Whisper)</span>
              <span className={status?.stt_available ? "text-green-400" : "text-red-400"}>
                {status?.stt_available ? 'Ready' : 'Offline'}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-white/50">TTS (pyttsx3)</span>
              <span className={status?.tts_available ? "text-green-400" : "text-red-400"}>
                {status?.tts_available ? 'Ready' : 'Offline'}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-white/50">Brain</span>
              <span className={status?.brain_status ? "text-green-400" : "text-red-400"}>
                {status?.brain_status ? (status.brain_status.model || 'Ready') : 'Offline'}
              </span>
            </div>
          </div>
        </div>

      </div>

      {/* Admin Button - Only for Admins */}
      {user?.role === 'admin' && (
        <div className="mt-auto pt-6 border-t border-white/10">
          <button
            onClick={onAdmin}
            className="w-full flex items-center justify-center gap-2 py-3 rounded-xl border border-purple-500/20 bg-purple-500/10 text-purple-400 hover:bg-purple-500/20 transition-all uppercase tracking-widest text-xs font-semibold mb-3"
          >
            <Shield size={16} /> Admin Dashboard
          </button>
        </div>
      )}

      {/* Logout Button */}
      <div className={user?.role === 'admin' ? "" : "mt-auto pt-6 border-t border-white/10"}>
        <button
          onClick={onLogout}
          className="w-full flex items-center justify-center gap-2 py-3 rounded-xl border border-red-500/20 bg-red-500/10 text-red-400 hover:bg-red-500/20 transition-all uppercase tracking-widest text-xs font-semibold"
        >
          <LogOut size={16} /> Disconnect
        </button>
      </div>

    </div>
  );
}
