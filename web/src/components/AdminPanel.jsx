import React, { useState, useEffect } from 'react';
import { Users, Trash2, ArrowLeft, RefreshCw, AlertCircle } from 'lucide-react';

export function AdminPanel({ onBack }) {
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const fetchUsers = async () => {
    setLoading(true);
    setError('');
    try {
      const res = await fetch('/api/admin/users');
      if (!res.ok) {
        throw new Error('Failed to fetch users');
      }
      const data = await res.json();
      setUsers(data);
    } catch (err) {
      setError('System Error: Unable to load users.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchUsers();
  }, []);

  const handleDelete = async (userId) => {
    if (!window.confirm(`Are you sure you want to delete user "${userId}"?`)) return;
    
    console.log('Attempting to delete user:', userId);
    try {
      // Use encodeURIComponent to handle special characters in userId
      const res = await fetch(`/api/admin/users/${encodeURIComponent(userId)}`, { 
        method: 'DELETE' 
      });
      
      const data = await res.json();
      
      if (!res.ok) {
        throw new Error(data.error || 'Failed to delete user');
      }
      
      setUsers(prev => prev.filter(u => u.user_id !== userId));
      alert(`User "${userId}" has been successfully purged from the system.`);
    } catch (err) {
      console.error('Delete error:', err);
      alert('System Error: ' + err.message);
    }
  };

  return (
    <div className="flex flex-col h-full bg-[#050a14] text-white overflow-hidden p-8">
      <div className="flex items-center justify-between mb-8">
        <div className="flex items-center gap-4">
          <button 
            onClick={onBack}
            className="p-2 rounded-full hover:bg-white/10 text-white/70 hover:text-white transition-colors"
          >
            <ArrowLeft className="w-6 h-6" />
          </button>
          <div className="flex items-center gap-3">
            <Users className="w-8 h-8 text-blue-400" />
            <h1 className="text-2xl font-light tracking-widest uppercase">Admin Dashboard</h1>
          </div>
        </div>
        
        <button 
          onClick={fetchUsers}
          disabled={loading}
          className="flex items-center gap-2 px-4 py-2 rounded-full bg-blue-500/20 hover:bg-blue-500/30 text-blue-400 border border-blue-500/30 transition-colors"
        >
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          Refresh
        </button>
      </div>

      <div className="flex-1 bg-white/5 border border-white/10 rounded-2xl overflow-hidden flex flex-col">
        {error ? (
          <div className="flex-1 flex items-center justify-center text-red-400 gap-2">
            <AlertCircle className="w-6 h-6" />
            <span>{error}</span>
          </div>
        ) : loading ? (
          <div className="flex-1 flex items-center justify-center">
            <RefreshCw className="w-8 h-8 text-blue-400 animate-spin" />
          </div>
        ) : users.length === 0 ? (
          <div className="flex-1 flex flex-col items-center justify-center text-white/50">
            <Users className="w-12 h-12 mb-4 opacity-50" />
            <p className="tracking-widest uppercase text-sm">No registered users found</p>
          </div>
        ) : (
          <div className="overflow-auto w-full flex-1 p-6">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="border-b border-white/10 text-white/50 text-xs tracking-widest uppercase">
                  <th className="pb-4 font-normal">Database ID</th>
                  <th className="pb-4 font-normal">Username</th>
                  <th className="pb-4 font-normal">Full Name</th>
                  <th className="pb-4 font-normal">Email</th>
                  <th className="pb-4 font-normal">Role</th>
                  <th className="pb-4 font-normal">Created At</th>
                  <th className="pb-4 font-normal text-right">Actions</th>
                </tr>
              </thead>
              <tbody>
                {users.map(u => (
                  <tr key={u.id} className="border-b border-white/5 hover:bg-white/5 transition-colors group">
                    <td className="py-4 text-white/30 text-sm">#{u.id}</td>
                    <td className="py-4 font-medium text-blue-400">{u.user_id}</td>
                    <td className="py-4 text-white/90">{u.name}</td>
                    <td className="py-4 text-white/70 text-sm">{u.email}</td>
                    <td className="py-4">
                      <span className={`text-[10px] px-2 py-0.5 rounded-full border ${
                        u.role === 'admin' 
                        ? 'bg-blue-500/10 border-blue-500/30 text-blue-400' 
                        : 'bg-white/5 border-white/10 text-white/50'
                      } uppercase tracking-widest font-bold`}>
                        {u.role || 'user'}
                      </span>
                    </td>
                    <td className="py-4 text-white/50 text-sm">
                      {new Date(u.created_at).toLocaleString()}
                    </td>
                    <td className="py-4 text-right">
                      {u.role !== 'admin' && (
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleDelete(u.user_id);
                          }}
                          className="p-2.5 rounded-xl text-red-500/60 hover:text-red-400 hover:bg-red-500/20 active:scale-95 transition-all cursor-pointer group"
                          title="Purge User"
                        >
                          <Trash2 className="w-5 h-5 group-hover:scale-110 transition-transform" />
                        </button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
