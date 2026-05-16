import React, { useState, useEffect } from 'react';
import { User, Mail, LogIn, AlertCircle, Eye, EyeOff, UserPlus, Key, Fingerprint } from 'lucide-react';

export function LoginScreen({ onLoginSuccess }) {
  const [view, setView] = useState('login');
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [userId, setUserId] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  
  const [showPassword, setShowPassword] = useState({ signup: false, confirm: false, login: false });
  const [error, setError] = useState('');
  const [successMsg, setSuccessMsg] = useState('');
  const [suggestions, setSuggestions] = useState([]);
  const [isReturning, setIsReturning] = useState(false);

  useEffect(() => {
    try {
      const activeSession = localStorage.getItem('fleaa_user_data');
      if (activeSession) {
        const parsed = JSON.parse(activeSession);
        if (parsed.userId) {
          setUserId(parsed.userId);
          setIsReturning(true);
        }
      }
    } catch (e) {}
  }, []);

  const switchView = (newView) => {
    setView(newView);
    setError('');
    setSuccessMsg('');
    setSuggestions([]);
    setPassword('');
    setConfirmPassword('');
    if (newView === 'signup') {
      setName('');
      setEmail('');
      setUserId('');
    }
  };

  const handleUserIdChange = (e) => {
    setUserId(e.target.value);
    setError('');
  };

  const handleSignUp = async () => {
    if (!name.trim() || !userId.trim() || !email.trim() || !password || !confirmPassword) {
      setError('Please fill out all fields.');
      return;
    }
    if (/\s/.test(userId)) {
      setError('Username cannot contain spaces.');
      return;
    }
    if (userId.length < 3) {
      setError('Username must be at least 3 characters.');
      return;
    }
    if (!email.includes('@') || !email.includes('.')) {
      setError('Invalid email format.');
      return;
    }
    if (password.length < 6) {
      setError('Password must be at least 6 characters.');
      return;
    }
    if (password !== confirmPassword) {
      setError('Passwords do not match.');
      return;
    }

    try {
      const res = await fetch('/api/auth/signup', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          userId: userId.trim(),
          name: name.trim(),
          email: email.trim(),
          password: password
        })
      });

      const data = await res.json();
      
      if (!res.ok) {
        setError(data.error || 'Registration failed.');
        return;
      }

      switchView('login');
      setSuccessMsg('Account created successfully! Please log in.');
    } catch (e) {
      setError('System Error: Unable to reach server.');
    }
  };

  const handleLogin = async () => {
    if (!userId.trim() || !password) {
      setError('Please enter your Username and Password.');
      return;
    }

    try {
      const res = await fetch('/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          userId: userId.trim(),
          password: password
        })
      });

      const data = await res.json();

      if (!res.ok) {
        setError(data.error || 'Login failed.');
        return;
      }

      const userData = {
        name: data.name,
        email: data.email,
        userId: data.userId,
        role: data.role,
        lastLogin: data.lastLogin
      };

      localStorage.setItem('fleaa_user_data', JSON.stringify(userData));
      onLoginSuccess(userData, isReturning);
    } catch (e) {
      setError('System Error: Unable to reach server.');
    }
  };

  const togglePassword = (field) => {
    setShowPassword(prev => ({ ...prev, [field]: !prev[field] }));
  };

  const renderInput = (icon, type, value, setter, placeholder, onEnter, passField = null) => {
    const isPassword = passField !== null;
    const isShowing = isPassword ? showPassword[passField] : false;
    const Icon = icon;
    
    return (
      <div className="relative">
        <Icon className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-white/40" />
        <input
          type={isPassword ? (isShowing ? 'text' : 'password') : type}
          value={value}
          onChange={(e) => {
            setter(e.target.value);
            setError('');
            setSuccessMsg('');
          }}
          placeholder={placeholder}
          className="w-full bg-black/40 border border-white/10 rounded-xl py-3 pl-12 pr-12 text-white placeholder-white/30 focus:outline-none focus:border-blue-500/50 focus:ring-1 focus:ring-blue-500/50 transition-all"
          onKeyDown={(e) => e.key === 'Enter' && onEnter()}
        />
        {isPassword && (
          <button 
            className="absolute right-4 top-1/2 -translate-y-1/2 text-white/40 hover:text-white/70"
            onClick={() => togglePassword(passField)}
          >
            {isShowing ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
          </button>
        )}
      </div>
    );
  };

  return (
    <div className="flex h-screen w-full items-center justify-center bg-[#050a14] relative overflow-hidden">
      <div className="absolute inset-0 bg-grid pointer-events-none opacity-50" />

      <div className="z-10 bg-white/5 border border-white/10 backdrop-blur-xl p-8 rounded-2xl w-full max-w-md shadow-2xl flex flex-col items-center">
        
        <div className="w-16 h-16 rounded-full mb-6 relative flex items-center justify-center bg-blue-500/10 border border-blue-500/30">
          <div className="absolute inset-0 rounded-full animate-ping bg-blue-500/20" style={{ animationDuration: '3s' }} />
          {view === 'login' ? <Fingerprint className="w-8 h-8 text-blue-400" /> : <UserPlus className="w-8 h-8 text-blue-400" />}
        </div>

        <h1 className="text-2xl font-light text-white mb-2 tracking-widest uppercase">
          Authentication
        </h1>
        <p className="text-sm text-white/50 mb-8 text-center uppercase tracking-widest">
          {view === 'login' ? (isReturning ? 'Welcome back' : 'Please log in to continue') : 'Create a new account'}
        </p>

        {successMsg && (
          <div className="w-full flex items-center gap-3 p-3 mb-6 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-sm">
            <span>{successMsg}</span>
          </div>
        )}

        {error && (
          <div className="w-full p-3 mb-6 rounded-lg bg-red-500/10 border border-red-500/20 text-red-400 text-sm">
            <div className="flex items-center gap-3">
              <AlertCircle className="w-5 h-5 shrink-0" />
              <span>{error}</span>
            </div>
          </div>
        )}

        {view === 'signup' ? (
          <div className="w-full flex flex-col gap-4">
            {renderInput(User, 'text', name, setName, 'Full Name', handleSignUp)}
            {renderInput(Fingerprint, 'text', userId, setUserId, 'Username', handleSignUp)}
            {renderInput(Mail, 'email', email, setEmail, 'Email Address', handleSignUp)}
            {renderInput(Key, 'password', password, setPassword, 'Password', handleSignUp, 'signup')}
            {renderInput(Key, 'password', confirmPassword, setConfirmPassword, 'Confirm Password', handleSignUp, 'confirm')}
            
            <button
              onClick={handleSignUp}
              className="w-full mt-2 bg-blue-500/20 hover:bg-blue-500/30 border border-blue-500/50 text-blue-400 py-3 rounded-xl font-medium tracking-widest uppercase transition-all flex items-center justify-center gap-2 group"
            >
              <UserPlus className="w-5 h-5 group-hover:scale-110 transition-transform" />
              Create Account
            </button>
            <button onClick={() => switchView('login')} className="text-white/40 hover:text-white/70 text-sm mt-2 uppercase tracking-wider transition-colors">
              Back to Login
            </button>
          </div>
        ) : (
          <div className="w-full flex flex-col gap-5">
            {renderInput(Fingerprint, 'text', userId, setUserId, 'Username', handleLogin)}
            {renderInput(Key, 'password', password, setPassword, 'Password', handleLogin, 'login')}

            <button
              onClick={handleLogin}
              className="w-full mt-2 bg-blue-500/20 hover:bg-blue-500/30 border border-blue-500/50 text-blue-400 py-3 rounded-xl font-medium tracking-widest uppercase transition-all flex items-center justify-center gap-2 group"
            >
              <LogIn className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
              {isReturning ? 'Log In' : 'Log In'}
            </button>
            <button onClick={() => switchView('signup')} className="text-white/40 hover:text-white/70 text-sm mt-2 uppercase tracking-wider transition-colors">
              Sign Up
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
