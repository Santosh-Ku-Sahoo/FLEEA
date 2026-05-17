import React, { useState, useRef, useEffect } from 'react';
import { Send, User, Bot, Trash2, X } from 'lucide-react';
import Markdown from 'react-markdown';

export function ChatPanel({ state, onSendText, onClose }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const messagesEndRef = useRef(null);
  const lastProcessedResponse = useRef(null);

  // Load history from localStorage
  useEffect(() => {
    const saved = localStorage.getItem('fleea_chat_history');
    if (saved) {
      try {
        setMessages(JSON.parse(saved));
      } catch (e) {
        console.error("Failed to parse chat history", e);
      }
    } else {
      setMessages([{
        id: 'welcome',
        role: 'assistant',
        content: "Welcome! I'm **FLEEA**, your local-first AI assistant. Ask me anything or click the orb to talk."
      }]);
    }
  }, []);

  // Save history to localStorage
  useEffect(() => {
    if (messages.length > 0) {
      localStorage.setItem('fleea_chat_history', JSON.stringify(messages));
    }
  }, [messages]);

  // Auto-scroll
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, state.transcript, state.response]);

  // Handle incoming responses (voice, text chat, and errors)
  useEffect(() => {
    // Capture responses from any non-thinking state that has both transcript and response
    const isTerminalState = state.status === 'speaking' || state.status === 'idle' || state.status === 'error';
    
    if (isTerminalState && state.transcript && state.response) {
      // Use ref to prevent duplicate processing of the same response
      if (lastProcessedResponse.current === state.response) return;
      lastProcessedResponse.current = state.response;

      setMessages(prev => [
        ...prev,
        { id: Date.now() + 'u', role: 'user', content: state.transcript },
        { id: Date.now() + 'a', role: 'assistant', content: state.response }
      ]);
    }
  }, [state.status, state.transcript, state.response]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!input.trim() || state.status === 'thinking') return;
    
    const text = input.trim();
    setInput('');
    
    // Optimistic UI update
    setMessages(prev => [...prev, { id: Date.now().toString(), role: 'user', content: text }]);
    
    // Send to backend
    onSendText(text);
  };

  const handleClear = () => {
    if (window.confirm("Clear chat history?")) {
      const welcome = [{
        id: 'welcome',
        role: 'assistant',
        content: "History cleared. How can I help you today?"
      }];
      setMessages(welcome);
      localStorage.setItem('fleea_chat_history', JSON.stringify(welcome));
    }
  };

  return (
    <div className="flex flex-col h-full max-h-screen bg-[#0a0f1a] lg:bg-[#0a0f1a]/60 backdrop-blur-md">
      
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-white/10 bg-white/5">
        <div className="flex items-center gap-2">
           <button 
             onClick={onClose}
             className="lg:hidden p-1.5 text-white/40 hover:text-white hover:bg-white/5 rounded"
           >
             <X size={18} />
           </button>
           <h3 className="text-sm font-medium tracking-wide text-white/70">Conversation History</h3>
        </div>
        <button 
          onClick={handleClear}
          className="p-1.5 text-white/30 hover:text-white/80 hover:bg-white/10 rounded transition-colors"
          title="Clear Chat"
        >
          <Trash2 size={16} />
        </button>
      </div>

      {/* Message List */}
      <div className="flex-1 overflow-y-auto p-4 space-y-6">
        {messages.map((msg) => (
          <div key={msg.id} className={`flex gap-3 max-w-[90%] ${msg.role === 'user' ? 'ml-auto flex-row-reverse' : ''}`}>
            
            {/* Avatar */}
            <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center border ${
              msg.role === 'user' 
                ? 'bg-blue-500/20 border-blue-500/50 text-blue-300' 
                : 'bg-purple-500/20 border-purple-500/50 text-purple-300'
            }`}>
              {msg.role === 'user' ? <User size={16} /> : <Bot size={16} />}
            </div>

            {/* Bubble */}
            <div className={`px-4 py-3 rounded-2xl ${
              msg.role === 'user'
                ? 'bg-blue-600/20 text-blue-50 rounded-tr-sm border border-blue-500/20'
                : 'bg-white/5 text-white/90 rounded-tl-sm border border-white/10 prose prose-invert prose-sm max-w-none'
            }`}>
              {msg.role === 'user' ? (
                <div className="whitespace-pre-wrap">{msg.content}</div>
              ) : (
                <Markdown>{msg.content}</Markdown>
              )}
            </div>
          </div>
        ))}

        {/* Real-time Voice / Thinking indicators */}
        {state.status === 'listening' && (
           <div className="flex gap-3 ml-auto flex-row-reverse max-w-[90%] opacity-70">
             <div className="w-8 h-8 rounded-full flex items-center justify-center border bg-blue-500/20 border-blue-500/50 text-blue-300">
               <User size={16} />
             </div>
             <div className="px-4 py-3 rounded-2xl bg-blue-600/20 text-blue-50/50 rounded-tr-sm border border-blue-500/20 italic">
               {state.transcript ? state.transcript + '...' : 'Listening...'}
             </div>
           </div>
        )}

        {state.status === 'thinking' && (
           <div className="flex gap-3 max-w-[90%] opacity-70">
             <div className="w-8 h-8 rounded-full flex items-center justify-center border bg-purple-500/20 border-purple-500/50 text-purple-300 animate-pulse">
               <Bot size={16} />
             </div>
             <div className="px-4 py-3 rounded-2xl bg-white/5 text-white/50 rounded-tl-sm border border-white/10 italic flex items-center gap-2">
               Thinking
               <span className="flex gap-1">
                 <span className="w-1.5 h-1.5 bg-white/50 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></span>
                 <span className="w-1.5 h-1.5 bg-white/50 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></span>
                 <span className="w-1.5 h-1.5 bg-white/50 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></span>
               </span>
             </div>
           </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="p-4 bg-white/5 border-t border-white/10">
        <form onSubmit={handleSubmit} className="relative flex items-center">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={state.status === 'thinking'}
            placeholder={state.status === 'thinking' ? "Thinking..." : "Type a message..."}
            className="w-full bg-[#050810]/50 border border-white/10 rounded-full py-3 pl-5 pr-12 text-sm text-white focus:outline-none focus:border-blue-500/50 transition-colors disabled:opacity-50"
          />
          <button
            type="submit"
            disabled={!input.trim() || state.status === 'thinking'}
            className="absolute right-2 p-2 bg-blue-500 hover:bg-blue-600 text-white rounded-full transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Send size={16} />
          </button>
        </form>
      </div>

    </div>
  );
}
