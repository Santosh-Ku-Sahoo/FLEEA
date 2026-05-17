import { useEffect, useRef, useState, useCallback } from 'react';
import { io } from 'socket.io-client';

export function useSocket() {
  const [state, setState] = useState({
    status: 'idle',
    transcript: '',
    response: '',
  });
  
  const [systemData, setSystemData] = useState(null);
  const socketRef = useRef(null);
  
  // Browser Voice State
  const recognitionRef = useRef(null);
  const isWebVoiceActive = useRef(false);

  useEffect(() => {
    // Initialize Web Speech API
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRecognition) {
      const recognition = new SpeechRecognition();
      recognition.continuous = true;
      recognition.interimResults = true;
      recognition.lang = 'en-US';

      recognition.onresult = (event) => {
        const transcript = Array.from(event.results)
          .map(result => result[0])
          .map(result => result.transcript)
          .join('');
        
        const isFinal = event.results[event.results.length - 1].isFinal;
        
        setState(prev => ({ ...prev, status: 'listening', transcript }));

        if (isFinal) {
          const finalTranscript = event.results[event.results.length - 1][0].transcript.trim();
          if (finalTranscript) {
             socketRef.current.emit('text_chat', { text: finalTranscript });
          }
        }
      };

      recognition.onerror = (event) => {
        console.error('Speech recognition error', event.error);
        if (event.error === 'not-allowed') {
          setState(prev => ({ ...prev, status: 'error', response: 'Microphone permission denied.' }));
          isWebVoiceActive.current = false;
        }
      };

      recognition.onend = () => {
        // If we're still supposed to be listening, restart it with a small delay
        // to avoid browser state errors (mimic continuous backend loop)
        if (isWebVoiceActive.current) {
          setTimeout(() => {
            try {
              if (isWebVoiceActive.current) recognition.start();
            } catch (e) {
              console.warn('Recognition restart failed:', e);
            }
          }, 250);
        }
      };

      recognitionRef.current = recognition;
    }

    // Vite proxy handles /socket.io in dev, Render handles it in prod.
    // Force websocket transport to bypass Render load balancer polling issues
    socketRef.current = io({ transports: ['websocket'], upgrade: false });

    socketRef.current.on('connect', () => {
      console.log('Connected to FLEEA Voice UI');
      socketRef.current.emit('get_status'); // Request initial status
    });

    socketRef.current.on('state', (data) => {
      setState(prev => {
        // Only update if changed or if we have new text
        if (prev.status === data.status && !data.transcript && !data.response) {
          return prev;
        }
        
        // Browser TTS Fallback
        if (data.response && isWebVoiceActive.current) {
          speakBrowser(data.response);
        }

        return {
          status: data.status || 'idle',
          transcript: data.transcript ?? prev.transcript,
          response: data.response ?? prev.response
        };
      });
      
      // If thinking or speaking, request status update to refresh memory stats
      if (data.status === 'thinking' || data.status === 'speaking') {
         socketRef.current.emit('get_status');
      }
    });

    socketRef.current.on('system_status', (data) => {
       setSystemData(data);
    });

    return () => {
      if (socketRef.current) {
        socketRef.current.disconnect();
      }
      if (recognitionRef.current) {
        recognitionRef.current.stop();
      }
      if (window.speechSynthesis) {
        window.speechSynthesis.cancel();
      }
    };
  }, []);

  const speakBrowser = (text) => {
    if (!window.speechSynthesis) return;
    
    // Cancel any current speech
    window.speechSynthesis.cancel();
    
    // Strip markdown for cleaner speech
    const clean = text.replace(/```[\s\S]*?```/g, '')
                      .replace(/`([^`]*)`/g, '$1')
                      .replace(/\[([^\]]+)\]\([^\)]+\)/g, '$1')
                      .replace(/[*#_]/g, '')
                      .trim();

    const utterance = new SpeechSynthesisUtterance(clean);
    utterance.rate = 1.0;
    utterance.pitch = 1.0;
    
    // Notify status change to visualizer
    utterance.onstart = () => {
      setState(prev => ({ ...prev, status: 'speaking' }));
      // PAUSE recognition while speaking to prevent feedback loop
      if (recognitionRef.current) {
        try { recognitionRef.current.stop(); } catch(e) {}
      }
    };

    utterance.onend = () => {
      setState(prev => ({ ...prev, status: 'idle' }));
      // RESUME recognition after speaking if it was supposed to be active
      if (isWebVoiceActive.current && recognitionRef.current) {
        setTimeout(() => {
          try {
            if (isWebVoiceActive.current) recognitionRef.current.start();
          } catch(e) {}
        }, 250);
      }
    };
    
    window.speechSynthesis.speak(utterance);
  };

  const startVoice = useCallback((username) => {
    if (!socketRef.current) return;

    // Logic: If backend STT/TTS is available (local), use it. 
    // Otherwise, use browser Web Speech API.
    const isLocalVoiceAvailable = systemData?.status?.stt_available && systemData?.status?.tts_available;

    if (isLocalVoiceAvailable) {
      isWebVoiceActive.current = false;
      socketRef.current.emit('start', { username });
    } else {
      if (recognitionRef.current) {
        isWebVoiceActive.current = true;
        setState(prev => ({ ...prev, status: 'listening', response: '' }));
        try {
          recognitionRef.current.start();
        } catch (e) {
          console.warn('Recognition already started');
        }
      } else {
        setState(prev => ({ ...prev, status: 'error', response: 'Browser voice not supported in this browser.' }));
      }
    }
  }, [systemData]);

  const stopVoice = useCallback(() => {
    isWebVoiceActive.current = false;
    if (recognitionRef.current) {
      recognitionRef.current.stop();
    }
    if (window.speechSynthesis) {
      window.speechSynthesis.cancel();
    }
    if (socketRef.current) {
      socketRef.current.emit('stop');
    }
    setState(prev => ({ ...prev, status: 'idle' }));
  }, []);

  const sendText = useCallback((text) => {
    if (socketRef.current) socketRef.current.emit('text_chat', { text });
  }, []);

  const greetUser = useCallback((name, isReturning) => {
    if (socketRef.current) socketRef.current.emit('greet', { name, isReturning });
  }, []);

  return { state, systemData, startVoice, stopVoice, sendText, greetUser };
}
