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

  useEffect(() => {
    // Vite proxy handles /socket.io
    socketRef.current = io();

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
    };
  }, []);

  const startVoice = useCallback((username) => {
    if (socketRef.current) socketRef.current.emit('start', { username });
  }, []);

  const stopVoice = useCallback(() => {
    if (socketRef.current) socketRef.current.emit('stop');
  }, []);

  const sendText = useCallback((text) => {
    if (socketRef.current) socketRef.current.emit('text_chat', { text });
  }, []);

  const greetUser = useCallback((name, isReturning) => {
    if (socketRef.current) socketRef.current.emit('greet', { name, isReturning });
  }, []);

  return { state, systemData, startVoice, stopVoice, sendText, greetUser };
}
