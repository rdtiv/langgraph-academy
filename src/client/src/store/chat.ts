import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import { Message, Thread } from '@/types';

interface ChatState {
  // Thread Management
  threads: Thread[];
  currentThread: Thread | null;
  
  // Message Management
  messages: Message[];
  isStreaming: boolean;
  streamingMessage: string;
  thinkingMessage: string;
  
  // UI State
  suggestions: string[];
  
  // Actions
  setThreads: (threads: Thread[]) => void;
  setCurrentThread: (thread: Thread | null) => void;
  addThread: (thread: Thread) => void;
  deleteThread: (threadId: string) => void;
  
  addMessage: (message: Message) => void;
  setMessages: (messages: Message[]) => void;
  clearMessages: () => void;
  
  setStreaming: (isStreaming: boolean) => void;
  appendToStreamingMessage: (content: string) => void;
  clearStreamingMessage: () => void;
  setThinkingMessage: (content: string) => void;
  
  setSuggestions: (suggestions: string[]) => void;
  clearSuggestions: () => void;
  
  reset: () => void;
}

const initialState = {
  threads: [],
  currentThread: null,
  messages: [],
  isStreaming: false,
  streamingMessage: '',
  thinkingMessage: '',
  suggestions: [],
};

export const useChatStore = create<ChatState>()(
  persist(
    (set, get) => ({
      ...initialState,
      
      // Thread Management
      setThreads: (threads) => set({ threads }),
      setCurrentThread: (thread) => set({ currentThread: thread }),
      addThread: (thread) => set((state) => ({ 
        threads: [thread, ...state.threads] 
      })),
      deleteThread: (threadId) => set((state) => ({
        threads: state.threads.filter(t => t.id !== threadId),
        currentThread: state.currentThread?.id === threadId ? null : state.currentThread,
      })),
      
      // Message Management
      addMessage: (message) => set((state) => ({ 
        messages: [...state.messages, message] 
      })),
      setMessages: (messages) => set({ messages }),
      clearMessages: () => set({ messages: [] }),
      
      // Streaming State
      setStreaming: (isStreaming) => set({ isStreaming }),
      appendToStreamingMessage: (content) => set((state) => ({ 
        streamingMessage: state.streamingMessage + content 
      })),
      clearStreamingMessage: () => set({ streamingMessage: '' }),
      setThinkingMessage: (content) => set({ thinkingMessage: content }),
      
      // Suggestions
      setSuggestions: (suggestions) => set({ suggestions }),
      clearSuggestions: () => set({ suggestions: [] }),
      
      // Reset
      reset: () => set(initialState),
    }),
    {
      name: 'langgraph-chat-storage',
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({
        threads: state.threads,
        currentThread: state.currentThread,
      }),
    }
  )
);