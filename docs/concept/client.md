# Next.js Client for LangGraph Implementation

**Created**: 2025-05-26  
**Updated**: 2025-05-27

## Overview

This document describes the Next.js client implementation for the LangGraph agent. The client provides a modern chat interface with real-time streaming, thinking pattern display, follow-up suggestions, and thread management.

## Architecture Overview

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Next.js Client │────▶│  API Proxy Route │────▶│ LangGraph Dev   │
│  (React + SSE)  │     │  (/api/langgraph)│     │  (Port 8123)    │
└─────────────────┘     └──────────────────┘     └─────────────────┘
     Browser                Next.js Server           Local/Production
```

## Project Setup

### 1. Dependencies

```json
// package.json dependencies
{
  "dependencies": {
    "next": "14.2.0",
    "react": "^18",
    "react-dom": "^18",
    "@tanstack/react-query": "^5.40.0",
    "zustand": "^4.5.2",
    "clsx": "^2.1.1",
    "tailwind-merge": "^2.3.0",
    "sonner": "^1.4.41",
    "lucide-react": "^0.378.0"
  },
  "devDependencies": {
    "typescript": "^5",
    "@types/react": "^18",
    "@types/node": "^20",
    "tailwindcss": "^3.4.1",
    "postcss": "^8",
    "autoprefixer": "^10.0.1"
  }
}
```

### 2. Actual Project Structure

```
client/
├── src/
│   ├── app/
│   │   ├── api/
│   │   │   └── langgraph/
│   │   │       └── [...path]/
│   │   │           └── route.ts      # API proxy
│   │   ├── layout.tsx
│   │   ├── page.tsx
│   │   └── providers.tsx
│   ├── components/
│   │   └── chat/
│   │       ├── ChatInterface.tsx
│   │       ├── ChatInput.tsx
│   │       ├── ChatPanel.tsx
│   │       ├── MessageBubble.tsx
│   │       ├── MessageList.tsx
│   │       ├── StreamingMessage.tsx
│   │       ├── SuggestionPills.tsx
│   │       ├── ThinkingIndicator.tsx
│   │       └── ThreadSidebar.tsx
│   ├── hooks/
│   │   └── useChat.ts
│   ├── lib/
│   │   └── langgraph-client.ts
│   ├── store/
│   │   └── chat.ts                   # Zustand store
│   └── types/
│       └── index.ts
├── package.json
├── tsconfig.json
└── .env.example
```

## Core Implementation

### 1. Core Types

```typescript
// types/index.ts
export interface Message {
  id: string;
  role: 'human' | 'assistant' | 'system';
  content: string;
  timestamp: string;
  metadata?: Record<string, any>;
}

export interface Thread {
  id: string;
  metadata: {
    title: string;
    created_at: string;
    updated_at?: string;
    [key: string]: any;
  };
  created_at: string;
  updated_at: string;
}

export interface StreamChunk {
  type: 'message' | 'thinking' | 'suggestion' | 'tool_use' | 'error' | 'done';
  content: string;
  metadata?: {
    tool?: string;
    query?: string;
    error_type?: string;
    [key: string]: any;
  };
}```

### 2. LangGraph Client

```typescript
// lib/langgraph-client.ts
import { Thread, Message, StreamChunk } from '@/types';

class LangGraphClient {
  private baseUrl: string;

  constructor(baseUrl: string = process.env.NEXT_PUBLIC_API_ENDPOINT || '/api/langgraph') {
    this.baseUrl = baseUrl;
  }

  async createThread(metadata?: Record<string, any>): Promise<Thread> {
    const response = await fetch(`${this.baseUrl}/threads`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        metadata: {
          title: `Chat ${new Date().toLocaleString()}`,
          created_at: new Date().toISOString(),
          ...metadata
        }
      }),
    });
    
    if (!response.ok) throw new Error('Failed to create thread');
    return response.json();
  }

  async listThreads(): Promise<Thread[]> {
    const response = await fetch(`${this.baseUrl}/threads`);
    if (!response.ok) throw new Error('Failed to list threads');
    return response.json();
  }

  async deleteThread(threadId: string): Promise<void> {
    const response = await fetch(`${this.baseUrl}/threads/${threadId}`, {
      method: 'DELETE',
    });
    if (!response.ok) throw new Error('Failed to delete thread');
  }

  async *streamChat(
    threadId: string,
    message: string
  ): AsyncGenerator<StreamChunk> {
    const response = await fetch(`${this.baseUrl}/threads/${threadId}/runs/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        assistant_id: 'agent',
        input: {
          messages: [{ role: 'human', content: message }],
        },
        stream_mode: 'events',
      }),
    });

    if (!response.ok) {
      throw new Error(`Stream error: ${response.status}`);
    }

    const reader = response.body!.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value);
      const lines = chunk.split('\n');

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            const data = JSON.parse(line.slice(6));
            
            // Validate StreamChunk structure
            if (data && typeof data === 'object' && 'type' in data && 'content' in data) {
              const validTypes = ['message', 'thinking', 'suggestion', 'tool_use', 'error', 'done'];
              if (validTypes.includes(data.type)) {
                yield data as StreamChunk;
              }
            }
          } catch (e) {
            console.error('Failed to parse chunk:', e);
          }
        }
      }
    }
  }

}

// Export singleton instance
export const langgraphClient = new LangGraphClient(
  process.env.NEXT_PUBLIC_API_ENDPOINT || '/api/langgraph'
);
```

### 3. Zustand Store

```typescript
// store/chat.ts
import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { Message, Thread } from '@/types';

interface ChatStore {
  // State
  threads: Thread[];
  currentThreadId: string | null;
  messages: Record<string, Message[]>;
  isStreaming: boolean;
  streamingMessage: string;
  thinkingMessage: string;
  suggestions: string[];
  
  // Actions
  setThreads: (threads: Thread[]) => void;
  addThread: (thread: Thread) => void;
  removeThread: (threadId: string) => void;
  setCurrentThreadId: (threadId: string | null) => void;
  addMessage: (threadId: string, message: Message) => void;
  setMessages: (threadId: string, messages: Message[]) => void;
  setStreaming: (isStreaming: boolean) => void;
  appendToStreamingMessage: (content: string) => void;
  setThinkingMessage: (content: string) => void;
  setSuggestions: (suggestions: string[]) => void;
  clearStreamingMessage: () => void;
}

export const useChatStore = create<ChatStore>()(
  persist(
    (set) => ({
      threads: [],
      currentThreadId: null,
      messages: {},
      isStreaming: false,
      streamingMessage: '',
      thinkingMessage: '',
      suggestions: [],

      setThreads: (threads) => set({ threads }),
      addThread: (thread) => 
        set((state) => ({ threads: [...state.threads, thread] })),
      removeThread: (threadId) =>
        set((state) => ({
          threads: state.threads.filter(t => t.id !== threadId),
          messages: { ...state.messages, [threadId]: undefined },
        })),
      setCurrentThreadId: (threadId) => set({ currentThreadId: threadId }),
      addMessage: (threadId, message) =>
        set((state) => ({
          messages: {
            ...state.messages,
            [threadId]: [...(state.messages[threadId] || []), message],
          },
        })),
      setMessages: (threadId, messages) =>
        set((state) => ({
          messages: { ...state.messages, [threadId]: messages },
        })),
      setStreaming: (isStreaming) => set({ isStreaming }),
      appendToStreamingMessage: (content) =>
        set((state) => ({ streamingMessage: state.streamingMessage + content })),
      setThinkingMessage: (content) => set({ thinkingMessage: content }),
      setSuggestions: (suggestions) => set({ suggestions }),
      clearStreamingMessage: () => set({ streamingMessage: '' }),
    }),
    {
      name: 'langgraph-chat',
      partialize: (state) => ({ 
        threads: state.threads,
        currentThreadId: state.currentThreadId,
        messages: state.messages,
      }),
    }
  )
);
```

### 4. Chat Hook

```typescript
// hooks/useChat.ts
import { useState, useCallback } from 'react';
import { useMutation } from '@tanstack/react-query';
import { langgraphClient } from '@/lib/langgraph-client';
import { useChatStore } from '@/store/chat';
import { toast } from 'sonner';

export function useChat() {
  const {
    currentThreadId,
    threads,
    addThread,
    setCurrentThreadId,
    addMessage,
    setStreaming,
    appendToStreamingMessage,
    clearStreamingMessage,
    setThinkingMessage,
    setSuggestions,
  } = useChatStore();

  const sendMessage = useMutation({
    mutationFn: async (message: string) => {
      // Get or create thread
      let threadId = currentThreadId;
      let thread = threads.find(t => t.id === threadId);
      
      if (!thread) {
        thread = await langgraphClient.createThread();
        addThread(thread);
        setCurrentThreadId(thread.id);
        threadId = thread.id;
      }

      // Add user message
      const userMessage = {
        id: Date.now().toString(),
        role: 'human' as const,
        content: message,
        timestamp: new Date().toISOString(),
      };
      addMessage(threadId, userMessage);

      // Reset streaming state
      setStreaming(true);
      clearStreamingMessage();
      setThinkingMessage('');
      setSuggestions([]);

      let assistantMessage = '';
      const suggestions: string[] = [];

      try {
        for await (const chunk of langgraphClient.streamChat(threadId, message)) {
          switch (chunk.type) {
            case 'thinking':
              setThinkingMessage(chunk.content);
              break;
            
            case 'message':
              assistantMessage += chunk.content;
              appendToStreamingMessage(chunk.content);
              break;
            
            case 'suggestion':
              suggestions.push(chunk.content);
              setSuggestions(suggestions);
              break;
            
            case 'tool_use':
              // Optionally display tool usage
              console.log('Tool usage:', chunk.content);
              break;
              
            case 'error':
              throw new Error(chunk.content);
              
            case 'done':
              // Stream complete
              break;
          }
        }

        // Save complete assistant message
        if (assistantMessage) {
          addMessage(threadId, {
            id: Date.now().toString(),
            role: 'assistant',
            content: assistantMessage,
            timestamp: new Date().toISOString(),
          });
        }

      } catch (error) {
        console.error('Chat error:', error);
        toast.error('Failed to send message');
        throw error;
      } finally {
        setStreaming(false);
        clearStreamingMessage();
        setThinkingMessage('');
      }
    },
  });

  return {
    sendMessage: sendMessage.mutate,
    isLoading: sendMessage.isPending,
  };
}
```

### 5. API Proxy Route

```typescript
// app/api/langgraph/[...path]/route.ts
import { NextRequest } from 'next/server';

const LANGGRAPH_ENDPOINT = process.env.LANGGRAPH_ENDPOINT || 'http://localhost:8123';
const LANGGRAPH_API_KEY = process.env.LANGGRAPH_API_KEY || 'dev-key';

export async function GET(
  request: NextRequest,
  { params }: { params: { path: string[] } }
) {
  const path = params.path.join('/');
  
  // Validate API key
  if (!LANGGRAPH_API_KEY) {
    return Response.json({ error: 'API key not configured' }, { status: 500 });
  }
  
  try {
    const response = await fetch(`${LANGGRAPH_ENDPOINT}/${path}`, {
      headers: {
        'X-API-Key': LANGGRAPH_API_KEY,
      },
    });

    if (!response.ok) {
      const error = await response.text();
      return Response.json({ error }, { status: response.status });
    }

    const data = await response.json();
    return Response.json(data);
  } catch (error) {
    return Response.json({ error: 'Failed to fetch from LangGraph' }, { status: 500 });
  }
}

export async function POST(
  request: NextRequest,
  { params }: { params: { path: string[] } }
) {
  const path = params.path.join('/');
  const body = await request.text();
  
  // Validate API key
  if (!LANGGRAPH_API_KEY) {
    return Response.json({ error: 'API key not configured' }, { status: 500 });
  }
  
  // Handle streaming endpoints
  if (path.includes('/runs/stream')) {
    try {
      const response = await fetch(`${LANGGRAPH_ENDPOINT}/${path}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-API-Key': LANGGRAPH_API_KEY,
        },
        body,
      });

      if (!response.ok) {
        const error = await response.text();
        return Response.json({ error }, { status: response.status });
      }

      // Return SSE stream with CORS headers
      return new Response(response.body, {
        headers: {
          'Content-Type': 'text/event-stream',
          'Cache-Control': 'no-cache',
          'Connection': 'keep-alive',
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
          'Access-Control-Allow-Headers': 'Content-Type',
        },
      });
    } catch (error) {
      return Response.json({ error: 'Failed to stream from LangGraph' }, { status: 500 });
    }
  }
  
  // Handle regular endpoints
  try {
    const response = await fetch(`${LANGGRAPH_ENDPOINT}/${path}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': LANGGRAPH_API_KEY,
      },
      body,
    });

    if (!response.ok) {
      const error = await response.text();
      return Response.json({ error }, { status: response.status });
    }

    const data = await response.json();
    return Response.json(data);
  } catch (error) {
    return Response.json({ error: 'Failed to post to LangGraph' }, { status: 500 });
  }
}
```

## Environment Variables

```bash
# .env.local
# For local development with langgraph dev
NEXT_PUBLIC_LANGGRAPH_API_URL=http://localhost:8123
NEXT_PUBLIC_LANGGRAPH_API_KEY=dev-key

# Server-side variables for API proxy route
LANGGRAPH_ENDPOINT=http://localhost:8123
LANGGRAPH_API_KEY=dev-key

# For production deployment
# NEXT_PUBLIC_LANGGRAPH_API_URL=https://your-deployment-url
# NEXT_PUBLIC_LANGGRAPH_API_KEY=your-actual-api-key
```

## Key Features

### 1. Real-time Streaming
- SSE-based streaming with proper chunk parsing
- Handles thinking, message, tool usage, and suggestion events
- Graceful error handling and timeout protection

### 2. Thread Management
- Persistent thread storage with Zustand
- Automatic thread creation on first message
- Thread switching and deletion support

### 3. UI Components
- Thinking indicator for transparency
- Follow-up suggestion pills
- Streaming message display
- Tool usage notifications

### 4. Type Safety
- Full TypeScript coverage
- Shared types between client and server
- Runtime validation for stream chunks

## Deployment

### Development
```bash
npm install
npm run dev
# Runs on http://localhost:3000
```

### Production
```bash
npm run build
npm start
```

### Vercel Deployment
```bash
vercel --prod
```

The client connects to the LangGraph agent via the API proxy route, providing a secure and efficient way to stream responses while keeping API keys server-side.
