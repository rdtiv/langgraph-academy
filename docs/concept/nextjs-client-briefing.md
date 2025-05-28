# Next.js Client for LangGraph: Implementation Briefing

**Created**: 2025-05-26  
**Last Modified**: 2025-05-26

## Overview

This briefing outlines how to build a production-ready Next.js application that serves as a client for your LangGraph deployment. The application will provide a ChatGPT-like interface with streaming messages, thinking indicators, smart suggestions, and thread management.

## Architecture Overview

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│                 │     │                  │     │                 │
│  Next.js App    │────▶│  Vercel Edge     │────▶│  LangGraph      │
│  (React UI)     │     │  Functions       │     │  Server         │
│                 │     │                  │     │                 │
└─────────────────┘     └──────────────────┘     └─────────────────┘
     Browser                API Routes              Your Backend
```

## Project Setup

### 1. Initialize Next.js Project

```bash
npx create-next-app@latest langgraph-chat --typescript --tailwind --app
cd langgraph-chat

# Install additional dependencies
npm install @tanstack/react-query axios clsx date-fns react-markdown 
npm install react-syntax-highlighter zustand sonner lucide-react
npm install @radix-ui/react-scroll-area @radix-ui/react-tooltip
npm install --save-dev @types/react-syntax-highlighter
```

### 2. Project Structure

```
langgraph-chat/
├── app/
│   ├── layout.tsx
│   ├── page.tsx
│   ├── api/
│   │   ├── threads/
│   │   │   ├── route.ts
│   │   │   └── [threadId]/
│   │   │       ├── route.ts
│   │   │       └── messages/route.ts
│   │   └── chat/route.ts
│   └── globals.css
├── components/
│   ├── chat/
│   │   ├── ChatInterface.tsx
│   │   ├── MessageList.tsx
│   │   ├── MessageItem.tsx
│   │   ├── ChatInput.tsx
│   │   ├── ThinkingIndicator.tsx
│   │   └── SuggestionsBar.tsx
│   ├── sidebar/
│   │   ├── ThreadList.tsx
│   │   ├── ThreadItem.tsx
│   │   └── NewThreadButton.tsx
│   └── ui/
│       └── (shadcn components)
├── lib/
│   ├── langgraph-client.ts
│   ├── hooks/
│   │   ├── useChat.ts
│   │   ├── useThreads.ts
│   │   └── useStreaming.ts
│   └── stores/
│       ├── chatStore.ts
│       └── threadStore.ts
├── types/
│   └── index.ts
└── utils/
    └── stream-parser.ts
```

## Core Implementation

### 1. LangGraph Client Wrapper

```typescript
// lib/langgraph-client.ts
import axios, { AxiosInstance } from 'axios';

export interface Thread {
  thread_id: string;
  created_at: string;
  metadata: {
    title?: string;
    last_message?: string;
  };
}

export interface Message {
  id: string;
  role: 'human' | 'assistant' | 'system' | 'thinking';
  content: string;
  timestamp: string;
  metadata?: Record<string, any>;
}

export interface StreamChunk {
  type: 'message' | 'thinking' | 'suggestion' | 'error';
  content: string;
  metadata?: Record<string, any>;
}

class LangGraphClient {
  private client: AxiosInstance;
  private baseURL: string;

  constructor(baseURL: string = process.env.NEXT_PUBLIC_LANGGRAPH_URL!) {
    this.baseURL = baseURL;
    this.client = axios.create({
      baseURL,
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': process.env.NEXT_PUBLIC_LANGGRAPH_API_KEY,
      },
    });
  }

  // Thread Management
  async createThread(metadata?: Record<string, any>): Promise<Thread> {
    const response = await this.client.post('/threads', { metadata });
    return response.data;
  }

  async getThreads(): Promise<Thread[]> {
    const response = await this.client.get('/threads');
    return response.data;
  }

  async getThread(threadId: string): Promise<Thread> {
    const response = await this.client.get(`/threads/${threadId}`);
    return response.data;
  }

  async deleteThread(threadId: string): Promise<void> {
    await this.client.delete(`/threads/${threadId}`);
  }

  // Streaming Chat
  async *streamChat(
    threadId: string,
    message: string,
    config?: Record<string, any>
  ): AsyncGenerator<StreamChunk> {
    const response = await fetch(`${this.baseURL}/runs/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': process.env.NEXT_PUBLIC_LANGGRAPH_API_KEY!,
      },
      body: JSON.stringify({
        thread_id: threadId,
        assistant_id: 'task_maistro',
        input: {
          messages: [{ role: 'human', content: message }],
        },
        config: {
          ...config,
          stream_mode: 'messages-tuple',
        },
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const reader = response.body!.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6);
          if (data === '[DONE]') return;
          
          try {
            const chunk = JSON.parse(data);
            yield this.parseChunk(chunk);
          } catch (e) {
            console.error('Failed to parse chunk:', e);
          }
        }
      }
    }
  }

  private parseChunk(chunk: any): StreamChunk {
    // Parse LangGraph streaming format
    if (chunk.type === 'message_chunk') {
      return {
        type: 'message',
        content: chunk.content,
        metadata: chunk.metadata,
      };
    } else if (chunk.type === 'thinking') {
      return {
        type: 'thinking',
        content: chunk.content,
      };
    } else if (chunk.type === 'suggestion') {
      return {
        type: 'suggestion',
        content: chunk.content,
      };
    }
    
    return {
      type: 'message',
      content: chunk.content || '',
    };
  }
}

export const langgraphClient = new LangGraphClient();
```

### 2. Chat Store with Zustand

```typescript
// lib/stores/chatStore.ts
import { create } from 'zustand';
import { Message, Thread } from '@/lib/langgraph-client';

interface ChatState {
  currentThread: Thread | null;
  messages: Message[];
  isStreaming: boolean;
  streamingMessage: string;
  thinkingMessage: string;
  suggestions: string[];
  
  // Actions
  setCurrentThread: (thread: Thread | null) => void;
  addMessage: (message: Message) => void;
  setMessages: (messages: Message[]) => void;
  setStreaming: (isStreaming: boolean) => void;
  appendToStreamingMessage: (content: string) => void;
  setThinkingMessage: (content: string) => void;
  setSuggestions: (suggestions: string[]) => void;
  clearStreamingMessage: () => void;
}

export const useChatStore = create<ChatState>((set) => ({
  currentThread: null,
  messages: [],
  isStreaming: false,
  streamingMessage: '',
  thinkingMessage: '',
  suggestions: [],

  setCurrentThread: (thread) => set({ currentThread: thread }),
  addMessage: (message) => 
    set((state) => ({ messages: [...state.messages, message] })),
  setMessages: (messages) => set({ messages }),
  setStreaming: (isStreaming) => set({ isStreaming }),
  appendToStreamingMessage: (content) =>
    set((state) => ({ streamingMessage: state.streamingMessage + content })),
  setThinkingMessage: (content) => set({ thinkingMessage: content }),
  setSuggestions: (suggestions) => set({ suggestions }),
  clearStreamingMessage: () => set({ streamingMessage: '' }),
}));
```

### 3. Streaming Hook

```typescript
// lib/hooks/useChat.ts
import { useMutation } from '@tanstack/react-query';
import { langgraphClient } from '@/lib/langgraph-client';
import { useChatStore } from '@/lib/stores/chatStore';
import { toast } from 'sonner';

export function useChat() {
  const {
    currentThread,
    addMessage,
    setStreaming,
    appendToStreamingMessage,
    clearStreamingMessage,
    setThinkingMessage,
    setSuggestions,
  } = useChatStore();

  const sendMessage = useMutation({
    mutationFn: async (message: string) => {
      if (!currentThread) {
        throw new Error('No thread selected');
      }

      // Add user message immediately
      const userMessage = {
        id: Date.now().toString(),
        role: 'human' as const,
        content: message,
        timestamp: new Date().toISOString(),
      };
      addMessage(userMessage);

      // Start streaming
      setStreaming(true);
      clearStreamingMessage();
      setThinkingMessage('');
      setSuggestions([]);

      let assistantMessage = '';
      const suggestions: string[] = [];

      try {
        for await (const chunk of langgraphClient.streamChat(
          currentThread.thread_id,
          message
        )) {
          switch (chunk.type) {
            case 'thinking':
              setThinkingMessage(chunk.content);
              break;
            
            case 'message':
              appendToStreamingMessage(chunk.content);
              assistantMessage += chunk.content;
              break;
            
            case 'suggestion':
              suggestions.push(chunk.content);
              setSuggestions(suggestions);
              break;
            
            case 'error':
              throw new Error(chunk.content);
          }
        }

        // Add complete assistant message
        addMessage({
          id: Date.now().toString(),
          role: 'assistant',
          content: assistantMessage,
          timestamp: new Date().toISOString(),
        });

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

### 4. Main Chat Interface

```typescript
// components/chat/ChatInterface.tsx
'use client';

import { useState } from 'react';
import { MessageList } from './MessageList';
import { ChatInput } from './ChatInput';
import { ThinkingIndicator } from './ThinkingIndicator';
import { SuggestionsBar } from './SuggestionsBar';
import { useChatStore } from '@/lib/stores/chatStore';
import { useChat } from '@/lib/hooks/useChat';
import { ScrollArea } from '@/components/ui/scroll-area';

export function ChatInterface() {
  const { 
    messages, 
    isStreaming, 
    streamingMessage, 
    thinkingMessage,
    suggestions,
    currentThread 
  } = useChatStore();
  const { sendMessage, isLoading } = useChat();
  const [input, setInput] = useState('');

  const handleSend = () => {
    if (input.trim() && !isLoading) {
      sendMessage(input);
      setInput('');
    }
  };

  const handleSuggestionClick = (suggestion: string) => {
    setInput(suggestion);
  };

  if (!currentThread) {
    return (
      <div className="flex h-full items-center justify-center">
        <p className="text-muted-foreground">
          Select a thread or create a new one to start chatting
        </p>
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col">
      {/* Messages Area */}
      <ScrollArea className="flex-1 p-4">
        <MessageList 
          messages={messages}
          streamingMessage={streamingMessage}
          isStreaming={isStreaming}
        />
        {thinkingMessage && (
          <ThinkingIndicator message={thinkingMessage} />
        )}
      </ScrollArea>

      {/* Suggestions */}
      {suggestions.length > 0 && (
        <SuggestionsBar 
          suggestions={suggestions}
          onSuggestionClick={handleSuggestionClick}
        />
      )}

      {/* Input Area */}
      <div className="border-t p-4">
        <ChatInput
          value={input}
          onChange={setInput}
          onSend={handleSend}
          disabled={isLoading}
          placeholder="Type a message..."
        />
      </div>
    </div>
  );
}
```

### 5. Message Display Component

```typescript
// components/chat/MessageItem.tsx
import { Message } from '@/lib/langgraph-client';
import { cn } from '@/lib/utils';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { User, Bot } from 'lucide-react';

interface MessageItemProps {
  message: Message;
  isStreaming?: boolean;
}

export function MessageItem({ message, isStreaming }: MessageItemProps) {
  const isHuman = message.role === 'human';

  return (
    <div className={cn(
      'group flex gap-3 py-4',
      isHuman ? 'flex-row-reverse' : 'flex-row'
    )}>
      {/* Avatar */}
      <div className={cn(
        'flex h-8 w-8 shrink-0 items-center justify-center rounded-full',
        isHuman ? 'bg-primary text-primary-foreground' : 'bg-muted'
      )}>
        {isHuman ? <User size={16} /> : <Bot size={16} />}
      </div>

      {/* Message Content */}
      <div className={cn(
        'flex-1 space-y-2',
        isHuman ? 'text-right' : 'text-left'
      )}>
        <div className={cn(
          'inline-block rounded-lg px-4 py-2',
          isHuman 
            ? 'bg-primary text-primary-foreground' 
            : 'bg-muted'
        )}>
          <ReactMarkdown
            className="prose prose-sm dark:prose-invert max-w-none"
            components={{
              code({ node, inline, className, children, ...props }) {
                const match = /language-(\w+)/.exec(className || '');
                return !inline && match ? (
                  <SyntaxHighlighter
                    style={oneDark}
                    language={match[1]}
                    PreTag="div"
                    {...props}
                  >
                    {String(children).replace(/\n$/, '')}
                  </SyntaxHighlighter>
                ) : (
                  <code className={className} {...props}>
                    {children}
                  </code>
                );
              },
            }}
          >
            {message.content}
          </ReactMarkdown>
          {isStreaming && (
            <span className="inline-block h-4 w-1 animate-pulse bg-current ml-1" />
          )}
        </div>
        <p className="text-xs text-muted-foreground">
          {new Date(message.timestamp).toLocaleTimeString()}
        </p>
      </div>
    </div>
  );
}
```

### 6. Thread Sidebar

```typescript
// components/sidebar/ThreadList.tsx
'use client';

import { useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { langgraphClient } from '@/lib/langgraph-client';
import { useChatStore } from '@/lib/stores/chatStore';
import { ThreadItem } from './ThreadItem';
import { NewThreadButton } from './NewThreadButton';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Skeleton } from '@/components/ui/skeleton';

export function ThreadList() {
  const { currentThread, setCurrentThread } = useChatStore();
  
  const { data: threads, isLoading } = useQuery({
    queryKey: ['threads'],
    queryFn: () => langgraphClient.getThreads(),
    refetchInterval: 30000, // Refresh every 30s
  });

  // Auto-select first thread if none selected
  useEffect(() => {
    if (!currentThread && threads && threads.length > 0) {
      setCurrentThread(threads[0]);
    }
  }, [threads, currentThread, setCurrentThread]);

  return (
    <div className="flex h-full flex-col">
      <div className="p-4">
        <NewThreadButton />
      </div>
      
      <ScrollArea className="flex-1">
        <div className="space-y-2 p-4 pt-0">
          {isLoading ? (
            <>
              <Skeleton className="h-16 w-full" />
              <Skeleton className="h-16 w-full" />
              <Skeleton className="h-16 w-full" />
            </>
          ) : (
            threads?.map((thread) => (
              <ThreadItem
                key={thread.thread_id}
                thread={thread}
                isActive={currentThread?.thread_id === thread.thread_id}
                onClick={() => setCurrentThread(thread)}
              />
            ))
          )}
        </div>
      </ScrollArea>
    </div>
  );
}
```

### 7. API Routes

```typescript
// app/api/chat/route.ts
import { NextRequest } from 'next/server';

const LANGGRAPH_URL = process.env.LANGGRAPH_URL!;
const LANGGRAPH_API_KEY = process.env.LANGGRAPH_API_KEY!;

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    
    // Forward to LangGraph server
    const response = await fetch(`${LANGGRAPH_URL}/runs/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': LANGGRAPH_API_KEY,
      },
      body: JSON.stringify(body),
    });

    // Return streaming response
    return new Response(response.body, {
      headers: {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
      },
    });
  } catch (error) {
    console.error('Chat API error:', error);
    return Response.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
```

### 8. Main Layout

```typescript
// app/layout.tsx
import { Inter } from 'next/font/google';
import { Providers } from './providers';
import { ThreadList } from '@/components/sidebar/ThreadList';
import { Toaster } from 'sonner';
import './globals.css';

const inter = Inter({ subsets: ['latin'] });

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="h-full">
      <body className={`${inter.className} h-full`}>
        <Providers>
          <div className="flex h-full">
            {/* Sidebar */}
            <aside className="w-64 border-r bg-muted/50">
              <ThreadList />
            </aside>
            
            {/* Main Content */}
            <main className="flex-1">
              {children}
            </main>
          </div>
          <Toaster position="top-right" />
        </Providers>
      </body>
    </html>
  );
}
```

## Deployment to Vercel

### 1. Environment Variables

Create `.env.local` for development:

```env
NEXT_PUBLIC_LANGGRAPH_URL=http://localhost:8123
NEXT_PUBLIC_LANGGRAPH_API_KEY=your-api-key
LANGGRAPH_URL=http://localhost:8123
LANGGRAPH_API_KEY=your-api-key
```

### 2. Vercel Configuration

```json
// vercel.json
{
  "functions": {
    "app/api/chat/route.ts": {
      "maxDuration": 60
    }
  },
  "env": {
    "LANGGRAPH_URL": "@langgraph-url",
    "LANGGRAPH_API_KEY": "@langgraph-api-key"
  }
}
```

### 3. Deploy to Vercel

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel

# Set environment variables
vercel env add LANGGRAPH_URL
vercel env add LANGGRAPH_API_KEY
vercel env add NEXT_PUBLIC_LANGGRAPH_URL
vercel env add NEXT_PUBLIC_LANGGRAPH_API_KEY
```

## Key Features Implementation

### 1. Thinking Messages

Show LLM's reasoning process:

```typescript
// components/chat/ThinkingIndicator.tsx
import { Brain } from 'lucide-react';

export function ThinkingIndicator({ message }: { message: string }) {
  return (
    <div className="flex items-start gap-2 py-2 text-muted-foreground">
      <Brain className="mt-1 h-4 w-4 animate-pulse" />
      <div className="flex-1 italic text-sm">
        {message || 'Thinking...'}
      </div>
    </div>
  );
}
```

### 2. Smart Suggestions

```typescript
// components/chat/SuggestionsBar.tsx
import { ArrowRight } from 'lucide-react';

interface SuggestionsBarProps {
  suggestions: string[];
  onSuggestionClick: (suggestion: string) => void;
}

export function SuggestionsBar({ 
  suggestions, 
  onSuggestionClick 
}: SuggestionsBarProps) {
  return (
    <div className="border-t bg-muted/50 p-4">
      <p className="mb-2 text-sm text-muted-foreground">
        Suggested follow-ups:
      </p>
      <div className="flex flex-wrap gap-2">
        {suggestions.map((suggestion, index) => (
          <button
            key={index}
            onClick={() => onSuggestionClick(suggestion)}
            className="flex items-center gap-1 rounded-full bg-background px-3 py-1 text-sm hover:bg-accent"
          >
            {suggestion}
            <ArrowRight className="h-3 w-3" />
          </button>
        ))}
      </div>
    </div>
  );
}
```

### 3. Streaming with Abort

```typescript
// lib/hooks/useChat.ts (enhanced)
export function useChat() {
  const abortControllerRef = useRef<AbortController | null>(null);

  const abort = () => {
    abortControllerRef.current?.abort();
  };

  const sendMessage = useMutation({
    mutationFn: async (message: string) => {
      abortControllerRef.current = new AbortController();
      
      // Pass signal to fetch
      const response = await fetch('/api/chat', {
        method: 'POST',
        signal: abortControllerRef.current.signal,
        // ... rest of config
      });
      
      // ... streaming logic
    },
  });

  return { sendMessage, abort, isLoading };
}
```

## Performance Optimizations

### 1. Message Virtualization

For long conversations:

```bash
npm install @tanstack/react-virtual
```

```typescript
// components/chat/VirtualMessageList.tsx
import { useVirtualizer } from '@tanstack/react-virtual';

export function VirtualMessageList({ messages }: { messages: Message[] }) {
  const parentRef = useRef<HTMLDivElement>(null);
  
  const virtualizer = useVirtualizer({
    count: messages.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 100,
    overscan: 5,
  });

  return (
    <div ref={parentRef} className="h-full overflow-auto">
      <div
        style={{
          height: `${virtualizer.getTotalSize()}px`,
          width: '100%',
          position: 'relative',
        }}
      >
        {virtualizer.getVirtualItems().map((virtualItem) => (
          <div
            key={virtualItem.key}
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              width: '100%',
              height: `${virtualItem.size}px`,
              transform: `translateY(${virtualItem.start}px)`,
            }}
          >
            <MessageItem message={messages[virtualItem.index]} />
          </div>
        ))}
      </div>
    </div>
  );
}
```

### 2. Optimistic Updates

```typescript
// Immediately show user message while waiting for response
const optimisticUpdate = () => {
  const tempId = `temp-${Date.now()}`;
  addMessage({
    id: tempId,
    role: 'human',
    content: input,
    timestamp: new Date().toISOString(),
  });
  
  // Remove temp message when real one arrives
  // ...
};
```

## Security Considerations

### 1. API Route Protection

```typescript
// middleware.ts
import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

export function middleware(request: NextRequest) {
  // Check for API key in production
  if (process.env.NODE_ENV === 'production') {
    const apiKey = request.headers.get('x-api-key');
    if (!apiKey || apiKey !== process.env.API_SECRET) {
      return NextResponse.json(
        { error: 'Unauthorized' },
        { status: 401 }
      );
    }
  }
  
  return NextResponse.next();
}

export const config = {
  matcher: '/api/:path*',
};
```

### 2. Rate Limiting

```typescript
// lib/rate-limit.ts
import { Ratelimit } from '@upstash/ratelimit';
import { Redis } from '@upstash/redis';

const redis = new Redis({
  url: process.env.UPSTASH_REDIS_REST_URL!,
  token: process.env.UPSTASH_REDIS_REST_TOKEN!,
});

export const ratelimit = new Ratelimit({
  redis,
  limiter: Ratelimit.slidingWindow(10, '1 m'), // 10 requests per minute
});
```

## Summary

This Next.js application provides:

1. **Real-time Streaming**: Server-sent events for smooth message streaming
2. **Thread Management**: Sidebar with conversation history
3. **Rich UI**: Markdown rendering, syntax highlighting, thinking indicators
4. **Smart Suggestions**: Context-aware follow-up prompts
5. **Production Ready**: Error handling, loading states, optimistic updates
6. **Vercel Optimized**: Edge functions, proper caching, environment management

The architecture separates concerns cleanly, uses modern React patterns (hooks, server components), and integrates seamlessly with your LangGraph backend. The UI follows ChatGPT/Claude patterns that users are familiar with while adding LangGraph-specific features like thinking messages and suggestions.