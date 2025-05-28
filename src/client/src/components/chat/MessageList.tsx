'use client';

import { useEffect, useRef } from 'react';
import { useChatStore } from '@/store/chat';
import { MessageBubble } from './MessageBubble';
import { ThinkingIndicator } from './ThinkingIndicator';
import { StreamingMessage } from './StreamingMessage';

export function MessageList() {
  const { messages, thinkingMessage, streamingMessage, isStreaming } = useChatStore();
  const bottomRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, streamingMessage, thinkingMessage]);

  return (
    <div className="h-full overflow-y-auto custom-scrollbar p-4">
      <div className="max-w-3xl mx-auto space-y-4">
        {/* Welcome message if no messages */}
        {messages.length === 0 && !isStreaming && (
          <div className="text-center py-8">
            <h2 className="text-2xl font-semibold mb-2">Welcome to LangGraph Chat</h2>
            <p className="text-muted-foreground">
              Start a conversation with Claude powered by LangGraph
            </p>
          </div>
        )}

        {/* Message List */}
        {messages.map((message) => (
          <MessageBubble key={message.id} message={message} />
        ))}

        {/* Thinking Indicator */}
        {thinkingMessage && <ThinkingIndicator content={thinkingMessage} />}

        {/* Streaming Message */}
        {streamingMessage && <StreamingMessage content={streamingMessage} />}

        {/* Scroll anchor */}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}