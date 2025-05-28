'use client';

import { Message } from '@/types';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { format } from 'date-fns';

interface MessageBubbleProps {
  message: Message;
}

export function MessageBubble({ message }: MessageBubbleProps) {
  const isHuman = message.role === 'human';

  return (
    <div className={`flex ${isHuman ? 'justify-end' : 'justify-start'}`}>
      <div
        className={`
          max-w-[80%] rounded-lg px-4 py-2
          ${isHuman ? 'bg-primary text-primary-foreground' : 'bg-muted'}
        `}
      >
        {/* Message Content */}
        <div className="prose prose-sm dark:prose-invert max-w-none">
          {isHuman ? (
            <p className="mb-0">{message.content}</p>
          ) : (
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {message.content}
            </ReactMarkdown>
          )}
        </div>

        {/* Timestamp */}
        <div
          className={`
            text-xs mt-1
            ${isHuman ? 'text-primary-foreground/70' : 'text-muted-foreground'}
          `}
        >
          {format(new Date(message.timestamp), 'h:mm a')}
        </div>
      </div>
    </div>
  );
}