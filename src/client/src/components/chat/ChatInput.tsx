'use client';

import { useState, useRef, useEffect } from 'react';
import { useChat } from '@/hooks/useChat';
import { useChatStore } from '@/store/chat';

export function ChatInput() {
  const [input, setInput] = useState('');
  const { sendMessage, isLoading } = useChat();
  const { isStreaming } = useChatStore();
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const disabled = isLoading || isStreaming || !input.trim();

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!disabled) {
      sendMessage(input.trim());
      setInput('');
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
    }
  }, [input]);

  return (
    <form onSubmit={handleSubmit} className="relative">
      <div className="relative">
        <textarea
          ref={textareaRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Message Claude..."
          disabled={isLoading || isStreaming}
          className="
            w-full px-4 py-3 pr-12
            bg-background border rounded-lg
            resize-none max-h-32 custom-scrollbar
            focus:outline-none focus:ring-2 focus:ring-primary
            disabled:opacity-50 disabled:cursor-not-allowed
          "
          rows={1}
        />
        
        {/* Send Button */}
        <button
          type="submit"
          disabled={disabled}
          className="
            absolute right-2 bottom-2
            p-2 rounded-md
            bg-primary text-primary-foreground
            hover:bg-primary/90
            disabled:opacity-50 disabled:cursor-not-allowed
            transition-all
          "
        >
          {isLoading || isStreaming ? (
            <svg
              className="w-5 h-5 animate-spin"
              fill="none"
              viewBox="0 0 24 24"
            >
              <circle
                className="opacity-25"
                cx="12"
                cy="12"
                r="10"
                stroke="currentColor"
                strokeWidth="4"
              />
              <path
                className="opacity-75"
                fill="currentColor"
                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
              />
            </svg>
          ) : (
            <svg
              className="w-5 h-5"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"
              />
            </svg>
          )}
        </button>
      </div>
      
      {/* Character count */}
      <div className="text-xs text-muted-foreground mt-1 text-right">
        {input.length} / 4000
      </div>
    </form>
  );
}