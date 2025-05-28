'use client';

import { useChat } from '@/hooks/useChat';

interface SuggestionPillsProps {
  suggestions: string[];
}

export function SuggestionPills({ suggestions }: SuggestionPillsProps) {
  const { sendMessage } = useChat();

  return (
    <div className="flex flex-wrap gap-2">
      {suggestions.map((suggestion, index) => (
        <button
          key={index}
          onClick={() => sendMessage(suggestion)}
          className="
            px-3 py-1.5 text-sm
            bg-secondary hover:bg-secondary/80
            rounded-full transition-colors
            border border-border
          "
        >
          {suggestion}
        </button>
      ))}
    </div>
  );
}