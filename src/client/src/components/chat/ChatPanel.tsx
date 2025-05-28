'use client';

import { MessageList } from './MessageList';
import { ChatInput } from './ChatInput';
import { SuggestionPills } from './SuggestionPills';
import { useChatStore } from '@/store/chat';

export function ChatPanel() {
  const { currentThread, suggestions } = useChatStore();

  return (
    <div className="flex-1 flex flex-col">
      {/* Messages Area */}
      <div className="flex-1 overflow-hidden">
        <MessageList />
      </div>

      {/* Suggestions */}
      {suggestions.length > 0 && (
        <div className="px-4 py-2 border-t">
          <SuggestionPills suggestions={suggestions} />
        </div>
      )}

      {/* Input Area */}
      <div className="border-t p-4">
        {!currentThread ? (
          <div className="text-center text-muted-foreground">
            Select or create a thread to start chatting
          </div>
        ) : (
          <ChatInput />
        )}
      </div>
    </div>
  );
}