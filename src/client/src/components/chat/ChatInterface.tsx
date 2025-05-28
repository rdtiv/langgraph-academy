'use client';

import { useState } from 'react';
import { ThreadSidebar } from './ThreadSidebar';
import { ChatPanel } from './ChatPanel';

export function ChatInterface() {
  const [sidebarOpen, setSidebarOpen] = useState(true);

  return (
    <div className="flex h-full">
      {/* Sidebar */}
      <div className={`${sidebarOpen ? 'w-64' : 'w-0'} transition-all duration-300 overflow-hidden`}>
        <ThreadSidebar />
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <div className="h-14 border-b flex items-center px-4">
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="p-2 hover:bg-muted rounded-md"
          >
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
                d="M4 6h16M4 12h16M4 18h16"
              />
            </svg>
          </button>
          <h1 className="ml-4 font-semibold">LangGraph Chat</h1>
        </div>

        {/* Chat Panel */}
        <ChatPanel />
      </div>
    </div>
  );
}