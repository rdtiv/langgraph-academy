'use client';

import { useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useChatStore } from '@/store/chat';
import { langgraphClient } from '@/lib/langgraph-client';
import { format } from 'date-fns';

export function ThreadSidebar() {
  const {
    threads,
    currentThread,
    setThreads,
    setCurrentThread,
    addThread,
    deleteThread,
    setMessages,
  } = useChatStore();

  // Fetch threads on mount
  const { data: fetchedThreads } = useQuery({
    queryKey: ['threads'],
    queryFn: () => langgraphClient.getThreads(),
  });

  useEffect(() => {
    if (fetchedThreads) {
      setThreads(fetchedThreads);
    }
  }, [fetchedThreads, setThreads]);

  const handleNewThread = async () => {
    try {
      const newThread = await langgraphClient.createThread();
      addThread(newThread);
      setCurrentThread(newThread);
      setMessages([]);
    } catch (error) {
      console.error('Failed to create thread:', error);
    }
  };

  const handleSelectThread = (thread: typeof threads[0]) => {
    setCurrentThread(thread);
    // In a real app, you'd fetch messages for this thread
    setMessages([]);
  };

  const handleDeleteThread = async (threadId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    try {
      await langgraphClient.deleteThread(threadId);
      deleteThread(threadId);
    } catch (error) {
      console.error('Failed to delete thread:', error);
    }
  };

  return (
    <div className="h-full bg-muted/50 p-4 flex flex-col">
      {/* New Thread Button */}
      <button
        onClick={handleNewThread}
        className="w-full py-2 px-4 bg-primary text-primary-foreground rounded-md hover:bg-primary/90 transition-colors mb-4"
      >
        New Chat
      </button>

      {/* Thread List */}
      <div className="flex-1 overflow-y-auto custom-scrollbar">
        <div className="space-y-2">
          {threads.map((thread) => (
            <div
              key={thread.id}
              onClick={() => handleSelectThread(thread)}
              className={`
                p-3 rounded-md cursor-pointer transition-colors group
                ${currentThread?.id === thread.id ? 'bg-primary/10' : 'hover:bg-muted'}
              `}
            >
              <div className="flex justify-between items-start">
                <div className="flex-1 min-w-0">
                  <h3 className="font-medium truncate">
                    {thread.metadata.title}
                  </h3>
                  <p className="text-sm text-muted-foreground">
                    {format(new Date(thread.created_at), 'MMM d, h:mm a')}
                  </p>
                </div>
                <button
                  onClick={(e) => handleDeleteThread(thread.id, e)}
                  className="opacity-0 group-hover:opacity-100 p-1 hover:bg-destructive/10 rounded transition-all"
                >
                  <svg
                    className="w-4 h-4 text-destructive"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                    />
                  </svg>
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}