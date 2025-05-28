import { useCallback } from 'react';
import { useMutation } from '@tanstack/react-query';
import { useChatStore } from '@/store/chat';
import { langgraphClient } from '@/lib/langgraph-client';
import { toast } from 'sonner';

export function useChat() {
  const {
    currentThread,
    setCurrentThread,
    addMessage,
    setStreaming,
    appendToStreamingMessage,
    clearStreamingMessage,
    setThinkingMessage,
    setSuggestions,
  } = useChatStore();

  const sendMessage = useMutation({
    mutationFn: async (message: string) => {
      // Create thread if needed
      let threadToUse = currentThread;
      if (!threadToUse) {
        try {
          const newThread = await langgraphClient.createThread();
          setCurrentThread(newThread);
          threadToUse = newThread;
        } catch (error) {
          throw new Error('Failed to create thread');
        }
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
          threadToUse.thread_id || threadToUse.id,
          message
        )) {
          switch (chunk.type) {
            case 'thinking':
              setThinkingMessage(chunk.content);
              break;
            
            case 'message':
              assistantMessage += chunk.content;
              appendToStreamingMessage(chunk.content);
              break;
            
            case 'tool_use':
              appendToStreamingMessage(`\n🔍 ${chunk.content}\n`);
              break;
            
            case 'suggestion':
              suggestions.push(chunk.content);
              setSuggestions(suggestions);
              break;
            
            case 'error':
              throw new Error(chunk.content);
              
            case 'done':
              // Streaming complete
              break;
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
    onError: (error) => {
      console.error('Send message error:', error);
      toast.error('Failed to send message. Please try again.');
    },
  });

  return {
    sendMessage: sendMessage.mutate,
    isLoading: sendMessage.isPending,
  };
}