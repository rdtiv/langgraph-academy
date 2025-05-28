/**
 * Core types for the LangGraph chat application
 */

/**
 * Message roles in the conversation
 */
export type MessageRole = 'human' | 'assistant' | 'system';

/**
 * Represents a message in the conversation
 */
export interface Message {
  id: string;
  role: MessageRole;
  content: string;
  timestamp: string;
  metadata?: Record<string, any>;
}

/**
 * Represents a conversation thread
 */
export interface Thread {
  id: string;
  thread_id?: string; // For backward compatibility
  metadata: {
    title: string;
    created_at: string;
    updated_at?: string;
    [key: string]: any;
  };
  created_at: string;
  updated_at: string;
}

/**
 * Streaming chunk format matching agent output
 */
export interface StreamChunk {
  type: 'message' | 'thinking' | 'suggestion' | 'tool_use' | 'error' | 'done';
  content: string;
  metadata?: {
    tool?: string;
    query?: string;
    error_type?: string;
    [key: string]: any;
  };
}

/**
 * Configuration for the chat interface
 */
export interface ChatConfig {
  streaming: {
    timeout: number;
    reconnectAttempts: number;
    bufferSize: number;
  };
  ui: {
    thinkingDisplay: boolean;
    suggestionsCount: number;
    messageGrouping: boolean;
  };
}

/**
 * Agent configuration passed to the server
 */
export interface AgentConfig {
  thread_id: string;
  temperature?: number;
  max_iterations?: number;
  thinking_visible?: boolean;
  search_depth?: 'basic' | 'advanced';
}