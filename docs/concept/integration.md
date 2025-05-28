# LangGraph Agent + Next.js Client Integration Guide

## Overview

This guide provides the definitive integration approach for connecting the ReAct LangGraph agent with the Next.js client. It ensures perfect synchronization between the agent's streaming output and the client's real-time UI updates.

## Architecture Overview

```
┌─────────────────┐     SSE Stream      ┌──────────────────┐
│                 │ ◄─────────────────── │                  │
│   Next.js       │                      │  LangGraph       │
│   Client        │ ────────────────────►│  Agent           │
│                 │    HTTP Requests     │                  │
└─────────────────┘                      └──────────────────┘
       │                                          │
       │                                          │
       ▼                                          ▼
┌─────────────────┐                      ┌──────────────────┐
│  Zustand Store  │                      │  Claude Sonnet 4 │
│  (UI State)     │                      │  + Tavily Search │
└─────────────────┘                      └──────────────────┘
```

## Environment Configuration

### Agent Environment Variables
```bash
# LangGraph Agent (.env)
ANTHROPIC_API_KEY=your_anthropic_key
TAVILY_API_KEY=your_tavily_key
LANGCHAIN_API_KEY=your_langchain_key
ANTHROPIC_MODEL=claude-sonnet-4-20250514  # Latest Sonnet
```

### Client Environment Variables
```bash
# Next.js Client (.env.local)
NEXT_PUBLIC_LANGGRAPH_API_KEY=your_langgraph_key
NEXT_PUBLIC_API_ENDPOINT=https://your-deployment.langgraph.com
```

## Streaming Event Contract

The agent and client communicate using Server-Sent Events (SSE) with a strict chunk format:

```typescript
// Shared chunk interface between agent and client
interface StreamChunk {
  type: 'thinking' | 'message' | 'tool_use' | 'suggestion' | 'error' | 'done';
  content: string;
  metadata?: {
    tool?: string;        // For tool_use: 'tavily_search'
    query?: string;       // For tool_use: search query
    error_type?: string;  // For error: exception class name
    [key: string]: any;   // Additional metadata
  };
}
```

## Agent Implementation Details

### Core Stream Function
```python
async def stream_events(app, thread_id: str, messages: List[BaseMessage]) -> AsyncIterator[Dict[str, Any]]:
    """Stream events with proper formatting for Next.js client.
    
    Args:
        app: The compiled LangGraph application
        thread_id: Unique identifier for the conversation thread
        messages: List of messages in the conversation
        
    Yields:
        Dict containing:
            - type: 'thinking' | 'message' | 'tool_use' | 'suggestion' | 'error' | 'done'
            - content: The actual content to display
            - metadata: Optional metadata (tool name, query, etc.)
    """
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        async for event in app.astream_events(
            {"messages": messages},
            config=config,
            version="v2"
        ):
            if event["event"] == "on_chat_model_stream":
                # Extract and stream thinking patterns
                if "claude-sonnet-4" in str(event.get("metadata", {}).get("model", "")):
                    content = event["data"]["chunk"].content
                    if content:
                        # Check for thinking pattern
                        thinking_match = re.search(r'<thinking>(.*?)</thinking>', content, re.DOTALL)
                        if thinking_match:
                            yield {
                                "type": "thinking",
                                "content": thinking_match.group(1).strip()
                            }
                        
                        # Stream regular content (excluding thinking)
                        cleaned_content = re.sub(r'<thinking>.*?</thinking>', '', content, flags=re.DOTALL)
                        if cleaned_content.strip():
                            yield {
                                "type": "message",
                                "content": cleaned_content
                            }
            
            elif event["event"] == "on_tool_start":
                # Stream tool usage
                tool_name = event["metadata"].get("tool_name", "unknown")
                if tool_name == "tavily_search_results":
                    tool_input = event["data"].get("input", {})
                    yield {
                        "type": "tool_use",
                        "content": f"Searching the web for: {tool_input.get('query', 'information')}",
                        "metadata": {
                            "tool": "tavily_search",
                            "query": tool_input.get("query", "")
                        }
                    }
        
        # Generate suggestions after message completes
        if messages:
            suggestions = await generate_suggestions(messages)
            for suggestion in suggestions:
                yield {
                    "type": "suggestion",
                    "content": suggestion
                }
        
        # Send done signal after all events
        yield {"type": "done", "content": ""}
        
    except Exception as e:
        yield {
            "type": "error",
            "content": str(e),
            "metadata": {"error_type": type(e).__name__}
        }
```

### Key Agent Features
1. **Thinking Extraction**: Uses regex to extract `<thinking>` tags from Claude's response
2. **Selective Tool Use**: Only invokes Tavily when web search is needed
3. **Suggestion Generation**: Uses Claude Haiku (`claude-3-5-haiku-latest`) for lightweight suggestions
4. **Error Handling**: Comprehensive try-catch with typed error responses

## Client Implementation Details

### API Client with Authentication
```typescript
class LangGraphClient {
  private client: AxiosInstance;

  constructor(baseURL: string = '/api/langgraph') {
    this.baseURL = baseURL;
    this.client = axios.create({
      baseURL,
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': process.env.NEXT_PUBLIC_LANGGRAPH_API_KEY || '',
      },
    });
  }

  // Stream chat with proper chunk parsing
  async *streamChat(threadId: string, message: string): AsyncGenerator<StreamChunk> {
    const response = await fetch(`${this.baseURL}/threads/${threadId}/runs/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': process.env.NEXT_PUBLIC_LANGGRAPH_API_KEY || '',
      },
      body: JSON.stringify({ message }),
    });

    const reader = response.body?.getReader();
    const decoder = new TextDecoder();

    if (!reader) throw new Error('No response body');

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value);
      const lines = chunk.split('\n');

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            const data = JSON.parse(line.slice(6));
            yield data as StreamChunk;
          } catch (e) {
            console.error('Failed to parse chunk:', e);
          }
        }
      }
    }
  }
}
```

### React Hook with Thread Management
```typescript
export function useChat() {
  const {
    currentThread,
    setCurrentThread,
    addMessage,
    setStreaming,
    // ... other store methods
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

      // Add user message
      const userMessage = {
        id: Date.now().toString(),
        role: 'human' as const,
        content: message,
        timestamp: new Date().toISOString(),
      };
      addMessage(userMessage);

      // Stream response
      setStreaming(true);
      let assistantMessage = '';

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
              // Display tool usage in UI
              appendToStreamingMessage(`\n🔍 ${chunk.content}\n`);
              break;
            
            case 'suggestion':
              // Collect suggestions
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
      } finally {
        setStreaming(false);
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

## Message Format Alignment

Both agent and client use the same message structure:

```python
# Agent (Python)
class Message(TypedDict):
    role: Literal["human", "assistant", "system"]
    content: str
    timestamp: str
    id: str
```

```typescript
// Client (TypeScript)
interface Message {
  id: string;
  role: 'human' | 'assistant' | 'system';
  content: string;
  timestamp: string;
}
```

## API Endpoints

### Thread Management
```typescript
// Create thread
POST /api/langgraph/threads
Body: { metadata?: Record<string, any> }
Response: { thread_id: string, metadata: Record<string, any> }

// List threads
GET /api/langgraph/threads
Response: Thread[]

// Get thread
GET /api/langgraph/threads/:threadId
Response: Thread
```

### Streaming Chat
```typescript
// Stream chat response
POST /api/langgraph/threads/:threadId/runs/stream
Headers: {
  'Content-Type': 'application/json',
  'X-API-Key': 'your_key'
}
Body: { message: string }
Response: Server-Sent Events stream
```

## SSE Format Specification

The agent sends Server-Sent Events in this exact format:
```
data: {"type": "thinking", "content": "Analyzing the user's request..."}\n\n
data: {"type": "message", "content": "Based on my analysis"}\n\n
data: {"type": "tool_use", "content": "Searching the web for: latest AI news", "metadata": {"tool": "tavily_search", "query": "latest AI news"}}\n\n
data: {"type": "suggestion", "content": "What specific AI developments interest you?"}\n\n
data: {"type": "done", "content": ""}\n\n
```

## Error Handling Strategy

### Agent-Side Error Handling
```python
try:
    # Process message
    async for event in app.astream_events(...):
        # Handle events
except Exception as e:
    yield {
        "type": "error",
        "content": str(e),
        "metadata": {"error_type": type(e).__name__}
    }
```

### Client-Side Error Handling
```typescript
try {
  for await (const chunk of langgraphClient.streamChat(...)) {
    if (chunk.type === 'error') {
      throw new Error(chunk.content);
    }
    // Process chunk
  }
} catch (error) {
  toast.error('Failed to send message');
  console.error('Chat error:', error);
} finally {
  setStreaming(false);
}
```

## Deployment Configuration

### LangGraph Platform Deployment

1. **Agent Configuration (langgraph.json)**
```json
{
  "name": "react-agent",
  "version": "1.0.0",
  "description": "ReAct agent with Anthropic Claude and Tavily search",
  "entry_point": "agent.py",
  "models": {
    "primary": "claude-sonnet-4-20250514",
    "suggestions": "claude-3-5-haiku-latest"
  },
  "tools": ["tavily_search"],
  "environment_variables": [
    "ANTHROPIC_API_KEY",
    "TAVILY_API_KEY",
    "LANGCHAIN_API_KEY"
  ]
}
```

2. **Deploy Command**
```bash
langgraph deploy --name react-agent
```

### Next.js Deployment

1. **Environment Setup**
```bash
# Production .env
NEXT_PUBLIC_LANGGRAPH_API_KEY=your_production_key
NEXT_PUBLIC_API_ENDPOINT=https://your-agent.langgraph.com
```

2. **API Route Proxy** (recommended for security)
```typescript
// app/api/langgraph/[...path]/route.ts
export async function POST(req: Request) {
  const path = req.url.split('/api/langgraph/')[1];
  
  const response = await fetch(`${process.env.LANGGRAPH_ENDPOINT}/${path}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-API-Key': process.env.LANGGRAPH_API_KEY!, // Server-side only
    },
    body: await req.text(),
  });

  // Return SSE stream
  return new Response(response.body, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
    },
  });
}
```

## Testing Strategy

### Integration Test Example
```typescript
describe('Agent-Client Integration', () => {
  it('should handle complete message flow', async () => {
    // 1. Create thread
    const thread = await client.createThread();
    expect(thread.id).toBeDefined();

    // 2. Send message
    const chunks: StreamChunk[] = [];
    for await (const chunk of client.streamChat(thread.id, 'Hello')) {
      chunks.push(chunk);
    }

    // 3. Verify chunk sequence
    expect(chunks.some(c => c.type === 'thinking')).toBe(true);
    expect(chunks.some(c => c.type === 'message')).toBe(true);
    expect(chunks[chunks.length - 1].type).toBe('done');
  });
});
```

## Performance Optimizations

1. **Message Virtualization**: For long conversations
2. **Request Debouncing**: Prevent rapid successive requests
3. **Chunk Batching**: Aggregate small chunks before rendering
4. **Connection Pooling**: Reuse SSE connections
5. **State Persistence**: Save threads to localStorage

## Security Considerations

1. **API Key Management**: Never expose server-side keys to client
2. **Input Validation**: Sanitize user messages before sending
3. **Rate Limiting**: Implement per-user request limits
4. **CORS Configuration**: Restrict origins in production
5. **Content Security Policy**: Prevent XSS attacks

## Common Issues and Solutions

### Issue: Thinking patterns not displaying
**Solution**: Ensure regex pattern matches exactly:
```python
thinking_match = re.search(r'<thinking>(.*?)</thinking>', content, re.DOTALL)
```

### Issue: Tool usage not showing in UI
**Solution**: Check tool name matching:
```python
if tool_name == "tavily_search_results":  # Exact match required
```

### Issue: Suggestions not appearing
**Solution**: Verify Haiku model is available and suggestions are yielded after main content

### Issue: Thread creation fails
**Solution**: Ensure client creates thread before sending first message:
```typescript
if (!threadToUse) {
  const newThread = await langgraphClient.createThread();
  threadToUse = newThread;
}
```

## Monitoring and Observability

1. **Agent Metrics**
   - Response time per message
   - Tool invocation frequency
   - Error rates by type
   - Token usage per conversation

2. **Client Metrics**
   - Message send success rate
   - Stream connection stability
   - UI render performance
   - User engagement with suggestions

## Future Enhancements

1. **Multi-modal Support**: Handle image inputs/outputs
2. **Voice Integration**: Add speech-to-text and text-to-speech
3. **Collaborative Features**: Multi-user thread support
4. **Advanced Memory**: Long-term memory with vector stores
5. **Custom Tools**: Extensible tool framework

---

This integration guide serves as the single source of truth for connecting the LangGraph agent with the Next.js client. All technical details are synchronized across agent.md, client.md, and this guide to ensure a flawless implementation.