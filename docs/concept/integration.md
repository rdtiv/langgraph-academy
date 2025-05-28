# LangGraph Agent + Next.js Client Integration Guide

**Created**: 2025-05-26  
**Updated**: 2025-05-27

## Overview

This guide provides the complete integration details for connecting the ReAct LangGraph agent with the Next.js client, ensuring perfect synchronization between agent streaming and client UI updates.

## Architecture

```
┌─────────────────┐     SSE Stream      ┌──────────────────┐     ┌─────────────────┐
│  Next.js Client │ ◄─────────────────── │  LangGraph Dev   │ ◄── │  ReAct Agent    │
│  (Port 3000)    │                      │  (Port 8123)     │     │  (Python)       │
│                 │ ────────────────────►│                  │     │                 │
└─────────────────┘    HTTP Requests     └──────────────────┘     └─────────────────┘
       │                                          │                        │
       ▼                                          ▼                        ▼
┌─────────────────┐                      ┌──────────────────┐     ┌─────────────────┐
│  Zustand Store  │                      │  API Proxy       │     │  Claude Sonnet  │
│  + Persistence  │                      │  Route Handler   │     │  + Tavily       │
└─────────────────┘                      └──────────────────┘     └─────────────────┘
```

## Environment Configuration

### Agent Environment
```bash
# /src/agent/.env
ANTHROPIC_API_KEY=your_anthropic_key
TAVILY_API_KEY=your_tavily_key
ANTHROPIC_MODEL=claude-sonnet-4-20250514
ANTHROPIC_SUGGESTIONS_MODEL=claude-3-5-haiku-latest
STREAM_TIMEOUT_SECONDS=300
```

### Client Environment
```bash
# /src/client/.env.local
# For local development
NEXT_PUBLIC_LANGGRAPH_API_URL=http://localhost:8123
NEXT_PUBLIC_LANGGRAPH_API_KEY=dev-key

# Server-side for API proxy
LANGGRAPH_ENDPOINT=http://localhost:8123
LANGGRAPH_API_KEY=dev-key
```

## Streaming Event Contract

### Shared Types
```typescript
// Both agent and client use this exact interface
interface StreamChunk {
  type: 'message' | 'thinking' | 'suggestion' | 'tool_use' | 'error' | 'done';
  content: string;
  metadata?: {
    tool?: string;
    query?: string;
    error_type?: string;
    [key: string]: any;
  };
}
```

### Event Flow
1. **User Message** → Client sends to `/threads/:id/runs/stream`
2. **Agent Processing** → ReAct loop with thinking extraction
3. **Streaming Response** → SSE chunks in order:
   - `thinking` - Agent's reasoning process
   - `tool_use` - When searching the web
   - `message` - Actual response content
   - `suggestion` - Follow-up questions
   - `done` - Stream complete

### Example Stream
```
data: {"type": "thinking", "content": "The user is asking about AI developments..."}
data: {"type": "tool_use", "content": "Searching the web for: latest AI news 2025", "metadata": {"tool": "tavily_search", "query": "latest AI news 2025"}}
data: {"type": "message", "content": "Based on my search, here are the latest developments..."}
data: {"type": "suggestion", "content": "What specific AI application interests you?"}
data: {"type": "done", "content": ""}
```

## Key Integration Points

### 1. Thread Management
- **Creation**: Client creates thread on first message
- **Persistence**: LangGraph handles thread state
- **Client Storage**: Zustand persists thread IDs locally

### 2. Message Streaming
- **Agent**: Uses `astream_events` with v2 API
- **Proxy**: Next.js API route forwards SSE stream
- **Client**: Parses chunks with validation

### 3. Error Handling
- **Timeout**: 5-minute default for long operations
- **Retry**: Client implements exponential backoff
- **Fallback**: Graceful degradation on failures

### 4. Thinking Pattern Extraction
```python
# Agent extracts thinking
thinking_match = re.search(r'<thinking>(.*?)</thinking>', content, re.DOTALL)
if thinking_match:
    yield {"type": "thinking", "content": thinking_match.group(1).strip()}

# Client displays in UI
case 'thinking':
  setThinkingMessage(chunk.content);
  break;
```

## API Endpoints

### LangGraph Platform API
```
POST   /threads                        # Create thread
GET    /threads                        # List threads
DELETE /threads/:id                    # Delete thread
POST   /threads/:id/runs/stream       # Stream chat
GET    /threads/:id/state             # Get state
```

### Next.js Proxy Routes
```
/api/langgraph/threads                # Thread operations
/api/langgraph/threads/:id/runs/stream # Streaming endpoint
```

## Deployment Configuration

### Agent Deployment
```json
// langgraph.json
{
  "dependencies": ["."],
  "graphs": {
    "agent": "agent:app"
  },
  "env": ".env"
}
```

### Client Deployment
```json
// vercel.json
{
  "functions": {
    "app/api/langgraph/[...path]/route.ts": {
      "maxDuration": 60
    }
  }
}
```

## Development Workflow

### Local Setup
```bash
# Terminal 1: Start agent
cd src/agent
pip install -r requirements.txt
langgraph dev

# Terminal 2: Start client
cd src/client
npm install
npm run dev
```

### Testing Integration
```bash
# Test connection
curl http://localhost:8123/health

# Test streaming
curl -X POST http://localhost:8123/threads \
  -H "X-API-Key: dev-key" \
  -H "Content-Type: application/json" \
  -d '{"metadata": {"title": "Test"}}'
```

## Common Issues & Solutions

### 1. CORS Errors
- **Issue**: Browser blocks cross-origin requests
- **Solution**: API proxy handles CORS headers

### 2. Stream Parsing
- **Issue**: Malformed SSE chunks
- **Solution**: Runtime validation in client

### 3. Thread State
- **Issue**: Lost conversation context
- **Solution**: Persistent checkpointer in agent

### 4. Timeout Errors
- **Issue**: Long searches timeout
- **Solution**: Configurable timeout + retry logic

## Performance Optimization

### Agent Side
- Prompt caching with Anthropic beta
- Selective search with tool description
- Async suggestions generation

### Client Side
- Zustand persistence for threads
- React Query for data fetching
- Streaming UI updates without re-renders

## Security Considerations

### Implemented
- API keys server-side only
- Proxy route validation
- Input sanitization

### Recommended (Not Implemented)
- Rate limiting with Redis
- User authentication
- Request signing

## Summary

This integration provides:
1. **Real-time streaming** with thinking patterns
2. **Thread persistence** across sessions
3. **Type safety** throughout the stack
4. **Production-ready** error handling
5. **Seamless deployment** to LangGraph Platform and Vercel

The architecture cleanly separates concerns while maintaining perfect synchronization between the intelligent agent backend and responsive client frontend.