# LangGraph + Next.js AI Assistant - Project Guide

**Created**: 2025-05-27  
**Last Modified**: 2025-05-27

## 🎯 Project Overview

This is a production-ready conversational AI system combining:
- **Backend**: ReAct pattern LangGraph agent with Claude Sonnet 4 + selective Tavily search
- **Frontend**: Next.js 14 client with real-time streaming and thinking transparency
- **Deployment**: LangGraph Platform (agent) + Vercel (client)

## 📂 Project Structure

```
src/
├── agent/                      # Python LangGraph agent
│   ├── agent.py               # Main ReAct implementation
│   ├── langgraph.json         # Deployment config
│   ├── requirements.txt       # Python dependencies
│   └── .env.example           # Environment template
├── client/                     # Next.js application
│   ├── src/
│   │   ├── app/               # App router pages
│   │   ├── components/        # React components
│   │   ├── hooks/            # Custom React hooks
│   │   ├── lib/              # Utilities & client
│   │   ├── store/            # Zustand state
│   │   └── types/            # TypeScript types
│   ├── package.json          # Node dependencies
│   └── .env.example          # Environment template
├── docs/                       # Documentation
│   ├── agent.md              # Agent implementation
│   ├── client.md             # Client implementation
│   ├── integration.md        # Integration guide
│   └── overview.md           # Executive summary
├── startup.md                  # Getting started guide
└── test_connection.sh         # Connection test script
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Copy environment files
cp agent/.env.example agent/.env
cp client/.env.example client/.env.local

# Add your API keys to agent/.env
ANTHROPIC_API_KEY=your_key
TAVILY_API_KEY=your_key
```

### 2. Start Development

```bash
# Terminal 1: Start agent
cd agent
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
langgraph dev

# Terminal 2: Start client
cd client
npm install
npm run dev
```

### 3. Test Connection

```bash
# From src directory
./test_connection.sh
```

## 🛠️ Development Commands

### Agent Commands
```bash
# Linting
ruff check agent/
ruff format agent/

# Type checking
mypy agent/agent.py

# Run locally
langgraph dev
```

### Client Commands
```bash
# Development
npm run dev

# Linting
npm run lint

# Type checking
npm run type-check

# Build
npm run build
```

## 🔑 Key Features

### Agent (agent.py)
- **ReAct Pattern**: Reasoning → Action → Observation loop
- **Thinking Extraction**: Captures `<thinking>` tags from Claude
- **Selective Search**: Only searches when needed (not for general knowledge)
- **Suggestions**: Haiku generates follow-up questions
- **Streaming**: Real-time event streaming with timeout protection

### Client
- **Thread Management**: Automatic creation and persistence
- **Real-time UI**: Streaming messages with thinking display
- **Type Safety**: Full TypeScript coverage
- **State Management**: Zustand with persistence
- **API Proxy**: Secure server-side API key handling

## 📋 Common Tasks

### Adding a New Tool to Agent

```python
# 1. Define tool
@tool
def my_tool(query: str) -> str:
    """Tool description."""
    return result

# 2. Add to tools list
tools = [selective_search, my_tool]

# 3. Update streaming to handle tool events
elif event["event"] == "on_tool_start":
    if event["metadata"].get("tool_name") == "my_tool":
        yield {...}
```

### Adding a New UI Component

```typescript
// 1. Create component in components/
export function MyComponent({ prop }: Props) {
  return <div>{prop}</div>
}

// 2. Import and use
import { MyComponent } from '@/components/MyComponent'
```

### Updating Stream Events

Both agent and client must agree on the StreamChunk interface:

```typescript
// types/index.ts
interface StreamChunk {
  type: 'message' | 'thinking' | 'suggestion' | 'tool_use' | 'error' | 'done' | 'new_type';
  content: string;
  metadata?: Record<string, any>;
}
```

## 🐛 Debugging

### Agent Issues
- Check logs: `langgraph dev` shows all events
- Verify environment variables are loaded
- Test with curl: `curl http://localhost:8123/health`

### Client Issues
- Check browser console for errors
- Verify API proxy is working: Network tab
- Check environment variables: `console.log(process.env.NEXT_PUBLIC_*)`

### Streaming Issues
- Verify SSE format: `data: {"type": "message", "content": "..."}\n\n`
- Check for timeout errors (5 minute default)
- Ensure proxy forwards headers correctly

## 🚢 Deployment

### Deploy Agent
```bash
cd agent
langgraph deploy
```

### Deploy Client
```bash
cd client
vercel --prod
```

## 📚 Documentation

- **agent.md**: Complete agent implementation details
- **client.md**: Next.js client architecture
- **integration.md**: How agent and client work together
- **overview.md**: Executive summary with glossary
- **startup.md**: Detailed setup instructions for Mac

## ⚠️ Important Notes

1. **API Keys**: Never commit .env files
2. **Models**: Claude Sonnet 4 for main agent, Haiku for suggestions
3. **Timeout**: 5 minute default for streaming operations
4. **Thread Creation**: Automatic on first message if no thread exists
5. **Error Handling**: Both agent and client handle errors gracefully

## 🔍 Quick Reference

### Environment Variables
```bash
# Agent
ANTHROPIC_API_KEY=
TAVILY_API_KEY=
ANTHROPIC_MODEL=claude-sonnet-4-20250514
ANTHROPIC_SUGGESTIONS_MODEL=claude-3-5-haiku-latest

# Client
NEXT_PUBLIC_LANGGRAPH_API_URL=http://localhost:8123
NEXT_PUBLIC_LANGGRAPH_API_KEY=dev-key
LANGGRAPH_ENDPOINT=http://localhost:8123
LANGGRAPH_API_KEY=dev-key
```

### API Endpoints
```
POST   /threads                        # Create thread
GET    /threads                        # List threads  
DELETE /threads/:id                    # Delete thread
POST   /threads/:id/runs/stream       # Stream chat
```

### Stream Event Types
- `thinking` - AI's reasoning process
- `message` - Actual response content
- `tool_use` - Tool execution (e.g., search)
- `suggestion` - Follow-up questions
- `error` - Error occurred
- `done` - Stream complete

---
*This file helps Claude understand the project structure and provide better assistance with development tasks.*