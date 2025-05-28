# Integration Guide: ReAct Agent + Next.js Client
**Created**: 2025-05-27  
**Purpose**: Complete integration guide for connecting the ReAct LangGraph Agent with Next.js Client

---

## Quick Start

This guide shows how to connect the ReAct LangGraph Agent (with Anthropic Claude + Tavily) to the Next.js client with full streaming support.

## System Overview

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Next.js Client │────▶│  Vercel/Local    │────▶│ LangGraph Agent │
│  (React + SSE)  │◀────│  API Routes      │◀────│  (ReAct + AI)   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
        UI                   Gateway                  Intelligence
```

## 1. Agent Setup

### File Structure
```
agent/
├── agent.py           # Main agent implementation
├── requirements.txt   # Dependencies
├── langgraph.json    # LangGraph configuration
└── .env              # Environment variables
```

### Complete Agent Implementation
```python
# agent.py
from typing import List, Dict, Any, Literal
from datetime import datetime
import re
import json
import os
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import BaseMessage, SystemMessage, AIMessage, HumanMessage
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# State definition
class ThinkingState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    thinking_trace: List[str]
    search_results: List[Dict[str, Any]]

# System prompt for thinking
THINKING_PROMPT = """You are an AI assistant with advanced reasoning capabilities.
When responding:
1. Think through problems step-by-step in <thinking> tags
2. Identify what information you need
3. Use the web search tool ONLY when you need current information beyond your knowledge
4. Make your reasoning transparent
5. Be explicit about when and why you're searching the web"""

# Initialize tools and LLM
tavily = TavilySearchResults(
    max_results=3,
    description="Search the web for current information. Use ONLY when you need up-to-date information beyond your training data."
)
tools = [tavily]

llm = ChatAnthropic(
    model="claude-3-5-sonnet-20241022",
    temperature=0.7,
    anthropic_beta="prompt-caching-2024-07-31"
)
llm_with_tools = llm.bind_tools(tools)

def agent_node(state: ThinkingState) -> Dict[str, Any]:
    """Process messages with Anthropic Claude and extract thinking.
    
    Args:
        state: Current conversation state with messages and thinking trace
        
    Returns:
        Dict with updated messages and thinking trace
    """
    messages = state["messages"]
    
    system_msg = SystemMessage(content=THINKING_PROMPT)
    messages_with_system = [system_msg] + messages
    
    response = llm_with_tools.invoke(messages_with_system)
    
    # Extract thinking
    thinking_trace = []
    clean_content = response.content
    
    if "<thinking>" in response.content:
        thinking_pattern = r'<thinking>(.*?)</thinking>'
        thinking_matches = re.findall(thinking_pattern, response.content, re.DOTALL)
        thinking_trace = thinking_matches
        clean_content = re.sub(thinking_pattern, '', response.content, flags=re.DOTALL).strip()
    
    clean_response = AIMessage(
        content=clean_content,
        tool_calls=response.tool_calls if hasattr(response, 'tool_calls') else []
    )
    
    return {
        "messages": [clean_response],
        "thinking_trace": state.get("thinking_trace", []) + thinking_trace
    }

def should_continue(state: ThinkingState) -> Literal["tools", "end"]:
    """Determine next step in ReAct pattern.
    
    Args:
        state: Current conversation state
        
    Returns:
        'tools' if the last message contains tool calls, 'end' otherwise
    """
    messages = state["messages"]
    last_message = messages[-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "end"

def tool_node_wrapper(state: ThinkingState) -> Dict[str, Any]:
    """Execute tools and track search results.
    
    Args:
        state: Current conversation state
        
    Returns:
        Dict with updated messages and search results
    """
    tool_node = ToolNode(tools)
    result = tool_node.invoke(state)
    
    search_results = state.get("search_results", [])
    last_message = result["messages"][-1]
    
    if "tavily" in str(last_message.content).lower():
        search_results.append({
            "timestamp": datetime.now().isoformat(),
            "content": last_message.content
        })
    
    return {
        "messages": result["messages"],
        "search_results": search_results
    }

def create_react_agent():
    """Create the ReAct agent graph with thinking extraction.
    
    Returns:
        Compiled LangGraph application with memory persistence
    """
    graph = StateGraph(ThinkingState)
    
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node_wrapper)
    
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
    graph.add_edge("tools", "agent")
    
    return graph.compile(checkpointer=MemorySaver())

# Streaming handler
async def stream_events(app, thread_id: str, messages: List[BaseMessage]):
    """Stream events with proper formatting for Next.js client"""
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        async for event in app.astream_events(
            {"messages": messages},
            config=config,
            version="v2"
        ):
            if event["event"] == "on_chat_model_stream":
                # Stream message chunks
                content = event["data"]["chunk"].get("content", "")
                if content:
                    yield {
                        "type": "message",
                        "content": content
                    }
            
            elif event["event"] == "on_chain_end" and event["name"] == "agent_node":
                # Stream thinking after agent completes
                state = event["data"]["output"]
                if state.get("thinking_trace"):
                    for thought in state["thinking_trace"]:
                        yield {
                            "type": "thinking",
                            "content": thought
                        }
            
            elif event["event"] == "on_tool_start":
                # Stream tool usage
                yield {
                    "type": "tool_use",
                    "content": f"Searching for: {event['data'].get('input', {}).get('query', 'information')}",
                    "metadata": {
                        "tool": event["name"],
                        "query": str(event["data"].get("input", {}).get("query", ""))
                    }
                }
            
            elif event["event"] == "on_chain_end" and event["name"] == "react_agent":
                # Generate suggestions based on conversation
                final_state = event["data"]["output"]
                if final_state.get("messages"):
                    suggestions = await generate_suggestions(final_state["messages"])
                    for suggestion in suggestions:
                        yield {
                            "type": "suggestion",
                            "content": suggestion
                        }
        
        # Send done signal
        yield {"type": "done", "content": ""}
        
    except Exception as e:
        yield {
            "type": "error",
            "content": str(e),
            "metadata": {"error_type": type(e).__name__}
        }

async def generate_suggestions(messages: List[BaseMessage]) -> List[str]:
    """Generate smart next-step suggestions using Haiku based on conversation"""
    # Get the last assistant message
    last_assistant_msg = None
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            last_assistant_msg = msg.content
            break
    
    if not last_assistant_msg:
        return []
    
    # Use Haiku for lightweight suggestion generation
    haiku = ChatAnthropic(
        model="claude-3-haiku-20240307",
        temperature=0.7,
        max_tokens=200
    )
    
    suggestion_prompt = f"""Based on this response, suggest 3 brief follow-up questions the user might ask to learn more.

Response: {last_assistant_msg[:500]}...

Return only the 3 questions, one per line, no numbering or formatting."""
    
    try:
        suggestions_response = await haiku.ainvoke([
            SystemMessage(content="You generate concise follow-up questions."),
            HumanMessage(content=suggestion_prompt)
        ])
        
        suggestions = [
            s.strip() 
            for s in suggestions_response.content.split('\n') 
            if s.strip()
        ][:3]
        
        return suggestions
    except Exception as e:
        print(f"Error generating suggestions: {e}")
        return []

# API handler for deployment
async def handle_stream_request(thread_id: str, message: str):
    """Handle streaming request from Next.js client.
    
    Args:
        thread_id: Unique identifier for the conversation thread
        message: The user's message to process
        
    Yields:
        SSE-formatted strings with JSON data chunks
    """
    app = create_react_agent()
    
    async for chunk in stream_events(app, thread_id, [HumanMessage(content=message)]):
        # Format for SSE
        yield f"data: {json.dumps(chunk)}\n\n"
    
    yield "data: [DONE]\n\n"
```

### Environment Configuration
```bash
# .env
ANTHROPIC_API_KEY=sk-ant-...
TAVILY_API_KEY=tvly-...
LANGSMITH_API_KEY=ls-...
ASSISTANT_ID=react-agent
```

### LangGraph Configuration
```json
// langgraph.json
{
  "python_version": "3.11",
  "dependencies": ["requirements.txt"],
  "graphs": {
    "react_agent": {
      "path": "agent.py:create_react_agent",
      "config_schemas": {
        "thread_id": {
          "type": "string",
          "description": "Conversation thread ID"
        }
      }
    }
  },
  "env": {
    "ASSISTANT_ID": "${ASSISTANT_ID:-react-agent}"
  }
}
```

### Dependencies
```txt
# requirements.txt
langgraph>=0.2.0
langchain-anthropic>=0.1.0
langchain-community>=0.2.0
tavily-python>=0.3.0
python-dotenv>=1.0.0
```

## 2. Client Setup

### Key Integration Points

1. **Environment Variables**
```env
# .env.local
NEXT_PUBLIC_LANGGRAPH_URL=http://localhost:8123
NEXT_PUBLIC_LANGGRAPH_API_KEY=your-api-key
NEXT_PUBLIC_ASSISTANT_ID=react-agent
```

2. **Stream Chunk Types**
```typescript
// Exact match with agent output
interface StreamChunk {
  type: 'thinking' | 'message' | 'tool_use' | 'suggestion' | 'error' | 'done';
  content: string;
  metadata?: {
    tool?: string;
    query?: string;
  };
}

// Message type with metadata for thinking
interface Message {
  id: string;
  role: 'human' | 'assistant' | 'system';
  content: string;
  timestamp: string;
  metadata?: {
    thinking?: string;
    tool_use?: string;
    [key: string]: any;
  };
}
```

3. **Enhanced Chat Hook**
```typescript
// lib/hooks/useChat.ts
export function useChat() {
  const { /* ... state ... */ } = useChatStore();

  const sendMessage = useMutation({
    mutationFn: async (message: string) => {
      // ... setup ...
      
      try {
        for await (const chunk of langgraphClient.streamChat(
          currentThread.thread_id,
          message
        )) {
          switch (chunk.type) {
            case 'thinking':
              setThinkingMessage(chunk.content);
              break;
            
            case 'message':
              appendToStreamingMessage(chunk.content);
              assistantMessage += chunk.content;
              break;
            
            case 'tool_use':
              // Show tool usage in UI
              setToolUseMessage(`🔍 ${chunk.content}`);
              break;
            
            case 'suggestion':
              suggestions.push(chunk.content);
              setSuggestions(suggestions);
              break;
            
            case 'error':
              throw new Error(chunk.content);
          }
        }
        
        // ... finalize message ...
      } catch (error) {
        console.error('Chat error:', error);
        toast.error(error.message || 'Failed to send message');
      }
    },
  });

  return { sendMessage: sendMessage.mutate, isLoading: sendMessage.isPending };
}
```

## 3. Development Setup

### Local Development with LangGraph Platform

```bash
# 1. Install LangGraph CLI
pip install langgraph-cli

# 2. Create langgraph.json in your agent directory
cat > langgraph.json << EOF
{
  "dependencies": ["./"],
  "graphs": {
    "react_agent": "./agent.py:create_react_agent"
  },
  "env": ".env"
}
EOF

# 3. Start LangGraph Platform locally
langgraph up

# 4. In another terminal, start Next.js
cd nextjs-client
npm run dev
```

The LangGraph Platform handles all infrastructure:
- Local PostgreSQL for state persistence
- Redis for caching
- Task queues for background jobs
- API server with proper routing

## 4. Testing the Integration

### 1. Start the Stack
```bash
# Start LangGraph Platform
langgraph up

# Start Next.js in another terminal
npm run dev
```

### 2. Test Streaming
```bash
# Test agent directly
curl -X POST http://localhost:8123/runs/stream \
  -H "Content-Type: application/json" \
  -d '{
    "thread_id": "test-thread",
    "assistant_id": "react-agent",
    "input": {"messages": [{"role": "human", "content": "What is the weather today?"}]}
  }'
```

### 3. Verify Integration Points

✅ **Checklist:**
- [ ] Agent extracts and streams thinking separately
- [ ] Tool usage appears in UI with search queries
- [ ] Suggestions generate from search results
- [ ] Errors propagate correctly to UI
- [ ] Thread state persists between messages
- [ ] Streaming is smooth without buffering issues

## 5. Production Deployment

### Agent Deployment (LangGraph Platform)
```bash
# Deploy to LangGraph Platform
langgraph deploy

# Your deployment will be available at:
# https://your-deployment.langgraph.app
```

### Client Deployment (Vercel)
```bash
# Deploy to Vercel
vercel --prod

# Set production environment variables
vercel env add LANGGRAPH_URL production
vercel env add LANGGRAPH_API_KEY production
vercel env add NEXT_PUBLIC_LANGGRAPH_URL production
vercel env add NEXT_PUBLIC_ASSISTANT_ID production
```

## 6. Common Issues & Solutions

### Issue: Thinking not displaying
**Solution**: Ensure agent extracts `<thinking>` tags and streams them as separate chunks.

### Issue: Tool usage not visible
**Solution**: Check that `tool_use` chunks are being parsed correctly in the client.

### Issue: Suggestions not appearing
**Solution**: 
- Verify Haiku model is accessible with your Anthropic API key
- Check that the last assistant message exists before generating suggestions
- Ensure async/await is properly handled in the streaming pipeline

### Issue: CORS errors in development
**Solution**: Add CORS headers to LangGraph server or use API routes as proxy.

### Issue: Tavily being called too frequently
**Solution**: 
- Review system prompt to emphasize "ONLY when needed"
- Check agent's thinking to ensure it's not defaulting to search
- Consider adding explicit examples of when NOT to search

## 7. Monitoring & Debugging

### Agent Side
```python
# Add logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Track in LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
```

### Client Side
```typescript
// Add debug logging
if (process.env.NODE_ENV === 'development') {
  console.log('Stream chunk:', chunk);
}
```

## Key Improvements in This Version

### 1. Selective Web Search
- Tavily is only called when the agent determines it needs current information
- System prompt explicitly states to use web search "ONLY when needed"
- Agent will rely on its training data for general knowledge questions

### 2. Smart Suggestions with Haiku
- Claude 3 Haiku generates contextual follow-up questions
- Based on the actual agent response, not search results
- Lightweight and fast for better UX
- Examples of generated suggestions:
  - "How does this compare to previous approaches?"
  - "What are the potential risks or limitations?"
  - "Can you provide a practical example?"

### 3. Cost Optimization
- Reduced Tavily API calls = lower costs
- Haiku for suggestions = ~10x cheaper than Sonnet
- Efficient streaming = better resource utilization

## Summary

This integration provides:
1. **Full streaming support** with thinking, tool use, and suggestions
2. **Intelligent tool usage** - web search only when necessary
3. **Smart suggestions** generated by Haiku based on context
4. **Cost-effective operation** with selective API usage
5. **Type safety** with matching interfaces

The key to success is ensuring the streaming protocol matches exactly between agent and client, with proper chunk types and error handling throughout the pipeline.