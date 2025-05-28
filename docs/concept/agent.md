# ReAct LangGraph Agent Briefing Plan
## Anthropic Sonnet with Thinking Enabled + Next.js Client + Tavily Search

**Created**: 2025-05-26  
**Purpose**: Proof of concept for production-ready AI agent deployed on LangGraph Platform

---

## Executive Summary

This briefing outlines the architecture and implementation plan for a React-based LangGraph agent that:
- Uses Anthropic Claude 3 Sonnet with thinking/reasoning enabled
- Integrates with a Next.js client via streaming API
- Deploys on LangGraph Platform as a production API
- Incorporates Tavily for web search capabilities

## System Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Next.js Client │────▶│  Vercel Edge Fn  │────▶│ LangGraph Server│
│  (React + SSE)  │◀────│  (API Gateway)   │◀────│  (Agent + API)  │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                            │
                                                            ▼
                                                   ┌─────────────────┐
                                                   │ External APIs   │
                                                   │ - Anthropic     │
                                                   │ - Tavily Search │
                                                   └─────────────────┘
```

## Core Components

### Graph Construction

```python
# ReAct Pattern Implementation
def create_react_agent():
    graph = StateGraph(ThinkingState)
    
    # Add nodes
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node_wrapper)
    
    # ReAct pattern edges
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "end": END}
    )
    graph.add_edge("tools", "agent")  # Loop back
    
    # Compile with memory
    return graph.compile(checkpointer=MemorySaver())
```

### 1. Agent Architecture

```python
# ReAct Pattern: START → Agent → Conditional → Tools → Agent → END
#                                     ↓
#                                    END

# Core components structure
class ThinkingState(TypedDict):
    """Extended state to track reasoning process"""
    messages: Annotated[List[BaseMessage], add_messages]
    thinking_trace: List[str]
    search_results: List[Dict[str, Any]]

# Agent node with thinking extraction
def agent_node(state: ThinkingState) -> Dict[str, Any]:
    """Process messages with thinking-enabled Anthropic"""
    messages = state["messages"]
    
    # Add system prompt for thinking
    system_msg = SystemMessage(content=THINKING_PROMPT)
    messages_with_system = [system_msg] + messages
    
    # Invoke Anthropic
    response = llm_with_tools.invoke(messages_with_system)
    
    # Extract thinking from response content
    thinking_trace = []
    clean_content = response.content
    
    if "<thinking>" in response.content:
        import re
        thinking_pattern = r'<thinking>(.*?)</thinking>'
        thinking_matches = re.findall(thinking_pattern, response.content, re.DOTALL)
        thinking_trace = thinking_matches
        # Remove thinking tags from final message
        clean_content = re.sub(thinking_pattern, '', response.content, flags=re.DOTALL).strip()
    
    # Create clean response without thinking tags
    clean_response = AIMessage(
        content=clean_content,
        tool_calls=response.tool_calls if hasattr(response, 'tool_calls') else []
    )
    
    return {
        "messages": [clean_response],
        "thinking_trace": state.get("thinking_trace", []) + thinking_trace
    }

# Conditional routing
def should_continue(state: ThinkingState) -> Literal["tools", "end"]:
    """Check if last message has tool calls"""
    # If tool_calls exist → "tools"
    # Otherwise → "end"

# Tool execution with result tracking
def tool_node_wrapper(state: ThinkingState) -> Dict[str, Any]:
    """Execute tools and track search results"""
    tool_node = ToolNode(tools)
    result = tool_node.invoke(state)
    
    # Extract search results if Tavily was used
    search_results = state.get("search_results", [])
    last_message = result["messages"][-1]
    
    if "tavily" in last_message.content.lower():
        # Parse search results for suggestions
        search_results.append({
            "timestamp": datetime.now().isoformat(),
            "content": last_message.content
        })
    
    return {
        "messages": result["messages"],
        "search_results": search_results
    }
```

### 2. Next.js Client Integration

```typescript
// Client wrapper for LangGraph API
interface LangGraphConfig {
  apiUrl: string;
  apiKey: string;
  assistantId: string;
}

class LangGraphClient {
  async createThread(): Promise<Thread>
  async streamChat(threadId: string, message: string): AsyncIterator<StreamChunk>
  async getThreadHistory(threadId: string): Promise<Message[]>
}

// Stream handling for thinking + responses
interface StreamChunk {
  type: 'thinking' | 'message' | 'tool_use' | 'error';
  content: string;
  metadata?: any;
}
```

### 3. Tool Integration

```python
# Tavily Search Tool Configuration
tavily_tool = TavilySearchResults(
    max_results=3,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=False,
    description="Search for current information"
)

# Additional Tools (optional)
tools = [
    tavily_tool,
    calculator_tool,
    # Add more tools as needed
]

# Bind tools to LLM
llm_with_tools = llm.bind_tools(tools)
```

## Implementation Phases

### Phase 1: Basic Chat + Search (Week 1)
- [ ] Set up LangGraph graph with Anthropic Sonnet
- [ ] Integrate Tavily search tool
- [ ] Basic message streaming
- [ ] Deploy to LangGraph Platform

### Phase 2: Thinking Display (Week 2)
- [ ] Implement reasoning trace capture
- [ ] Stream thinking process to client
- [ ] UI components for reasoning display
- [ ] Optimize prompt engineering

### Phase 3: Production Features (Week 3)
- [ ] Add conversation memory (checkpointer)
- [ ] Implement user profiles (Store API)
- [ ] Error handling and retry logic
- [ ] Performance optimization

### Phase 4: Advanced Capabilities (Week 4)
- [ ] Multi-turn reasoning chains
- [ ] Parallel search operations
- [ ] Smart suggestions system
- [ ] Analytics and monitoring

## Technical Specifications

### API Endpoints

```yaml
# LangGraph Platform API structure
POST   /threads                                  # Create thread
POST   /threads/:threadId/runs/stream          # Stream chat
GET    /threads/:threadId/messages             # Get history
PATCH  /threads/:threadId/state                # Update state
GET    /threads                                 # List threads
DELETE /threads/:threadId                      # Delete thread
```

### Streaming Protocol

```python
# Custom streaming handler for proper chunk formatting
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
    
    async for event in app.astream_events(
        {"messages": messages},
        config=config,
        version="v2"
    ):
        if event["event"] == "on_chat_model_stream":
            # Stream message chunks
            yield {
                "type": "message",
                "content": event["data"]["chunk"]["content"]
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
        
        # Send done signal after all events
        yield {"type": "done", "content": ""}
        
    except Exception as e:
        yield {
            "type": "error",
            "content": str(e),
            "metadata": {"error_type": type(e).__name__}
        }

async def generate_suggestions(messages: List[BaseMessage]) -> List[str]:
    """Generate smart next-step suggestions using Haiku based on conversation.
    
    Args:
        messages: Conversation history
        
    Returns:
        List of up to 3 suggested follow-up questions
    """
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
        model="claude-3-5-haiku-latest",
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
```

Server-Sent Events format:
```
data: {"type": "thinking", "content": "Analyzing user query..."}
data: {"type": "tool_use", "tool": "tavily_search", "query": "latest AI news"}
data: {"type": "message", "content": "Based on my search..."}
data: {"type": "suggestion", "content": "Tell me more about..."}
data: {"type": "done", "metadata": {"tokens": 523}}
```

### Configuration Schema

```python
# Agent configuration structure
class AgentConfig:
    thread_id: str = "default"  # Conversation thread
    temperature: float = 0.7     # LLM temperature
    max_iterations: int = 10     # Max ReAct loops
    thinking_visible: bool = True # Show reasoning to user
    search_depth: Literal["basic", "advanced"] = "advanced"
    
# Anthropic configuration
anthropicConfig = {
    "model": "claude-sonnet-4-20250514",  # Latest Sonnet
    "temperature": 0.7,
    "max_tokens": 4096,
    "anthropic_beta": "prompt-caching-2024-07-31"
}

# System prompt for thinking
THINKING_PROMPT = """
You are an AI assistant with advanced reasoning capabilities.
When responding:
1. Think through problems step-by-step in <thinking> tags
2. Identify what information you need
3. Use tools when you need current information
4. Make your reasoning transparent
"""
```

## Deployment Strategy

### LangGraph Platform Deployment

With LangGraph Platform now in general availability, deployment is simplified:

```bash
# 1. Configure your project
# langgraph.json
{
  "dependencies": ["./"],
  "graphs": {
    "react_agent": "./agent.py:create_react_agent"
  },
  "env": ".env"
}

# 2. Deploy to LangGraph Platform
langgraph deploy

# 3. Your agent is now available at the platform endpoint
# https://your-deployment.langgraph.app
```

No Docker or infrastructure management needed - LangGraph Platform handles:
- Scaling and load balancing
- State persistence (PostgreSQL/Redis)
- Long-running task management
- Connection pooling
- Monitoring and observability

### Environment Variables

```bash
# .env.production
LANGGRAPH_API_URL=https://api.langgraph.com
LANGGRAPH_API_KEY=lsv2_...
ANTHROPIC_API_KEY=sk-ant-...
TAVILY_API_KEY=tvly-...
LANGSMITH_API_KEY=ls-...
ASSISTANT_ID=react-agent  # Configurable assistant ID
```

## UI/UX Considerations

### Thinking Display
- Collapsible reasoning section
- Real-time streaming updates
- Color-coded for different reasoning types

### Search Integration
- Show when searches are happening
- Display source citations
- Preview search results inline

### Conversation Flow
- Smart follow-up suggestions
- Context-aware prompts
- Thread management sidebar

## Success Metrics

### Performance
- Response time < 2s for first token
- Streaming rate > 50 tokens/second
- Search latency < 1s

### Quality
- Reasoning transparency score > 90%
- Search relevance score > 85%
- User satisfaction rating > 4.5/5

### Scale
- Support 1000+ concurrent users
- Handle 10k messages/hour
- 99.9% uptime

## Security & Compliance

### API Security
- Rate limiting: 100 req/min per user
- API key rotation every 90 days
- Request signing with HMAC

### Data Privacy
- No PII in logs
- Conversation encryption at rest
- GDPR-compliant data retention

## Resources & References

### Documentation
- [LangGraph Platform Guide](./langgraph-platform-complete-guide.md)
- [Next.js Client Briefing](./nextjs-client-briefing.md)
- [Agent Patterns](./agents-and-assistants.md)

### Code Examples
- Python agent: `module-1/studio/agent.py`
- Deployment: `module-6/deployment/`
- Memory systems: `module-5/studio/`

### Implementation Structure
- **Main Agent**: Python file implementing ReAct pattern with thinking extraction
- **Configuration**: LangGraph JSON config for deployment settings
- **Deployment**: Docker Compose stack with PostgreSQL + Redis + LangGraph Server
- **Dependencies**: Core packages - langgraph, langchain-anthropic, tavily-python
- **Environment**: API keys for Anthropic, Tavily, and LangSmith

### External Links
- [Anthropic API Docs](https://docs.anthropic.com)
- [Tavily API Docs](https://docs.tavily.com)
- [LangGraph Studio](https://smith.langchain.com)

## Next Steps

1. **Immediate**: Review and approve architecture
2. **Week 1**: Implement Phase 1 POC
3. **Week 2**: User testing with thinking display
4. **Week 3**: Production deployment
5. **Week 4**: Launch and iterate

## Integration with Next.js Client

### Complete Working Example

```python
# agent.py - Full implementation
from typing import List, Dict, Any, Literal, AsyncIterator, Optional
from datetime import datetime
import re
import json
import os
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage, SystemMessage, AIMessage, HumanMessage, ToolMessage
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# State definition
class ThinkingState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    thinking_trace: List[str]
    search_results: List[Dict[str, Any]]

# System prompt
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
    model="claude-sonnet-4-20250514",
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

# Streaming handler for API (if deploying custom endpoint)
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
        # Format for SSE with double newline
        yield f"data: {json.dumps(chunk)}\n\n"
    
    yield "data: [DONE]\n\n"
```

### API Configuration

```json
// langgraph.json
{
  "python_version": "3.11",
  "dependencies": ["requirements.txt"],
  "graphs": {
    "react_agent": {
      "path": "agent.py:create_react_agent",
      "config_schemas": {
        "assistant_id": {
          "type": "string",
          "default": "react-agent"
        },
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

### Client-Agent Contract

```typescript
// Shared types between agent and client
interface StreamChunk {
  type: 'thinking' | 'message' | 'tool_use' | 'suggestion' | 'error';
  content: string;
  metadata?: {
    tool?: string;
    query?: string;
    tokens?: number;
  };
}

// Agent outputs these exact chunk types
// Client parses these exact chunk types
```

---

*This briefing plan provides a comprehensive roadmap for building a production-ready LangGraph agent with advanced reasoning capabilities, fully integrated with the Next.js client.*