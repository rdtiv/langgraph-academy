# ReAct LangGraph Agent Implementation
## Anthropic Claude Sonnet 4 with Thinking + Next.js Client + Tavily Search

**Created**: 2025-05-26  
**Updated**: 2025-05-27  
**Purpose**: Production-ready AI agent with thinking patterns, deployed on LangGraph Platform

---

## Executive Summary

This document describes the implementation of a ReAct pattern LangGraph agent that:
- Uses Anthropic Claude Sonnet 4 (claude-sonnet-4-20250514) with thinking patterns
- Streams responses to Next.js client via Server-Sent Events (SSE)
- Deploys on LangGraph Platform with `langgraph dev` for local development
- Incorporates selective Tavily search with advanced search depth
- Generates follow-up suggestions using Claude 3.5 Haiku

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
def create_graph():
    """Create the ReAct agent graph with selective search."""
    # System prompt with thinking pattern
    system_message = SystemMessage(content=AGENT_SYSTEM_PROMPT)
    
    # Use Anthropic's Claude Sonnet 4 as our LLM
    llm = ChatAnthropic(
        model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
        temperature=0.7,
        anthropic_beta="prompt-caching-2024-07-31"  # Enable prompt caching
    )
    
    # Create selective search tool
    search_tool = create_selective_search_tool()
    tools = [search_tool]
    
    # Create React agent with system message
    graph = create_react_agent(
        llm,
        tools,
        state_modifier=lambda messages: [system_message] + messages
    )
    
    # Compile with memory
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)
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

# Streaming with thinking extraction
async def stream_events(app, thread_id: str, messages: List[BaseMessage]) -> AsyncIterator[Dict[str, Any]]:
    """Stream events with proper formatting for Next.js client."""
    import asyncio
    
    config = {"configurable": {"thread_id": thread_id}}
    timeout = float(os.getenv("STREAM_TIMEOUT_SECONDS", "300"))  # 5 minute default
    
    try:
        # Add timeout protection
        async with asyncio.timeout(timeout):
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
            
    except asyncio.TimeoutError:
        yield {
            "type": "error",
            "content": "Stream timeout exceeded. Please try again.",
            "metadata": {"error_type": "TimeoutError"}
        }
```

### 2. Selective Search Tool

```python
def create_selective_search_tool():
    """Create a search tool that decides when to search."""
    search = TavilySearchResults(
        max_results=3,
        search_depth="advanced",
        include_raw_content=False,
        description="Search the web for current information when needed"
    )
    
    @tool
    def selective_search(query: str) -> str:
        """Searches for information only when necessary.
        
        Use this when:
        - User asks about current events, news, or recent information
        - Specific facts need verification
        - Real-time data is requested
        
        Skip this when:
        - The question is about general knowledge
        - You can answer from your training data
        - The query is about concepts or explanations
        """
        results = search.invoke({"query": query})
        return f"Search results for '{query}': {results}"
    
    return selective_search
```

### 3. Suggestions Generation

```python
async def generate_suggestions(messages: List[BaseMessage]) -> List[str]:
    """Generate smart next-step suggestions using Haiku."""
    # Get the last assistant message
    last_assistant_msg = None
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            last_assistant_msg = msg
            break
    
    if not last_assistant_msg:
        return []
    
    # Use Haiku for lightweight suggestion generation
    haiku = ChatAnthropic(
        model=os.getenv("ANTHROPIC_SUGGESTIONS_MODEL", "claude-3-5-haiku-latest"),
        temperature=0.7,
        max_tokens=200
    )
    
    suggestion_prompt = f"""Based on this response, suggest 3 brief follow-up questions.

Response: {last_assistant_msg.content}

Provide exactly 3 short questions as a JSON array."""
    
    suggestions_response = await haiku.ainvoke([HumanMessage(content=suggestion_prompt)])
    return extract_suggestions(suggestions_response.content)
```

### 4. Environment Configuration

```bash
# Required environment variables
ANTHROPIC_API_KEY=your_anthropic_key
TAVILY_API_KEY=your_tavily_key

# Optional model configuration
ANTHROPIC_MODEL=claude-sonnet-4-20250514
ANTHROPIC_SUGGESTIONS_MODEL=claude-3-5-haiku-latest
STREAM_TIMEOUT_SECONDS=300
```

### 5. Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run with LangGraph dev server
langgraph dev

# Server runs on http://localhost:8123
```

## Key Features

### 1. Thinking Pattern Extraction
- Real-time extraction of `<thinking>` tags from Claude's responses
- Separate streaming of thinking vs. visible content
- Clean message display without thinking artifacts

### 2. Selective Web Search
- Smart decision-making about when to search
- Optimized Tavily configuration for quality results
- Tool usage streaming for transparency

### 3. Follow-up Suggestions
- Haiku-powered lightweight suggestion generation
- Context-aware questions based on conversation
- Asynchronous generation after main response

### 4. Production-Ready Features
- Thread-based conversation management
- Persistent memory across sessions
- Timeout protection for long-running operations
- Comprehensive error handling

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
        

## Stream Event Format

```typescript
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

Example SSE stream:
```
data: {"type": "thinking", "content": "Analyzing the user's question about recent AI developments..."}
data: {"type": "tool_use", "content": "Searching the web for: latest AI news 2025", "metadata": {"tool": "tavily_search", "query": "latest AI news 2025"}}
data: {"type": "message", "content": "Based on my search, here are the latest AI developments..."}
data: {"type": "suggestion", "content": "What specific AI application interests you most?"}
data: {"type": "done", "content": ""}
```

## System Prompt

```python
AGENT_SYSTEM_PROMPT = """
You are Claude, an AI assistant created by Anthropic.

Important instructions:
1. Use <thinking> tags to show your reasoning process
2. Be helpful, harmless, and honest
3. Search the web ONLY when you need current information beyond your training data
4. When you do search, be explicit about why you're searching
5. Provide clear, well-structured responses

Remember: Your knowledge cutoff is April 2024, so search for recent events.
"""
```

## Deployment

### Local Development with LangGraph Dev

```bash
# Install dependencies
pip install -r requirements.txt

# Run development server
langgraph dev

# Server runs on http://localhost:8123
```

### Production Deployment

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

```bash
# Deploy to LangGraph Cloud
langgraph deploy
```

## Required Dependencies

```txt
langchain==0.3.14
langchain-anthropic==0.3.0
langchain-community==0.3.14
langgraph==0.2.53
langgraph-cli==0.1.75
tavily-python==0.5.0
python-dotenv==1.0.1
pydantic==2.10.4
httpx==0.28.1
```

## Summary

This implementation provides:

1. **ReAct Agent**: Claude Sonnet 4 with thinking patterns and selective search
2. **Streaming**: Real-time SSE with thinking extraction and tool usage visibility  
3. **Suggestions**: Haiku-powered follow-up questions after each response
4. **Production Ready**: Thread management, timeout protection, error handling

The agent integrates seamlessly with the Next.js client via the LangGraph Platform API, providing a complete end-to-end solution for conversational AI with transparency and reasoning capabilities.