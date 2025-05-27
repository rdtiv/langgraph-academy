# React LangGraph Agent Briefing Plan
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
    # 1. Extract messages from state
    # 2. Add system prompt for step-by-step thinking
    # 3. Invoke Anthropic with tools bound
    # 4. Extract <thinking> tags from response
    # 5. Return updated messages and thinking trace

# Conditional routing
def should_continue(state: ThinkingState) -> Literal["tools", "end"]:
    """Check if last message has tool calls"""
    # If tool_calls exist → "tools"
    # Otherwise → "end"

# Tool execution with result tracking
def tool_node_wrapper(state: ThinkingState) -> Dict[str, Any]:
    """Execute tools and track search results"""
    # 1. Run ToolNode with available tools
    # 2. Extract Tavily search results if used
    # 3. Update search_results in state
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
POST   /assistants/{assistant_id}/threads          # Create thread
POST   /threads/{thread_id}/runs/stream           # Stream chat
GET    /threads/{thread_id}/messages              # Get history
PATCH  /threads/{thread_id}/state                 # Update state
```

### Streaming Protocol

```typescript
// Server-Sent Events format
data: {"type": "thinking", "content": "Analyzing user query..."}
data: {"type": "tool_use", "tool": "tavily_search", "query": "latest AI news"}
data: {"type": "message", "content": "Based on my search..."}
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
    "model": "claude-3-5-sonnet-20241022",  # Latest Sonnet
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

### LangGraph Platform Setup

```yaml
# docker-compose.yml
version: '3.8'
services:
  langgraph:
    image: langchain/langgraph-server:latest
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - TAVILY_API_KEY=${TAVILY_API_KEY}
      - LANGSMITH_API_KEY=${LANGSMITH_API_KEY}
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://postgres:password@db:5432/langgraph
    ports:
      - "8123:8000"
    
  redis:
    image: redis:alpine
    
  db:
    image: postgres:15
    environment:
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=langgraph
```

### Environment Variables

```bash
# .env.production
LANGGRAPH_API_URL=https://api.langgraph.com
LANGGRAPH_API_KEY=lsv2_...
ANTHROPIC_API_KEY=sk-ant-...
TAVILY_API_KEY=tvly-...
LANGSMITH_API_KEY=ls-...
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

---

*This briefing plan provides a comprehensive roadmap for building a production-ready LangGraph agent with advanced reasoning capabilities.*