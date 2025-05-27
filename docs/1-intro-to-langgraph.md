# 1: Introduction to LangGraph - Learning Notes

**Created**: 2025-05-26  
**Last Modified**: 2025-05-26

## Overview
Module 1 introduces the fundamental concepts of LangGraph, a low-level orchestration framework for building stateful, multi-agent applications. This module progresses from simple state graphs to production-ready agents with memory and tool-calling capabilities.

**Current LangGraph Version Focus**: This summary reflects LangGraph patterns as of 2024, emphasizing prebuilt components and simplified patterns for rapid development.

## Core Concepts Progression

### 1. **Simple Graph Construction** (simple-graph.ipynb)
The foundation of LangGraph - building stateful applications as graphs.

#### Key Components:
- **State**: Defined using `TypedDict` to specify the shape of data flowing through the graph
- **Nodes**: Python functions that process state and return updates
- **Edges**: Connections between nodes (can be direct or conditional)
- **Special Nodes**: `START` (entry point) and `END` (termination)

#### Essential Pattern:
```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# Define state structure
class State(TypedDict):
    messages: list[str]
    count: int

# Define node functions
def process_node(state: State) -> dict:
    # Return partial state updates
    return {"count": state["count"] + 1}

# Build graph
graph = StateGraph(State)
graph.add_node("processor", process_node)
graph.add_edge(START, "processor")
graph.add_edge("processor", END)
app = graph.compile()
```

### 2. **LLM Integration with Messages** (chain.ipynb)
Integrating Large Language Models with structured message handling.

#### Key Concepts:
- **MessagesState**: Pre-built state for managing conversation history
- **Message Types**: HumanMessage, AIMessage, SystemMessage, ToolMessage
- **State Reducers**: Functions that define how state updates are merged (e.g., `add_messages`)
- **Tool Binding**: Attaching functions to LLMs for structured output

#### Essential Pattern:
```python
from langgraph.graph import MessagesState
from langchain_openai import ChatOpenAI

# Use built-in MessagesState
class AgentState(MessagesState):
    pass  # Can extend with additional fields

# Bind tools to LLM
llm = ChatOpenAI()
llm_with_tools = llm.bind_tools([my_tool])

# Node that uses LLM
def assistant(state: MessagesState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}
```

### 3. **Intelligent Routing** (router.ipynb)
Creating conditional flows based on LLM decisions.

#### Key Concepts:
- **ToolNode**: Pre-built node that executes tool calls from LLM responses
- **tools_condition**: Pre-built router that checks if LLM response contains tool calls
- **Conditional Edges**: Dynamic routing based on state or function output

#### Essential Pattern:
```python
from langgraph.prebuilt import ToolNode, tools_condition

# Add tool node
graph.add_node("tools", ToolNode(tools))

# Add conditional routing
graph.add_conditional_edges(
    "assistant",
    tools_condition,  # Routes to "tools" if tool call, END otherwise
)
```

### 4. **ReAct Agent Pattern** (agent.ipynb)
Building agents that can reason, act, and observe in a loop.

#### Key Concepts:
- **Agent Loop**: Creating cycles in the graph (tools → assistant → tools)
- **Multiple Tools**: Managing various tool functions
- **Sequential Execution**: Tools execute one at a time, results feed back
- **System Messages**: Guiding agent behavior

#### Architecture:
```
START → Assistant → (Tool Call?) → Tools → Assistant → (Done?) → END
              ↑                             ↓
              └─────────────────────────────┘
```

### 5. **Memory and Persistence** (agent-memory.ipynb)
Adding conversation memory across multiple invocations.

#### Key Concepts:
- **Checkpointer**: Saves state after each node execution
- **Thread ID**: Identifies conversation sessions
- **MemorySaver**: In-memory implementation (SQLite for production)
- **State Recovery**: Resuming conversations from saved state

#### Essential Pattern:
```python
from langgraph.checkpoint.memory import MemorySaver

# Add checkpointer
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# Invoke with thread
config = {"configurable": {"thread_id": "conversation-123"}}
result = graph.invoke({"messages": [...]}, config)
```

### 6. **Production Deployment** (deployment.ipynb)
Running graphs in production environments.

#### Key Concepts:
- **LangGraph Studio**: Browser-based local development (via `langgraph dev`)
- **LangGraph Cloud**: Managed hosting platform with GitHub integration
- **LangGraph SDK**: Client library for interacting with deployed graphs
- **Streaming**: Real-time response streaming with different modes

#### Deployment Options:
1. **Local Development**: `langgraph dev` for browser-based Studio
2. **Cloud Deployment**: GitHub integration for automatic deployment
3. **Self-hosted**: Docker containers with LangGraph API

## Python Patterns to Master

### 1. **TypedDict Usage**

**What**: TypedDict creates dictionary-like classes with fixed keys and type hints, providing structure and type safety for state management.

**Why**: LangGraph uses TypedDict for state because it:
- Provides IDE autocomplete and type checking
- Ensures consistent state structure across nodes
- Supports partial updates (crucial for node returns)
- Works seamlessly with Python's type system

**How**:
```python
from typing import TypedDict, Optional, Annotated
from langgraph.graph import add_messages

# Basic TypedDict for state
class GraphState(TypedDict):
    count: int
    messages: list[str]
    error: Optional[str]  # Optional field

# Advanced: Using Annotated with reducers
class AgentState(TypedDict):
    # Annotated allows custom merge behavior
    messages: Annotated[list, add_messages]
    summary: str
    tool_calls: list[dict]

# Node function with partial updates
def process_node(state: GraphState) -> dict:
    # ❌ Don't mutate state directly
    # state["count"] += 1  # WRONG!
    
    # ✅ Return only fields that changed
    return {
        "count": state["count"] + 1,
        # Don't need to return unchanged fields
    }

# Extending MessagesState
from langgraph.graph import MessagesState

class CustomState(MessagesState):
    # Inherits 'messages' field with add_messages reducer
    user_info: dict
    context: Optional[str] = None  # Can have defaults
```

### 2. **Function Annotations for Tools**

**What**: Properly annotated functions that LLMs can call, with type hints and docstrings that get converted to tool schemas.

**Why**: LangGraph/LangChain automatically converts these annotations into schemas that:
- Tell the LLM what parameters to provide
- Validate inputs before execution
- Generate clear error messages
- Enable reliable tool calling

**How**:
```python
# ✅ Properly annotated tool function
def search_database(
    query: str,
    limit: int = 10,
    category: Optional[str] = None
) -> list[dict]:
    """Search the product database for items.
    
    Args:
        query: The search query string
        limit: Maximum number of results to return (default: 10)
        category: Optional category filter (e.g., 'electronics', 'books')
    
    Returns:
        List of product dictionaries matching the search criteria
    """
    # Implementation here
    pass

# ❌ Poor tool annotation (avoid this)
def search(q):  # No type hints
    "searches stuff"  # Unclear docstring
    pass

# Advanced: Using Pydantic for complex inputs
from pydantic import BaseModel, Field

class SearchParams(BaseModel):
    """Parameters for advanced search."""
    query: str = Field(description="The search query")
    filters: dict[str, str] = Field(
        default_factory=dict,
        description="Key-value pairs for filtering"
    )
    
def advanced_search(params: SearchParams) -> list[dict]:
    """Perform advanced search with filters.
    
    Args:
        params: Search parameters including query and filters
    """
    # LangChain converts Pydantic model to tool schema
    pass

# Binding tools to LLM
tools = [search_database, advanced_search]
llm_with_tools = llm.bind_tools(tools)
```

### 3. **Async Patterns**

**What**: Asynchronous programming patterns for concurrent operations and better performance in I/O-bound scenarios.

**Why**: Essential for production LangGraph applications because:
- Enables handling multiple requests concurrently
- Prevents blocking on API calls (LLM, database, tools)
- Improves response times and throughput
- Required for streaming responses

**How**:
```python
import asyncio
from langchain_core.messages import HumanMessage

# Async node function
async def async_assistant(state: MessagesState) -> dict:
    # Async LLM call
    response = await llm_with_tools.ainvoke(state["messages"])
    return {"messages": [response]}

# Async tool function
async def fetch_weather(city: str) -> dict:
    """Fetch weather data asynchronously."""
    async with aiohttp.ClientSession() as session:
        async with session.get(f"https://api.weather.com/{city}") as resp:
            return await resp.json()

# Building async graph
graph = StateGraph(MessagesState)
graph.add_node("assistant", async_assistant)  # Can add async nodes
app = graph.compile()

# Async invocation
async def run_graph():
    result = await app.ainvoke({
        "messages": [HumanMessage(content="What's the weather?")]
    })
    return result

# Async streaming
async def stream_graph():
    async for event in app.astream({
        "messages": [HumanMessage(content="Tell me a story")]
    }):
        print(event)  # Handle each event as it arrives

# Running async code
# In Jupyter: await run_graph()
# In scripts: asyncio.run(run_graph())
```

### 4. **Generator Functions for Streaming**

**What**: Functions that yield results incrementally rather than returning everything at once.

**Why**: Critical for LangGraph streaming because:
- Provides real-time feedback to users
- Reduces memory usage for large responses
- Enables progressive rendering in UIs
- Allows early termination if needed

**How**:
```python
from typing import Iterator, AsyncIterator

# Sync generator for streaming
def process_items(state: GraphState) -> Iterator[dict]:
    """Process items one at a time, yielding updates."""
    items = state.get("items", [])
    
    for i, item in enumerate(items):
        # Process each item
        result = expensive_operation(item)
        
        # Yield intermediate update
        yield {
            "processed_count": i + 1,
            "last_result": result
        }
    
    # Final yield with summary
    yield {
        "status": "completed",
        "total_processed": len(items)
    }

# Async generator for streaming
async def async_process_stream(
    state: MessagesState
) -> AsyncIterator[dict]:
    """Stream tokens from LLM response."""
    # Stream from LLM
    async for chunk in llm.astream(state["messages"]):
        # Yield each chunk as it arrives
        yield {"messages": [chunk]}

# Using generators in graphs
def streaming_node(state: GraphState) -> Iterator[dict]:
    # Can return generator for streaming updates
    return process_items(state)

# Consuming streams from compiled graph
app = graph.compile()

# Sync streaming
for event in app.stream({"items": [1, 2, 3]}):
    print(f"Update: {event}")

# Async streaming with event filtering
async for event in app.astream(
    {"messages": [HumanMessage(content="Hello")]},
    stream_mode="values"  # or "updates", "debug"
):
    if "assistant" in event:
        print(event["assistant"])
```

### 5. **Error Handling Patterns** (Bonus)

**What**: Robust error handling specific to LangGraph applications.

**Why**: Essential because:
- Tools can fail (API errors, timeouts)
- LLMs can produce invalid tool calls
- State updates might conflict
- Network issues are common

**How**:
```python
from typing import Any
import logging

logger = logging.getLogger(__name__)

# Tool with error handling
def safe_api_call(endpoint: str) -> dict:
    """Make API call with proper error handling."""
    try:
        response = requests.get(endpoint, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.Timeout:
        logger.error(f"Timeout calling {endpoint}")
        return {"error": "Request timed out", "status": "failed"}
    except requests.RequestException as e:
        logger.error(f"API error: {e}")
        return {"error": str(e), "status": "failed"}

# Node with error state management
def resilient_node(state: GraphState) -> dict:
    try:
        # Risky operation
        result = process_data(state["data"])
        return {
            "result": result,
            "error": None  # Clear any previous errors
        }
    except Exception as e:
        logger.exception("Processing failed")
        return {
            "error": f"Processing failed: {str(e)}",
            "status": "error"
        }

# Conditional edge based on error state
def route_on_error(state: GraphState) -> str:
    if state.get("error"):
        return "error_handler"
    return "continue"

graph.add_conditional_edges(
    "process",
    route_on_error,
    {
        "error_handler": "handle_error",
        "continue": "next_step"
    }
)
```

## Key Takeaways

### 1. **Graphs > Chains**

**What**: LangGraph uses directed graphs instead of linear chains, enabling complex workflows with branching, loops, and parallel execution.

**Why This Matters**:
- Real-world AI applications rarely follow linear paths
- Enables decision trees, retry logic, and error handling flows
- Supports human-in-the-loop and conditional processing
- Allows multiple paths to execute in parallel for efficiency

**How to Apply**:
```python
# ❌ Chain Thinking (Limited)
# prompt → llm → parse → tool → response

# ✅ Graph Thinking (Flexible)
def route_based_on_intent(state: MessagesState) -> str:
    last_message = state["messages"][-1]
    
    if "search" in last_message.content.lower():
        return "search_branch"
    elif "calculate" in last_message.content.lower():
        return "math_branch"
    else:
        return "general_assistant"

# Build graph with multiple paths
builder = StateGraph(MessagesState)
builder.add_node("classifier", classify_intent)
builder.add_conditional_edges(
    "classifier",
    route_based_on_intent,
    {
        "search_branch": "web_search",
        "math_branch": "calculator",
        "general_assistant": "chat"
    }
)

# Note: Parallel execution in Module 1 examples actually uses
# parallel_tool_calls=False to ensure sequential execution
# Advanced parallel patterns are covered in Module 4
```

### 2. **State is Central**

**What**: All communication between nodes happens through a shared state object that flows through the graph.

**Why This Matters**:
- Single source of truth for the entire workflow
- Enables nodes to be independent and reusable
- Simplifies debugging (inspect state at any point)
- Supports both synchronous and asynchronous execution
- Allows for state persistence and recovery

**How to Apply**:
```python
# Define comprehensive state
class WorkflowState(TypedDict):
    messages: Annotated[list, add_messages]
    user_context: dict
    extracted_data: Optional[dict]
    validation_errors: list[str]
    processing_status: Literal["pending", "processing", "complete", "error"]
    metadata: dict

# Nodes communicate only through state
def extract_node(state: WorkflowState) -> dict:
    # Read from state
    messages = state["messages"]
    context = state.get("user_context", {})
    
    # Process
    extracted = extract_information(messages, context)
    
    # Write to state (partial update)
    return {
        "extracted_data": extracted,
        "processing_status": "processing"
    }

def validate_node(state: WorkflowState) -> dict:
    # Read previous node's output from state
    data = state.get("extracted_data")
    if not data:
        return {"validation_errors": ["No data to validate"]}
    
    errors = validate_data(data)
    return {
        "validation_errors": errors,
        "processing_status": "complete" if not errors else "error"
    }

# State inspection for debugging
result = app.invoke(initial_state)
print(f"Final status: {result['processing_status']}")
print(f"Errors: {result['validation_errors']}")
```

### 3. **Reducers Control State Updates**

**What**: Reducer functions define how state updates from nodes are merged into the existing state, especially important for list and dict fields.

**Why This Matters**:
- Prevents accidental state overwrites
- Enables sophisticated update patterns (append, merge, replace)
- Handles concurrent updates correctly
- Critical for message history management
- Allows custom business logic for state evolution

**How to Apply**:
```python
from typing import Annotated
from operator import add

# Built-in reducer: add_messages
class ChatState(MessagesState):
    # Messages are appended, not replaced
    summary: str  # Replaced by default
    tags: Annotated[list[str], add]  # Custom reducer

# Custom reducer function
def merge_metadata(existing: dict, update: dict) -> dict:
    """Merge metadata dicts, preserving existing keys unless explicitly updated."""
    result = existing.copy()
    for key, value in update.items():
        if value is not None:  # Only update non-None values
            result[key] = value
    return result

class AdvancedState(TypedDict):
    messages: Annotated[list, add_messages]
    metadata: Annotated[dict, merge_metadata]
    scores: Annotated[list[float], add]  # Accumulate scores
    
# How reducers work in practice
def node1(state: AdvancedState) -> dict:
    return {
        "messages": [AIMessage(content="First")],
        "metadata": {"source": "node1"},
        "scores": [0.9]
    }

def node2(state: AdvancedState) -> dict:
    return {
        "messages": [AIMessage(content="Second")],
        "metadata": {"timestamp": "2024-01-01"},
        "scores": [0.8]
    }

# After both nodes:
# messages: [Human(...), AI("First"), AI("Second")]  # Appended
# metadata: {"source": "node1", "timestamp": "2024-01-01"}  # Merged
# scores: [0.9, 0.8]  # Accumulated
```

### 4. **Tools Enable Agency**

**What**: Tools are functions that LLMs can call to interact with external systems, APIs, or perform computations.

**Why This Matters**:
- Transforms LLMs from "just chat" to autonomous agents
- Enables real-world actions (search, calculate, create, modify)
- Provides structured interfaces for reliable execution
- Allows agents to gather information they don't have
- Makes outputs verifiable and reproducible

**How to Apply**:
```python
# Tool definition best practices
def create_calendar_event(
    title: str,
    date: str,  # Format: YYYY-MM-DD
    time: str,  # Format: HH:MM
    duration_minutes: int = 60,
    attendees: Optional[list[str]] = None
) -> dict:
    """Create a calendar event with the specified details.
    
    Args:
        title: Event title/subject
        date: Event date in YYYY-MM-DD format
        time: Start time in 24-hour HH:MM format
        duration_minutes: Duration in minutes (default: 60)
        attendees: List of email addresses to invite
        
    Returns:
        Dict with event_id and status
    """
    # Validation
    try:
        event_date = datetime.strptime(f"{date} {time}", "%Y-%m-%d %H:%M")
    except ValueError:
        return {"error": "Invalid date/time format"}
    
    # Create event (mock implementation)
    event_id = create_event_in_calendar(
        title=title,
        start=event_date,
        duration=duration_minutes,
        attendees=attendees or []
    )
    
    return {
        "event_id": event_id,
        "status": "created",
        "details": {
            "title": title,
            "start": event_date.isoformat(),
            "attendees_count": len(attendees or [])
        }
    }

# Multiple tools for comprehensive agency
tools = [
    create_calendar_event,
    search_emails,
    send_message,
    create_task,
    search_knowledge_base
]

# Tool selection strategy
llm_with_tools = llm.bind_tools(tools)

# Agents can chain tools
def agent_node(state: MessagesState) -> dict:
    # LLM might:
    # 1. search_emails("meeting with John")
    # 2. Extract date/time from results
    # 3. create_calendar_event(...)
    # 4. send_message("John", "Calendar invite sent!")
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}
```

### 5. **Memory Enables Conversations**

**What**: Checkpointers persist state between invocations, enabling multi-turn conversations and workflow recovery.

**Why This Matters**:
- Users expect context retention across messages
- Enables complex multi-step workflows
- Allows interruption and resumption of tasks
- Supports user-specific personalization
- Critical for production deployment
- Enables debugging via state history

**How to Apply**:
```python
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.postgres import PostgresSaver
import uuid

# Development: In-memory
memory = MemorySaver()

# Production: Persistent storage
# SQLite for single-instance
memory = SqliteSaver.from_conn_string("agent.db")

# PostgreSQL for multi-instance
conn_string = "postgresql://user:pass@localhost:5432/agentdb"
memory = PostgresSaver.from_conn_string(conn_string)

# Compile with memory
app = builder.compile(checkpointer=memory)

# Thread management patterns
def get_thread_id(user_id: str, conversation_id: str) -> str:
    """Generate consistent thread IDs."""
    return f"{user_id}-{conversation_id}"

# Session management
class ConversationManager:
    def __init__(self, app, checkpointer):
        self.app = app
        self.checkpointer = checkpointer
    
    async def start_conversation(self, user_id: str) -> str:
        """Start new conversation for user."""
        conversation_id = str(uuid.uuid4())
        thread_id = get_thread_id(user_id, conversation_id)
        
        # Initialize with user context
        initial_state = {
            "messages": [
                SystemMessage(content=f"User ID: {user_id}"),
                HumanMessage(content="Hello")
            ]
        }
        
        config = {"configurable": {"thread_id": thread_id}}
        await self.app.ainvoke(initial_state, config)
        
        return conversation_id
    
    def get_history(self, user_id: str, conversation_id: str):
        """Retrieve conversation history."""
        thread_id = get_thread_id(user_id, conversation_id)
        return self.checkpointer.get_tuple(thread_id)

# Usage with memory
config = {"configurable": {"thread_id": "user123-chat456"}}

# First message
app.invoke(
    {"messages": [HumanMessage(content="Remember that I prefer Python")]},
    config
)

# Later message - agent remembers preference
app.invoke(
    {"messages": [HumanMessage(content="Show me a code example")]},
    config
)  # Will show Python example based on memory
```

### 6. **Production Ready**

**What**: LangGraph provides built-in deployment options from local development to cloud-scale production.

**Why This Matters**:
- Reduces time from prototype to production
- Handles scaling, monitoring, and reliability
- Provides consistent APIs across environments
- Includes observability and debugging tools
- Supports various deployment strategies
- Enterprise-ready with security and compliance

**How to Apply**:
```python
# 1. Local Development with Studio
# langgraph.json
{
    "graphs": {
        "agent": {
            "module": "agent",
            "class": "graph"
        }
    },
    "dependencies": [".", "../shared"]
}

# 2. Self-Hosted Deployment
# Dockerfile
FROM python:3.11
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["langgraph", "api", "--host", "0.0.0.0", "--port", "8000"]

# 3. Cloud Deployment
# deploy.py
from langgraph_sdk import get_client

client = get_client(url="https://api.langgraph.com")
client.deployments.create(
    name="production-agent",
    graph_id="agent",
    config={
        "environment_variables": {
            "OPENAI_API_KEY": "{{secrets.openai_key}}"
        },
        "resource_allocation": {
            "memory": "2Gi",
            "cpu": "1000m"
        }
    }
)

# 4. Production client usage
async def production_app():
    from langgraph_sdk import get_client
    
    # Initialize client
    client = get_client(
        url="https://your-deployment.langgraph.com",
        api_key=os.environ["LANGGRAPH_API_KEY"]
    )
    
    # Create assistant
    assistant = await client.assistants.create(
        graph_id="agent",
        config={
            "model": "gpt-4",
            "temperature": 0.7
        }
    )
    
    # Run with streaming
    thread = await client.threads.create()
    
    async for event in client.runs.stream(
        thread_id=thread.id,
        assistant_id=assistant.id,
        input={"messages": [{"role": "human", "content": "Hello"}]},
        stream_mode="events"
    ):
        if event.event == "messages/partial":
            print(event.data["content"], end="")

# 5. Monitoring and observability
builder.compile(
    checkpointer=checkpointer,
    debug=True,  # Enable debug logging
    tracing=True  # Enable LangSmith tracing
)
```

## Common Pitfalls to Avoid

### 1. **State Mutation**

**What**: Directly modifying the state dictionary passed to node functions instead of returning new update dictionaries.

**Why This Is Bad**:
- Breaks LangGraph's state management system
- Can cause race conditions in parallel execution
- Makes debugging difficult (state changes aren't tracked)
- Violates functional programming principles that LangGraph relies on

**How to Avoid**:
```python
# ❌ WRONG: Mutating state directly
def bad_node(state: GraphState) -> dict:
    state["count"] += 1  # Modifying in-place
    state["messages"].append("new message")  # Mutating list
    return state  # Returning the entire state

# ✅ CORRECT: Return only updates
def good_node(state: GraphState) -> dict:
    # Create new values, don't modify existing
    new_count = state["count"] + 1
    new_messages = state["messages"] + ["new message"]
    
    # Return only the fields that changed
    return {
        "count": new_count,
        "messages": new_messages
    }

# ✅ CORRECT: Using list operations safely
def safe_list_node(state: MessagesState) -> dict:
    # For MessagesState, the reducer handles list merging
    from langchain_core.messages import AIMessage
    return {
        "messages": [AIMessage(content="New message")]
        # add_messages reducer will append this properly
    }
```

### 2. **Missing or Incorrect Type Hints**

**What**: Writing tool functions without proper type annotations, making them unreliable for LLM tool calling.

**Why This Is Bad**:
- LLMs can't understand how to call the function
- No input validation before execution
- Poor error messages when things go wrong
- IDE can't provide helpful autocomplete

**How to Avoid**:
```python
# ❌ WRONG: Missing or vague annotations
def bad_tool(data):
    """Process some data"""  # Vague description
    return {"result": data}  # Unknown return type

# ❌ WRONG: Using generic types
def generic_tool(items: list) -> dict:  # list of what?
    """Process items"""
    return {"count": len(items)}

# ✅ CORRECT: Specific type hints and clear docstring
def good_tool(
    user_id: str,
    product_ids: list[str],
    include_reviews: bool = False
) -> dict[str, Any]:
    """Fetch user's product recommendations.
    
    Args:
        user_id: The unique identifier for the user
        product_ids: List of product IDs to base recommendations on
        include_reviews: Whether to include review data (default: False)
    
    Returns:
        Dictionary containing:
        - recommendations: List of recommended product IDs
        - confidence_scores: Confidence score for each recommendation
        - user_profile: Brief user preference summary
    """
    # Implementation
    return {
        "recommendations": ["prod_123", "prod_456"],
        "confidence_scores": [0.95, 0.87],
        "user_profile": "Tech enthusiast"
    }

# ✅ Using Literal for constrained choices
from typing import Literal

def search_products(
    query: str,
    category: Literal["electronics", "books", "clothing", "food"],
    sort_by: Literal["price", "rating", "date"] = "rating"
) -> list[dict]:
    """Search products with category constraints."""
    pass
```

### 3. **Forgetting Memory/Checkpointers**

**What**: Building conversational agents without adding checkpointer, resulting in no conversation memory.

**Why This Is Bad**:
- Agent forgets everything between invocations
- Can't build multi-turn conversations
- No ability to resume interrupted workflows
- Users have to repeat context constantly

**How to Avoid**:
```python
# ❌ WRONG: No checkpointer
def build_agent_no_memory():
    builder = StateGraph(MessagesState)
    builder.add_node("assistant", assistant_node)
    return builder.compile()  # No memory!

# ❌ WRONG: Checkpointer but no thread_id
app = builder.compile(checkpointer=MemorySaver())
# Using without thread_id - each call is isolated
result = app.invoke({"messages": [...]})  # No config

# ✅ CORRECT: Checkpointer with proper configuration
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

def build_agent_with_memory():
    # For development
    memory = MemorySaver()
    
    # For production
    # memory = SqliteSaver.from_conn_string("agent.db")
    
    builder = StateGraph(MessagesState)
    builder.add_node("assistant", assistant_node)
    builder.add_edge(START, "assistant")
    
    return builder.compile(checkpointer=memory)

# ✅ CORRECT: Using with thread configuration
app = build_agent_with_memory()

# Each conversation gets unique thread_id
config = {"configurable": {"thread_id": "user-123-session-456"}}

# First message
app.invoke(
    {"messages": [HumanMessage(content="Hi, I'm Alice")]}, 
    config
)

# Follow-up - agent remembers previous context
app.invoke(
    {"messages": [HumanMessage(content="What's my name?")]}, 
    config  # Same thread_id
)
```

### 4. **Creating Infinite Loops**

**What**: Building graphs with cycles that have no termination conditions, causing infinite execution.

**Why This Is Bad**:
- Graph never completes execution
- Consumes resources indefinitely
- API timeouts and cost overruns
- Poor user experience (no response)

**How to Avoid**:
```python
# ❌ WRONG: Unconditional loop
def build_infinite_loop():
    builder = StateGraph(GraphState)
    builder.add_node("process", process_node)
    builder.add_edge(START, "process")
    builder.add_edge("process", "process")  # Always loops!
    return builder.compile()

# ❌ WRONG: Condition that never terminates
def always_continue(state: GraphState) -> str:
    return "process"  # Never returns "end"

# ✅ CORRECT: Loop with termination condition
def should_continue(state: GraphState) -> str:
    # Check termination conditions
    if state.get("iterations", 0) >= 5:
        return "end"
    if state.get("found_answer"):
        return "end"
    if state.get("error"):
        return "error_handler"
    return "continue"

def build_safe_loop():
    builder = StateGraph(GraphState)
    
    def process_with_counter(state: GraphState) -> dict:
        return {
            "iterations": state.get("iterations", 0) + 1,
            "result": do_processing(state)
        }
    
    builder.add_node("process", process_with_counter)
    builder.add_conditional_edges(
        "process",
        should_continue,
        {
            "continue": "process",  # Loop back
            "end": END,
            "error_handler": "handle_error"
        }
    )
    
    return builder.compile()

# ✅ CORRECT: Using recursion limit
app = builder.compile()
# Set maximum steps to prevent runaway execution
result = app.invoke(
    initial_state,
    {"recursion_limit": 25}  # Max 25 steps
)
```

### 5. **Poor Tool Error Handling**

**What**: Not handling tool execution failures, causing the entire graph to crash when a tool fails.

**Why This Is Bad**:
- Single tool failure crashes entire agent
- No graceful degradation
- Poor user experience (cryptic errors)
- Can't retry or use alternative approaches

**How to Avoid**:
```python
# ❌ WRONG: No error handling in tool
def unsafe_api_tool(query: str) -> dict:
    response = requests.get(f"https://api.example.com/{query}")
    return response.json()  # Can fail many ways!

# ❌ WRONG: Catching but not handling properly
def bad_error_tool(query: str) -> dict:
    try:
        # risky operation
        return fetch_data(query)
    except:
        return {}  # Empty dict confuses LLM

# ✅ CORRECT: Comprehensive error handling
def robust_tool(query: str, max_retries: int = 3) -> dict:
    """Search with proper error handling and retries."""
    import time
    from typing import Optional
    
    last_error: Optional[Exception] = None
    
    for attempt in range(max_retries):
        try:
            response = requests.get(
                f"https://api.example.com/search",
                params={"q": query},
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            return {
                "status": "success",
                "results": data.get("results", []),
                "count": len(data.get("results", []))
            }
            
        except requests.Timeout:
            last_error = TimeoutError(f"Request timed out (attempt {attempt + 1})")
            time.sleep(2 ** attempt)  # Exponential backoff
            
        except requests.HTTPError as e:
            if e.response.status_code == 429:  # Rate limit
                time.sleep(5)
                continue
            return {
                "status": "error",
                "error": f"API error: {e.response.status_code}",
                "message": "The search service returned an error"
            }
            
        except Exception as e:
            last_error = e
            logger.exception(f"Unexpected error in tool: {e}")
    
    # All retries failed
    return {
        "status": "error",
        "error": str(last_error),
        "message": "Unable to complete search after multiple attempts"
    }

# ✅ CORRECT: Node that handles tool errors
def assistant_with_error_handling(state: MessagesState) -> dict:
    try:
        response = llm_with_tools.invoke(state["messages"])
        
        # Check if tool calls failed
        if response.tool_calls:
            for tool_call in response.tool_calls:
                if tool_call.get("error"):
                    # Add explanation for user
                    error_msg = AIMessage(
                        content=f"I encountered an error while using the {tool_call['name']} tool. "
                                f"Let me try a different approach."
                    )
                    return {"messages": [error_msg]}
        
        return {"messages": [response]}
        
    except Exception as e:
        # Fallback response
        fallback = AIMessage(
            content="I encountered an issue processing your request. "
                    "Could you please rephrase or try again?"
        )
        return {"messages": [fallback]}
```

### 6. **Blocking Operations in Async Context** (Bonus)

**What**: Using synchronous blocking operations in async graphs, defeating the purpose of async execution.

**Why This Is Bad**:
- Blocks the event loop
- Prevents concurrent execution
- Degrades performance
- Can cause timeouts in production

**How to Avoid**:
```python
# ❌ WRONG: Blocking I/O in async function
async def bad_async_node(state: GraphState) -> dict:
    # Blocking call in async function
    data = requests.get("https://api.example.com/data").json()
    
    # Blocking file I/O
    with open("data.txt", "r") as f:
        content = f.read()
    
    return {"data": data}

# ✅ CORRECT: Proper async I/O
import aiohttp
import aiofiles

async def good_async_node(state: GraphState) -> dict:
    # Async HTTP request
    async with aiohttp.ClientSession() as session:
        async with session.get("https://api.example.com/data") as response:
            data = await response.json()
    
    # Async file I/O
    async with aiofiles.open("data.txt", "r") as f:
        content = await f.read()
    
    return {"data": data, "file_content": content}

# ✅ CORRECT: Running sync code in async context
import asyncio
from functools import partial

async def hybrid_node(state: GraphState) -> dict:
    # For CPU-bound sync operations
    loop = asyncio.get_event_loop()
    
    # Run sync function in thread pool
    result = await loop.run_in_executor(
        None,  # Default executor
        partial(expensive_sync_function, state["input"])
    )
    
    return {"result": result}
```

## Important Notes on LangGraph Evolution

### Current Module 1 Patterns (2024)
The module currently emphasizes:
- **Prebuilt components** (`ToolNode`, `tools_condition`) for rapid development
- **Simple conditional routing** without complex patterns
- **MessagesState** as the primary state pattern
- **Sequential tool execution** (parallel_tool_calls=False)
- **Browser-based Studio** via `langgraph dev`

### Advanced Patterns (Not in Module 1)
These patterns exist in LangGraph but are introduced in later modules:
- **Command objects** for state updates with navigation
- **Send API** for dynamic edge generation
- **Complex parallel execution** (covered in Module 4)
- **Custom state schemas** beyond MessagesState (Module 2)
- **Memory Store** patterns (Module 5)

## Next Steps
- Practice building variations of the agent pattern
- Experiment with different tool combinations
- Try different checkpointer backends
- Deploy a simple agent to LangGraph Studio using `langgraph dev`
- Focus on mastering prebuilt components before custom patterns

---
*Note: This summary reflects both the conceptual capabilities of LangGraph and the actual patterns taught in Module 1. The module intentionally starts with simpler, production-ready patterns before introducing advanced concepts.*