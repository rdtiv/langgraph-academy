# LangGraph Quick Reference Guide

**Created**: 2025-05-26  
**Last Modified**: 2025-05-27

## Essential Imports

```python
# Core Graph Building
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.postgres import PostgresSaver

# Messages and State
from langchain_core.messages import (
    HumanMessage, AIMessage, SystemMessage, ToolMessage,
    RemoveMessage, trim_messages, filter_messages
)
from typing import TypedDict, Annotated, Optional, Literal, List, Any
from operator import add

# Human-in-the-Loop (Module 3)
from langgraph.errors import NodeInterrupt

# Parallelization (Module 4)
from langgraph.constants import Send

# Memory Systems (Module 5)
from langgraph.store.memory import InMemoryStore
from langgraph.store.postgres import PostgresStore
from langgraph.prebuilt import TrustCall

# Deployment (Module 6)
from langgraph_sdk import get_client
from langgraph.constants import CONFIG
```

## Core Patterns

### Basic Graph Structure
```python
# 1. Define State with Reducers
class AgentState(TypedDict):
    messages: Annotated[list, add]  # Auto-concatenates
    results: Annotated[list, add]   # Merges lists from parallel nodes
    count: int                       # Overwrites (default behavior)
    metadata: Annotated[dict, lambda x,y: {**x, **y}]  # Merges dicts

# 2. Create Nodes (always return partial state)
def agent_node(state: AgentState) -> dict:
    # Process state
    response = llm.invoke(state["messages"])
    # Return ONLY updates, not full state
    return {"messages": [response], "count": state["count"] + 1}

# 3. Build Graph
builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_edge(START, "agent")
builder.add_edge("agent", END)

# 4. Compile with Features
app = builder.compile(
    checkpointer=MemorySaver(),     # For conversation memory
    store=InMemoryStore(),           # For long-term memory
    interrupt_before=["sensitive"],  # Human-in-the-loop
    debug=False                      # Production setting
)

# 5. Invoke with Configuration
config = {"configurable": {"thread_id": "conv-123", "user_id": "user-456"}}
result = app.invoke({"messages": [HumanMessage("Hello")]}, config)
```

### Agent with Tools
```python
# Define tools with proper annotations
def search_flights(origin: str, destination: str, date: str) -> str:
    """Search for flights between cities."""
    return f"Found 5 flights from {origin} to {destination}"

def book_flight(flight_id: str, passenger: str) -> str:
    """Book a specific flight."""
    return f"Booked flight {flight_id} for {passenger}"

# Set up LLM with tools
tools = [search_flights, book_flight]
llm = ChatOpenAI(model="gpt-4o", temperature=0)
llm_with_tools = llm.bind_tools(tools)

# Agent node that can use tools
def assistant(state: MessagesState) -> dict:
    # Add system prompt for behavior
    system = SystemMessage("You are a helpful travel assistant.")
    messages = [system] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# Build graph with tool execution
builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    tools_condition,  # Routes to "tools" if tool_calls present
)
builder.add_edge("tools", "assistant")  # Loop back after tool execution

app = builder.compile()
```

## Module 3: Human-in-the-Loop

### Static Breakpoints
```python
# Compile with breakpoints
app = builder.compile(
    checkpointer=checkpointer,
    interrupt_before=["sensitive_action"],  # Pause before node
    interrupt_after=["llm_call"]           # Pause after node
)

# Run until breakpoint
for event in app.stream(input_state, config):
    print(event)
    
# Examine state at breakpoint
current_state = app.get_state(config)
print(f"Next: {current_state.next}")
print(f"Values: {current_state.values}")

# Option 1: Continue as-is
for event in app.stream(None, config):  # None = continue
    print(event)

# Option 2: Update state first
app.update_state(
    config,
    {"messages": [HumanMessage("Actually, do this instead")]},
    as_node="human"  # Record who made the update
)
# Then continue
for event in app.stream(None, config):
    print(event)
```

### Dynamic Interrupts
```python
def review_node(state: State) -> dict:
    # Conditionally interrupt based on runtime logic
    if state["risk_score"] > 0.8:
        raise NodeInterrupt(f"High risk detected: {state['risk_score']}")
    
    return {"approved": True}

# Handle interrupt
try:
    result = app.invoke(input_state, config)
except NodeInterrupt as e:
    print(f"Interrupted: {e}")
    # Get human input or modify state
    app.update_state(config, {"risk_override": True})
    # Resume
    result = app.invoke(None, config)
```

### Time Travel
```python
# Get conversation history
states = list(app.get_state_history(config))

# Find specific checkpoint
for state in states:
    if "booking confirmed" in str(state.values):
        target_state = state
        break

# Fork from that point
app.update_state(
    target_state.config,
    {"messages": [HumanMessage("Let's try different dates")]}
)

# Continue on new timeline
result = app.invoke(None, target_state.config)
```

## Module 4: Parallelization

### Fan-out/Fan-in Pattern
```python
class ParallelState(TypedDict):
    query: str
    results: Annotated[list, add]  # REQUIRED: Reducer for parallel writes
    
def search_web(state: ParallelState) -> dict:
    return {"results": [f"Web: {state['query']}"]}

def search_docs(state: ParallelState) -> dict:
    return {"results": [f"Docs: {state['query']}"]}

def search_news(state: ParallelState) -> dict:
    return {"results": [f"News: {state['query']}"]}

# Build with parallel execution
builder = StateGraph(ParallelState)
builder.add_node("search_web", search_web)
builder.add_node("search_docs", search_docs)  
builder.add_node("search_news", search_news)
builder.add_node("synthesize", synthesize_results)

# Fan-out
builder.add_edge(START, "search_web")
builder.add_edge(START, "search_docs")
builder.add_edge(START, "search_news")

# Fan-in
builder.add_edge("search_web", "synthesize")
builder.add_edge("search_docs", "synthesize")
builder.add_edge("search_news", "synthesize")
```

### Map-Reduce with Send() API
```python
class MapReduceState(TypedDict):
    documents: List[str]
    analyses: Annotated[List[dict], add]  # Collects results
    summary: str

# Map phase - distribute work dynamically
def distribute_analysis(state: MapReduceState) -> List[Send]:
    # Create parallel tasks based on runtime data
    return [
        Send("analyze_doc", {"doc": doc, "doc_id": i}) 
        for i, doc in enumerate(state["documents"])
    ]

# Worker node
def analyze_doc(state: dict) -> dict:
    analysis = {"id": state["doc_id"], "insights": f"Analysis of {state['doc'][:20]}..."}
    return {"analyses": [analysis]}  # Appends to parent state

# Reduce phase  
def summarize(state: MapReduceState) -> dict:
    summary = f"Analyzed {len(state['analyses'])} documents"
    return {"summary": summary}

# Build graph
builder = StateGraph(MapReduceState)
builder.add_node("analyze_doc", analyze_doc)
builder.add_node("summarize", summarize)

# Conditional edge returns Send objects
builder.add_conditional_edges(START, distribute_analysis, ["analyze_doc"])
builder.add_edge("analyze_doc", "summarize")
```

### Subgraph Pattern
```python
# Subgraph with its own state schema
class SubgraphState(TypedDict):
    task: str  # Input from parent
    subtask_results: List[str]
    final_result: str  # Output to parent

def create_subgraph():
    sub_builder = StateGraph(SubgraphState)
    sub_builder.add_node("process", process_subtask)
    sub_builder.add_node("finalize", finalize_subtask)
    sub_builder.add_edge(START, "process")
    sub_builder.add_edge("process", "finalize")
    sub_builder.add_edge("finalize", END)
    return sub_builder.compile()

# Parent graph
class ParentState(TypedDict):
    tasks: List[str]
    task: str  # Overlapping key for communication
    final_result: str  # Receives from subgraph
    all_results: Annotated[List[str], add]

# Use subgraph in parent
parent_builder = StateGraph(ParentState)
parent_builder.add_node("delegate", create_subgraph())

# Send multiple tasks to subgraph
def distribute_tasks(state: ParentState) -> List[Send]:
    return [Send("delegate", {"task": task}) for task in state["tasks"]]

parent_builder.add_conditional_edges(START, distribute_tasks, ["delegate"])
```

## Module 5: Memory Systems

### Store API Basics
```python
# Initialize stores
store = InMemoryStore()  # Development
store = PostgresStore(connection_string)  # Production

# Namespace MUST be tuple
namespace = ("user-123", "preferences")
namespace = ("user-123", "conversations", "2024-12")

# Basic operations
store.put(namespace, "favorite_color", {"color": "blue"})
item = store.get(namespace, "favorite_color")
store.delete(namespace, "old_key")

# Search namespace
items = store.search(namespace)  # Returns all items

# In a node with store
def memory_node(state: State, config: CONFIG, store: BaseStore) -> dict:
    user_id = config["configurable"]["user_id"]
    namespace = (user_id, "profile")
    
    # Get existing profile
    profile_data = store.get(namespace, "main")
    profile = profile_data.value if profile_data else {}
    
    # Update and save
    profile["last_seen"] = datetime.now().isoformat()
    store.put(namespace, "main", profile)
    
    return {"profile_loaded": True}
```

### Trustcall for Safe Updates
```python
from langgraph.prebuilt import TrustCall
from pydantic import BaseModel, Field

# Define schema
class UserProfile(BaseModel):
    name: Optional[str] = Field(None, description="User's name")
    location: Optional[str] = Field(None, description="City")
    preferences: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

# Initialize Trustcall
trustcall = TrustCall(UserProfile)

# In your node - NEVER regenerate, always patch
def update_profile_node(state: State, store: BaseStore) -> dict:
    # Get existing profile
    namespace = (state["user_id"], "profile")
    existing = store.get(namespace, "main")
    profile = UserProfile(**existing.value) if existing else UserProfile()
    
    # Extract patches from conversation
    result = trustcall.extract_patches(
        messages=[m.model_dump() for m in state["messages"][-5:]],
        existing=profile,
        instructions="Extract only new facts about the user"
    )
    
    # Apply patches (preserves all existing data)
    if result.patches:
        updated = result.apply_patches(profile)
        store.put(namespace, "main", updated.model_dump())
        
    return {"profile_updated": len(result.patches) > 0}
```

### Memory Agent Pattern
```python
class MemoryAgentState(MessagesState):
    user_id: str
    memory_type: Optional[str]

def analyze_memory_need(state: MemoryAgentState) -> str:
    """Route to appropriate memory handler"""
    last_msg = state["messages"][-1].content.lower()
    
    if any(phrase in last_msg for phrase in ["my name is", "i prefer", "i like"]):
        return "update_profile"
    elif any(phrase in last_msg for phrase in ["remind me", "todo", "task"]):
        return "add_todo"
    elif any(phrase in last_msg for phrase in ["remember that", "don't forget"]):
        return "save_fact"
    else:
        return "no_memory"

# Build memory-aware agent
builder = StateGraph(MemoryAgentState)
builder.add_node("chat", chat_with_memory)
builder.add_node("update_profile", update_profile_node)
builder.add_node("add_todo", add_todo_node)
builder.add_node("save_fact", save_fact_node)

builder.add_edge(START, "chat")
builder.add_conditional_edges(
    "chat", 
    analyze_memory_need,
    {
        "update_profile": "update_profile",
        "add_todo": "add_todo",
        "save_fact": "save_fact",
        "no_memory": END
    }
)
```

## Module 6: Production Deployment

### Local Development
```python
# langgraph.json
{
  "dependencies": [".", "langchain-openai"],
  "graphs": {
    "my_assistant": "./graph:app"
  },
  "env": {
    "OPENAI_API_KEY": "${OPENAI_API_KEY}",
    "LANGCHAIN_TRACING_V2": "true"
  }
}

# Run locally with hot reload
# $ langgraph dev
```

### SDK Client Usage
```python
from langgraph_sdk import get_client

# Initialize client
client = get_client(url="http://localhost:8123")

# Create thread for conversation
thread = await client.threads.create(
    metadata={"user_id": "user-123", "channel": "web"}
)

# Run with streaming
async for chunk in client.runs.stream(
    thread_id=thread["thread_id"],
    assistant_id="my_assistant",
    input={"messages": [HumanMessage("Hello!")]},
    config={"configurable": {"user_id": "user-123"}},
    stream_mode="values"
):
    print(chunk)

# Handle double-texting
run = await client.runs.create(
    thread_id=thread["thread_id"],
    assistant_id="my_assistant", 
    input={"messages": [HumanMessage("Wait, actually...")]},
    multitask_strategy="interrupt"  # or: reject, enqueue, rollback
)
```

### Configuration Pattern
```python
from dataclasses import dataclass

@dataclass  
class Configuration:
    """Runtime configuration"""
    model: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 2000
    tools_enabled: bool = True
    memory_enabled: bool = True
    
    @classmethod
    def from_configurable(cls, config: dict) -> "Configuration":
        configurable = config.get("configurable", {})
        return cls(**{k: v for k, v in configurable.items() if hasattr(cls, k)})

# Use in node
def configurable_node(state: State, config: CONFIG) -> dict:
    conf = Configuration.from_configurable(config)
    
    llm = ChatOpenAI(model=conf.model, temperature=conf.temperature)
    # ... use configuration
```

### Production Graph Pattern
```python
# Production-ready graph with monitoring
from prometheus_client import Counter, Histogram
import logging

# Metrics
node_executions = Counter('node_executions_total', 'Total node executions')
node_duration = Histogram('node_duration_seconds', 'Node execution duration')

def monitored_node(name: str):
    """Decorator for production monitoring"""
    def decorator(func):
        async def wrapper(state, config=None):
            start = time.time()
            node_executions.inc()
            
            try:
                result = await func(state, config) if config else await func(state)
                duration = time.time() - start
                node_duration.observe(duration)
                logging.info(f"{name} completed in {duration:.2f}s")
                return result
                
            except Exception as e:
                logging.error(f"{name} failed: {e}", exc_info=True)
                raise
                
        return wrapper
    return decorator

@monitored_node("process_request")
async def process_request(state: State, config: CONFIG) -> dict:
    # Your logic here
    return {"processed": True}
```

## Best Practices

### State Management
✅ Always use reducers for parallel node writes
✅ Return partial state updates, not full state
✅ Use MessagesState for conversation apps
✅ Design minimal state - only what's needed
✅ Use Annotated types for custom reducers

### Memory Patterns  
✅ Use tuples for namespaces: `(user_id, category)`
✅ Never regenerate schemas - use Trustcall patches
✅ Separate short-term (thread) and long-term (store) memory
✅ Implement retention policies for memory cleanup
✅ Use Pydantic models for memory schemas

### Human-in-the-Loop
✅ Place breakpoints before sensitive operations
✅ Use dynamic interrupts for conditional checks
✅ Always pass None to continue from interrupts
✅ Use `as_node` parameter when updating state
✅ Test time-travel locally before production

### Parallelization
✅ Always use reducers for parallel writes
✅ Keep Send state minimal and complete
✅ Design independent parallel operations
✅ Use subgraphs for reusable components
✅ Implement batching for large workloads

### Production Deployment
✅ Handle double-texting with appropriate strategy
✅ Use assistants for configuration management
✅ Implement health checks and monitoring
✅ Use environment variables for secrets
✅ Test with `langgraph dev` before deploying

## Common Pitfalls

### ❌ State Mistakes
```python
# WRONG: Mutating state
state["messages"].append(new_message)  # Never mutate!

# RIGHT: Return new state
return {"messages": [new_message]}
```

### ❌ Memory Mistakes  
```python
# WRONG: String namespace
store.put("user-123", key, value)  # Must be tuple!

# RIGHT: Tuple namespace
store.put(("user-123",), key, value)
```

### ❌ Parallel Mistakes
```python
# WRONG: No reducer
class State(TypedDict):
    results: list  # Parallel writes will conflict!

# RIGHT: With reducer
class State(TypedDict):
    results: Annotated[list, add]
```

### ❌ Deployment Mistakes
```python
# WRONG: Hardcoded configuration
llm = ChatOpenAI(model="gpt-4", temperature=0.7)

# RIGHT: Configurable
def node(state, config):
    model = config["configurable"].get("model", "gpt-4")
    llm = ChatOpenAI(model=model)
```

## Module Progression Path

1. **Start Here**: Basic graphs → Agents with tools → Checkpointing
2. **Add Intelligence**: State schemas → Reducers → Message management  
3. **Add Control**: Breakpoints → Dynamic interrupts → Time travel
4. **Add Scale**: Parallelization → Send API → Subgraphs
5. **Add Memory**: Store API → Trustcall → Memory agents
6. **Add Production**: Deployment → Monitoring → Double-texting

---
*LangGraph Academy Quick Reference - Your companion for building production AI systems*