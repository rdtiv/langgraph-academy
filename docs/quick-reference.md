# LangGraph Quick Reference Guide

**Created**: 2025-05-26  
**Last Modified**: 2025-05-26

## Essential Imports
```python
# Core
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

# Messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.messages import RemoveMessage, trim_messages, merge_message_runs

# Types
from typing import TypedDict, Annotated, Optional, Literal, List
from typing_extensions import TypedDict
from operator import add

# Human-in-the-loop (Module 3)
from langgraph.errors import NodeInterrupt

# Parallelization (Module 4)
from langgraph.constants import Send

# Memory stores (Module 5)
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore

# Deployment (Module 6)
from langgraph_sdk import get_client
from langgraph.pregel.remote import RemoteGraph
```

## Basic Graph Pattern
```python
# 1. Define State
class State(MessagesState):
    # Additional fields beyond messages
    custom_field: str

# 2. Define Nodes
def my_node(state: State) -> dict:
    # Always return partial state updates
    return {"custom_field": "updated_value"}

# 3. Build Graph
builder = StateGraph(State)
builder.add_node("node_name", my_node)
builder.add_edge(START, "node_name")
builder.add_edge("node_name", END)

# 4. Compile
app = builder.compile()

# 5. Run
result = app.invoke({"messages": [HumanMessage(content="Hello")]})
```

## Agent with Tools Pattern
```python
# 1. Define Tools
def search(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"

tools = [search]

# 2. Setup LLM with Tools
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools(tools)

# 3. Define Agent Node
def agent(state: MessagesState) -> dict:
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# 4. Build Graph with Tool Execution
builder = StateGraph(MessagesState)
builder.add_node("agent", agent)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "agent")
builder.add_conditional_edges(
    "agent",
    tools_condition,  # Prebuilt condition
)
builder.add_edge("tools", "agent")

app = builder.compile()
```

## Memory/Persistence Pattern
```python
# In-memory (development)
checkpointer = MemorySaver()

# SQLite (production)
checkpointer = SqliteSaver.from_conn_string("agent.db")

# Compile with checkpointer
app = builder.compile(checkpointer=checkpointer)

# Use with thread ID
config = {"configurable": {"thread_id": "conversation-123"}}
result = app.invoke(
    {"messages": [HumanMessage(content="Hello")]},
    config
)
```

## Conditional Routing Pattern
```python
def route_decision(state: MessagesState) -> str:
    last_message = state["messages"][-1]
    
    if "urgent" in last_message.content.lower():
        return "high_priority"
    else:
        return "normal_priority"

builder.add_conditional_edges(
    "source_node",
    route_decision,
    {
        "high_priority": "urgent_handler",
        "normal_priority": "normal_handler"
    }
)
```

## State with Reducers Pattern
```python
from typing import Annotated
from operator import add

class State(TypedDict):
    messages: Annotated[list, add_messages]  # Special message reducer
    results: Annotated[list, add]  # Concatenate lists
    count: int  # Overwrites (default)
    metadata: Annotated[dict, lambda x, y: {**x, **y}]  # Merge dicts
```

## Streaming Pattern
```python
# Stream events
for event in app.stream({"messages": [msg]}, config):
    print(event)

# Stream specific values
for event in app.stream({"messages": [msg]}, config, stream_mode="values"):
    print(event)

# Async streaming
async for event in app.astream({"messages": [msg]}, config):
    print(event)
```

## Module 3: Human-in-the-Loop Patterns

### Breakpoints Pattern
```python
# Compile with breakpoint before a node
app = builder.compile(
    checkpointer=checkpointer,
    interrupt_before=["tools"]  # Pause before tools node
)

# Run until breakpoint
for event in app.stream(input, config):
    print(event)

# Get current state
state = app.get_state(config)
print("Next node:", state.next)

# Continue from breakpoint
for event in app.stream(None, config):  # None continues from last state
    print(event)
```

### Dynamic Breakpoints Pattern
```python
def my_node(state: State) -> dict:
    # Conditionally interrupt based on logic
    if len(state["input"]) > 100:
        raise NodeInterrupt(f"Input too long: {len(state['input'])}")
    
    return {"processed": True}

# Resume after addressing the interrupt
app.update_state(config, {"input": "shorter input"})
for event in app.stream(None, config):
    print(event)
```

### Edit State & Human Feedback Pattern
```python
# Update state at a breakpoint
app.update_state(
    config,
    {"messages": [HumanMessage(content="Actually, do this instead")]},
    as_node="human"  # Optional: specify which node made the update
)

# Continue execution
for event in app.stream(None, config):
    print(event)
```

### Time Travel Pattern
```python
# Get state history
states = list(app.get_state_history(config))

# Access a previous state
previous_state = states[2]  # 3rd most recent state
print(previous_state.values)

# Fork from a previous state
app.update_state(
    previous_state.config,
    {"messages": [HumanMessage(content="Let's try something different")]}
)

# Continue from forked state
for event in app.stream(None, previous_state.config):
    print(event)
```

## Module 4: Parallelization & Advanced Control Flow

### Parallel Node Execution Pattern
```python
class State(TypedDict):
    # Use reducer for parallel updates
    results: Annotated[list, operator.add]
    
# Fan-out pattern
builder.add_edge("orchestrator", "worker_1")
builder.add_edge("orchestrator", "worker_2")
builder.add_edge("orchestrator", "worker_3")

# Fan-in pattern
builder.add_edge("worker_1", "aggregator")
builder.add_edge("worker_2", "aggregator")
builder.add_edge("worker_3", "aggregator")
```

### Map-Reduce Pattern with Send API
```python
from langgraph.constants import Send

class OverallState(TypedDict):
    subjects: list[str]
    jokes: Annotated[list[str], operator.add]

# Map step - create parallel tasks
def continue_to_jokes(state: OverallState):
    return [Send("generate_joke", {"subject": s}) for s in state["subjects"]]

# Individual task state
class JokeState(TypedDict):
    subject: str

def generate_joke(state: JokeState) -> dict:
    joke = f"Joke about {state['subject']}"
    return {"jokes": [joke]}  # Adds to parent state

# Reduce step
def best_joke(state: OverallState) -> dict:
    return {"best": max(state["jokes"], key=len)}

# Build graph
builder.add_conditional_edges("generate_topics", continue_to_jokes, ["generate_joke"])
builder.add_edge("generate_joke", "best_joke")
```

### Subgraph Pattern
```python
# Define subgraph states with overlapping keys
class SubGraphState(TypedDict):
    shared_data: list  # Accessed from parent
    local_result: str  # Only in subgraph
    
class SubGraphOutputState(TypedDict):
    local_result: str  # Only output this

# Build subgraph
sub_builder = StateGraph(SubGraphState, output=SubGraphOutputState)
sub_builder.add_node("process", process_node)
sub_graph = sub_builder.compile()

# Add subgraph to parent
parent_builder = StateGraph(ParentState)
parent_builder.add_node("sub_task", sub_graph)
```

## Module 5: Memory Store Patterns

### Basic Memory Store Pattern
```python
# Initialize store
store = InMemoryStore()

# In node with store access
def node_with_memory(state: State, config: RunnableConfig, store: BaseStore):
    user_id = config["configurable"]["user_id"]
    namespace = ("memories", user_id)
    
    # Save memory
    memory_id = str(uuid.uuid4())
    store.put(namespace, memory_id, {"fact": "User likes pizza"})
    
    # Retrieve memories
    memories = store.search(namespace)
    
    # Get specific memory
    memory = store.get(namespace, memory_id)
    
    return {"result": "processed"}

# Compile with store
app = builder.compile(checkpointer=checkpointer, store=store)
```

### Profile Schema Pattern (with Trustcall)
```python
from trustcall import create_extractor
from pydantic import BaseModel, Field

class UserProfile(BaseModel):
    name: str = Field(description="User's name")
    preferences: List[str] = Field(description="User preferences")

# Create extractor for structured updates
extractor = create_extractor(
    model,
    tools=[UserProfile],
    tool_choice="UserProfile"
)

# Update existing profile
result = extractor.invoke({
    "messages": conversation,
    "existing": {"UserProfile": current_profile.model_dump()}
})
```

### Collection Schema Pattern
```python
class Memory(BaseModel):
    content: str = Field(description="Memory content")

# Enable inserts for collections
extractor = create_extractor(
    model,
    tools=[Memory],
    tool_choice="Memory",
    enable_inserts=True  # Allow adding new items
)

# Save memories to store
for memory in result["responses"]:
    store.put(namespace, str(uuid.uuid4()), memory.model_dump())
```

### Memory Agent Pattern
```python
# Tools for memory decisions
class UpdateMemory(TypedDict):
    update_type: Literal['profile', 'todo', 'instructions']

# Conditional routing based on memory type
def route_memory_update(state) -> str:
    update_type = state["messages"][-1].tool_calls[0]["args"]["update_type"]
    return f"update_{update_type}"

# Build with memory routers
builder.add_conditional_edges("decide", route_memory_update)
```

## Module 6: Production Deployment Patterns

### LangGraph SDK Pattern
```python
from langgraph_sdk import get_client

# Connect to deployment
client = get_client(url="http://localhost:8123")

# Create thread
thread = await client.threads.create()

# Run graph
async for chunk in client.runs.stream(
    thread["thread_id"],
    assistant_id="my_graph",
    input={"messages": [HumanMessage(content="Hello")]},
    config={"configurable": {"user_id": "123"}},
    stream_mode="messages-tuple"
):
    print(chunk)

# Get thread state
state = await client.threads.get_state(thread["thread_id"])

# Search store items
items = await client.store.search_items(
    ("memories", "user_123"),
    limit=10
)
```

### Remote Graph Pattern
```python
from langgraph.pregel.remote import RemoteGraph

# Connect to deployed graph
graph = RemoteGraph("my_graph", url="http://localhost:8123")

# Use like local graph
config = {"configurable": {"thread_id": "123"}}
result = graph.invoke({"messages": [msg]}, config)
```

### Assistants Pattern
```python
# Create assistant with configuration
assistant = await client.assistants.create(
    "my_graph",
    config={"configurable": {
        "user_id": "123",
        "category": "work",
        "system_prompt": "You are a helpful assistant"
    }}
)

# Update assistant (creates new version)
updated = await client.assistants.update(
    assistant["assistant_id"],
    config={"configurable": {"category": "personal"}}
)

# Use specific assistant
await client.runs.stream(
    thread["thread_id"],
    assistant_id=assistant["assistant_id"],
    input={"messages": [msg]}
)
```

### Double-Texting Patterns
```python
# Strategy 1: Reject concurrent runs
run = await client.runs.create(
    thread["thread_id"],
    "my_graph",
    input={"messages": [msg]},
    multitask_strategy="reject"
)

# Strategy 2: Enqueue runs
run = await client.runs.create(
    thread["thread_id"], 
    "my_graph",
    input={"messages": [msg]},
    multitask_strategy="enqueue"
)

# Strategy 3: Interrupt current run
run = await client.runs.create(
    thread["thread_id"],
    "my_graph", 
    input={"messages": [msg]},
    multitask_strategy="interrupt"
)

# Strategy 4: Rollback and restart
run = await client.runs.create(
    thread["thread_id"],
    "my_graph",
    input={"messages": [msg]},
    multitask_strategy="rollback"
)
```

## Deployment Commands
```bash
# Local development with Studio
langgraph dev

# Test graph
langgraph test

# Build Docker image
langgraph build -t my-image

# Using docker-compose (add .env with API keys)
docker compose up

# Deploy to LangGraph Cloud
# (Configure via GitHub integration)
```

### Deployment Structure
```
my-app/
├── langgraph.json       # Configuration
├── my_graph.py         # Graph implementation  
├── requirements.txt    # Dependencies
├── .env               # Environment variables
└── docker-compose.yml # Optional: multi-container setup
```

## Best Practices Checklist

### Core Practices
- [ ] Use `MessagesState` for conversation-based graphs
- [ ] Return partial state updates from nodes
- [ ] Add proper type hints to tool functions
- [ ] Use prebuilt components when available
- [ ] Configure checkpointer for production
- [ ] Set recursion_limit to prevent infinite loops
- [ ] Handle errors in tool functions gracefully
- [ ] Use thread_id for conversation continuity
- [ ] Test locally with `langgraph dev` before deploying

### Human-in-the-Loop (Module 3)
- [ ] Use breakpoints for user approval workflows
- [ ] Implement dynamic interrupts for conditional pauses
- [ ] Save state before critical operations for rollback
- [ ] Use `as_node` parameter when updating state externally

### Parallelization (Module 4)
- [ ] Use reducers for parallel state updates
- [ ] Design independent subgraph states with minimal overlap
- [ ] Use Send API for dynamic fan-out operations
- [ ] Consider task dependencies when parallelizing

### Memory Systems (Module 5)
- [ ] Namespace memories by (category, user_id) minimum
- [ ] Use Trustcall for complex schema updates
- [ ] Separate short-term (thread) and long-term (store) memory
- [ ] Define clear memory schemas with Pydantic

### Production (Module 6)
- [ ] Configure proper multitask strategies for double-texting
- [ ] Use assistants for versioned configurations
- [ ] Set up proper monitoring with LangSmith
- [ ] Test deployment locally before cloud deployment

## Common Pitfalls to Avoid

### Basic Pitfalls
1. ❌ Mutating state directly in nodes
2. ❌ Forgetting to add checkpointer for memory
3. ❌ Missing type hints on tool functions
4. ❌ Creating infinite loops without exit conditions
5. ❌ Not handling tool errors
6. ❌ Returning full state instead of updates

### Advanced Pitfalls (Modules 3-6)
7. ❌ Not using reducers for parallel state updates
8. ❌ Forgetting to pass `None` to continue from breakpoint
9. ❌ Using same state keys in parallel without reducers
10. ❌ Not providing output schema for subgraphs
11. ❌ Regenerating profiles instead of using Trustcall updates
12. ❌ Missing namespace hierarchy in memory store
13. ❌ Not handling concurrent runs (double-texting)
14. ❌ Forgetting to version assistants for experiments

## Module Progression

1. **Module 1**: Basic graphs, agents, tools, memory
   - StateGraph fundamentals
   - Building agents with tools
   - Basic checkpointing and persistence

2. **Module 2**: Advanced state, reducers, schemas
   - Custom state schemas
   - Reducer functions (add, merge, filter)
   - Message management and trimming

3. **Module 3**: Human-in-the-loop, breakpoints, time-travel
   - Static and dynamic breakpoints
   - State editing and human feedback
   - Time-travel debugging and forking

4. **Module 4**: Parallelization, map-reduce, subgraphs
   - Fan-out/fan-in patterns
   - Send API for dynamic parallelization
   - Subgraph composition and state overlap

5. **Module 5**: Memory stores, collections, profiles
   - Long-term memory with BaseStore
   - Profile vs collection schemas
   - Trustcall for schema updates
   - Memory agents with routing

6. **Module 6**: Deployment, assistants, production patterns
   - LangGraph Platform deployment
   - SDK and Remote Graph usage
   - Assistants for configuration
   - Double-texting strategies

---
*LangGraph Academy Quick Reference - Updated 2025-05-26*