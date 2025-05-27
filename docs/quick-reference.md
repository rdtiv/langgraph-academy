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
from langchain_core.messages import RemoveMessage, trim_messages

# Types
from typing import TypedDict, Annotated, Optional, Literal
from operator import add
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

## Common Deployment Commands
```bash
# Local development with Studio
langgraph dev

# Test graph
langgraph test

# Build for deployment
langgraph build

# Deploy to LangGraph Cloud
# (Configure via GitHub integration)
```

## Best Practices Checklist
- [ ] Use `MessagesState` for conversation-based graphs
- [ ] Return partial state updates from nodes
- [ ] Add proper type hints to tool functions
- [ ] Use prebuilt components when available
- [ ] Configure checkpointer for production
- [ ] Set recursion_limit to prevent infinite loops
- [ ] Handle errors in tool functions gracefully
- [ ] Use thread_id for conversation continuity
- [ ] Test locally with `langgraph dev` before deploying

## Common Pitfalls to Avoid
1. ❌ Mutating state directly in nodes
2. ❌ Forgetting to add checkpointer for memory
3. ❌ Missing type hints on tool functions
4. ❌ Creating infinite loops without exit conditions
5. ❌ Not handling tool errors
6. ❌ Returning full state instead of updates

## Progression
1. **1**: Basic graphs, agents, tools, memory
2. **2**: Advanced state, reducers, schemas
3. **3**: Human-in-the-loop, breakpoints
4. **4**: Parallelization, subgraphs
5. **5**: Memory systems, stores
6. **6**: Production deployment

---
*Quick reference for LangGraph Academy - Updated 2024*