# Agents and Assistants in LangGraph - Learning Notes

**Created**: 2025-05-26  
**Last Modified**: 2025-05-26

## Overview
This document explores the concepts of agents and assistants in LangGraph, explaining what they are, why they're important, and how to implement them effectively. We'll cover the architecture patterns, Python implementation details, and key patterns you need to understand.

## What Are Agents and Assistants?

### Agents
An **agent** is a system that uses an LLM to decide the control flow of an application. Instead of following hard-coded logic, agents dynamically determine what actions to take based on the current state and user input.

**Key Characteristics:**
- Uses LLM for decision-making
- Can call tools and process their outputs
- Maintains state across interactions
- Implements loops for iterative problem-solving

### Assistants
**Assistants** are instances of a graph with specific configurations. They build on LangGraph's configuration system and are available only in LangGraph Platform (not the open-source library).

**Key Characteristics:**
- Multiple assistants can reference the same graph
- Each assistant can have different configurations (prompts, models, tools)
- Support versioning to track changes over time
- Tightly coupled to deployed graphs

## Why Use Agents and Assistants?

### Benefits of Agents

1. **Dynamic Control Flow**: Agents adapt their behavior based on context, unlike static workflows
2. **Tool Integration**: Seamlessly integrate and orchestrate multiple tools
3. **Iterative Problem Solving**: Can reason about results and take follow-up actions
4. **Error Recovery**: Built-in state management enables recovery from failures

### Benefits of Assistants

1. **Rapid Experimentation**: Modify agent behavior without changing code
2. **Multiple Personas**: Create specialized versions for different use cases
3. **Version Control**: Track configuration changes over time
4. **Production Flexibility**: Deploy once, configure many times

## How to Implement Agents and Assistants

### Basic Agent Architecture

#### 1. Simple Tool-Calling Agent

```python
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage

# Define your tools
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

# Set up the LLM with tools
tools = [add, multiply]
llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools(tools)

# Define the assistant node
def assistant(state: MessagesState):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# Build the graph
builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# Add edges with tool routing
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    tools_condition,  # Routes to tools or END based on tool calls
)
builder.add_edge("tools", "assistant")  # Loop back after tool execution

# Compile
graph = builder.compile()
```

#### 2. ReAct Agent Pattern

The ReAct (Reasoning + Acting) pattern is a powerful agent architecture:

```python
from langchain_core.messages import SystemMessage

# System prompt for ReAct behavior
sys_msg = SystemMessage(
    content="""You are a helpful assistant that can:
    1. Reason about what actions to take
    2. Use tools to gather information
    3. Observe the results
    4. Decide next steps based on observations
    
    Always explain your reasoning before taking actions."""
)

def react_assistant(state: MessagesState):
    # Include system message with each call
    messages = [sys_msg] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# Update the graph with ReAct assistant
builder = StateGraph(MessagesState)
builder.add_node("assistant", react_assistant)
builder.add_node("tools", ToolNode(tools))
# ... rest of the graph setup remains the same
```

### Implementing Assistants

#### 1. Configuration Setup

First, create a configuration file for your graph:

```python
# configuration.py
from typing import TypedDict

class ConfigSchema(TypedDict):
    """Configuration for the assistant."""
    model_name: str
    system_prompt: str
    temperature: float
    max_retries: int
```

#### 2. Using Configuration in Your Graph

```python
from langgraph.graph import StateGraph
from langgraph.constants import CONFIG

def configurable_assistant(state: MessagesState, config: CONFIG):
    # Access configuration values
    model_name = config.get("configurable", {}).get("model_name", "gpt-4o")
    system_prompt = config.get("configurable", {}).get("system_prompt", "You are helpful.")
    temperature = config.get("configurable", {}).get("temperature", 0.7)
    
    # Use configured LLM
    llm = ChatOpenAI(model=model_name, temperature=temperature)
    llm_with_tools = llm.bind_tools(tools)
    
    # Include configured system prompt
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    
    return {"messages": [response]}

# Build graph with configuration support
builder = StateGraph(MessagesState)
builder.add_node("assistant", configurable_assistant)
# ... rest of setup
```

#### 3. Creating and Using Assistants (Platform Only)

```python
from langgraph_sdk import get_client

# Connect to deployed graph
client = get_client(url="http://localhost:8123")

# Create a personal assistant
personal_assistant = await client.assistants.create(
    "my_graph_name",  # Name of deployed graph
    config={
        "configurable": {
            "model_name": "gpt-4o",
            "system_prompt": "You are a friendly personal assistant.",
            "temperature": 0.9,
            "max_retries": 3
        }
    }
)

# Create a work assistant with different config
work_assistant = await client.assistants.create(
    "my_graph_name",
    config={
        "configurable": {
            "model_name": "gpt-4o",
            "system_prompt": "You are a professional work assistant.",
            "temperature": 0.3,
            "max_retries": 5
        }
    }
)

# Use the assistant
thread = await client.threads.create()
async for chunk in client.runs.stream(
    thread["thread_id"],
    personal_assistant["assistant_id"],
    input={"messages": [HumanMessage(content="Help me plan my day")]},
    stream_mode="values"
):
    # Process responses
    pass
```

## Python Patterns to Master

### 1. **Type Hints and TypedDict**

Understanding Python's type system is crucial for LangGraph:

```python
from typing import TypedDict, Annotated, Sequence
from typing_extensions import TypedDict  # For Python < 3.8

class AgentState(TypedDict):
    messages: Sequence[BaseMessage]
    current_task: str
    completed_tasks: list[str]
    metadata: dict[str, Any]
```

### 2. **Async/Await Patterns**

Many LangGraph operations are asynchronous:

```python
import asyncio

async def process_with_agent(query: str):
    thread = await client.threads.create()
    
    # Stream results
    async for chunk in client.runs.stream(
        thread["thread_id"],
        assistant_id,
        input={"messages": [HumanMessage(content=query)]}
    ):
        if chunk.event == "values":
            # Process chunk
            pass

# Run async function
asyncio.run(process_with_agent("Hello"))
```

### 3. **Function Decorators and Annotations**

Tools in LangGraph use function annotations:

```python
from langchain_core.tools import tool

@tool
def search_database(query: str, limit: int = 10) -> list[dict]:
    """Search the database for relevant records.
    
    Args:
        query: Search query string
        limit: Maximum number of results to return
        
    Returns:
        List of matching records
    """
    # Implementation
    return results

# The decorator extracts the schema from type hints and docstring
```

### 4. **State Management Patterns**

Understanding reducers and state updates:

```python
from typing import Annotated
from langgraph.graph import add_messages

class StateWithReducer(TypedDict):
    # Reducer function aggregates messages
    messages: Annotated[list[BaseMessage], add_messages]
    # Simple assignment for other fields
    current_step: str
    
def node_function(state: StateWithReducer):
    # Return partial state updates
    return {
        "messages": [AIMessage(content="Done")],
        "current_step": "completed"
    }
```

### 5. **Conditional Routing**

Pattern for dynamic flow control:

```python
from typing import Literal

def route_on_completion(state: MessagesState) -> Literal["continue", "end"]:
    """Route based on the last message."""
    last_message = state["messages"][-1]
    
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "continue"
    return "end"

# Add conditional edge
builder.add_conditional_edges(
    "assistant",
    route_on_completion,
    {
        "continue": "tools",
        "end": END
    }
)
```

### 6. **Context Managers**

For resource management in agents:

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def agent_session(assistant_id: str):
    """Manage an agent session."""
    thread = await client.threads.create()
    try:
        yield thread
    finally:
        # Cleanup if needed
        pass

# Usage
async with agent_session(assistant_id) as thread:
    result = await client.runs.create(
        thread["thread_id"],
        assistant_id,
        input={"messages": messages}
    )
```

## Advanced Patterns

### Multi-Agent Systems

```python
# Define specialized agents as subgraphs
def create_research_agent():
    builder = StateGraph(MessagesState)
    # ... build research-specific graph
    return builder.compile()

def create_writer_agent():
    builder = StateGraph(MessagesState)
    # ... build writing-specific graph
    return builder.compile()

# Orchestrator that coordinates agents
class OrchestratorState(MessagesState):
    current_agent: str
    research_complete: bool
    
def orchestrator(state: OrchestratorState):
    if not state.get("research_complete", False):
        return {"current_agent": "researcher"}
    return {"current_agent": "writer"}

# Main graph combining agents
main_builder = StateGraph(OrchestratorState)
main_builder.add_node("orchestrator", orchestrator)
main_builder.add_node("researcher", create_research_agent())
main_builder.add_node("writer", create_writer_agent())
```

### Memory Patterns

```python
from langgraph.checkpoint.memory import MemorySaver

# Short-term memory (conversation state)
checkpointer = MemorySaver()

# Long-term memory (cross-conversation)
class MemoryState(MessagesState):
    user_preferences: dict
    conversation_summaries: list[str]
    
def remember_preferences(state: MemoryState):
    # Extract and store user preferences
    messages = state["messages"]
    # ... extraction logic
    return {"user_preferences": preferences}
```

## Best Practices

1. **Start Simple**: Begin with basic tool-calling agents before complex architectures
2. **Type Everything**: Use type hints for better IDE support and fewer runtime errors
3. **Handle Errors**: Implement proper error handling in nodes
4. **Test Incrementally**: Test each node independently before assembling the graph
5. **Use Streaming**: For better UX, stream responses instead of waiting for completion
6. **Version Assistants**: Track configuration changes for debugging and rollback
7. **Monitor Performance**: Log and monitor agent decisions and tool usage

## Summary

Agents and assistants in LangGraph provide powerful abstractions for building intelligent, adaptive applications. Agents handle the dynamic control flow and decision-making, while assistants (in LangGraph Platform) enable rapid experimentation through configuration. Master the Python patterns shown here—especially async programming, type hints, and state management—to build robust agent systems.

Remember: agents are about creating intelligent control flow, while assistants are about making that intelligence configurable and versionable for production use.