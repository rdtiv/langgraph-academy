# LangGraph Platform Complete API Guide

**Created**: 2025-05-26  
**Last Modified**: 2025-05-26

## Table of Contents
1. [Platform Overview](#platform-overview)
2. [Authentication & Setup](#authentication--setup)
3. [Deployment Options](#deployment-options)
4. [Core APIs](#core-apis)
5. [Advanced Features](#advanced-features)
6. [Infrastructure & Monitoring](#infrastructure--monitoring)
7. [Migration & Best Practices](#migration--best-practices)

## Platform Overview

LangGraph Platform is a production-ready service for deploying stateful AI agents. It consists of:

- **LangGraph Server**: HTTP API server (FastAPI + PostgreSQL + Redis)
- **LangGraph Studio**: Visual IDE for debugging and testing
- **SDKs**: Python (`langgraph-sdk`) and JavaScript (`@langchain/langgraph-sdk`)

## Authentication & Setup

### 1. Authentication
```python
from langgraph_sdk import get_client

# All requests require API key
client = get_client(
    url="https://your-deployment.api.langchain.com",
    api_key="your-langsmith-api-key"  # Required for all deployments
)
```

### 2. Local Development
```bash
# Start local server
langgraph up

# Access at http://localhost:8123
# API docs at http://localhost:8123/docs
# Studio at http://localhost:8123/studio
```

## Deployment Options

| Option | Cost | Best For | Infrastructure |
|--------|------|----------|----------------|
| **Self-Hosted Lite** | Free (1M nodes/month) | Development, small projects | Your servers |
| **Cloud SaaS** | Free during beta* | Production without ops overhead | Fully managed |
| **BYOC** | Contact sales | Enterprise with data residency | Your AWS VPC |
| **Self-Hosted Enterprise** | Contact sales | Full control requirements | Your infrastructure |

*Free for LangSmith Plus/Enterprise users during beta

## Core APIs

### 1. Assistants API

```python
# Create assistant with configuration
assistant = await client.assistants.create(
    graph_id="my_graph",
    config={
        "configurable": {
            "model": "gpt-4",
            "temperature": 0.7,
            "system_prompt": "You are a helpful assistant"
        }
    },
    metadata={"version": "1.0", "purpose": "customer_support"}
)

# Update assistant (creates new version)
updated = await client.assistants.update(
    assistant["assistant_id"],
    config={"configurable": {"temperature": 0.5}}
)

# Search assistants
assistants = await client.assistants.search(
    metadata={"purpose": "customer_support"},
    limit=10
)

# Get specific version
version = await client.assistants.get_version(
    assistant["assistant_id"], 
    version=2
)

# Delete assistant
await client.assistants.delete(assistant["assistant_id"])
```

### 2. Threads API

```python
# Create thread with metadata
thread = await client.threads.create(
    metadata={"user_id": "123", "session_type": "support"}
)

# Get thread state
state = await client.threads.get_state(thread["thread_id"])

# Get thread history
history = await client.threads.get_history(thread["thread_id"])

# Update thread state (human-in-the-loop)
await client.threads.update_state(
    thread["thread_id"],
    values={"approved": True, "next_step": "process_payment"}
)

# Copy/fork thread
new_thread = await client.threads.copy(
    thread["thread_id"],
    checkpoint_id="specific-checkpoint"  # Optional: copy from specific point
)

# Delete thread
await client.threads.delete(thread["thread_id"])
```

### 3. Runs API

```python
# Stream run with all options
async for chunk in client.runs.stream(
    thread_id=thread["thread_id"],
    assistant_id=assistant["assistant_id"],
    input={"messages": [{"role": "human", "content": "Hello"}]},
    stream_mode="values",  # or "updates", "messages", "debug"
    stream_subgraphs=True,  # Include subgraph events
    interrupt_before=["human_approval"],  # Breakpoints
    interrupt_after=["tool_execution"],
    metadata={"run_type": "interactive"},
    config={"callbacks": [...]}  # LangChain callbacks
):
    print(f"Event: {chunk.event}, Data: {chunk.data}")

# Create background run
run = await client.runs.create(
    thread_id=thread["thread_id"],
    assistant_id=assistant["assistant_id"],
    input={"task": "analyze_data", "data_id": "12345"},
    webhook="https://myapp.com/webhook",
    metadata={"priority": "high"}
)

# Get run status
status = await client.runs.get(run["run_id"])

# List runs
runs = await client.runs.list(
    thread_id=thread["thread_id"],
    limit=20,
    status="success"  # Filter by status
)

# Cancel run
await client.runs.cancel(run["run_id"])

# Wait for run completion
result = await client.runs.wait(run["run_id"], timeout=300)
```

### 4. Crons API

```python
# Create cron job
cron = await client.runs.crons.create(
    assistant_id=assistant["assistant_id"],
    schedule="0 9 * * MON",  # Every Monday at 9am
    input={"task": "weekly_report"},
    metadata={"report_type": "sales"}
)

# List crons
crons = await client.runs.crons.list()

# Get specific cron
cron_details = await client.runs.crons.get(cron["cron_id"])

# Update cron
await client.runs.crons.update(
    cron["cron_id"],
    schedule="0 10 * * MON"  # Change to 10am
)

# Delete cron (IMPORTANT: to avoid charges)
await client.runs.crons.delete(cron["cron_id"])
```

### 5. Store API (Long-term Memory)

```python
# Put items in store
await client.store.put(
    namespace=["users", "123", "preferences"],
    key="theme",
    value={"mode": "dark", "accent": "blue"}
)

# Batch put
await client.store.put_batch([
    {
        "namespace": ["users", "123", "history"],
        "key": "session_1",
        "value": {"duration": 3600, "topics": ["pricing", "features"]}
    },
    {
        "namespace": ["users", "123", "history"],
        "key": "session_2",
        "value": {"duration": 1800, "topics": ["support"]}
    }
])

# Get items
item = await client.store.get(
    namespace=["users", "123", "preferences"],
    key="theme"
)

# Search store
results = await client.store.search(
    namespace_prefix=["users", "123"],
    filter={"duration": {"$gt": 1000}},
    limit=10
)

# Delete items
await client.store.delete(
    namespace=["users", "123", "preferences"],
    key="theme"
)
```

## Advanced Features

### 1. Double-Texting Strategies

Handle users sending multiple messages before response completes:

```python
# Configure strategy when creating run
await client.runs.stream(
    thread_id=thread["thread_id"],
    assistant_id=assistant["assistant_id"],
    input={"messages": [...]},
    double_text_strategy="interrupt"  # Options: reject, enqueue, interrupt, rollback
)
```

### 2. Streaming Modes Explained

```python
# Mode: "values" - Full state after each step
async for chunk in client.runs.stream(..., stream_mode="values"):
    # chunk.data = complete current state
    current_state = chunk.data

# Mode: "updates" - Only changes
async for chunk in client.runs.stream(..., stream_mode="updates"):
    # chunk.data = just what changed
    updates = chunk.data

# Mode: "messages" - LangChain message format
async for chunk in client.runs.stream(..., stream_mode="messages"):
    # chunk.data = (message, metadata) tuple
    message, metadata = chunk.data

# Mode: "debug" - Detailed execution info
async for chunk in client.runs.stream(..., stream_mode="debug"):
    # Includes internal processing details
    debug_info = chunk.data
```

### 3. Human-in-the-Loop Patterns

```python
# Set breakpoints
run = await client.runs.create(
    thread_id=thread["thread_id"],
    assistant_id=assistant["assistant_id"],
    input={"action": "delete_user", "user_id": "123"},
    interrupt_before=["confirm_deletion"]
)

# Check if waiting for human input
state = await client.threads.get_state(thread["thread_id"])
if state.next == ["human_confirmation"]:
    # Get user approval
    user_approved = await get_user_approval()
    
    # Update state and continue
    await client.threads.update_state(
        thread["thread_id"],
        values={"approved": user_approved}
    )
    
    # Resume run
    await client.runs.create(
        thread_id=thread["thread_id"],
        assistant_id=assistant["assistant_id"],
        input=None  # Continue from current state
    )
```

### 4. Subgraph Support

```python
# Stream with subgraph events
async for chunk in client.runs.stream(
    thread_id=thread["thread_id"],
    assistant_id=assistant["assistant_id"],
    input={"task": "complex_analysis"},
    stream_mode="values",
    stream_subgraphs=True  # Get events from nested graphs
):
    if chunk.event == "values":
        print(f"Main graph: {chunk.data}")
    elif chunk.event.startswith("values:"):
        subgraph = chunk.event.split(":")[1]
        print(f"Subgraph {subgraph}: {chunk.data}")
```

## Infrastructure & Monitoring

### 1. Rate Limits & Quotas

| Plan | Free Quota | Rate Limit |
|------|------------|------------|
| Developer | 100k nodes/month | 100 req/min |
| Self-Hosted Lite | 1M nodes total | Configurable |
| Cloud/BYOC | Based on plan | Configurable |

### 2. Monitoring Integration

```python
# LangSmith integration (automatic)
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-key"

# Custom callbacks
from langchain.callbacks import BaseCallbackHandler

class CustomMonitor(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        # Log to your monitoring system
        pass

# Use in run
await client.runs.create(
    ...,
    config={"callbacks": [CustomMonitor()]}
)
```

### 3. Error Handling

```python
from langgraph_sdk.errors import (
    GraphInterrupt,  # Human-in-the-loop required
    GraphTimeout,    # Execution timeout
    InvalidInput,    # Bad input data
    RateLimitError   # Too many requests
)

try:
    async for chunk in client.runs.stream(...):
        process(chunk)
except GraphInterrupt as e:
    # Handle human intervention needed
    print(f"Waiting for input at: {e.breakpoint}")
except GraphTimeout as e:
    # Handle timeout
    print(f"Run timed out after {e.timeout}s")
except RateLimitError as e:
    # Back off and retry
    await asyncio.sleep(e.retry_after)
```

## Migration & Best Practices

### From LangServe to LangGraph Platform

```python
# LangServe runnable
from langserve import Runnable

class MyChain(Runnable):
    def invoke(self, input):
        # Your logic
        pass

# Wrap in LangGraph node
from langgraph.graph import StateGraph

def langserve_node(state):
    chain = MyChain()
    result = chain.invoke(state["input"])
    return {"output": result}

graph = StateGraph(State)
graph.add_node("chain", langserve_node)
```

### Best Practices

1. **State Management**
   - Keep state minimal and serializable
   - Use reducers for complex state updates
   - Implement state validation

2. **Performance**
   - Use streaming for real-time responses
   - Implement proper pagination for lists
   - Cache frequently accessed data in Store

3. **Error Handling**
   - Always handle GraphInterrupt for human-in-the-loop
   - Implement exponential backoff for rate limits
   - Log errors with context for debugging

4. **Security**
   - Validate all inputs before processing
   - Use environment variables for sensitive data
   - Implement proper access controls

5. **Monitoring**
   - Set up alerts for failed runs
   - Monitor API usage against quotas
   - Track performance metrics

## Complete Example: Customer Support Bot

```python
from langgraph_sdk import get_client
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, MessagesState

# 1. Setup
client = get_client(url="...", api_key="...")

# 2. Create specialized assistants
support_assistant = await client.assistants.create(
    "customer_support_graph",
    config={
        "configurable": {
            "model": "gpt-4",
            "system_prompt": "You are a helpful customer support agent.",
            "tools": ["search_kb", "create_ticket", "escalate"]
        }
    }
)

# 3. Handle customer conversation
async def handle_customer(user_id: str, message: str):
    # Get or create thread for user
    threads = await client.threads.search(
        metadata={"user_id": user_id}
    )
    
    if threads:
        thread = threads[0]
    else:
        thread = await client.threads.create(
            metadata={"user_id": user_id}
        )
    
    # Stream response
    response = ""
    async for chunk in client.runs.stream(
        thread["thread_id"],
        support_assistant["assistant_id"],
        input={"messages": [{"role": "human", "content": message}]},
        stream_mode="messages",
        interrupt_before=["escalate_to_human"]  # Require approval for escalation
    ):
        if chunk.event == "messages":
            msg, _ = chunk.data
            if msg.type == "ai":
                response += msg.content
    
    return response

# 4. Background analysis
analysis_cron = await client.runs.crons.create(
    assistant_id=support_assistant["assistant_id"],
    schedule="0 0 * * *",  # Daily at midnight
    input={"task": "analyze_support_metrics"}
)

# 5. Cleanup
async def cleanup():
    await client.runs.crons.delete(analysis_cron["cron_id"])
```

## Summary

LangGraph Platform provides a complete, production-ready API for deploying stateful AI agents with:
- Comprehensive state management (threads, store)
- Flexible execution models (streaming, background, scheduled)
- Human-in-the-loop support
- Multiple deployment options
- Built-in monitoring and debugging

Start with basic thread + run patterns, add assistants for configuration, use store for persistence, and leverage advanced features as needed.