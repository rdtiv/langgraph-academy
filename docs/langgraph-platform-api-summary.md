# LangGraph Platform API Summary

**Created**: 2025-05-26  
**Last Modified**: 2025-05-26

## Overview
This document provides a clear, practical summary of LangGraph Platform APIs, cutting through the confusion to show you exactly what each API does and how to use them together.

## Core Concepts Quick Reference

| Concept | What It Is | When to Use |
|---------|------------|-------------|
| **Assistant** | A configured version of your graph | When you need different behaviors from the same graph |
| **Thread** | A conversation with persistent state | For maintaining context across messages |
| **Run** | An execution of your graph | Every time you want to process input |
| **Cron** | A scheduled graph execution | For periodic tasks (reports, cleanup, etc.) |
| **Webhook** | Completion notification URL | When you need to know when long tasks finish |

## The API Hierarchy

```
LangGraph Platform
├── Assistants (configure your graphs)
├── Threads (maintain conversation state)
├── Runs (execute graphs)
│   ├── Stream (real-time responses)
│   ├── Create (background tasks)
│   └── Crons (scheduled execution)
└── Webhooks (completion callbacks)
```

## Essential API Patterns

### 1. Basic Conversation Flow

```python
from langgraph_sdk import get_client

# Connect to your deployment
client = get_client(url="http://localhost:8123")

# Step 1: Create a thread (conversation container)
thread = await client.threads.create()

# Step 2: Send messages and get responses
async for chunk in client.runs.stream(
    thread["thread_id"],
    "assistant_name",  # or assistant_id
    input={"messages": [{"role": "human", "content": "Hello!"}]},
    stream_mode="values"  # or "updates", "messages"
):
    if chunk.event == "values":
        print(chunk.data)
```

### 2. Creating Custom Assistants

```python
# Create specialized versions of your graph
work_assistant = await client.assistants.create(
    "my_graph",  # The base graph name
    config={
        "configurable": {
            "system_prompt": "You are a work assistant",
            "model": "gpt-4",
            "temperature": 0.3
        }
    }
)

# Use the custom assistant
await client.runs.stream(
    thread_id,
    work_assistant["assistant_id"],
    input={"messages": [...]}
)
```

### 3. Background Tasks with Webhooks

```python
# Start a long-running task
run = await client.runs.create(
    thread_id,
    assistant_id,
    input={"task": "analyze_large_dataset"},
    webhook="https://myapp.com/webhook/task-complete"
)

# Your server will receive a POST when done:
# {
#   "run_id": "...",
#   "thread_id": "...",
#   "status": "success",
#   "result": {...}
# }
```

### 4. Scheduled Tasks (Crons)

```python
# Schedule a weekly report
cron = await client.runs.crons.create(
    assistant_id="report_generator",
    schedule="0 9 * * MON",  # Every Monday at 9am
    input={"report_type": "weekly"}
)

# List active crons
crons = await client.runs.crons.list()

# Delete when no longer needed (important!)
await client.runs.crons.delete(cron["cron_id"])
```

## Stream Modes Explained

When streaming runs, choose the right mode:

```python
# Mode 1: "values" - Get full state after each step
async for chunk in client.runs.stream(..., stream_mode="values"):
    # chunk.data contains complete current state
    print(chunk.data["messages"][-1])  # Latest message

# Mode 2: "updates" - Get only what changed
async for chunk in client.runs.stream(..., stream_mode="updates"):
    # chunk.data contains just the updates
    print(chunk.data)  # {"messages": [new_message]}

# Mode 3: "messages" - Get LangChain message format
async for chunk in client.runs.stream(..., stream_mode="messages"):
    # chunk.data is a message tuple: (message, metadata)
    message, metadata = chunk.data
```

## Common Patterns

### Pattern 1: Stateful Chat Application

```python
class ChatApp:
    def __init__(self, assistant_id):
        self.client = get_client(url="...")
        self.assistant_id = assistant_id
        self.thread = None
    
    async def start_conversation(self):
        self.thread = await self.client.threads.create()
    
    async def send_message(self, message):
        responses = []
        async for chunk in self.client.runs.stream(
            self.thread["thread_id"],
            self.assistant_id,
            input={"messages": [{"role": "human", "content": message}]},
            stream_mode="values"
        ):
            if chunk.event == "values":
                last_message = chunk.data["messages"][-1]
                if last_message["type"] == "ai":
                    responses.append(last_message["content"])
        return responses
```

### Pattern 2: Multi-Assistant System

```python
# Create specialized assistants
assistants = {
    "researcher": await client.assistants.create(
        "base_graph",
        config={"configurable": {"role": "researcher"}}
    ),
    "writer": await client.assistants.create(
        "base_graph", 
        config={"configurable": {"role": "writer"}}
    )
}

# Route to appropriate assistant
async def handle_request(request_type, message):
    assistant = assistants.get(request_type, assistants["researcher"])
    thread = await client.threads.create()
    
    return await client.runs.stream(
        thread["thread_id"],
        assistant["assistant_id"],
        input={"messages": [message]}
    )
```

### Pattern 3: Human-in-the-Loop Approval

```python
# Start a run that may need approval
thread = await client.threads.create()
run = await client.runs.create(
    thread["thread_id"],
    assistant_id,
    input={"task": "make_purchase", "amount": 1000}
)

# Check if waiting for approval
state = await client.threads.get_state(thread["thread_id"])
if state.next == ["human_approval"]:
    # Update state with approval
    await client.threads.update_state(
        thread["thread_id"],
        {"approved": True}
    )
    
    # Continue the run
    await client.runs.create(
        thread["thread_id"],
        assistant_id,
        input=None  # Continue from current state
    )
```

## Key Differences: Platform vs SDK

| Feature | LangGraph SDK | LangGraph Platform |
|---------|---------------|-------------------|
| **Purpose** | Build graphs locally | Deploy graphs to production |
| **State Management** | Manual (checkpointer) | Automatic (threads) |
| **Scaling** | Single process | Horizontal scaling |
| **Streaming** | Basic | Optimized streaming APIs |
| **Scheduling** | Manual | Built-in crons |
| **Configuration** | In code | Assistants API |
| **Monitoring** | Manual | LangGraph Studio |

## Quick Tips

1. **Always delete crons** when done - they incur charges even when idle
2. **Use webhooks** for long-running tasks instead of polling
3. **Choose the right stream mode** - "values" for full state, "updates" for changes
4. **Version your assistants** - each update creates a new version
5. **Reuse threads** for conversations - don't create new ones for each message
6. **Set timeouts** for runs to avoid hanging on errors

## Error Handling

```python
try:
    async for chunk in client.runs.stream(...):
        process(chunk)
except Exception as e:
    # Common errors:
    # - GraphInterrupt: Awaiting human input
    # - TimeoutError: Run took too long
    # - ValueError: Invalid configuration
    handle_error(e)
```

## Summary

The LangGraph Platform APIs provide a production-ready layer on top of LangGraph SDK:
- **Assistants** configure your graphs
- **Threads** maintain conversation state  
- **Runs** execute your graphs (streaming or background)
- **Crons** schedule periodic execution
- **Webhooks** notify on completion

Start with basic thread + run streaming, then add assistants for configuration, and finally crons/webhooks for advanced patterns.