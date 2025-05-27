# 5: Memory Systems

**Created**: 2025-05-26  
**Last Modified**: 2025-05-26

## What You'll Learn

Module 5 explores LangGraph's sophisticated memory systems for building context-aware AI applications. You'll master the Store API for cross-thread persistence, memory schemas with Trustcall for safe updates, and how to integrate long-term memory into production agents that remember users across conversations.

## Why It Matters

Memory transforms stateless AI into intelligent assistants that:
- **Remember Context**: Maintain user preferences and history across sessions
- **Personalize Responses**: Adapt behavior based on accumulated knowledge
- **Build Relationships**: Create coherent, long-term user experiences
- **Scale Efficiently**: Handle thousands of users with isolated memory spaces

## How It Works

### Core Concepts

1. **Memory Store Architecture**
   - BaseStore provides persistent key-value storage
   - Namespaces organize memories hierarchically (like folders)
   - Keys identify specific memories within namespaces
   - Values store structured memory data

2. **Memory Types**
   - **Within-thread**: Session-specific via checkpointers
   - **Cross-thread**: Persistent via LangGraph Store
   - **Semantic**: Facts and knowledge about users
   - **Procedural**: Instructions for system behavior

3. **Schema Patterns**
   - **Profile Schema**: Single evolving user profile
   - **Collection Schema**: Multiple independent memories
   - Both use Pydantic for structure and validation

4. **Trustcall Integration**
   - Safe schema updates via JSON Patch
   - Avoids regeneration data loss
   - Self-correcting with validation
   - Efficient partial updates

5. **Memory Agents**
   - Decide when/what to remember
   - Route updates by memory type
   - Provide visibility via tool calls
   - Integrate seamlessly with chat flow

### Python Patterns

#### Basic Memory Store
```python
from langgraph.store.memory import InMemoryStore
from langgraph.graph import StateGraph
from typing import TypedDict

# Initialize store
store = InMemoryStore()

class State(TypedDict):
    messages: list
    user_id: str

# Save memory
def save_fact(state: State):
    namespace = (state["user_id"], "facts")
    key = "preferences"
    value = {"favorite_color": "blue", "interests": ["coding"]}
    store.put(namespace, key, value)
    return state

# Retrieve memory
def get_facts(state: State):
    namespace = (state["user_id"], "facts")
    memories = store.search(namespace)
    return {"facts": memories}
```

#### Profile Schema with Trustcall
```python
from pydantic import BaseModel
from langgraph.prebuilt import Trustcall
from typing import Optional

class UserProfile(BaseModel):
    name: Optional[str] = None
    location: Optional[str] = None
    interests: list[str] = []
    preferences: dict = {}

# Create trustworthy updater
tc = Trustcall(UserProfile)

def update_profile(state: State):
    """Use LLM to extract profile updates"""
    namespace = (state["user_id"], "profile")
    
    # Get existing profile
    existing = store.get(namespace, "main")
    profile_data = existing.value if existing else {}
    profile = UserProfile(**profile_data)
    
    # Extract updates from conversation
    patches = tc.extract_patches(
        messages=state["messages"],
        existing=profile
    )
    
    # Apply patches and save
    updated_profile = tc.apply_patches(profile, patches)
    store.put(namespace, "main", updated_profile.model_dump())
    
    return {"profile_updated": True}
```

#### Collection Schema Pattern
```python
class Memory(BaseModel):
    content: str
    timestamp: str
    category: str
    importance: float

class MemoryCollection(BaseModel):
    memories: list[Memory] = []

def save_memory(state: State):
    """Add new memory to collection"""
    namespace = (state["user_id"], "memories")
    
    # Get existing collection
    existing = store.get(namespace, "all")
    collection_data = existing.value if existing else {}
    collection = MemoryCollection(**collection_data)
    
    # Add new memory
    new_memory = Memory(
        content=state["new_memory"],
        timestamp=datetime.now().isoformat(),
        category="conversation",
        importance=0.8
    )
    collection.memories.append(new_memory)
    
    # Save updated collection
    store.put(namespace, "all", collection.model_dump())
    return state
```

#### Memory Agent Integration
```python
# Configure memory types
memory_config = {
    "within_thread": MemorySaver(),  # Session memory
    "across_thread": InMemoryStore()  # Persistent memory
}

# Build graph with memories
builder = StateGraph(State)
builder.add_node("chat", chat_with_memory)
builder.add_node("update_profile", update_profile)
builder.add_node("save_memory", save_memory)

# Conditional routing for memory updates
def should_update_memory(state: State):
    # Agent decides what to remember
    if "personal info" in str(state["messages"][-1]):
        return "update_profile"
    elif "remember this" in str(state["messages"][-1]):
        return "save_memory"
    return "continue"

builder.add_conditional_edges(
    "chat",
    should_update_memory,
    ["update_profile", "save_memory", "continue"]
)

# Compile with both memory types
graph = builder.compile(
    checkpointer=memory_config["within_thread"],
    store=memory_config["across_thread"]
)
```

#### Configuration for Multi-User
```python
from dataclasses import dataclass
from langgraph.graph import RunnableConfig

@dataclass
class Configuration:
    user_id: str = "default-user"
    
    @classmethod
    def from_runnable_config(cls, config: RunnableConfig):
        return cls(user_id=config.get("configurable", {}).get("user_id", "default-user"))

# Use configuration in nodes
def personalized_chat(state: State, config: RunnableConfig):
    conf = Configuration.from_runnable_config(config)
    namespace = (conf.user_id, "profile")
    
    # Get user-specific memories
    profile = store.get(namespace, "main")
    
    # Personalize response based on profile
    return chat_with_context(state, profile)
```

### Common Pitfalls

1. **Regeneration Data Loss**: Creating new schemas instead of updating
   ```python
   # BAD - Loses existing data
   new_profile = extract_profile(messages)
   store.put(namespace, "profile", new_profile)
   
   # GOOD - Updates existing data
   patches = tc.extract_patches(messages, existing_profile)
   updated = tc.apply_patches(existing_profile, patches)
   ```

2. **Namespace Confusion**: Incorrect namespace structure
   ```python
   # BAD - String namespace
   namespace = f"{user_id}/memories"
   
   # GOOD - Tuple namespace
   namespace = (user_id, "memories")
   ```

3. **Memory Sprawl**: Saving everything without strategy
   ```python
   # BAD - Save every message
   for msg in messages:
       store.put(namespace, str(uuid4()), msg)
   
   # GOOD - Strategic memory selection
   if is_important(message):
       save_to_appropriate_schema(message)
   ```

4. **Missing Error Handling**: Not handling store failures
   ```python
   # BAD - No error handling
   memory = store.get(namespace, key).value
   
   # GOOD - Handle missing memories
   result = store.get(namespace, key)
   memory = result.value if result else default_memory()
   ```

5. **Inefficient Updates**: Rewriting entire collections
   ```python
   # BAD - Regenerate entire list
   todos = generate_all_todos(messages)
   
   # GOOD - Patch specific items
   patches = tc.extract_patches(messages, existing_todos)
   ```

## Key Takeaways

1. **Design Memory Schemas Carefully**: Structure determines capability
2. **Use Trustcall for Updates**: Prevents data loss and ensures validity
3. **Separate Memory Types**: Within-thread vs cross-thread for different needs
4. **Let Agents Decide**: What to remember is as important as how
5. **Namespace by User**: Ensure memory isolation in multi-user systems
6. **Monitor Memory Growth**: Implement cleanup strategies for production

## Next Steps

With robust memory systems mastered, Module 6 will show you how to deploy these sophisticated agents to production, handling real users at scale.