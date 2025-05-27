# 5: Memory Systems

**Created**: 2025-05-26  
**Last Modified**: 2025-05-26

## What You'll Learn

Module 5 dives deep into LangGraph's memory systems - the foundation for building AI applications that truly understand and remember their users. You'll master the Store API for persistent cross-thread memory, learn to design effective memory schemas using Pydantic models, implement safe updates with the Trustcall library, and build production-ready agents that maintain context across thousands of conversations. By the end, you'll understand how to transform stateless LLMs into intelligent assistants with genuine long-term memory.

## Why It Matters

Memory is the difference between a chatbot and an intelligent assistant. Without memory, every conversation starts from zero - users must repeatedly explain their preferences, restate their goals, and rebuild context. This creates frustrating, impersonal experiences that fail to leverage the true potential of AI systems.

With proper memory systems, your applications can:
- **Build Relationships**: Remember user preferences, history, and context across months of interaction
- **Provide Continuity**: Pick up conversations where they left off, even weeks later
- **Personalize Deeply**: Adapt responses based on accumulated knowledge about each user
- **Learn and Improve**: Track patterns and preferences to provide better assistance over time
- **Scale Efficiently**: Handle thousands of concurrent users with isolated memory spaces

## How It Works

### Core Concepts

#### 1. **Memory Store Architecture**

LangGraph's memory system is built on a flexible key-value store architecture that supports both in-memory and persistent storage backends. At its core is the `BaseStore` class, which provides a consistent interface for memory operations regardless of the underlying storage mechanism.

The store uses a hierarchical namespace system similar to file directories. Namespaces are tuples that create logical groupings of related memories. For example, `("user-123", "preferences")` might contain all preference-related memories for user 123, while `("user-123", "conversations", "2024-05")` could store conversation summaries from May 2024.

Within each namespace, individual memories are identified by unique keys. The combination of namespace and key provides a precise address for any piece of information in the system. Values stored can be any JSON-serializable data, though structured Pydantic models are recommended for complex data.

#### 2. **Memory Types and Persistence Levels**

LangGraph distinguishes between several types of memory based on scope and persistence:

**Within-Thread Memory** operates at the conversation level, maintaining state throughout a single session. This includes the message history, temporary calculations, and session-specific context. It's implemented through checkpointers that can persist this state, allowing conversations to be resumed even after interruptions.

**Cross-Thread Memory** persists across all conversations with a user. This is where you store long-term facts, preferences, and accumulated knowledge. It survives session boundaries and forms the foundation of personalized experiences. The Store API is specifically designed for this type of memory.

**Semantic Memory** contains factual information about users - their names, locations, preferences, and interests. This is typically structured data that can be queried and updated systematically.

**Procedural Memory** stores instructions and patterns for how the system should behave. This might include user-specific communication preferences ("always be formal") or task-specific instructions ("when I ask about stocks, include the ticker symbol").

#### 3. **Schema Patterns for Memory Organization**

The module introduces two primary patterns for organizing memories, each suited to different use cases:

**Profile Schema Pattern** is used when you need a single, continuously evolving representation of a user. Think of it as a living document that gets updated with new information while preserving existing data. A user profile might start empty and gradually accumulate fields like name, location, interests, and communication preferences. The key insight is that you're always working with one profile object per user, updating it rather than replacing it.

**Collection Schema Pattern** is ideal for memories that are naturally multiple and independent - like a list of todo items, conversation summaries, or saved preferences. Each memory in the collection is self-contained, and new memories are added without modifying existing ones. Collections can grow over time and support operations like filtering, sorting, and searching.

#### 4. **Trustcall: Safe Schema Updates**

One of the biggest challenges in memory systems is updating complex schemas without losing data. The naive approach - regenerating the entire schema from scratch - risks losing information not mentioned in the current conversation.

Trustcall solves this with JSON Patch operations. Instead of regenerating entire objects, it identifies specific changes and applies them surgically. For example, if a user mentions they've moved to Seattle, Trustcall generates a patch operation to update just the location field, leaving all other profile data intact.

The library also provides self-correction capabilities. If the LLM generates an invalid update (like trying to add a string to a list field), Trustcall can detect the error and retry with corrected instructions. This dramatically improves reliability in production systems.

#### 5. **Memory Agents: Intelligent Persistence**

Not every piece of information should be remembered. Memory agents act as intelligent gatekeepers, deciding what information is worth persisting and how it should be stored. They analyze conversations in real-time, extract relevant information, and route it to appropriate memory stores.

The module's `task_mAIstro` agent demonstrates this pattern. It monitors conversations for different types of memorable information - personal facts go to the user profile, action items go to a todo list, and behavioral preferences go to an instructions store. This selective approach prevents memory bloat while ensuring important information is captured.

### Python Patterns

#### Building a Complete Memory System

Let's build a production-ready memory system step by step, starting with the foundation and adding sophistication as we go.

```python
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import json

# Initialize both memory types
within_thread_memory = MemorySaver()  # For session persistence
cross_thread_memory = InMemoryStore()  # For long-term memory

class State(TypedDict):
    messages: Annotated[list, lambda x, y: x + y]
    user_id: str
    memory_updates: list  # Track what was remembered
```

#### Implementing Profile Schema with Trustcall

The profile schema pattern requires careful design to be both flexible and maintainable:

```python
from langgraph.prebuilt import TrustCall
from typing import Optional, List, Dict

class UserProfile(BaseModel):
    """Comprehensive user profile that evolves over time"""
    # Basic information
    name: Optional[str] = Field(None, description="User's preferred name")
    email: Optional[str] = Field(None, description="Contact email")
    location: Optional[str] = Field(None, description="Current city or region")
    timezone: Optional[str] = Field(None, description="User's timezone")
    
    # Preferences and interests
    interests: List[str] = Field(default_factory=list, description="Topics of interest")
    communication_style: Optional[str] = Field(None, description="Preferred communication style")
    language_preferences: List[str] = Field(default_factory=list, description="Preferred languages")
    
    # Professional information
    occupation: Optional[str] = Field(None, description="Current job or field")
    skills: List[str] = Field(default_factory=list, description="Professional skills")
    current_projects: List[Dict[str, str]] = Field(default_factory=list, description="Active projects")
    
    # Behavioral preferences
    preferred_meeting_times: List[str] = Field(default_factory=list, description="Best times for meetings")
    learning_style: Optional[str] = Field(None, description="How user prefers to learn")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    interaction_count: int = Field(0, description="Number of conversations")

# Create the Trustcall instance with the schema
profile_trustcall = TrustCall(UserProfile)

def update_user_profile(state: State) -> State:
    """Intelligently update user profile from conversation"""
    user_id = state["user_id"]
    namespace = (user_id, "profile")
    
    # Retrieve existing profile
    stored_profile = cross_thread_memory.get(namespace, "main")
    if stored_profile:
        profile = UserProfile(**stored_profile.value)
    else:
        profile = UserProfile()
    
    # Extract updates from recent messages
    recent_messages = state["messages"][-10:]  # Look at recent context
    
    # Use Trustcall to generate patches
    result = profile_trustcall.extract_patches(
        messages=[{"role": m.role, "content": m.content} for m in recent_messages],
        existing=profile,
        instructions="""
        Extract factual information about the user from the conversation.
        Only include information explicitly stated by the user.
        Do not infer or assume information.
        Focus on persistent facts, not temporary states.
        """
    )
    
    if result.patches:
        # Apply patches to create updated profile
        updated_profile = result.apply_patches(profile)
        updated_profile.last_updated = datetime.utcnow()
        updated_profile.interaction_count += 1
        
        # Store updated profile
        cross_thread_memory.put(
            namespace, 
            "main", 
            updated_profile.model_dump()
        )
        
        # Track what was updated for transparency
        state["memory_updates"].append({
            "type": "profile",
            "patches": len(result.patches),
            "timestamp": datetime.utcnow().isoformat()
        })
    
    return state
```

#### Implementing Collection Schema Pattern

Collections require different handling - each item is independent:

```python
class TodoItem(BaseModel):
    """Individual todo item with metadata"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    content: str = Field(..., description="What needs to be done")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    due_date: Optional[datetime] = None
    priority: Optional[str] = Field(None, pattern="^(high|medium|low)$")
    category: Optional[str] = None
    completed: bool = False
    completed_at: Optional[datetime] = None

class TodoCollection(BaseModel):
    """Collection of todo items"""
    items: List[TodoItem] = Field(default_factory=list)
    
    def add_item(self, content: str, **kwargs) -> TodoItem:
        """Add new todo item"""
        item = TodoItem(content=content, **kwargs)
        self.items.append(item)
        return item
    
    def complete_item(self, item_id: str) -> Optional[TodoItem]:
        """Mark item as completed"""
        for item in self.items:
            if item.id == item_id:
                item.completed = True
                item.completed_at = datetime.utcnow()
                return item
        return None
    
    def get_active_items(self) -> List[TodoItem]:
        """Get all incomplete items"""
        return [item for item in self.items if not item.completed]

# Trustcall for todo updates
todo_trustcall = TrustCall(TodoCollection)

def manage_todos(state: State) -> State:
    """Add or update todo items based on conversation"""
    user_id = state["user_id"]
    namespace = (user_id, "todos")
    
    # Get existing todos
    stored_todos = cross_thread_memory.get(namespace, "all")
    if stored_todos:
        todos = TodoCollection(**stored_todos.value)
    else:
        todos = TodoCollection()
    
    # Extract todo operations from conversation
    result = todo_trustcall.extract_patches(
        messages=[{"role": m.role, "content": m.content} for m in state["messages"][-5:]],
        existing=todos,
        instructions="""
        Look for:
        1. New tasks the user wants to remember
        2. Updates to existing tasks (completion, priority changes)
        3. Due dates or deadlines mentioned
        
        Only create todos for actionable items the user explicitly wants tracked.
        """
    )
    
    if result.patches:
        # Apply updates
        updated_todos = result.apply_patches(todos)
        
        # Store updated collection
        cross_thread_memory.put(
            namespace,
            "all",
            updated_todos.model_dump()
        )
        
        # Track updates
        state["memory_updates"].append({
            "type": "todos",
            "patches": len(result.patches),
            "active_items": len(updated_todos.get_active_items()),
            "timestamp": datetime.utcnow().isoformat()
        })
    
    return state
```

#### Building a Memory-Aware Agent

Now let's create an agent that intelligently decides when and how to use memory:

```python
from typing import Literal
from langgraph.prebuilt import ToolNode

def analyze_memory_needs(state: State) -> Literal["update_profile", "manage_todos", "continue"]:
    """Decide what type of memory update is needed"""
    last_message = state["messages"][-1].content.lower()
    
    # Simple routing logic - in production, use an LLM for this
    if any(keyword in last_message for keyword in ["my name is", "i live in", "i work as", "i prefer"]):
        return "update_profile"
    elif any(keyword in last_message for keyword in ["remind me", "todo", "task", "need to"]):
        return "manage_todos"
    else:
        return "continue"

def chat_with_memory(state: State) -> State:
    """Chat node that uses memory for context"""
    user_id = state["user_id"]
    
    # Gather memory context
    context_parts = []
    
    # Get user profile
    profile_data = cross_thread_memory.get((user_id, "profile"), "main")
    if profile_data:
        profile = UserProfile(**profile_data.value)
        context_parts.append(f"User Profile: {profile.name or 'Unknown'}, {profile.location or 'Unknown location'}")
        if profile.interests:
            context_parts.append(f"Interests: {', '.join(profile.interests)}")
    
    # Get active todos
    todos_data = cross_thread_memory.get((user_id, "todos"), "all")
    if todos_data:
        todos = TodoCollection(**todos_data.value)
        active_items = todos.get_active_items()
        if active_items:
            context_parts.append(f"Active todos: {len(active_items)}")
    
    # Add context to system message
    context = "\n".join(context_parts) if context_parts else "No previous context available."
    
    # Generate response with context
    response = llm.invoke([
        {"role": "system", "content": f"You are a helpful assistant. User context:\n{context}"},
        *state["messages"]
    ])
    
    state["messages"].append(response)
    return state

# Build the complete graph
def build_memory_graph():
    builder = StateGraph(State)
    
    # Add nodes
    builder.add_node("chat", chat_with_memory)
    builder.add_node("update_profile", update_user_profile)
    builder.add_node("manage_todos", manage_todos)
    
    # Add edges
    builder.add_edge(START, "chat")
    builder.add_conditional_edges(
        "chat",
        analyze_memory_needs,
        {
            "update_profile": "update_profile",
            "manage_todos": "manage_todos",
            "continue": END
        }
    )
    builder.add_edge("update_profile", END)
    builder.add_edge("manage_todos", END)
    
    # Compile with both memory types
    return builder.compile(
        checkpointer=within_thread_memory,
        store=cross_thread_memory
    )

# Usage
app = build_memory_graph()

# Run with user configuration
config = {
    "configurable": {
        "thread_id": "conversation-123",
        "user_id": "user-456"
    }
}

result = app.invoke(
    {"messages": [HumanMessage("My name is Alice and I work as a data scientist")], 
     "user_id": "user-456",
     "memory_updates": []},
    config=config
)
```

#### Advanced Memory Patterns

For production systems, you'll need more sophisticated patterns:

```python
class MemoryAgent:
    """Production-ready memory management system"""
    
    def __init__(self, store: BaseStore):
        self.store = store
        self.profile_tc = TrustCall(UserProfile)
        self.todo_tc = TrustCall(TodoCollection)
        
    def search_semantic_memory(self, user_id: str, query: str, limit: int = 5) -> List[Dict]:
        """Search memories using semantic similarity"""
        # In production, use embeddings for semantic search
        namespace = (user_id, "semantic_memories")
        all_memories = self.store.search(namespace)
        
        # Simple keyword matching for demo
        # Replace with vector similarity in production
        results = []
        query_lower = query.lower()
        
        for memory in all_memories:
            content = memory.value.get("content", "").lower()
            if any(word in content for word in query_lower.split()):
                results.append(memory.value)
        
        return results[:limit]
    
    def summarize_conversation(self, messages: List[Dict], user_id: str) -> None:
        """Create and store conversation summary"""
        if len(messages) < 5:  # Don't summarize short conversations
            return
            
        # Generate summary using LLM
        summary_prompt = f"""
        Summarize this conversation in 2-3 sentences.
        Focus on key topics discussed and any decisions made.
        Messages: {json.dumps([m['content'] for m in messages[-20:]])}
        """
        
        summary = llm.invoke(summary_prompt)
        
        # Store summary with metadata
        namespace = (user_id, "conversation_summaries")
        key = f"summary_{datetime.utcnow().isoformat()}"
        
        self.store.put(namespace, key, {
            "summary": summary.content,
            "message_count": len(messages),
            "timestamp": datetime.utcnow().isoformat(),
            "topics": self._extract_topics(messages)
        })
    
    def _extract_topics(self, messages: List[Dict]) -> List[str]:
        """Extract main topics from conversation"""
        # In production, use NLP or LLM for topic extraction
        # This is a simplified version
        common_topics = ["work", "family", "travel", "health", "technology", "hobbies"]
        found_topics = []
        
        text = " ".join([m['content'] for m in messages]).lower()
        for topic in common_topics:
            if topic in text:
                found_topics.append(topic)
                
        return found_topics
    
    def cleanup_old_memories(self, user_id: str, days: int = 90) -> int:
        """Remove old, low-value memories"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        removed_count = 0
        
        # Check conversation summaries
        namespace = (user_id, "conversation_summaries")
        summaries = self.store.search(namespace)
        
        for summary in summaries:
            timestamp = datetime.fromisoformat(summary.value['timestamp'])
            if timestamp < cutoff_date:
                self.store.delete(namespace, summary.key)
                removed_count += 1
        
        return removed_count
```

### Common Pitfalls

#### 1. **Data Loss Through Regeneration**

The most common and dangerous pitfall is regenerating entire data structures instead of updating them. This happens when developers treat memory updates like stateless operations:

```python
# DANGEROUS - Loses all existing profile data
def bad_update_profile(state: State):
    # This extracts a new profile from scratch every time
    new_profile = llm.invoke(
        "Extract user profile from: " + str(state["messages"])
    )
    # All previous data is lost!
    store.put(namespace, "profile", new_profile)

# CORRECT - Preserves existing data
def good_update_profile(state: State):
    # Get existing profile
    existing = store.get(namespace, "profile")
    profile = UserProfile(**existing.value) if existing else UserProfile()
    
    # Extract only updates
    patches = trustcall.extract_patches(
        messages=state["messages"],
        existing=profile
    )
    
    # Apply updates to existing profile
    updated = patches.apply_patches(profile)
    store.put(namespace, "profile", updated.model_dump())
```

#### 2. **Namespace Structure Errors**

Namespaces must be tuples, not strings. This is a subtle but critical requirement:

```python
# WRONG - String namespace causes errors
namespace = f"{user_id}/memories"  # This will fail
namespace = "user:123:profile"      # This will also fail

# CORRECT - Tuple namespace
namespace = (user_id, "memories")
namespace = (user_id, "profile", "preferences")  # Nested is fine
```

#### 3. **Unbounded Memory Growth**

Without careful management, memory stores can grow indefinitely:

```python
# PROBLEMATIC - No limits on memory growth
def save_every_message(state: State):
    for i, msg in enumerate(state["messages"]):
        store.put(
            namespace=(state["user_id"], "messages"),
            key=f"msg_{i}_{datetime.utcnow().timestamp()}",
            value=msg.dict()
        )

# BETTER - Strategic memory with limits
def save_important_messages(state: State):
    # Only save messages that contain important information
    important_keywords = ["decision", "preference", "important", "remember"]
    
    for msg in state["messages"][-10:]:  # Only recent messages
        if any(keyword in msg.content.lower() for keyword in important_keywords):
            # Save with expiration metadata
            store.put(
                namespace=(state["user_id"], "important_messages"),
                key=str(uuid4()),
                value={
                    "content": msg.content,
                    "timestamp": datetime.utcnow().isoformat(),
                    "expires_at": (datetime.utcnow() + timedelta(days=30)).isoformat()
                }
            )
```

#### 4. **Synchronization Issues**

When multiple agents or threads access the same memory, synchronization becomes critical:

```python
# RISKY - No synchronization
def concurrent_update(state: State):
    profile = store.get(namespace, "profile").value
    profile["counter"] += 1  # Race condition!
    store.put(namespace, "profile", profile)

# SAFER - Use atomic operations or locking
def safe_concurrent_update(state: State):
    # Use Trustcall for safe updates
    result = trustcall.extract_patches(
        messages=[{"role": "system", "content": "Increment counter by 1"}],
        existing=profile
    )
    # Patches are applied atomically
    updated = result.apply_patches(profile)
    store.put(namespace, "profile", updated.model_dump())
```

#### 5. **Missing Memory Context**

Forgetting to use available memory context leads to poor user experience:

```python
# POOR - Ignores user's stored preferences
def chat_without_memory(state: State):
    response = llm.invoke(state["messages"])
    return {"messages": state["messages"] + [response]}

# EXCELLENT - Uses full memory context
def chat_with_full_context(state: State):
    # Gather all relevant context
    profile = get_user_profile(state["user_id"])
    recent_summaries = get_recent_summaries(state["user_id"], limit=3)
    active_todos = get_active_todos(state["user_id"])
    
    # Build rich context
    context = f"""
    User: {profile.name} from {profile.location}
    Interests: {', '.join(profile.interests)}
    Communication style: {profile.communication_style}
    Recent topics: {', '.join([s['topics'] for s in recent_summaries])}
    Active tasks: {len(active_todos)}
    """
    
    # Generate response with full context
    response = llm.invoke([
        {"role": "system", "content": f"User context:\n{context}"},
        *state["messages"]
    ])
    
    return {"messages": state["messages"] + [response]}
```

## Key Takeaways

1. **Design Schemas for Evolution**: Your memory schemas will need to grow and change. Design them with optional fields and clear extension points. Use Pydantic models for validation and documentation.

2. **Patch, Don't Replace**: Always update existing memories rather than regenerating them. Trustcall's JSON Patch approach preserves data while allowing precise updates.

3. **Namespace Strategically**: Use hierarchical namespaces to organize memories logically. Common patterns include `(user_id, memory_type, subcategory)` for deep organization.

4. **Let Agents Decide**: Not everything should be remembered. Build intelligent agents that analyze conversations and selectively persist truly important information.

5. **Plan for Scale**: Memory systems must handle thousands of users. Design with isolation, cleanup strategies, and efficient search from the start.

6. **Monitor and Maintain**: Track memory usage, implement retention policies, and regularly clean up old or low-value memories. Production systems need active maintenance.

7. **Combine Memory Types**: Use within-thread memory for conversation state and cross-thread memory for persistence. Both are necessary for complete user experiences.

## Next Steps

Module 5 has equipped you with the knowledge to build AI systems with genuine long-term memory. Your agents can now remember users across conversations, build rich profiles over time, and provide truly personalized experiences.

Module 6 will show you how to deploy these sophisticated memory-aware agents to production, handling real users at scale with proper infrastructure, monitoring, and maintenance strategies. You'll learn to turn these powerful prototypes into reliable, production-ready services.