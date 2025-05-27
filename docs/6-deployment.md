# 6: Deployment

**Created**: 2025-05-26  
**Last Modified**: 2025-05-26

## What You'll Learn

Module 6 transforms your LangGraph experiments into production-ready services. You'll master the complete deployment lifecycle - from local development with LangGraph Studio to cloud deployment with Docker and Kubernetes. You'll learn to handle real-world challenges like concurrent requests (double-texting), implement sophisticated assistant systems with versioning, manage configuration across environments, and build scalable infrastructure that can handle thousands of users. By the end, you'll understand not just how to deploy LangGraph applications, but how to operate them reliably at scale.

## Why It Matters

The gap between a working prototype and a production service is vast. A prototype might impress in a demo, but production demands answers to hard questions: How do you handle 10,000 concurrent users? What happens when users send multiple requests before the first completes? How do you update your agent's behavior without breaking existing conversations? How do you monitor, debug, and maintain AI systems that never stop running?

Production deployment is where AI applications succeed or fail. Without proper deployment:
- **Reliability Suffers**: Crashes, timeouts, and inconsistent behavior frustrate users
- **Scale Limits Growth**: Poor architecture prevents handling increased load
- **Maintenance Becomes Nightmare**: No visibility into problems or ability to fix them
- **Costs Explode**: Inefficient resource usage burns through budgets
- **Security Fails**: Exposed endpoints and poor isolation create vulnerabilities

With proper deployment, your LangGraph applications become:
- **Reliable Services**: Handle failures gracefully, maintain consistency
- **Scalable Platforms**: Grow from 10 to 10,000 users seamlessly
- **Maintainable Systems**: Update, monitor, and debug without downtime
- **Cost-Efficient**: Use resources optimally, scale dynamically
- **Production-Ready**: Secure, monitored, and professionally operated

## How It Works

### Core Concepts

#### 1. **LangGraph Platform Architecture**

LangGraph's deployment architecture is built on a foundation of battle-tested technologies, carefully orchestrated to provide reliability, scalability, and maintainability. Understanding this architecture is crucial for successful deployments.

At the heart of the system is the **LangGraph Server**, an HTTP and WebSocket server that packages your graphs into a production-ready API. This server doesn't work alone - it's supported by two critical components:

**Redis** serves as the high-performance message broker, handling all asynchronous communication between components. When you stream tokens from an LLM, trigger background runs, or coordinate between multiple workers, Redis ensures messages flow efficiently. Its pub/sub capabilities enable real-time streaming while its queue structures support reliable task distribution.

**PostgreSQL** provides durable state storage, maintaining conversation history, checkpoints, configuration, and long-term memory. Every message, every state transition, and every piece of user data is safely persisted here. PostgreSQL's ACID guarantees ensure your data remains consistent even under heavy load or system failures.

The beauty of this architecture is its flexibility. You can deploy it as:
- **LangGraph Cloud**: Fully managed service where LangChain handles all infrastructure
- **LangGraph Studio**: Local development environment for testing and debugging  
- **Self-Hosted**: Your own infrastructure using Docker, Kubernetes, or cloud services

#### 2. **Deployment Primitives**

LangGraph introduces several key primitives that form the building blocks of deployed applications:

**Runs** are single atomic executions of your graph. Each run has a unique ID, processes specific input, and produces output. Runs can be synchronous (wait for completion), asynchronous (fire-and-forget), or streaming (real-time updates). Think of runs as individual function calls to your AI system.

**Threads** represent ongoing conversations or sessions. Unlike runs which are one-shot, threads maintain state across multiple interactions. Each thread has its own checkpoint history, allowing users to have persistent, contextual conversations. Threads are isolated - one user's conversation never affects another's.

**Assistants** are configured versions of your graphs. They separate configuration from code, allowing you to experiment with different prompts, parameters, or behaviors without changing your graph implementation. Assistants support versioning, so you can test new behaviors while keeping stable versions for production.

**The Store** provides cross-thread memory, implementing the long-term memory systems from Module 5. While threads maintain conversation state, the Store maintains user profiles, preferences, and accumulated knowledge that persists across all interactions.

#### 3. **Double-Texting: Handling Concurrent Requests**

One of the most challenging aspects of production AI systems is handling "double-texting" - when users send multiple messages before receiving a response. This happens constantly in real applications: impatient users, accidental double-clicks, or genuine follow-up thoughts.

LangGraph provides four strategies to handle this:

**Reject** simply denies the second request with a 409 Conflict error. This is appropriate when you want to ensure users wait for responses, preventing confusion or wasted computation. It's the simplest strategy but can frustrate users who expect immediate acknowledgment.

**Enqueue** queues the second request to run after the first completes. This ensures all user input is processed in order, maintaining conversation coherence. It's ideal for scenarios where every message matters and order is important, like step-by-step tutorials or workflows.

**Interrupt** stops the current run, saves its progress as a checkpoint, and starts processing the new request. This mimics natural conversation where a follow-up thought takes precedence. The interrupted state is preserved, allowing potential resumption later.

**Rollback** is the most aggressive strategy - it deletes the current run entirely and starts fresh with the new input. This is useful when the user's second message completely changes direction, making the first response irrelevant.

#### 4. **Configuration Management**

Production systems need flexible configuration that can vary by environment, user, or deployment without code changes. LangGraph's configuration system provides multiple layers of customization:

**Graph-Level Configuration** defines the structure and capabilities of your agent - what tools it has access to, how state is managed, and core behavioral parameters. This is typically static and defined in code.

**Assistant Configuration** adds a layer of runtime customization. You might have different assistants for different user tiers, departments, or use cases, all running the same underlying graph but with different prompts, parameters, or restrictions.

**User Configuration** provides per-user customization through the Configuration dataclass pattern. This allows passing user IDs, preferences, feature flags, or any other user-specific data into your graph execution.

**Environment Configuration** handles deployment-specific settings like API keys, database connections, and service endpoints. These are typically managed through environment variables and deployment manifests.

#### 5. **Production Infrastructure**

Deploying LangGraph applications requires careful consideration of infrastructure components and their interactions:

**Container Architecture** packages your application into Docker images that include your code, dependencies, and runtime environment. This ensures consistency across development, staging, and production environments. The standard deployment includes three containers: the API server, Redis, and PostgreSQL.

**Service Orchestration** coordinates these containers using Docker Compose for simple deployments or Kubernetes for complex, scalable systems. Orchestration handles service discovery, health checking, restart policies, and resource allocation.

**Load Balancing** distributes requests across multiple API server instances, ensuring no single server becomes a bottleneck. This can be handled by cloud load balancers, Kubernetes services, or dedicated proxies like nginx.

**Monitoring and Observability** provides visibility into system health and performance. LangSmith integration enables tracing every LLM call, graph execution, and state transition. Metrics, logs, and traces help identify issues before they impact users.

### Python Patterns

#### Building a Production-Ready Graph

Let's build a complete production deployment, starting with a sophisticated graph that demonstrates key patterns:

```python
# task_maistro.py - Production graph implementation
from typing import TypedDict, Annotated, Literal
from datetime import datetime
from dataclasses import dataclass
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.store.memory import BaseStore
from langchain_core.messages import HumanMessage, AIMessage
import logging

# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration with validation and defaults
@dataclass
class Configuration:
    """Production configuration with validation"""
    user_id: str = "default-user"
    assistant_role: str = "helpful"
    max_tokens: int = 2000
    temperature: float = 0.7
    enable_memory: bool = True
    enable_tools: bool = True
    allowed_tools: list[str] = None
    
    def __post_init__(self):
        # Validate configuration
        if self.temperature < 0 or self.temperature > 2:
            raise ValueError(f"Invalid temperature: {self.temperature}")
        if self.max_tokens < 100 or self.max_tokens > 10000:
            raise ValueError(f"Invalid max_tokens: {self.max_tokens}")
        if self.allowed_tools is None:
            self.allowed_tools = ["search", "calculator", "weather"]
    
    @classmethod
    def from_runnable_config(cls, config: dict) -> "Configuration":
        """Create configuration from runtime config"""
        configurable = config.get("configurable", {})
        return cls(
            user_id=configurable.get("user_id", "default-user"),
            assistant_role=configurable.get("assistant_role", "helpful"),
            max_tokens=configurable.get("max_tokens", 2000),
            temperature=configurable.get("temperature", 0.7),
            enable_memory=configurable.get("enable_memory", True),
            enable_tools=configurable.get("enable_tools", True),
            allowed_tools=configurable.get("allowed_tools")
        )

# State with production considerations
class State(TypedDict):
    messages: Annotated[list, lambda x, y: x + y]
    user_id: str
    error: str | None
    metadata: dict
    
class PrivateState(State):
    """Internal state with sensitive data"""
    memory_context: dict
    tool_calls: list
    processing_time: float

# Production error handling
def error_handler(func):
    """Decorator for consistent error handling"""
    def wrapper(state: State, config: dict = None):
        try:
            start_time = datetime.utcnow()
            result = func(state, config)
            
            # Track processing time
            if hasattr(state, 'processing_time'):
                state['processing_time'] = (
                    datetime.utcnow() - start_time
                ).total_seconds()
            
            return result
            
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
            state["error"] = f"Processing error: {str(e)}"
            return state
    
    return wrapper

# Memory integration
class MemoryManager:
    """Production memory management"""
    
    def __init__(self, store: BaseStore):
        self.store = store
        
    def get_user_context(self, user_id: str) -> dict:
        """Retrieve user context with caching"""
        try:
            # Get user profile
            profile_ns = (user_id, "profile")
            profile_data = self.store.get(profile_ns, "main")
            
            # Get recent interactions
            interactions_ns = (user_id, "interactions")
            recent = self.store.search(interactions_ns, limit=5)
            
            return {
                "profile": profile_data.value if profile_data else {},
                "recent_interactions": [r.value for r in recent],
                "last_seen": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Memory retrieval error: {e}")
            return {"profile": {}, "recent_interactions": []}
    
    def save_interaction(self, user_id: str, messages: list, response: str):
        """Save interaction for future context"""
        try:
            ns = (user_id, "interactions")
            key = datetime.utcnow().isoformat()
            
            self.store.put(ns, key, {
                "timestamp": key,
                "message_count": len(messages),
                "last_human_message": messages[-1].content if messages else "",
                "assistant_response": response[:500],  # Truncate for storage
                "metadata": {
                    "model": "gpt-4",
                    "temperature": 0.7
                }
            })
        except Exception as e:
            logger.error(f"Memory save error: {e}")

# Graph nodes with production features
@error_handler
def process_input(state: PrivateState, config: dict) -> PrivateState:
    """Process and validate input"""
    conf = Configuration.from_runnable_config(config)
    
    # Validate message content
    if not state["messages"]:
        state["error"] = "No messages provided"
        return state
    
    last_message = state["messages"][-1]
    if not isinstance(last_message, HumanMessage):
        state["error"] = "Expected human message"
        return state
    
    # Check message length
    if len(last_message.content) > 10000:
        state["error"] = "Message too long (max 10000 characters)"
        return state
    
    # Add metadata
    state["metadata"] = {
        "user_id": conf.user_id,
        "timestamp": datetime.utcnow().isoformat(),
        "message_length": len(last_message.content)
    }
    
    logger.info(f"Processing input from user {conf.user_id}")
    return state

@error_handler
def enrich_with_memory(state: PrivateState, config: dict, memory_manager: MemoryManager) -> PrivateState:
    """Add memory context to state"""
    conf = Configuration.from_runnable_config(config)
    
    if not conf.enable_memory:
        state["memory_context"] = {}
        return state
    
    # Get user context
    context = memory_manager.get_user_context(conf.user_id)
    state["memory_context"] = context
    
    logger.info(f"Enriched with {len(context['recent_interactions'])} recent interactions")
    return state

@error_handler
def generate_response(state: PrivateState, config: dict, llm) -> PrivateState:
    """Generate AI response with full context"""
    conf = Configuration.from_runnable_config(config)
    
    # Build system prompt with configuration
    system_prompt = f"""You are a {conf.assistant_role} AI assistant.
    
User Profile: {state.get('memory_context', {}).get('profile', {})}
Recent Context: {len(state.get('memory_context', {}).get('recent_interactions', []))} previous interactions

Guidelines:
- Respond in a {conf.assistant_role} manner
- Keep responses under {conf.max_tokens} tokens
- Use available tools when helpful: {conf.allowed_tools if conf.enable_tools else 'No tools available'}
"""
    
    # Generate response
    messages = [
        {"role": "system", "content": system_prompt},
        *state["messages"]
    ]
    
    response = llm.invoke(
        messages,
        temperature=conf.temperature,
        max_tokens=conf.max_tokens
    )
    
    state["messages"].append(response)
    logger.info(f"Generated response with {len(response.content)} characters")
    
    return state

@error_handler
def save_interaction_node(state: PrivateState, config: dict, memory_manager: MemoryManager) -> State:
    """Save interaction and prepare final state"""
    conf = Configuration.from_runnable_config(config)
    
    if conf.enable_memory and len(state["messages"]) >= 2:
        # Save the interaction
        last_ai_message = state["messages"][-1]
        memory_manager.save_interaction(
            conf.user_id,
            state["messages"][:-1],
            last_ai_message.content
        )
    
    # Clean private state for output
    public_state = {
        "messages": state["messages"],
        "user_id": state["user_id"],
        "error": state.get("error"),
        "metadata": state["metadata"]
    }
    
    return public_state

# Build production graph
def create_graph(llm, tools: list, store: BaseStore):
    """Create production-ready graph"""
    memory_manager = MemoryManager(store)
    
    # Create graph with private state
    builder = StateGraph(PrivateState, output=State)
    
    # Add nodes with dependency injection
    builder.add_node(
        "process_input",
        lambda s, c: process_input(s, c)
    )
    builder.add_node(
        "enrich_memory",
        lambda s, c: enrich_with_memory(s, c, memory_manager)
    )
    builder.add_node(
        "generate",
        lambda s, c: generate_response(s, c, llm)
    )
    builder.add_node(
        "save_interaction",
        lambda s, c: save_interaction_node(s, c, memory_manager)
    )
    
    # Tool node with filtering
    if tools:
        filtered_tools = ToolNode(tools)
        builder.add_node("tools", filtered_tools)
    
    # Build flow with error handling
    builder.add_edge(START, "process_input")
    
    # Check for input errors
    builder.add_conditional_edges(
        "process_input",
        lambda s: "end" if s.get("error") else "continue",
        {
            "end": END,
            "continue": "enrich_memory"
        }
    )
    
    builder.add_edge("enrich_memory", "generate")
    
    # Tool routing
    if tools:
        builder.add_conditional_edges(
            "generate",
            lambda s: "tools" if s["messages"][-1].tool_calls else "save",
            {
                "tools": "tools",
                "save": "save_interaction"
            }
        )
        builder.add_edge("tools", "generate")
    else:
        builder.add_edge("generate", "save_interaction")
    
    builder.add_edge("save_interaction", END)
    
    return builder.compile()

# Export for deployment
graph = None  # Will be initialized by deployment infrastructure

def initialize_graph(llm=None, tools=None, store=None):
    """Initialize graph with dependencies"""
    global graph
    
    # Use provided dependencies or defaults
    if llm is None:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-4", temperature=0.7)
    
    if tools is None:
        tools = []  # Add your tools here
    
    if store is None:
        from langgraph.store.memory import InMemoryStore
        store = InMemoryStore()
    
    graph = create_graph(llm, tools, store)
    logger.info("Graph initialized successfully")
    
    return graph
```

#### Configuration File for Deployment

```python
# configuration.py - Runtime configuration
from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class AssistantConfiguration:
    """Assistant-specific configuration"""
    name: str = "Task Maistro"
    description: str = "A helpful AI assistant for task management"
    version: str = "1.0.0"
    
    # Model configuration
    model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 2000
    
    # Feature flags
    enable_memory: bool = True
    enable_tools: bool = True
    enable_web_search: bool = False
    
    # Behavioral configuration
    personality: str = "professional and friendly"
    response_style: str = "concise but thorough"
    
    # Tool configuration
    allowed_tools: List[str] = field(default_factory=lambda: [
        "search", "calculator", "weather"
    ])
    tool_timeout: int = 30  # seconds
    
    # Memory configuration
    memory_retention_days: int = 90
    max_memory_items: int = 1000
    
    # Safety configuration
    content_filter: bool = True
    pii_detection: bool = True
    
    @classmethod
    def from_dict(cls, data: dict) -> "AssistantConfiguration":
        """Create configuration from dictionary"""
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})
    
    def to_dict(self) -> dict:
        """Export configuration as dictionary"""
        return {
            k: getattr(self, k) 
            for k in dir(self) 
            if not k.startswith('_') and not callable(getattr(self, k))
        }

# Deployment environments
CONFIGURATIONS = {
    "development": AssistantConfiguration(
        name="Task Maistro (Dev)",
        temperature=0.9,
        enable_web_search=True,
        content_filter=False
    ),
    "staging": AssistantConfiguration(
        name="Task Maistro (Staging)",
        temperature=0.7,
        enable_web_search=True,
        content_filter=True
    ),
    "production": AssistantConfiguration(
        name="Task Maistro",
        temperature=0.7,
        enable_web_search=False,
        content_filter=True,
        pii_detection=True
    )
}

def get_configuration(environment: str = "production") -> AssistantConfiguration:
    """Get configuration for specific environment"""
    return CONFIGURATIONS.get(environment, CONFIGURATIONS["production"])
```

#### Deployment Configuration

```json
// langgraph.json - Deployment manifest
{
  "dependencies": [".", "langchain-openai", "tavily-python"],
  "graphs": {
    "task_maistro": "./task_maistro:graph"
  },
  "env": {
    "OPENAI_API_KEY": "${OPENAI_API_KEY}",
    "TAVILY_API_KEY": "${TAVILY_API_KEY}",
    "LANGSMITH_API_KEY": "${LANGSMITH_API_KEY}",
    "ENVIRONMENT": "production"
  },
  "python_version": "3.11",
  "pip_config_file": "./pip.conf",
  "dockerfile_lines": [
    "RUN apt-get update && apt-get install -y libpq-dev",
    "RUN pip install --upgrade pip"
  ]
}
```

#### Docker Deployment

```yaml
# docker-compose.yml - Production deployment
version: '3.8'

services:
  langgraph-redis:
    image: redis:7-alpine
    restart: unless-stopped
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: >
      redis-server
      --appendonly yes
      --appendfsync everysec
      --maxmemory 512mb
      --maxmemory-policy allkeys-lru
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 5

  langgraph-postgres:
    image: postgres:16-alpine
    restart: unless-stopped
    environment:
      POSTGRES_DB: langgraph
      POSTGRES_USER: langgraph
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-langgraph123}
      PGDATA: /var/lib/postgresql/data/pgdata
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U langgraph"]
      interval: 10s
      timeout: 5s
      retries: 5
    command: >
      postgres
      -c max_connections=200
      -c shared_buffers=256MB
      -c effective_cache_size=1GB
      -c maintenance_work_mem=64MB
      -c checkpoint_completion_target=0.9
      -c wal_buffers=16MB
      -c default_statistics_target=100
      -c random_page_cost=1.1
      -c effective_io_concurrency=200

  langgraph-api:
    build:
      context: .
      dockerfile: Dockerfile
    restart: unless-stopped
    ports:
      - "8123:8000"
    depends_on:
      langgraph-redis:
        condition: service_healthy
      langgraph-postgres:
        condition: service_healthy
    environment:
      # Database configuration
      POSTGRES_URI: postgresql://langgraph:${POSTGRES_PASSWORD:-langgraph123}@langgraph-postgres:5432/langgraph
      REDIS_URI: redis://langgraph-redis:6379
      
      # API configuration
      LANGSERVE_AUTH_SECRET: ${LANGSERVE_AUTH_SECRET}
      JWT_SECRET: ${JWT_SECRET}
      
      # LangChain configuration
      LANGCHAIN_TRACING_V2: "true"
      LANGCHAIN_API_KEY: ${LANGSMITH_API_KEY}
      LANGCHAIN_PROJECT: "task-maistro-production"
      
      # Model configuration
      OPENAI_API_KEY: ${OPENAI_API_KEY}
      ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY}
      
      # Application configuration
      ENVIRONMENT: production
      LOG_LEVEL: INFO
      WORKERS: 4
      
    volumes:
      - ./logs:/app/logs
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G

  # Optional: Nginx reverse proxy for production
  nginx:
    image: nginx:alpine
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - langgraph-api
    healthcheck:
      test: ["CMD", "wget", "--spider", "http://localhost/health"]
      interval: 30s
      timeout: 10s
      retries: 3

volumes:
  postgres_data:
    driver: local
  redis_data:
    driver: local

networks:
  default:
    name: langgraph-network
    driver: bridge
```

#### Production Client Implementation

```python
# client.py - Production client with error handling
import asyncio
import logging
from typing import Optional, AsyncIterator, Dict, Any
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from langgraph_sdk import get_client
from langgraph_sdk.client import LangGraphClient
import backoff

logger = logging.getLogger(__name__)

class ProductionClient:
    """Production-ready LangGraph client with retries and monitoring"""
    
    def __init__(
        self,
        url: str,
        api_key: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 3
    ):
        self.url = url
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self._client: Optional[LangGraphClient] = None
        
    @asynccontextmanager
    async def get_client(self) -> LangGraphClient:
        """Get client with connection management"""
        if self._client is None:
            self._client = get_client(
                url=self.url,
                api_key=self.api_key
            )
        
        try:
            yield self._client
        except Exception as e:
            logger.error(f"Client error: {e}")
            raise
    
    @backoff.on_exception(
        backoff.expo,
        Exception,
        max_tries=3,
        max_time=30
    )
    async def create_thread(self, metadata: Dict[str, Any] = None) -> str:
        """Create thread with retry logic"""
        async with self.get_client() as client:
            thread = await client.threads.create(
                metadata=metadata or {
                    "created_at": datetime.utcnow().isoformat(),
                    "source": "production-client"
                }
            )
            logger.info(f"Created thread: {thread['thread_id']}")
            return thread["thread_id"]
    
    async def run_with_timeout(
        self,
        thread_id: str,
        input_data: Dict[str, Any],
        config: Dict[str, Any] = None,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """Run graph with timeout and error handling"""
        timeout = timeout or self.timeout
        
        try:
            async with self.get_client() as client:
                # Start run
                run = await client.runs.create(
                    thread_id=thread_id,
                    assistant_id="task_maistro",
                    input=input_data,
                    config=config,
                    metadata={
                        "started_at": datetime.utcnow().isoformat(),
                        "timeout": timeout
                    }
                )
                
                # Wait for completion with timeout
                result = await asyncio.wait_for(
                    client.runs.join(thread_id, run["run_id"]),
                    timeout=timeout
                )
                
                return result
                
        except asyncio.TimeoutError:
            logger.error(f"Run timed out after {timeout}s")
            # Attempt to cancel the run
            try:
                async with self.get_client() as client:
                    await client.runs.cancel(thread_id, run["run_id"])
            except:
                pass
            raise
        except Exception as e:
            logger.error(f"Run failed: {e}")
            raise
    
    async def stream_run(
        self,
        thread_id: str,
        input_data: Dict[str, Any],
        config: Dict[str, Any] = None,
        stream_mode: str = "messages-tuple"
    ) -> AsyncIterator[Any]:
        """Stream graph execution with error recovery"""
        async with self.get_client() as client:
            try:
                async for chunk in client.runs.stream(
                    thread_id=thread_id,
                    assistant_id="task_maistro",
                    input=input_data,
                    config=config,
                    stream_mode=stream_mode
                ):
                    yield chunk
                    
            except Exception as e:
                logger.error(f"Stream error: {e}")
                # Yield error to client
                yield {"error": str(e), "type": "stream_error"}
    
    async def handle_double_text(
        self,
        thread_id: str,
        input_data: Dict[str, Any],
        strategy: str = "interrupt"
    ) -> Dict[str, Any]:
        """Handle double-texting with specified strategy"""
        config = {
            "multitask_strategy": strategy,
            "configurable": {
                "thread_id": thread_id
            }
        }
        
        return await self.run_with_timeout(
            thread_id=thread_id,
            input_data=input_data,
            config=config
        )
    
    async def get_thread_state(self, thread_id: str) -> Dict[str, Any]:
        """Get current thread state with caching consideration"""
        async with self.get_client() as client:
            state = await client.threads.get_state(thread_id)
            return state
    
    async def update_thread_state(
        self,
        thread_id: str,
        values: Dict[str, Any]
    ) -> None:
        """Update thread state for human-in-the-loop"""
        async with self.get_client() as client:
            await client.threads.update_state(
                thread_id=thread_id,
                values=values,
                as_node="human_feedback"
            )
    
    async def search_memories(
        self,
        user_id: str,
        query: str,
        limit: int = 10
    ) -> list:
        """Search user memories"""
        async with self.get_client() as client:
            namespace = (user_id, "memories")
            results = await client.store.search_items(
                namespace=namespace,
                query=query,
                limit=limit
            )
            return results

# Usage example
async def production_example():
    """Example production usage with full error handling"""
    client = ProductionClient(
        url="http://localhost:8123",
        api_key="your-api-key",
        timeout=30
    )
    
    try:
        # Create thread for user
        thread_id = await client.create_thread(
            metadata={"user_id": "user-123", "session": "web"}
        )
        
        # Run with double-text handling
        result = await client.handle_double_text(
            thread_id=thread_id,
            input_data={
                "messages": [{"role": "user", "content": "Help me plan my day"}]
            },
            strategy="interrupt"
        )
        
        # Stream follow-up
        async for chunk in client.stream_run(
            thread_id=thread_id,
            input_data={
                "messages": [{"role": "user", "content": "Add a meeting at 2pm"}]
            }
        ):
            if "error" not in chunk:
                print(chunk)
                
    except Exception as e:
        logger.error(f"Production error: {e}")
        # Handle error appropriately
```

#### Monitoring and Health Checks

```python
# monitoring.py - Production monitoring
from fastapi import FastAPI, Response
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import psutil
import asyncio
from datetime import datetime

# Metrics
request_count = Counter('langgraph_requests_total', 'Total requests', ['method', 'endpoint'])
request_duration = Histogram('langgraph_request_duration_seconds', 'Request duration')
active_threads = Gauge('langgraph_active_threads', 'Active conversation threads')
memory_usage = Gauge('langgraph_memory_usage_bytes', 'Memory usage in bytes')
error_count = Counter('langgraph_errors_total', 'Total errors', ['type'])

# Health check endpoint
app = FastAPI()

@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    try:
        checks = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "checks": {}
        }
        
        # Check database
        try:
            # Your database check here
            checks["checks"]["database"] = "healthy"
        except Exception as e:
            checks["checks"]["database"] = f"unhealthy: {str(e)}"
            checks["status"] = "degraded"
        
        # Check Redis
        try:
            # Your Redis check here
            checks["checks"]["redis"] = "healthy"
        except Exception as e:
            checks["checks"]["redis"] = f"unhealthy: {str(e)}"
            checks["status"] = "degraded"
        
        # Check memory usage
        memory_percent = psutil.virtual_memory().percent
        checks["checks"]["memory"] = f"{memory_percent}% used"
        if memory_percent > 90:
            checks["status"] = "degraded"
        
        # Check CPU
        cpu_percent = psutil.cpu_percent(interval=1)
        checks["checks"]["cpu"] = f"{cpu_percent}% used"
        if cpu_percent > 90:
            checks["status"] = "degraded"
        
        status_code = 200 if checks["status"] == "healthy" else 503
        return Response(
            content=json.dumps(checks),
            status_code=status_code,
            media_type="application/json"
        )
        
    except Exception as e:
        return Response(
            content=json.dumps({
                "status": "unhealthy",
                "error": str(e)
            }),
            status_code=503,
            media_type="application/json"
        )

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    # Update runtime metrics
    memory_usage.set(psutil.Process().memory_info().rss)
    
    return Response(
        content=generate_latest(),
        media_type="text/plain"
    )

# Middleware for request tracking
@app.middleware("http")
async def track_requests(request, call_next):
    """Track all requests for monitoring"""
    start_time = datetime.utcnow()
    
    # Track request
    request_count.labels(
        method=request.method,
        endpoint=request.url.path
    ).inc()
    
    try:
        response = await call_next(request)
        
        # Track duration
        duration = (datetime.utcnow() - start_time).total_seconds()
        request_duration.observe(duration)
        
        return response
        
    except Exception as e:
        error_count.labels(type=type(e).__name__).inc()
        raise
```

### Common Pitfalls

#### 1. **Ignoring Double-Texting Scenarios**

The most common production issue is failing to handle concurrent requests properly. Users will double-text, and your system must handle it gracefully:

```python
# PROBLEMATIC - No double-text handling
async def basic_run(client, thread_id, message):
    # This will fail with 409 errors when users double-text
    return await client.runs.create(
        thread_id=thread_id,
        input={"messages": [message]}
    )

# ROBUST - Proper double-text handling
async def production_run(client, thread_id, message, strategy="interrupt"):
    """Handle double-texting based on use case"""
    config = {
        "multitask_strategy": strategy,  # reject, enqueue, interrupt, or rollback
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": f"conversation_{datetime.now().date()}"
        }
    }
    
    try:
        return await client.runs.create(
            thread_id=thread_id,
            input={"messages": [message]},
            config=config
        )
    except HTTPStatusError as e:
        if e.response.status_code == 409:
            # Handle based on strategy
            if strategy == "reject":
                return {"error": "Please wait for current response"}
            elif strategy == "enqueue":
                # Retry with backoff
                await asyncio.sleep(1)
                return await production_run(client, thread_id, message, strategy)
        raise
```

#### 2. **Memory Leaks in Long-Running Services**

Production services run for weeks or months. Memory leaks that seem minor in development can crash production:

```python
# MEMORY LEAK - Unbounded cache
class BadCache:
    def __init__(self):
        self.cache = {}  # Never cleaned!
    
    def add(self, key, value):
        self.cache[key] = value  # Grows forever

# PRODUCTION-READY - Bounded cache with TTL
from cachetools import TTLCache
import threading

class ProductionCache:
    def __init__(self, maxsize=1000, ttl=3600):
        self.cache = TTLCache(maxsize=maxsize, ttl=ttl)
        self.lock = threading.Lock()
        
    def add(self, key, value):
        with self.lock:
            self.cache[key] = value
    
    def get(self, key, default=None):
        with self.lock:
            return self.cache.get(key, default)
    
    def cleanup(self):
        """Periodic cleanup of expired items"""
        with self.lock:
            # TTLCache handles this automatically
            self.cache.expire()
```

#### 3. **Poor Error Recovery**

Production systems face network issues, service outages, and unexpected errors. Poor error handling leads to cascading failures:

```python
# FRAGILE - No error recovery
async def fragile_operation(client, data):
    response = await client.process(data)  # Fails on any error
    return response

# RESILIENT - Comprehensive error handling
async def resilient_operation(client, data, max_retries=3):
    """Production operation with full error recovery"""
    last_error = None
    
    for attempt in range(max_retries):
        try:
            # Exponential backoff
            if attempt > 0:
                await asyncio.sleep(2 ** attempt)
            
            # Attempt operation
            response = await client.process(data)
            
            # Validate response
            if not response or "error" in response:
                raise ValueError(f"Invalid response: {response}")
            
            return response
            
        except asyncio.TimeoutError:
            last_error = "Operation timed out"
            logger.warning(f"Timeout on attempt {attempt + 1}")
            continue
            
        except NetworkError as e:
            last_error = f"Network error: {e}"
            logger.warning(f"Network error on attempt {attempt + 1}: {e}")
            continue
            
        except Exception as e:
            last_error = f"Unexpected error: {e}"
            logger.error(f"Unexpected error on attempt {attempt + 1}", exc_info=True)
            
            # Don't retry on certain errors
            if "rate limit" in str(e).lower():
                raise
            continue
    
    # All retries failed
    raise OperationFailedError(
        f"Operation failed after {max_retries} attempts. Last error: {last_error}"
    )
```

#### 4. **Configuration Drift**

Different environments (dev, staging, prod) with inconsistent configurations cause mysterious production issues:

```python
# RISKY - Hardcoded configuration
class HardcodedService:
    def __init__(self):
        self.api_key = "sk-abc123"  # Works in dev, fails in prod
        self.timeout = 10  # Too short for production load
        self.model = "gpt-3.5-turbo"  # Different behavior

# MANAGED - Environment-aware configuration
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class ServiceConfig:
    """Configuration with validation and defaults"""
    api_key: str
    timeout: int = 30
    model: str = "gpt-4"
    max_retries: int = 3
    environment: str = "production"
    
    @classmethod
    def from_environment(cls) -> "ServiceConfig":
        """Load configuration from environment with validation"""
        api_key = os.getenv("API_KEY")
        if not api_key:
            raise ValueError("API_KEY environment variable required")
        
        return cls(
            api_key=api_key,
            timeout=int(os.getenv("TIMEOUT", "30")),
            model=os.getenv("MODEL", "gpt-4"),
            max_retries=int(os.getenv("MAX_RETRIES", "3")),
            environment=os.getenv("ENVIRONMENT", "production")
        )
    
    def validate(self):
        """Validate configuration for environment"""
        if self.environment == "production":
            assert self.timeout >= 30, "Production timeout too short"
            assert self.max_retries >= 3, "Production needs more retries"
            assert "gpt-4" in self.model, "Production should use GPT-4"
```

#### 5. **Resource Exhaustion**

Production systems must handle varying load without exhausting resources:

```python
# DANGEROUS - Unbounded resource usage
async def dangerous_parallel_processing(items):
    # Creates unlimited concurrent tasks
    tasks = [process_item(item) for item in items]
    return await asyncio.gather(*tasks)  # Could create 10,000 tasks!

# SAFE - Bounded concurrency
async def safe_parallel_processing(items, max_concurrent=10):
    """Process items with bounded concurrency"""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_with_limit(item):
        async with semaphore:
            return await process_item(item)
    
    # Process in batches
    results = []
    for i in range(0, len(items), max_concurrent):
        batch = items[i:i + max_concurrent]
        batch_results = await asyncio.gather(
            *[process_with_limit(item) for item in batch]
        )
        results.extend(batch_results)
    
    return results

# Connection pooling
from asyncpg import create_pool

class DatabaseManager:
    """Production database management with pooling"""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.pool = None
    
    async def initialize(self):
        """Initialize connection pool"""
        self.pool = await create_pool(
            self.database_url,
            min_size=5,      # Minimum connections
            max_size=20,     # Maximum connections
            max_queries=50000,
            max_inactive_connection_lifetime=300,
            command_timeout=60
        )
    
    async def execute_query(self, query: str, *args):
        """Execute query with connection from pool"""
        async with self.pool.acquire() as connection:
            return await connection.fetch(query, *args)
```

## Key Takeaways

1. **Design for Failure**: Production systems face network issues, service outages, and unexpected load. Build resilience into every component with retries, timeouts, and graceful degradation.

2. **Handle Double-Texting**: Users will send multiple messages before receiving responses. Choose the right strategy (reject, enqueue, interrupt, rollback) based on your use case and implement it consistently.

3. **Configuration Management**: Use environment-aware configuration with validation. Never hardcode values that might change between environments. Version your assistant configurations for safe experimentation.

4. **Monitor Everything**: Implement comprehensive health checks, metrics, and logging. You can't fix what you can't see. Use tools like Prometheus, Grafana, and LangSmith for observability.

5. **Resource Management**: Bound all resources - connections, memory, concurrent operations. Production systems must handle varying load without exhausting resources or degrading performance.

6. **Container Architecture**: Use Docker for consistency across environments. Orchestrate with Docker Compose for simple deployments or Kubernetes for scale. Always include health checks and resource limits.

7. **Secure by Default**: Never expose raw endpoints. Use API keys, implement rate limiting, validate all inputs, and follow security best practices. Store secrets properly and rotate them regularly.

## Next Steps

Congratulations! You've completed the LangGraph Academy and are ready to build production AI systems. You now understand:

- **Graph Construction**: Building stateful, multi-agent applications
- **State Management**: Sophisticated patterns for maintaining context
- **Human-in-the-Loop**: Safe AI with human oversight
- **Parallelization**: High-performance concurrent processing
- **Memory Systems**: Long-term context and personalization
- **Production Deployment**: Reliable, scalable services

Your journey doesn't end here. The field of AI agents is rapidly evolving. Stay current with LangGraph updates, experiment with new patterns, and most importantly - build amazing things that help real users solve real problems.

Remember: great AI applications aren't just about powerful models - they're about thoughtful design, robust engineering, and deep understanding of user needs. You now have all the tools to build them.

Happy building!