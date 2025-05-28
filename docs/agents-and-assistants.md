# Agents and Assistants: Building Intelligent AI Systems

**Created**: 2025-05-26  
**Last Modified**: 2025-05-27

## What You'll Learn

This guide explores agents and assistants in LangGraph—the core abstractions for building intelligent, adaptive AI applications. You'll master:

- **Agent Architecture**: How agents use LLMs for dynamic control flow
- **Assistant Pattern**: Configuration-driven deployment with versioning
- **Implementation Patterns**: From simple tool-calling to complex multi-agent systems
- **Production Strategies**: Scaling, monitoring, and maintaining agent systems
- **Advanced Techniques**: Memory, delegation, and specialized architectures

## Why It Matters

The difference between a chatbot and an intelligent assistant is agency—the ability to reason, plan, and act autonomously:

**Traditional Chatbot:**
```python
def chatbot(message):
    if "weather" in message:
        return get_weather()
    elif "news" in message:
        return get_news()
    else:
        return "I don't understand"
```

**LangGraph Agent:**
```python
# Agent reasons about user intent, searches multiple sources,
# synthesizes information, and can ask clarifying questions
"What should I wear today?"
→ Checks weather
→ Considers your calendar
→ Remembers your preferences
→ Provides personalized recommendation
```

Agents and assistants enable:
- **Autonomous Problem Solving**: Break down complex tasks into steps
- **Dynamic Adaptation**: Change behavior based on context and results
- **Tool Orchestration**: Intelligently combine multiple capabilities
- **Personalization at Scale**: Deploy once, configure per user/use-case
- **Continuous Improvement**: Learn from interactions and feedback

## How It Works

### Understanding Agents

#### What is an Agent?

An agent is a system that uses an LLM to control application flow dynamically. Instead of following predetermined paths, agents:

1. **Observe** the current state
2. **Reason** about what to do next
3. **Act** by calling tools or generating responses
4. **Iterate** based on results

```python
# Traditional flow - rigid and brittle
if user_wants_flight:
    search_flights()
    if flights_found:
        show_flights()
    else:
        show_error()

# Agent flow - adaptive and intelligent
while not task_complete:
    action = llm.decide_next_action(state)
    result = execute_action(action)
    state = update_state(result)
    if llm.is_satisfied(state):
        task_complete = True
```

#### Core Agent Patterns

**1. ReAct (Reasoning + Acting)**
```python
# The agent explicitly reasons before acting
Thought: User wants to book a flight. I need to know their destination and dates.
Action: ask_user("Where would you like to fly to and when?")
Observation: User said "Paris next Friday"
Thought: I have destination but need return date. Let me search one-way first.
Action: search_flights(dest="Paris", date="next Friday", one_way=True)
Observation: Found 15 flights ranging from $200-$800
Thought: I should ask about return date and budget preferences
Action: ask_user("When would you like to return? Do you have a budget?")
```

**2. Tool-Calling Agent**
```python
# Agent seamlessly orchestrates multiple tools
User: "Find me a good Italian restaurant for tonight and make a reservation"

Agent → search_restaurants(cuisine="Italian", date="tonight")
     → check_availability(restaurant_id="123", time="7pm")
     → get_user_preferences()  # Remembers user prefers quiet places
     → filter_results(noise_level="quiet")
     → make_reservation(restaurant="La Bella", party_size=2)
     → send_confirmation(details=reservation)
```

**3. Multi-Agent Orchestration**
```python
# Specialized agents work together
Research Agent: Gathers information from multiple sources
Writer Agent: Creates comprehensive report
Reviewer Agent: Checks accuracy and completeness
Coordinator: Manages workflow between agents
```

### Understanding Assistants

#### What is an Assistant?

An assistant is a configured instance of an agent graph. While agents define capabilities, assistants define personalities, behaviors, and constraints:

```python
# Same agent graph, different assistants
CustomerServiceAssistant = {
    "system_prompt": "You are a helpful customer service representative...",
    "temperature": 0.3,  # More consistent responses
    "tools": ["order_lookup", "refund_processing"],
    "guardrails": ["no_technical_details", "escalate_complaints"]
}

TechnicalSupportAssistant = {
    "system_prompt": "You are a technical expert...",
    "temperature": 0.7,  # More creative problem-solving
    "tools": ["system_diagnostics", "knowledge_base"],
    "guardrails": ["explain_technical_clearly"]
}
```

### Implementation Guide

#### Building Your First Agent

Let's build a complete agent from scratch:

```python
from typing import Annotated, Literal, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
import json

# 1. Define the agent's state
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], lambda x, y: x + y]
    current_task: str
    task_complete: bool
    metadata: dict

# 2. Create tools the agent can use
def search_flights(origin: str, destination: str, date: str) -> str:
    """Search for available flights."""
    # Simulate flight search
    return json.dumps({
        "flights": [
            {"airline": "United", "time": "08:00", "price": "$350"},
            {"airline": "Delta", "time": "14:30", "price": "$280"},
        ]
    })

def book_flight(flight_id: str, passenger_name: str) -> str:
    """Book a specific flight."""
    return f"Flight {flight_id} booked for {passenger_name}"

def check_weather(city: str, date: str) -> str:
    """Check weather for a city on a specific date."""
    return f"Weather in {city} on {date}: Sunny, 72°F"

# 3. Set up the LLM with tools
tools = [search_flights, book_flight, check_weather]
model = ChatOpenAI(model="gpt-4o", temperature=0)
model_with_tools = model.bind_tools(tools)

# 4. Define the agent's reasoning system
AGENT_PROMPT = """You are a helpful travel assistant. Your goal is to help users plan and book travel.

Key behaviors:
1. Always gather complete information before searching
2. Present options clearly and wait for user choice
3. Confirm details before booking
4. Be proactive about checking weather and providing tips

Current task: {current_task}
Task complete: {task_complete}

Based on the conversation, decide what to do next."""

def agent_node(state: AgentState) -> AgentState:
    """Main agent reasoning node."""
    # Create system message with current context
    system_msg = SystemMessage(
        content=AGENT_PROMPT.format(
            current_task=state.get("current_task", "Assist with travel planning"),
            task_complete=state.get("task_complete", False)
        )
    )
    
    # Get response from LLM
    messages = [system_msg] + state["messages"]
    response = model_with_tools.invoke(messages)
    
    # Check if task is complete
    task_complete = False
    if "booking confirmed" in response.content.lower():
        task_complete = True
    
    return {
        "messages": [response],
        "task_complete": task_complete
    }

# 5. Add decision-making logic
def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """Decide whether to use tools or end."""
    messages = state["messages"]
    last_message = messages[-1]
    
    # Check if task is complete
    if state.get("task_complete", False):
        return "end"
    
    # Check if agent wants to use tools
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    
    return "end"

# 6. Build the agent graph
def create_travel_agent():
    builder = StateGraph(AgentState)
    
    # Add nodes
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(tools))
    
    # Add edges
    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END
        }
    )
    builder.add_edge("tools", "agent")  # Loop back after tool use
    
    return builder.compile()

# 7. Use the agent
agent = create_travel_agent()

# Example conversation
initial_state = {
    "messages": [HumanMessage(content="I need to fly to Paris next Friday")],
    "current_task": "Book flight to Paris",
    "task_complete": False,
    "metadata": {"user_id": "user-123"}
}

# Run the agent
result = agent.invoke(initial_state)
print(result["messages"][-1].content)
```

#### Creating Configurable Assistants

Now let's make our agent configurable for different use cases:

```python
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from langgraph.constants import CONFIG

@dataclass
class AssistantConfig:
    """Configuration schema for travel assistant."""
    personality: Literal["professional", "friendly", "concise"]
    language: str = "en"
    max_search_results: int = 5
    require_confirmation: bool = True
    allowed_airlines: Optional[List[str]] = None
    budget_limit: Optional[float] = None
    preferred_times: Optional[Dict[str, str]] = None

# Personality templates
PERSONALITIES = {
    "professional": {
        "prompt": "You are a professional travel agent. Be formal and thorough.",
        "temperature": 0.3
    },
    "friendly": {
        "prompt": "You are a friendly travel buddy! Be casual and enthusiastic.",
        "temperature": 0.7
    },
    "concise": {
        "prompt": "You are an efficient assistant. Be brief and to the point.",
        "temperature": 0.5
    }
}

def configurable_agent_node(state: AgentState, config: CONFIG) -> AgentState:
    """Agent node that uses configuration."""
    # Extract configuration
    assistant_config = AssistantConfig(**config.get("configurable", {}))
    
    # Get personality settings
    personality = PERSONALITIES[assistant_config.personality]
    
    # Create configured model
    model = ChatOpenAI(
        model="gpt-4o",
        temperature=personality["temperature"]
    )
    model_with_tools = model.bind_tools(tools)
    
    # Build system prompt with configuration
    system_content = f"""
{personality["prompt"]}

Configuration:
- Language: {assistant_config.language}
- Max search results: {assistant_config.max_search_results}
- Require confirmation: {assistant_config.require_confirmation}
- Budget limit: ${assistant_config.budget_limit or 'No limit'}
- Allowed airlines: {assistant_config.allowed_airlines or 'All airlines'}

Current task: {state.get("current_task", "Assist with travel")}
"""
    
    system_msg = SystemMessage(content=system_content)
    messages = [system_msg] + state["messages"]
    
    response = model_with_tools.invoke(messages)
    
    return {"messages": [response]}

# Create configurable graph
def create_configurable_assistant():
    builder = StateGraph(AgentState)
    
    builder.add_node("agent", configurable_agent_node)
    builder.add_node("tools", ToolNode(tools))
    
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", should_continue)
    builder.add_edge("tools", "agent")
    
    return builder.compile()

# Deploy with different configurations
assistant = create_configurable_assistant()

# Business traveler configuration
business_config = {
    "configurable": {
        "personality": "professional",
        "require_confirmation": True,
        "allowed_airlines": ["United", "Delta", "American"],
        "preferred_times": {"morning": "before 9am", "evening": "after 6pm"}
    }
}

# Budget traveler configuration  
budget_config = {
    "configurable": {
        "personality": "friendly",
        "budget_limit": 500,
        "max_search_results": 10
    }
}

# Use with configuration
result = assistant.invoke(initial_state, config=business_config)
```

### Advanced Agent Patterns

#### Multi-Agent Systems

Build systems where specialized agents collaborate:

```python
from langgraph.graph import StateGraph, Send

class ResearchState(TypedDict):
    topic: str
    subtopics: List[str]
    research_results: Annotated[List[dict], lambda x, y: x + y]
    final_report: str

# Specialized research agent
def research_agent(state: dict) -> dict:
    """Agent specialized in research."""
    topic = state["topic"]
    # Research logic here
    return {"research_results": [{"topic": topic, "findings": "..."}]}

# Specialized writing agent  
def writing_agent(state: ResearchState) -> dict:
    """Agent specialized in writing."""
    research = state["research_results"]
    # Create comprehensive report
    report = f"# Report on {state['topic']}\n\n"
    for finding in research:
        report += f"## {finding['topic']}\n{finding['findings']}\n\n"
    return {"final_report": report}

# Coordinator agent that delegates
def coordinator_agent(state: ResearchState) -> List[Send]:
    """Orchestrates other agents."""
    # Delegate research to multiple agents in parallel
    return [
        Send("research_agent", {"topic": subtopic})
        for subtopic in state["subtopics"]
    ]

# Build multi-agent system
def create_research_system():
    builder = StateGraph(ResearchState)
    
    builder.add_node("coordinator", coordinator_agent)
    builder.add_node("research_agent", research_agent)
    builder.add_node("writing_agent", writing_agent)
    
    builder.add_edge(START, "coordinator")
    builder.add_conditional_edges(
        "coordinator",
        lambda x: x,  # Send objects
        ["research_agent"]
    )
    builder.add_edge("research_agent", "writing_agent")
    builder.add_edge("writing_agent", END)
    
    return builder.compile()
```

#### Agents with Long-Term Memory

Integrate persistent memory for truly intelligent agents:

```python
from langgraph.store.memory import InMemoryStore

class MemoryAgentState(AgentState):
    user_id: str
    user_preferences: dict
    conversation_history: List[dict]

# Initialize memory store
memory_store = InMemoryStore()

def memory_agent_node(state: MemoryAgentState) -> dict:
    """Agent with long-term memory."""
    user_id = state["user_id"]
    
    # Retrieve user memory
    namespace = (user_id, "preferences")
    stored_prefs = memory_store.get(namespace, "main")
    
    if stored_prefs:
        user_preferences = stored_prefs.value
    else:
        user_preferences = {}
    
    # Update state with memory
    state["user_preferences"] = user_preferences
    
    # Generate response using memory
    system_prompt = f"""
You are a personalized travel assistant.
User preferences: {json.dumps(user_preferences, indent=2)}

Remember to:
1. Use their preferred airlines: {user_preferences.get('airlines', 'Any')}
2. Respect budget: {user_preferences.get('budget', 'Flexible')}
3. Consider dietary restrictions: {user_preferences.get('dietary', 'None')}
"""
    
    # ... rest of agent logic ...
    
    # Extract and save new preferences
    new_preferences = extract_preferences(state["messages"])
    if new_preferences:
        updated_prefs = {**user_preferences, **new_preferences}
        memory_store.put(namespace, "main", updated_prefs)
    
    return {"messages": [response]}

def extract_preferences(messages: List[BaseMessage]) -> dict:
    """Extract user preferences from conversation."""
    preferences = {}
    
    # Analyze messages for preferences
    # This could use NLP or another LLM call
    for msg in messages:
        if isinstance(msg, HumanMessage):
            content = msg.content.lower()
            if "i prefer" in content or "i like" in content:
                # Extract preference logic
                pass
    
    return preferences
```

#### Adaptive Learning Agents

Agents that improve through experience:

```python
class LearningAgentState(AgentState):
    feedback_history: List[dict]
    performance_metrics: dict
    adaptation_rules: List[dict]

def learning_agent_node(state: LearningAgentState) -> dict:
    """Agent that learns from feedback."""
    
    # Analyze past performance
    feedback = state.get("feedback_history", [])
    success_rate = calculate_success_rate(feedback)
    
    # Adapt behavior based on performance
    if success_rate < 0.7:
        # Adjust strategy
        temperature = 0.3  # Be more conservative
        prompt_modifier = "Be more careful and ask for confirmation"
    else:
        temperature = 0.7  # Be more creative
        prompt_modifier = "Feel free to make suggestions"
    
    # Create adaptive prompt
    system_prompt = f"""
You are an adaptive travel assistant.
Current success rate: {success_rate:.2%}
Behavior modifier: {prompt_modifier}

Recent feedback:
{format_recent_feedback(feedback[-5:])}

Learn from past interactions and improve your responses.
"""
    
    # Generate response with adapted behavior
    model = ChatOpenAI(temperature=temperature)
    # ... rest of logic ...
    
    return {"messages": [response]}

def calculate_success_rate(feedback: List[dict]) -> float:
    """Calculate agent's success rate from feedback."""
    if not feedback:
        return 0.5  # Default neutral
    
    positive = sum(1 for f in feedback if f.get("rating", 0) >= 4)
    return positive / len(feedback)
```

### Production Deployment Patterns

#### Monitoring and Observability

```python
import logging
from datetime import datetime
from prometheus_client import Counter, Histogram, Gauge

# Metrics
agent_invocations = Counter('agent_invocations_total', 'Total agent invocations')
agent_duration = Histogram('agent_duration_seconds', 'Agent execution duration')
tool_usage = Counter('tool_usage_total', 'Tool usage', ['tool_name'])
active_conversations = Gauge('active_conversations', 'Number of active conversations')

def monitored_agent_node(state: AgentState) -> dict:
    """Agent with comprehensive monitoring."""
    start_time = datetime.utcnow()
    agent_invocations.inc()
    active_conversations.inc()
    
    try:
        # Log entry
        logging.info(f"Agent invoked for user {state.get('user_id', 'unknown')}")
        
        # Execute agent logic
        response = agent_logic(state)
        
        # Track tool usage
        if hasattr(response, "tool_calls"):
            for tool_call in response.tool_calls:
                tool_usage.labels(tool_name=tool_call["name"]).inc()
        
        # Log success
        duration = (datetime.utcnow() - start_time).total_seconds()
        agent_duration.observe(duration)
        logging.info(f"Agent completed in {duration:.2f}s")
        
        return {"messages": [response]}
        
    except Exception as e:
        logging.error(f"Agent error: {str(e)}", exc_info=True)
        raise
        
    finally:
        active_conversations.dec()
```

#### Scaling Strategies

```python
# Horizontal scaling with specialized agents
class AgentPool:
    """Pool of specialized agents for different tasks."""
    
    def __init__(self):
        self.agents = {
            "travel": create_travel_agent(),
            "research": create_research_agent(),
            "support": create_support_agent(),
        }
        self.router = create_router_agent()
        
    async def handle_request(self, message: str, context: dict) -> str:
        """Route request to appropriate agent."""
        # Determine which agent to use
        agent_type = await self.router.classify(message, context)
        
        # Get specialized agent
        agent = self.agents.get(agent_type, self.agents["support"])
        
        # Process with appropriate agent
        result = await agent.ainvoke({
            "messages": [HumanMessage(content=message)],
            **context
        })
        
        return result["messages"][-1].content

# Caching for performance
from functools import lru_cache
from cachetools import TTLCache

# Cache tool results
tool_cache = TTLCache(maxsize=1000, ttl=3600)  # 1 hour TTL

@lru_cache(maxsize=128)
def cached_search_flights(origin: str, destination: str, date: str) -> str:
    """Cached flight search to reduce API calls."""
    cache_key = f"{origin}-{destination}-{date}"
    
    if cache_key in tool_cache:
        return tool_cache[cache_key]
    
    result = search_flights(origin, destination, date)
    tool_cache[cache_key] = result
    return result
```

### Testing Agent Systems

```python
import pytest
from unittest.mock import Mock, patch

@pytest.fixture
def mock_llm():
    """Mock LLM for testing."""
    mock = Mock()
    mock.invoke.return_value = AIMessage(
        content="I'll search for flights to Paris.",
        tool_calls=[{
            "name": "search_flights",
            "args": {"destination": "Paris", "date": "2024-06-15"}
        }]
    )
    return mock

def test_agent_handles_flight_request(mock_llm):
    """Test agent correctly handles flight requests."""
    # Create agent with mocked LLM
    with patch('langchain_openai.ChatOpenAI', return_value=mock_llm):
        agent = create_travel_agent()
    
    # Test input
    state = {
        "messages": [HumanMessage(content="I need to fly to Paris on June 15")],
        "current_task": "Book flight",
        "task_complete": False
    }
    
    # Run agent
    result = agent.invoke(state)
    
    # Verify behavior
    assert len(result["messages"]) > 1
    assert "search_flights" in str(result["messages"][-1])
    assert not result["task_complete"]

@pytest.mark.asyncio
async def test_agent_with_memory():
    """Test agent memory integration."""
    memory_store = InMemoryStore()
    agent = create_memory_agent(memory_store)
    
    # First interaction
    result1 = await agent.ainvoke({
        "messages": [HumanMessage(content="I prefer United Airlines")],
        "user_id": "test-user"
    })
    
    # Verify preference saved
    stored = memory_store.get(("test-user", "preferences"), "main")
    assert stored.value["airlines"] == "United Airlines"
    
    # Second interaction should remember
    result2 = await agent.ainvoke({
        "messages": [HumanMessage(content="Book me a flight")],
        "user_id": "test-user"
    })
    
    # Should use United preference
    assert "United" in str(result2["messages"])
```

## Best Practices

1. **Start Simple, Iterate**
   - Begin with basic tool-calling agents
   - Add complexity gradually (memory, multi-agent, learning)
   - Test each component thoroughly

2. **Design for Failure**
   - Implement fallback behaviors
   - Add retry logic for tools
   - Graceful degradation when services unavailable

3. **Monitor Everything**
   - Track agent decisions and tool usage
   - Log conversation flows for debugging
   - Measure performance and success rates

4. **Security First**
   - Validate all tool inputs
   - Implement rate limiting
   - Audit agent actions
   - Never expose sensitive data in prompts

5. **Configuration Over Code**
   - Make behavior configurable
   - Use assistants for different personas
   - Version configurations for rollback

6. **Test Comprehensively**
   - Unit test individual nodes
   - Integration test agent flows
   - Load test for production scale
   - Test edge cases and failures

## Key Takeaways

1. **Agents Enable Intelligence**: Dynamic control flow based on reasoning, not rigid rules

2. **Assistants Enable Flexibility**: Same capabilities, different configurations

3. **Composition is Power**: Combine simple agents into sophisticated systems

4. **Memory Makes the Difference**: Persistent context transforms user experience

5. **Production Requires Planning**: Monitor, scale, and maintain from the start

6. **Testing is Critical**: Agents are complex; thorough testing prevents surprises

## Next Steps

You now understand how to build intelligent agents and deploy them as configurable assistants. The power of LangGraph lies in combining these concepts with the state management, human-in-the-loop, parallelization, and memory systems you've learned throughout the academy.

Start building your own agents. Experiment with different architectures. Deploy assistants for various use cases. Most importantly, focus on solving real problems for real users—that's where agents truly shine.

Remember: Great agents don't just execute tasks—they understand intent, adapt to context, and improve through experience. Build agents that delight users by being genuinely helpful, not just technically capable.