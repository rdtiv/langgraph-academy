"""
ReAct Agent with Anthropic Claude and Selective Tavily Search
This agent uses Claude Sonnet 4 for reasoning and Tavily for web search when needed.
"""

import os
import re
from typing import List, Dict, Any, Literal, Optional
from typing_extensions import TypedDict, Annotated
from langchain_core.messages import BaseMessage, SystemMessage, AIMessage, HumanMessage, ToolMessage
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from collections.abc import AsyncIterator

# State definition with thinking trace
class ThinkingState(TypedDict):
    """State that includes conversation history and thinking trace."""
    messages: Annotated[List[BaseMessage], add_messages]
    thinking_trace: Optional[str]
    search_results: Optional[Dict[str, Any]]

# System prompt optimized for Claude's thinking pattern
SYSTEM_PROMPT = """You are Claude, a helpful AI assistant created by Anthropic.

For complex questions or when the user needs current information:
1. First, think through the problem in <thinking> tags
2. If you need current information beyond your knowledge cutoff, use the tavily_search tool
3. Provide a clear, helpful response

Remember:
- Be concise and direct
- Only search when you truly need current information
- Show your reasoning transparently"""

async def agent_node(state: ThinkingState) -> Dict[str, Any]:
    """Process user input with Claude Sonnet 4.
    
    Args:
        state: Current conversation state
        
    Returns:
        Updated state with agent response
    """
    messages = state["messages"]
    
    # Initialize Claude with latest Sonnet
    model = ChatAnthropic(
        model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
        temperature=0.7,
        anthropic_beta="prompt-caching-2024-07-31"
    )
    
    # Add system message if not present
    if not messages or messages[0].type != "system":
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
    
    # Get response with potential tool use
    tavily = TavilySearchResults(
        max_results=5,
        search_depth="advanced",
        include_answer=True,
        include_raw_content=False
    )
    tools = [tavily]
    
    llm_with_tools = model.bind_tools(tools)
    response = await llm_with_tools.ainvoke(messages)
    
    # Extract thinking trace if present
    thinking_trace = None
    if hasattr(response, 'content') and response.content:
        thinking_match = re.search(r'<thinking>(.*?)</thinking>', response.content, re.DOTALL)
        if thinking_match:
            thinking_trace = thinking_match.group(1).strip()
    
    return {
        "messages": [response],
        "thinking_trace": thinking_trace
    }

def tool_node_wrapper(state: ThinkingState) -> Dict[str, Any]:
    """Wrapper for tool node to handle state properly.
    
    Args:
        state: Current conversation state
        
    Returns:
        Updated state with tool results
    """
    tool_node = ToolNode([TavilySearchResults(max_results=5)])
    result = tool_node(state)
    
    # Store search results if Tavily was used
    search_results = None
    if result.get("messages"):
        last_message = result["messages"][-1]
        if hasattr(last_message, 'content'):
            search_results = {"tavily": last_message.content}
    
    return {
        "messages": result.get("messages", []),
        "search_results": search_results
    }

def should_use_tools(state: ThinkingState) -> Literal["tools", END]:
    """Determine if tools should be used based on the last message.
    
    Args:
        state: Current conversation state
        
    Returns:
        "tools" if tools should be used, END otherwise
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    return END

# Build the graph
def create_graph():
    """Create the ReAct pattern graph.
    
    Returns:
        Compiled graph with memory checkpointer
    """
    graph = StateGraph(ThinkingState)
    
    # Add nodes
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node_wrapper)
    
    # ReAct pattern edges
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        should_use_tools,
        {
            "tools": "tools",
            END: END
        }
    )
    graph.add_edge("tools", "agent")
    
    # Compile with memory
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)

async def stream_events(app, thread_id: str, messages: List[BaseMessage]) -> AsyncIterator[Dict[str, Any]]:
    """Stream events with proper formatting for Next.js client.
    
    Args:
        app: The compiled LangGraph application
        thread_id: Unique identifier for the conversation thread
        messages: List of messages in the conversation
        
    Yields:
        Dict containing:
            - type: 'thinking' | 'message' | 'tool_use' | 'suggestion' | 'error' | 'done'
            - content: The actual content to display
            - metadata: Optional metadata (tool name, query, etc.)
    """
    import asyncio
    
    config = {"configurable": {"thread_id": thread_id}}
    timeout = float(os.getenv("STREAM_TIMEOUT_SECONDS", "300"))  # 5 minute default
    
    try:
        # Add timeout protection
        async with asyncio.timeout(timeout):
            async for event in app.astream_events(
                {"messages": messages},
                config=config,
                version="v2"
            ):
                if event["event"] == "on_chat_model_stream":
                    # Extract and stream thinking patterns
                    if "claude-sonnet-4" in str(event.get("metadata", {}).get("model", "")):
                        content = event["data"]["chunk"].content
                        if content:
                            # Check for thinking pattern
                            thinking_match = re.search(r'<thinking>(.*?)</thinking>', content, re.DOTALL)
                            if thinking_match:
                                yield {
                                    "type": "thinking",
                                    "content": thinking_match.group(1).strip()
                                }
                            
                            # Stream regular content (excluding thinking)
                            cleaned_content = re.sub(r'<thinking>.*?</thinking>', '', content, flags=re.DOTALL)
                            if cleaned_content.strip():
                                yield {
                                    "type": "message",
                                    "content": cleaned_content
                                }
                
                elif event["event"] == "on_tool_start":
                    # Stream tool usage
                    tool_name = event["metadata"].get("tool_name", "unknown")
                    if tool_name == "tavily_search_results":
                        tool_input = event["data"].get("input", {})
                        yield {
                            "type": "tool_use",
                            "content": f"Searching the web for: {tool_input.get('query', 'information')}",
                            "metadata": {
                                "tool": "tavily_search",
                                "query": tool_input.get("query", "")
                            }
                        }
            
            # Generate suggestions after message completes
            if messages:
                suggestions = await generate_suggestions(messages)
                for suggestion in suggestions:
                    yield {
                        "type": "suggestion",
                        "content": suggestion
                    }
            
            # Send done signal after all events
            yield {"type": "done", "content": ""}
        
    except asyncio.TimeoutError:
        yield {
            "type": "error",
            "content": "Stream timeout exceeded. Please try again.",
            "metadata": {"error_type": "TimeoutError"}
        }
    except Exception as e:
        yield {
            "type": "error",
            "content": str(e),
            "metadata": {"error_type": type(e).__name__}
        }

async def generate_suggestions(messages: List[BaseMessage]) -> List[str]:
    """Generate smart next-step suggestions using Haiku based on conversation.
    
    Args:
        messages: Conversation history
        
    Returns:
        List of up to 3 suggested follow-up questions
    """
    # Get the last assistant message
    last_assistant_msg = None
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            last_assistant_msg = msg
            break
    
    if not last_assistant_msg:
        return []
    
    # Use Haiku for lightweight suggestion generation
    haiku = ChatAnthropic(
        model=os.getenv("ANTHROPIC_SUGGESTIONS_MODEL", "claude-3-5-haiku-latest"),
        temperature=0.7,
        max_tokens=200
    )
    
    suggestion_prompt = f"""Based on this response, suggest 3 brief follow-up questions the user might ask to learn more.

Response: {last_assistant_msg.content}

Format: Return only the 3 questions, one per line. Keep them short and conversational."""
    
    try:
        suggestions_response = await haiku.ainvoke([
            SystemMessage(content="You are a helpful assistant that suggests follow-up questions."),
            HumanMessage(content=suggestion_prompt)
        ])
        
        # Parse suggestions
        suggestions = []
        if suggestions_response.content:
            lines = suggestions_response.content.strip().split('\n')
            for line in lines[:3]:  # Take max 3 suggestions
                cleaned = line.strip().lstrip('•-123456789.')
                if cleaned:
                    suggestions.append(cleaned)
        
        return suggestions
    except Exception as e:
        print(f"Error generating suggestions: {e}")
        return []

# Export the app for LangGraph dev server
app = create_graph()

# The LangGraph dev server will automatically detect and serve this app
# Access it at http://localhost:8123 when running `langgraph dev`

if __name__ == "__main__":
    # Direct execution for testing
    import asyncio
    
    async def test_agent():
        test_messages = [
            HumanMessage(content="What is LangGraph and how does it work?")
        ]
        
        print("Testing agent directly...")
        async for chunk in stream_events(app, "test-thread", test_messages):
            print(f"{chunk['type']}: {chunk['content'][:50]}...")
    
    asyncio.run(test_agent())