# 4: Parallelization

**Created**: 2025-05-26  
**Last Modified**: 2025-05-27

## What You'll Learn

Module 4 demonstrates LangGraph's parallelization capabilities for building high-performance AI systems. You'll master:

- **Fan-out/Fan-in patterns** for static parallel execution
- **Map-Reduce with Send() API** for dynamic task distribution
- **State reducers** to safely merge parallel results
- **Subgraph composition** for modular, reusable components
- **Research Assistant** - a sophisticated multi-agent system combining all patterns

## Why It Matters

Sequential processing creates critical bottlenecks in AI applications. Consider these scenarios:

1. **Web Search**: Searching multiple sources sequentially takes 3x longer than parallel search
2. **Document Analysis**: Processing 100 documents one-by-one vs. in batches of 10
3. **Multi-Agent Systems**: Coordinating specialist agents that can work independently
4. **API Aggregation**: Calling multiple APIs where order doesn't matter

Parallelization enables:
- **10x Performance**: Execute independent operations simultaneously
- **Dynamic Scalability**: Handle variable workloads without code changes
- **Modular Architecture**: Compose complex systems from simple, reusable parts
- **Resource Efficiency**: Better CPU/API utilization through concurrent execution
- **Fault Isolation**: Parallel branches fail independently

## How It Works

### Core Concepts Deep Dive

#### 1. Fan-out/Fan-in Pattern
The simplest form of parallelization where multiple nodes execute simultaneously and converge at a single point.

**What**: Static parallel execution paths defined at compile time
**Why**: When you know exactly what parallel operations you need
**How**: Add multiple edges from one node to many, then back to one

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from operator import add

class SearchState(TypedDict):
    query: str
    results: Annotated[list, add]  # Reducer merges parallel results
    best_result: str

def search_web(state: SearchState):
    """Search general web for query"""
    # Simulated web search
    return {"results": [f"Web result for: {state['query']}"]}

def search_academic(state: SearchState):
    """Search academic papers"""
    return {"results": [f"Academic result for: {state['query']}"]}

def search_news(state: SearchState):
    """Search recent news"""
    return {"results": [f"News result for: {state['query']}"]}

def select_best(state: SearchState):
    """Analyze all results and pick best one"""
    all_results = state["results"]
    # Logic to score and rank results
    return {"best_result": all_results[0]}  # Simplified

# Build graph
graph = StateGraph(SearchState)

# Add nodes
graph.add_node("web_search", search_web)
graph.add_node("academic_search", search_academic)  
graph.add_node("news_search", search_news)
graph.add_node("select_best", select_best)

# Fan-out: START → all search nodes
graph.add_edge(START, "web_search")
graph.add_edge(START, "academic_search")
graph.add_edge(START, "news_search")

# Fan-in: all search nodes → select_best
graph.add_edge("web_search", "select_best")
graph.add_edge("academic_search", "select_best")
graph.add_edge("news_search", "select_best")

graph.add_edge("select_best", END)

app = graph.compile()
```

#### 2. Map-Reduce Pattern with Send() API

Dynamic parallelization that scales based on runtime data. This is LangGraph's most powerful pattern.

**What**: Distribute work dynamically, process in parallel, aggregate results
**Why**: When parallel work depends on runtime data (unknown at compile time)
**How**: Use Send() to spawn parallel instances, reducers to merge results

```python
from langgraph.constants import Send
from typing import List

class AnalysisState(TypedDict):
    documents: List[str]
    analyses: Annotated[List[dict], add]  # Accumulates all analyses
    summary: str

def distribute_work(state: AnalysisState):
    """Map phase - create parallel tasks for each document"""
    # This returns a list of Send objects
    # Each Send creates a new instance of the target node
    return [
        Send(
            "analyze_document",  # Target node
            {"document": doc, "doc_id": i}  # State for this instance
        ) 
        for i, doc in enumerate(state["documents"])
    ]

def analyze_document(state: dict):
    """Process a single document (runs in parallel)"""
    doc = state["document"]
    doc_id = state["doc_id"]
    
    # Expensive analysis operation
    analysis = {
        "id": doc_id,
        "length": len(doc),
        "sentiment": "positive",  # Simplified
        "key_points": ["point1", "point2"]
    }
    
    return {"analyses": [analysis]}

def summarize_analyses(state: AnalysisState):
    """Reduce phase - aggregate all analyses"""
    all_analyses = state["analyses"]
    
    # Aggregate insights
    total_docs = len(all_analyses)
    avg_length = sum(a["length"] for a in all_analyses) / total_docs
    
    summary = f"Analyzed {total_docs} documents. Average length: {avg_length}"
    return {"summary": summary}

# Build graph
graph = StateGraph(AnalysisState)

graph.add_node("analyze_document", analyze_document)
graph.add_node("summarize", summarize_analyses)

# Conditional edge with Send
graph.add_conditional_edges(
    START,
    distribute_work,  # Returns list of Send objects
    ["analyze_document"]  # Possible destinations
)

# After all analyses complete, summarize
graph.add_edge("analyze_document", "summarize")
graph.add_edge("summarize", END)
```

#### 3. Advanced Send() Patterns

Send() API supports sophisticated patterns beyond simple map-reduce:

```python
# Pattern 1: Conditional Parallelization
def maybe_parallelize(state: State):
    """Conditionally spawn parallel tasks"""
    if len(state["items"]) > 10:
        # Parallel processing for large batches
        return [Send("process_batch", {"batch": batch}) 
                for batch in chunks(state["items"], 5)]
    else:
        # Sequential for small batches
        return "process_sequential"

# Pattern 2: Nested Send (Send within Send)
def analyze_topic(state: dict):
    """Each topic analysis spawns sub-analyses"""
    topic = state["topic"]
    
    # This node itself uses Send!
    return [
        Send("analyze_subtopic", {"subtopic": sub})
        for sub in get_subtopics(topic)
    ]

# Pattern 3: Mixed Static and Dynamic
def route_analysis(state: State):
    """Combine conditional routing with Send"""
    if state["priority"] == "high":
        # Parallel fast-track
        return [
            Send("quick_analysis", {"data": state["data"]}),
            Send("alert_team", {"data": state["data"]})
        ]
    else:
        # Normal sequential flow
        return "standard_analysis"
```

#### 4. State Reducers In-Depth

Reducers are critical for safe parallel state updates. Without them, parallel writes conflict.

```python
from typing import Annotated
from operator import add
from langchain_core.messages import BaseMessage

# Built-in reducer: add (for lists)
class State1(TypedDict):
    results: Annotated[list, add]  # Appends lists

# Custom reducer for dictionaries
def merge_dicts(a: dict, b: dict) -> dict:
    """Merge two dictionaries, b overwrites a"""
    return {**a, **b}

class State2(TypedDict):
    metadata: Annotated[dict, merge_dicts]

# Custom reducer for complex logic
def merge_analyses(existing: list, new: list) -> list:
    """Merge analyses, removing duplicates"""
    seen = {a["id"] for a in existing}
    return existing + [a for a in new if a["id"] not in seen]

class State3(TypedDict):
    analyses: Annotated[list, merge_analyses]

# Messages have a special built-in reducer
from langgraph.graph import MessagesState

class ChatState(MessagesState):  # Inherits messages with reducer
    # messages: Annotated[list[BaseMessage], add_messages]
    summary: str
```

#### 5. Subgraph Patterns

Subgraphs enable modular, composable architectures:

```python
# Subgraph with its own state
class EmailAnalysisState(TypedDict):
    email_content: str
    sentiment: str
    urgent: bool
    summary: str

def create_email_analyzer():
    """Create a reusable email analysis subgraph"""
    subgraph = StateGraph(EmailAnalysisState)
    
    subgraph.add_node("extract_sentiment", analyze_sentiment)
    subgraph.add_node("check_urgency", detect_urgency)
    subgraph.add_node("summarize", create_summary)
    
    # Parallel sentiment and urgency detection
    subgraph.add_edge(START, "extract_sentiment")
    subgraph.add_edge(START, "check_urgency")
    
    # Both feed into summary
    subgraph.add_edge("extract_sentiment", "summarize")
    subgraph.add_edge("check_urgency", "summarize")
    subgraph.add_edge("summarize", END)
    
    return subgraph.compile()

# Parent graph uses subgraph
class CustomerServiceState(TypedDict):
    customer_emails: List[str]
    email_content: str  # Overlapping key!
    urgent: bool       # Overlapping key!
    analyses: Annotated[List[dict], add]

parent = StateGraph(CustomerServiceState)

# Add subgraph as a node
email_analyzer = create_email_analyzer()
parent.add_node("analyze_email", email_analyzer)

# Send emails to subgraph in parallel
def distribute_emails(state: CustomerServiceState):
    return [
        Send("analyze_email", {"email_content": email})
        for email in state["customer_emails"]
    ]

parent.add_conditional_edges(
    START, 
    distribute_emails,
    ["analyze_email"]
)
```

### Real-World Example: Research Assistant

The research assistant demonstrates all patterns working together:

```python
# Simplified version showing key patterns

class ResearchGraphState(TypedDict):
    topic: str
    max_analysts: int
    analysts: List[Analyst]  # Created dynamically
    sections: Annotated[list, add]  # Parallel interview results
    final_report: str

# 1. Dynamic analyst creation
def create_analysts(state: ResearchGraphState):
    """Create N analysts based on topic"""
    # LLM generates diverse analyst perspectives
    analysts = generate_analyst_personas(
        topic=state["topic"],
        count=state["max_analysts"]
    )
    return {"analysts": analysts}

# 2. Parallel interviews using Send()
def initiate_all_interviews(state: ResearchGraphState):
    """Launch parallel interviews with each analyst"""
    return [
        Send(
            "conduct_interview",  # Subgraph!
            {
                "analyst": analyst,
                "messages": [f"Tell me about {state['topic']}"]
            }
        ) 
        for analyst in state["analysts"]
    ]

# 3. Interview subgraph with parallel search
interview_builder = StateGraph(InterviewState)

# Parallel search during each Q&A turn
interview_builder.add_edge("ask_question", "search_web")
interview_builder.add_edge("ask_question", "search_wikipedia")
interview_builder.add_edge("search_web", "answer_question")
interview_builder.add_edge("search_wikipedia", "answer_question")

# 4. Parallel report writing
builder.add_edge("conduct_interview", "write_report")
builder.add_edge("conduct_interview", "write_introduction")
builder.add_edge("conduct_interview", "write_conclusion")

# All sections complete before finalization
builder.add_edge(
    ["write_conclusion", "write_report", "write_introduction"], 
    "finalize_report"
)
```

### Performance Optimization Patterns

```python
# 1. Batching for API limits
def batch_process(state: State):
    """Process in batches to avoid rate limits"""
    items = state["items"]
    batch_size = 10  # API concurrent limit
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i+batch_size]
        yield [Send("process_item", {"item": item}) for item in batch]

# 2. Early termination
def check_results(state: State):
    """Stop parallel processing if good result found"""
    for result in state["results"]:
        if result["confidence"] > 0.95:
            return "finalize"  # Skip remaining parallel work
    return "continue_searching"

# 3. Hierarchical parallelization
def hierarchical_analysis(state: State):
    """Multi-level parallel processing"""
    # Level 1: Parallel by region
    regions = state["data_by_region"]
    
    return [
        Send("analyze_region", {
            "region": region,
            "data": data,
            # Each region will spawn city-level parallel analysis
        })
        for region, data in regions.items()
    ]
```

### Common Pitfalls and Solutions

1. **Missing Reducers**
   ```python
   # ❌ BAD - Parallel writes conflict
   class State(TypedDict):
       results: list  # No reducer!
   
   # ✅ GOOD - Reducer handles merging
   class State(TypedDict):
       results: Annotated[list, add]
   ```

2. **Order Dependencies**
   ```python
   # ❌ BAD - Assumes execution order
   def process(state):
       if state["results"][0] == "web":  # What if web finishes second?
           return process_web_first(state)
   
   # ✅ GOOD - Order-independent
   def process(state):
       web_results = [r for r in state["results"] if r["source"] == "web"]
       if web_results:
           return process_web_results(web_results)
   ```

3. **State Propagation in Send()**
   ```python
   # ❌ BAD - Missing required state
   Send("process", {"data": item})  # process needs 'config' too!
   
   # ✅ GOOD - Complete state
   Send("process", {
       "data": item, 
       "config": state["config"],
       "user_id": state["user_id"]
   })
   ```

4. **Subgraph State Isolation**
   ```python
   # ❌ BAD - Subgraph modifies unexpected parent state
   def subgraph_node(state):
       state["parent_data"] = []  # Accidentally clears parent data!
       return state
   
   # ✅ GOOD - Only modify subgraph-specific keys
   def subgraph_node(state):
       return {"subgraph_result": process(state["input"])}
   ```

5. **Resource Exhaustion**
   ```python
   # ❌ BAD - Unbounded parallelization
   def process_all(state):
       # Could spawn 10,000 parallel tasks!
       return [Send("analyze", {"item": i}) for i in state["huge_list"]]
   
   # ✅ GOOD - Controlled parallelization
   def process_batched(state):
       MAX_PARALLEL = 50
       items = state["huge_list"]
       
       if len(items) > MAX_PARALLEL:
           # Process in chunks
           return "batch_processor"
       else:
           return [Send("analyze", {"item": i}) for i in items]
   ```

## Best Practices

1. **Design for Independence**: Parallel nodes should not depend on each other's outputs
2. **Use Reducers Everywhere**: Any parallel writes need reducers
3. **Profile Before Optimizing**: Measure to ensure parallelization helps
4. **Handle Partial Failures**: Design for resilience when some parallel tasks fail
5. **Monitor Resource Usage**: Watch API limits, memory, and CPU
6. **Test Edge Cases**: Empty lists, single items, very large batches

## Key Takeaways

1. **Start with fan-out/fan-in** for simple parallel operations
2. **Use Send() API** for dynamic, data-driven parallelization  
3. **Always add reducers** for parallel state updates
4. **Leverage subgraphs** for modular, reusable components
5. **Combine patterns** for sophisticated multi-agent systems
6. **Test thoroughly** - parallel systems have more edge cases

## Next Steps

Module 4 provides the foundation for building high-performance AI systems. Module 5 explores memory systems for long-term context retention and retrieval across graph executions.