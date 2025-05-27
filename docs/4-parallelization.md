# 4: Parallelization

**Created**: 2025-05-26  
**Last Modified**: 2025-05-26

## What You'll Learn

Module 4 demonstrates LangGraph's parallelization capabilities for building high-performance AI systems. You'll master fan-out/fan-in patterns, the Send() API for dynamic parallelization, subgraph composition, and how to combine these patterns into sophisticated multi-agent systems like a research assistant.

## Why It Matters

Sequential processing creates bottlenecks in AI applications. Parallelization enables:
- **Performance**: Execute multiple operations simultaneously
- **Scalability**: Handle variable workloads dynamically
- **Modularity**: Compose complex systems from reusable subgraphs
- **Efficiency**: Reduce overall execution time dramatically

## How It Works

### Core Concepts

1. **Fan-out/Fan-in Pattern**
   - Multiple nodes execute in parallel
   - Results merge automatically via reducers
   - Static parallelization defined at compile time

2. **Send() API**
   - Dynamic parallelization based on runtime data
   - Map phase distributes work across nodes
   - Reduce phase aggregates results
   - Handles variable workloads elegantly

3. **Reducers for Parallel State**
   - Required when parallel nodes write to same key
   - Prevent state conflicts automatically
   - Common patterns: append, merge, select

4. **Subgraphs**
   - Encapsulated graphs with own state schema
   - Communicate via overlapping state keys
   - Enable modular, reusable components
   - Support independent development

5. **Combined Patterns**
   - Research assistant uses all techniques
   - Parallel interviews with subgraphs
   - Map-reduce for report generation
   - Human oversight throughout

### Python Patterns

#### Basic Parallelization
```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from operator import add

class State(TypedDict):
    results: Annotated[list, add]  # Reducer for parallel writes

graph = StateGraph(State)

# Parallel nodes
graph.add_node("search_web", search_web)
graph.add_node("search_wiki", search_wikipedia)
graph.add_node("aggregate", aggregate_results)

# Fan-out
graph.add_edge(START, "search_web")
graph.add_edge(START, "search_wiki")

# Fan-in
graph.add_edge("search_web", "aggregate")
graph.add_edge("search_wiki", "aggregate")
graph.add_edge("aggregate", END)
```

#### Send API for Map-Reduce
```python
from langgraph.constants import Send

def continue_to_jokes(state: State):
    """Map phase - distribute to multiple instances"""
    return [
        Send("generate_joke", {"subject": s}) 
        for s in state["subjects"]
    ]

graph.add_conditional_edges(
    "generate_subjects",
    continue_to_jokes,  # Returns list of Send objects
    ["generate_joke"]   # Destination node
)

# Reduce phase
def aggregate_jokes(state: State):
    """Automatically receives all results"""
    return {"best_joke": select_best(state["jokes"])}
```

#### Subgraph Pattern
```python
# Define subgraph with own state
class SubgraphState(TypedDict):
    logs: list
    summary: str

def create_log_analyzer():
    subgraph = StateGraph(SubgraphState)
    subgraph.add_node("analyze", analyze_logs)
    subgraph.add_node("summarize", create_summary)
    subgraph.add_edge(START, "analyze")
    subgraph.add_edge("analyze", "summarize")
    subgraph.add_edge("summarize", END)
    return subgraph.compile()

# Use in parent graph
class ParentState(TypedDict):
    logs: list  # Overlapping key for communication
    reports: Annotated[list, add]

parent = StateGraph(ParentState)
parent.add_node("analyzer", create_log_analyzer())
```

### Common Pitfalls

1. **Missing Reducers**: Forgetting reducer when parallel nodes write to same key
   ```python
   # BAD - No reducer
   class State(TypedDict):
       results: list
   
   # GOOD - With reducer
   class State(TypedDict):
       results: Annotated[list, add]
   ```

2. **Order Assumptions**: Expecting parallel nodes to execute in specific order
   ```python
   # BAD - Assumes search_1 before search_2
   if "search_1" in state["completed"]:
       # Process search_2 differently
   
   # GOOD - Order-independent logic
   results = state.get("results", [])
   # Process all results equally
   ```

3. **Send State Mismatches**: Sending wrong state structure
   ```python
   # BAD - Missing required key
   Send("process", {"data": item})
   
   # GOOD - Complete state
   Send("process", {"data": item, "config": state["config"]})
   ```

4. **Subgraph State Conflicts**: Not handling state overlap correctly
   ```python
   # BAD - Subgraph overwrites parent state
   return {"logs": []}  # Clears parent's logs
   
   # GOOD - Preserve parent state
   return {"summary": summary}  # Only update what's needed
   ```

5. **Resource Exhaustion**: Creating too many parallel tasks
   ```python
   # BAD - Unbounded parallelization
   return [Send("process", {"item": i}) for i in huge_list]
   
   # GOOD - Batched processing
   for batch in chunks(huge_list, size=10):
       return [Send("process", {"item": i}) for i in batch]
   ```

## Key Takeaways

1. **Start Simple**: Master fan-out/fan-in before Send() API
2. **Always Use Reducers**: Required for parallel state updates
3. **Design for Independence**: Parallel nodes shouldn't depend on each other
4. **Leverage Subgraphs**: Modular components improve maintainability
5. **Monitor Performance**: More parallelization isn't always better
6. **Test Thoroughly**: Parallel systems have more edge cases

## Next Steps

This module equips you to build high-performance AI systems. Module 5 will explore memory systems for long-term context retention and retrieval.