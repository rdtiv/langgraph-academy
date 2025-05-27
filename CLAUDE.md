# LangGraph Academy - Project Guide for Claude

**Created**: 2025-05-26  
**Last Modified**: 2025-05-26

## 🎯 Project Overview

This is the LangGraph Academy repository - a comprehensive learning resource for building production-ready AI applications with LangGraph. The project contains 7 modules (0-6) covering everything from basic graph construction to advanced deployment patterns.

## 📂 Project Structure

```
langgraph-academy/
├── README.md                    # Main project readme
├── requirements.txt             # Python dependencies
├── docs/                        # Comprehensive documentation
│   ├── README.md               # Documentation index
│   ├── 1-intro-to-langgraph.md # Module 1 summary
│   ├── 2-state-and-memory.md   # Module 2 summary
│   ├── 3-human-in-the-loop.md  # Module 3 summary
│   ├── quick-reference.md      # Quick code patterns
│   └── updates-and-naming-schema.md
├── module-0/                    # Python basics
│   └── basics.ipynb
├── module-1/                    # Introduction to LangGraph
│   ├── *.ipynb                 # Jupyter notebooks
│   └── studio/                 # LangGraph Studio apps
├── module-2/                    # State and Memory
├── module-3/                    # Human-in-the-Loop
├── module-4/                    # Parallelization
├── module-5/                    # Memory Systems
└── module-6/                    # Deployment
```

## 🛠️ Development Commands

### Environment Setup
```bash
# The project uses a virtual environment at ./academy/
source academy/bin/activate  # Activate virtual environment

# Install dependencies
pip install -r requirements.txt
```

### Code Quality Commands

#### Linting (Python)
```bash
# Using ruff (recommended for this project)
ruff check .                    # Check for linting issues
ruff check --fix .             # Auto-fix linting issues
ruff format .                  # Format code

# Alternative: flake8
flake8 .                       # Check Python code style

# Alternative: pylint  
pylint module-*/studio/*.py    # Lint studio Python files
```

#### Type Checking
```bash
# Using mypy
mypy module-*/studio/*.py      # Type check studio files
mypy --strict .                # Strict type checking

# Using pyright
pyright                        # Alternative type checker
```

### Testing Commands
```bash
# Run Jupyter notebook tests
pytest --nbval module-*/*.ipynb  # Test notebook outputs

# Run Python tests (if any exist)
pytest                           # Run all tests
pytest -v                        # Verbose output
pytest module-1/                 # Test specific module
```

### LangGraph-Specific Commands
```bash
# LangGraph Studio
langgraph up                     # Start LangGraph Studio
langgraph dev                    # Development mode

# Inside module studio directories
cd module-1/studio
langgraph up                     # Run studio app
```

### Jupyter Notebook Commands
```bash
jupyter notebook                 # Start Jupyter server
jupyter lab                      # Start JupyterLab
```

## 📚 Module Overview

### Module 0: Python Basics
- Python fundamentals for LangGraph

### Module 1: Introduction to LangGraph
- Simple graphs, LLM integration, routing
- ReAct agents, memory, deployment
- Key files: `agent.py`, `router.py`, `simple.py`

### Module 2: State and Memory
- State schemas (TypedDict, Pydantic)
- Message management, summarization
- External memory with SQLite
- Key file: `chatbot.py`

### Module 3: Human-in-the-Loop
- Breakpoints, state editing
- Streaming with interruption
- Time travel debugging
- Key files: `agent.py`, `dynamic_breakpoints.py`

### Module 4: Parallelization
- Map-reduce patterns
- Subgraphs, parallel execution
- Key files: `map_reduce.py`, `parallelization.py`, `research_assistant.py`

### Module 5: Memory Systems
- Memory stores and schemas
- Collections and profiles
- Key files: `memory_agent.py`, `memory_store.py`

### Module 6: Deployment
- Production deployment patterns
- Docker deployment
- Key file: `task_maistro.py`

## 🔍 Common Patterns

### Basic Graph Structure
```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class State(TypedDict):
    messages: list

graph = StateGraph(State)
graph.add_node("node_name", node_function)
graph.add_edge(START, "node_name")
graph.add_edge("node_name", END)
app = graph.compile()
```

### Working with Studio Apps
Each `studio/` directory contains:
- `langgraph.json` - Configuration
- `requirements.txt` - Dependencies
- Python files - Graph implementations

## ⚠️ Important Notes

1. **Virtual Environment**: Always activate the `academy` environment before working
2. **Dependencies**: Each studio app may have its own requirements.txt
3. **Documentation**: Comprehensive summaries in `docs/` folder
4. **Notebooks**: Interactive examples in each module
5. **Database**: Module 2 uses SQLite for external memory

## 🚀 Quick Start

1. Activate environment: `source academy/bin/activate`
2. Navigate to a module: `cd module-1`
3. Open notebooks: `jupyter notebook`
4. Run studio apps: `cd studio && langgraph up`

## 📝 Documentation Standards

- Module summaries follow what-why-how format
- Code examples are simplified for clarity
- Common pitfalls are documented
- Each module builds on previous concepts

## 🔧 Troubleshooting

- Check virtual environment is activated
- Ensure all dependencies are installed
- For studio apps, check `langgraph.json` configuration
- Reference docs for conceptual understanding

---
*This file helps Claude understand the project structure and provide better assistance with development tasks.*