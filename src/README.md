# LangGraph Chat Monorepo

This monorepo contains both the LangGraph agent and Next.js client for a production-ready conversational AI system.

## Project Structure

```
src/
├── agent/                 # LangGraph ReAct agent
│   ├── agent.py          # Main agent implementation
│   ├── requirements.txt  # Python dependencies
│   ├── langgraph.json    # LangGraph deployment config
│   └── .env.example      # Environment variables template
│
└── client/               # Next.js chat interface
    ├── src/             # Source code
    │   ├── app/         # Next.js app router
    │   ├── components/  # React components
    │   ├── hooks/       # Custom React hooks
    │   ├── lib/         # Utilities and API client
    │   ├── store/       # Zustand state management
    │   └── types/       # TypeScript types
    ├── package.json     # Node dependencies
    └── .env.example     # Environment variables template
```

## Quick Start

### 1. Set up the Agent

```bash
cd agent
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Copy and configure environment variables
cp .env.example .env
# Edit .env with your API keys
```

### 2. Set up the Client

```bash
cd ../client
npm install

# Copy and configure environment variables
cp .env.example .env.local
# Edit .env.local with your API keys
```

### 3. Run Locally

**Agent (using LangGraph dev server):**
```bash
cd agent
# Install LangGraph CLI if not already installed
pip install langgraph-cli
# Start the development server
langgraph dev
```

The LangGraph dev server provides:
- API endpoints at http://localhost:8123
- Visual debugging studio at http://localhost:8123/studio
- API documentation at http://localhost:8123/docs

**Client:**
```bash
cd client
npm run dev
```

Visit http://localhost:3000 to see the chat interface.

## Deployment

### Deploy Agent to LangGraph Platform

```bash
cd agent
langgraph deploy --name react-agent
```

### Deploy Client to Vercel

```bash
cd client
vercel --prod
```

## Key Features

- **ReAct Agent**: Claude Sonnet 4 with selective Tavily web search
- **Real-time Streaming**: Server-Sent Events for instant feedback
- **Thinking Display**: Shows AI reasoning in real-time
- **Smart Suggestions**: Claude Haiku generates follow-up questions
- **Thread Management**: Persistent conversation history
- **Type Safety**: Full TypeScript and Python type hints

## Environment Variables

### Agent (.env)
- `ANTHROPIC_API_KEY`: Your Anthropic API key
- `TAVILY_API_KEY`: Your Tavily search API key
- `LANGCHAIN_API_KEY`: Your LangChain API key
- `ANTHROPIC_MODEL`: Model name (default: claude-sonnet-4-20250514)

### Client (.env.local)
- `NEXT_PUBLIC_LANGGRAPH_API_KEY`: LangGraph API key
- `NEXT_PUBLIC_API_ENDPOINT`: API endpoint URL
- `LANGGRAPH_ENDPOINT`: Server-side LangGraph endpoint
- `LANGGRAPH_API_KEY`: Server-side API key

## Development

### Running Tests

**Agent:**
```bash
cd agent
pytest
```

**Client:**
```bash
cd client
npm test
```

### Type Checking

**Agent:**
```bash
cd agent
mypy agent.py
```

**Client:**
```bash
cd client
npm run type-check
```

## Architecture

See the documentation in `/docs/concept/` for detailed architecture information:
- `agent.md`: Agent implementation details
- `client.md`: Client implementation details
- `integration.md`: Integration guide
- `overview.md`: Complete system overview

## License

MIT