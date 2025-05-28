# Detailed Startup Guide for LangGraph Chat

This guide provides step-by-step instructions to get both the agent and client running on your Mac development machine using Cursor IDE.

## Prerequisites Check

First, let's make sure you have the necessary tools installed.

### 1. Check Python Setup

Open Terminal in Cursor (Cmd+J) and check:

```bash
# Check if pyenv is installed
pyenv --version

# If not installed, install it:
brew install pyenv

# Check if uv is installed
uv --version

# If not installed, install it:
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Check Node.js Setup

```bash
# Check Node.js (should be 18+ for Next.js)
node --version

# Check npm
npm --version

# If not installed or old version:
brew install node
```

### 3. Check Git

```bash
git --version
# If not installed: xcode-select --install
```

## Part 1: Setting Up the Python Agent

### Step 1: Navigate to Agent Directory

In Cursor's terminal:

```bash
cd /Users/dant/dev/langgraph-academy/src/agent
```

### Step 2: Set Up Python Environment with pyenv and uv

```bash
# Install Python 3.11 with pyenv (recommended for LangChain)
pyenv install 3.11.7
pyenv local 3.11.7

# Verify Python version
python --version  # Should show 3.11.7

# Create virtual environment with uv
uv venv

# You should see a new 'venv' folder appear in the agent directory
```

### Step 3: Activate Virtual Environment

```bash
# On Mac, activate the environment:
source venv/bin/activate

# You should see (venv) appear in your terminal prompt
# Like this: (venv) ➜  agent git:(main)
```

### Step 4: Install Dependencies with uv

```bash
# Use uv to install requirements (it's faster than pip)
uv pip install -r requirements.txt

# This will install:
# - langchain
# - langchain-anthropic
# - langchain-community
# - langgraph
# - tavily-python
# - python-dotenv
# - pydantic
# - httpx
```

### Step 5: Set Up Environment Variables

```bash
# Copy the example env file
cp .env.example .env

# Open .env in Cursor
# In Cursor, click on the .env file in the file explorer
# Or use: code .env
```

Now edit the `.env` file with your actual API keys:

```bash
# Replace these with your actual keys
ANTHROPIC_API_KEY=sk-ant-api03-YOUR-ACTUAL-KEY-HERE
TAVILY_API_KEY=tvly-YOUR-ACTUAL-KEY-HERE
LANGCHAIN_API_KEY=ls-YOUR-ACTUAL-KEY-HERE
ANTHROPIC_MODEL=claude-sonnet-4-20250514

# Optional but recommended for debugging
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=langgraph-chat-dev
```

**Where to get these keys:**
- **Anthropic API Key**: https://console.anthropic.com/settings/keys
- **Tavily API Key**: https://app.tavily.com/home (free tier available)
- **LangChain API Key**: https://smith.langchain.com/settings

### Step 6: Install LangGraph CLI

```bash
# Make sure you're in the agent directory with venv activated
uv pip install langgraph-cli

# Verify installation
langgraph --version
```

### Step 7: Start LangGraph Development Server

```bash
# Make sure you're in the agent directory
langgraph dev

# You should see output like:
# 🚀 Starting LangGraph development server...
# 📍 API: http://localhost:8123
# 📊 Studio: http://localhost:8123/studio
# 📝 Docs: http://localhost:8123/docs
#
# Press Ctrl+C to stop
```

The LangGraph dev server provides:
- **API endpoints** for your agent at http://localhost:8123
- **LangGraph Studio** for visual debugging at http://localhost:8123/studio
- **API documentation** at http://localhost:8123/docs

If you see errors:
- `ModuleNotFoundError`: Run `uv pip install -r requirements.txt` again
- `API key not found`: Check your .env file
- `langgraph: command not found`: Install with `uv pip install langgraph-cli`
- `Port already in use`: Use `langgraph dev --port 8124`

## Part 2: Setting Up the Next.js Client

### Step 1: Open New Terminal Tab

In Cursor:
- Press `Cmd+Shift+P`
- Type "Terminal: Create New Terminal"
- Or click the `+` button in the terminal panel

### Step 2: Navigate to Client Directory

```bash
cd /Users/dant/dev/langgraph-academy/src/client
```

### Step 3: Install Node Dependencies

```bash
# Install all dependencies
npm install

# This will create node_modules folder and install:
# - Next.js
# - React
# - TypeScript
# - Tailwind CSS
# - Zustand
# - And many more...

# This might take 2-3 minutes
```

### Step 4: Set Up Client Environment Variables

```bash
# Copy the example env file
cp .env.example .env.local

# Open in Cursor
code .env.local
```

Edit `.env.local`:

```bash
# For local development with langgraph dev (default port 8123)
NEXT_PUBLIC_LANGGRAPH_API_URL=http://localhost:8123
NEXT_PUBLIC_LANGGRAPH_API_KEY=dev-key

# Server-side variables for API proxy route
LANGGRAPH_ENDPOINT=http://localhost:8123
LANGGRAPH_API_KEY=dev-key
```

### Step 5: Understanding the Next.js Structure

Before running, here's what you're looking at:

```
client/
├── src/
│   ├── app/          # Pages and routing
│   │   ├── page.tsx  # Home page (/)
│   │   └── layout.tsx # Root layout
│   ├── components/   # React components
│   │   └── chat/     # Chat UI components
│   ├── hooks/        # Custom React hooks
│   ├── lib/          # Utilities
│   └── store/        # State management
├── public/           # Static files
└── package.json      # Dependencies
```

### Step 6: Start the Development Server

```bash
# Make sure you're in the client directory
npm run dev

# You'll see output like:
# ▲ Next.js 14.0.4
# - Local:        http://localhost:3000
# - Environments: .env.local
# 
# ✓ Ready in 2.1s
```

### Step 7: Open in Browser

1. Cmd+Click on `http://localhost:3000` in the terminal
2. Or open your browser and go to: http://localhost:3000

You should see the LangGraph Chat interface!

## Part 3: Connecting Agent and Client

Now we'll connect the LangGraph development server with the Next.js client.

### Step 1: Verify LangGraph Server is Running

Make sure your agent terminal shows:
```
🚀 LangGraph development server running
📍 API: http://localhost:8123
```

### Step 2: Update Client to Use LangGraph Dev Server

Go to your client terminal and update the `.env.local`:

```bash
# Edit client/.env.local
cd /Users/dant/dev/langgraph-academy/src/client
code .env.local
```

Update these values:
```bash
# Point to LangGraph dev server
NEXT_PUBLIC_API_ENDPOINT=http://localhost:8123

# For the API proxy
LANGGRAPH_ENDPOINT=http://localhost:8123
LANGGRAPH_API_KEY=dev-key  # LangGraph dev server doesn't require a real key
```

### Step 3: (Optional) Test the API Directly

You can test the LangGraph API directly:

```bash
# Create a thread
curl -X POST http://localhost:8123/threads \
  -H "Content-Type: application/json" \
  -d '{"metadata": {"title": "Test Chat"}}'

# You should get back a thread_id
```

### Step 4: Open LangGraph Studio

While developing, you can use LangGraph Studio to debug:

1. Open http://localhost:8123/studio in your browser
2. You'll see a visual representation of your agent graph
3. You can inspect message flows and state changes

### Step 5: Restart Next.js Client

```bash
# In client terminal, restart the server
# Press Ctrl+C to stop
npm run dev
```

## Testing Everything Together

### Quick Connection Test

We've included a test script to verify everything is connected properly:

```bash
# From the src directory
cd /Users/dant/dev/langgraph-academy/src
./test_connection.sh
```

This will check:
- LangGraph dev server is running
- API endpoints are accessible  
- Thread creation works
- Next.js client is running

### Manual Testing

1. Make sure the LangGraph dev server is running (Terminal 1 shows "API: http://localhost:8123")
2. Make sure the Next.js client is running (Terminal 2 shows "Ready")
3. Open http://localhost:3000 in your browser
4. Click "New Chat" to create a thread
5. Type a message like "What is LangGraph?"
6. You should see:
   - Your message appear
   - Thinking indicator (if the agent is reasoning)
   - The response streaming in
   - Suggestions appear after the response

**Debugging Tips:**
- Check LangGraph Studio at http://localhost:8123/studio to see the agent execution
- Check API docs at http://localhost:8123/docs
- Browser DevTools Network tab to see API calls

## Troubleshooting Common Issues

### Python/Agent Issues

**Issue: "command not found: uv"**
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
# Restart terminal
```

**Issue: "No module named 'langchain'"**
```bash
# Make sure venv is activated (you see (venv) in prompt)
source venv/bin/activate
# Reinstall dependencies
uv pip install -r requirements.txt
```

**Issue: "ANTHROPIC_API_KEY not found"**
```bash
# Check .env file exists and has keys
cat .env
# Make sure no spaces around = sign
# Correct: ANTHROPIC_API_KEY=sk-ant-123
# Wrong: ANTHROPIC_API_KEY = sk-ant-123
```

### Next.js/Client Issues

**Issue: "npm: command not found"**
```bash
# Install Node.js
brew install node
```

**Issue: "Cannot find module 'react'"**
```bash
# Make sure you're in client directory
cd /Users/dant/dev/langgraph-academy/src/client
# Reinstall dependencies
rm -rf node_modules
npm install
```

**Issue: "Port 3000 already in use"**
```bash
# Find what's using port 3000
lsof -ti:3000
# Kill the process
kill -9 $(lsof -ti:3000)
# Or use different port
npm run dev -- -p 3001
```

### Connection Issues

**Issue: "Failed to fetch" or no response**
1. Check LangGraph server is running (http://localhost:8123)
2. Check browser console for errors (Right-click → Inspect → Console)
3. Try accessing http://localhost:8123/docs directly
4. Check that `.env.local` points to correct port (8123)

**Issue: Chat not working**
1. Check both terminals for error messages
2. Make sure you created a thread (click "New Chat")
3. Check API keys are correct in agent/.env
4. Open LangGraph Studio (http://localhost:8123/studio) to debug

**Issue: "Port 8123 already in use"**
```bash
# Use a different port
langgraph dev --port 8124
# Then update client/.env.local to use port 8124
```

## Daily Development Workflow

Once everything is set up, here's your daily workflow:

### Starting Up

1. Open Cursor IDE
2. Open integrated terminal (Cmd+J)
3. Split terminal (click + button)

**Terminal 1 - Agent:**
```bash
cd /Users/dant/dev/langgraph-academy/src/agent
source venv/bin/activate
langgraph dev
```

**Terminal 2 - Client:**
```bash
cd /Users/dant/dev/langgraph-academy/src/client
npm run dev
```

4. Open http://localhost:3000

### Making Changes

- **Agent changes**: Edit files in `src/agent/`, LangGraph dev server auto-reloads
- **Client changes**: Edit files in `src/client/`, Next.js auto-reloads

Both servers support hot-reloading, so you don't need to restart them when you make changes!

### Shutting Down

1. In each terminal, press `Ctrl+C`
2. Agent: `deactivate` to exit virtual environment

## Next Steps

1. **Customize the UI**: Edit components in `src/client/src/components/`
2. **Modify agent behavior**: Edit `src/agent/agent.py`
3. **Add new features**: Check the documentation in `/docs/concept/`
4. **Deploy to production**: See deployment guides in the docs

## Getting Help

- **Python/LangGraph issues**: Check the LangGraph docs at https://python.langchain.com/docs/langgraph
- **Next.js issues**: Check https://nextjs.org/docs
- **This project**: Review docs in `/docs/concept/` folder

Remember: 
- Always activate the Python virtual environment for agent work
- Keep both servers running while developing
- Check the terminal outputs for error messages
- The browser console (F12) is your friend for debugging client issues

Happy coding! 🚀