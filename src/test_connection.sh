#!/bin/bash

# Test script to verify LangGraph dev server connection

echo "🧪 Testing LangGraph Dev Server Connection"
echo "=========================================="

# Check if LangGraph dev server is running
echo -n "1. Checking LangGraph dev server at http://localhost:8123... "
if curl -s -o /dev/null -w "%{http_code}" http://localhost:8123/health | grep -q "200"; then
    echo "✅ Running"
else
    echo "❌ Not running"
    echo "   Please run 'langgraph dev' in the agent directory"
    exit 1
fi

# Check API docs
echo -n "2. Checking API docs at http://localhost:8123/docs... "
if curl -s -o /dev/null -w "%{http_code}" http://localhost:8123/docs | grep -q "200"; then
    echo "✅ Available"
else
    echo "⚠️  Not available (this is okay for some versions)"
fi

# Test thread creation
echo -n "3. Testing thread creation... "
RESPONSE=$(curl -s -X POST http://localhost:8123/threads \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-key" \
  -d '{"metadata": {"title": "Test Thread"}}')

if echo "$RESPONSE" | grep -q "thread_id"; then
    echo "✅ Success"
    THREAD_ID=$(echo "$RESPONSE" | grep -o '"thread_id":"[^"]*' | grep -o '[^"]*$')
    echo "   Created thread: $THREAD_ID"
else
    echo "❌ Failed"
    echo "   Response: $RESPONSE"
fi

# Check Next.js client
echo -n "4. Checking Next.js client at http://localhost:3000... "
if curl -s -o /dev/null -w "%{http_code}" http://localhost:3000 | grep -q "200"; then
    echo "✅ Running"
else
    echo "⚠️  Not running"
    echo "   Run 'npm run dev' in the client directory"
fi

echo ""
echo "=========================================="
echo "✨ Connection test complete!"
echo ""
echo "If all checks pass, your development environment is ready."
echo "Open http://localhost:3000 to start chatting!"