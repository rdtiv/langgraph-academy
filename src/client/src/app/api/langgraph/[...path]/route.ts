import { NextRequest } from 'next/server';

const LANGGRAPH_ENDPOINT = process.env.LANGGRAPH_ENDPOINT || 'http://localhost:8123';
const LANGGRAPH_API_KEY = process.env.LANGGRAPH_API_KEY || 'dev-key';

export async function GET(
  request: NextRequest,
  { params }: { params: { path: string[] } }
) {
  const path = params.path.join('/');
  
  // Validate API key
  if (!LANGGRAPH_API_KEY) {
    return Response.json({ error: 'API key not configured' }, { status: 500 });
  }
  
  try {
    const response = await fetch(`${LANGGRAPH_ENDPOINT}/${path}`, {
      headers: {
        'X-API-Key': LANGGRAPH_API_KEY,
      },
    });

    if (!response.ok) {
      const error = await response.text();
      return Response.json({ error }, { status: response.status });
    }

    const data = await response.json();
    return Response.json(data);
  } catch (error) {
    return Response.json({ error: 'Failed to fetch from LangGraph' }, { status: 500 });
  }
}

export async function POST(
  request: NextRequest,
  { params }: { params: { path: string[] } }
) {
  const path = params.path.join('/');
  const body = await request.text();
  
  // Validate API key
  if (!LANGGRAPH_API_KEY) {
    return Response.json({ error: 'API key not configured' }, { status: 500 });
  }
  
  // Handle streaming endpoints
  if (path.includes('/runs/stream')) {
    try {
      const response = await fetch(`${LANGGRAPH_ENDPOINT}/${path}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-API-Key': LANGGRAPH_API_KEY,
        },
        body,
      });

      if (!response.ok) {
        const error = await response.text();
        return Response.json({ error }, { status: response.status });
      }

      // Return SSE stream with CORS headers
      return new Response(response.body, {
        headers: {
          'Content-Type': 'text/event-stream',
          'Cache-Control': 'no-cache',
          'Connection': 'keep-alive',
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
          'Access-Control-Allow-Headers': 'Content-Type',
        },
      });
    } catch (error) {
      return Response.json({ error: 'Failed to stream from LangGraph' }, { status: 500 });
    }
  }
  
  // Handle regular endpoints
  try {
    const response = await fetch(`${LANGGRAPH_ENDPOINT}/${path}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': LANGGRAPH_API_KEY,
      },
      body,
    });

    if (!response.ok) {
      const error = await response.text();
      return Response.json({ error }, { status: response.status });
    }

    const data = await response.json();
    return Response.json(data);
  } catch (error) {
    return Response.json({ error: 'Failed to post to LangGraph' }, { status: 500 });
  }
}

export async function DELETE(
  request: NextRequest,
  { params }: { params: { path: string[] } }
) {
  const path = params.path.join('/');
  
  await fetch(`${LANGGRAPH_ENDPOINT}/${path}`, {
    method: 'DELETE',
    headers: {
      'X-API-Key': LANGGRAPH_API_KEY,
    },
  });

  return new Response(null, { status: 204 });
}