import { NextRequest } from 'next/server';

const LANGGRAPH_ENDPOINT = process.env.LANGGRAPH_ENDPOINT || 'http://localhost:8123';
const LANGGRAPH_API_KEY = process.env.LANGGRAPH_API_KEY || 'dev-key';

export async function GET(
  request: NextRequest,
  { params }: { params: { path: string[] } }
) {
  const path = params.path.join('/');
  
  const response = await fetch(`${LANGGRAPH_ENDPOINT}/${path}`, {
    headers: {
      'X-API-Key': LANGGRAPH_API_KEY,
    },
  });

  const data = await response.json();
  return Response.json(data);
}

export async function POST(
  request: NextRequest,
  { params }: { params: { path: string[] } }
) {
  const path = params.path.join('/');
  const body = await request.text();
  
  // Handle streaming endpoints
  if (path.includes('/runs/stream')) {
    const response = await fetch(`${LANGGRAPH_ENDPOINT}/${path}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': LANGGRAPH_API_KEY,
      },
      body,
    });

    // Return SSE stream
    return new Response(response.body, {
      headers: {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
      },
    });
  }
  
  // Handle regular endpoints
  const response = await fetch(`${LANGGRAPH_ENDPOINT}/${path}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-API-Key': LANGGRAPH_API_KEY,
    },
    body,
  });

  const data = await response.json();
  return Response.json(data);
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