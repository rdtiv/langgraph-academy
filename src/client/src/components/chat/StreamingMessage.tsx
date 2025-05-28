'use client';

import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

interface StreamingMessageProps {
  content: string;
}

export function StreamingMessage({ content }: StreamingMessageProps) {
  return (
    <div className="flex justify-start">
      <div className="max-w-[80%] rounded-lg px-4 py-2 bg-muted">
        <div className="prose prose-sm dark:prose-invert max-w-none">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>
            {content}
          </ReactMarkdown>
        </div>
        <div className="flex items-center space-x-1 mt-2">
          <div className="w-2 h-2 bg-primary rounded-full animate-pulse" />
          <div className="w-2 h-2 bg-primary rounded-full animate-pulse animation-delay-200" />
          <div className="w-2 h-2 bg-primary rounded-full animate-pulse animation-delay-400" />
        </div>
      </div>
    </div>
  );
}