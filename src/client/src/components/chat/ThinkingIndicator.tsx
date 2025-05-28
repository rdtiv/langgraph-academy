'use client';

interface ThinkingIndicatorProps {
  content: string;
}

export function ThinkingIndicator({ content }: ThinkingIndicatorProps) {
  return (
    <div className="flex justify-start">
      <div className="max-w-[80%] rounded-lg px-4 py-2 bg-accent/50 thinking-animation">
        <div className="flex items-start space-x-2">
          <div className="mt-1">
            <svg
              className="w-4 h-4 text-accent-foreground/70"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
              />
            </svg>
          </div>
          <div className="flex-1">
            <p className="text-sm font-medium text-accent-foreground/70 mb-1">
              Thinking...
            </p>
            <p className="text-sm text-accent-foreground/60 italic">
              {content}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}