import React, { useEffect, useRef } from 'react';

// mermaid is ~1MB minified, so it is dynamically imported and shared through
// a module-level promise: the Experiments bundle stays small and the library
// is initialized exactly once, only when a diagram is actually rendered.
let mermaidLoader: Promise<typeof import('mermaid').default> | null = null;

function loadMermaid(): Promise<typeof import('mermaid').default> {
  if (!mermaidLoader) {
    mermaidLoader = import('mermaid').then((mod) => {
      const mermaid = mod.default;
      const dark =
        typeof document !== 'undefined' &&
        document.documentElement.classList.contains('dark');
      mermaid.initialize({
        startOnLoad: false,
        securityLevel: 'strict',
        theme: dark ? 'dark' : 'default',
      });
      return mermaid;
    });
  }
  return mermaidLoader;
}

// mermaid draws flowchart labels as HTML inside <foreignObject> (htmlLabels),
// so the serialized SVG can contain markup that is NOT well-formed XML —
// e.g. an un-closed `<br>` from a `<br/>` line break in a label, which made
// `parseFromString(svg, 'image/svg+xml')` return a <parsererror> document.
// Parsing as HTML tolerates that label markup; DOMParser executes no scripts
// and returns the subtree inert, so mounting it stays XSS-safe.
export function svgFromString(svg: string): SVGSVGElement | null {
  const doc = new DOMParser().parseFromString(svg, 'text/html');
  return doc.querySelector('svg');
}

let diagramCounter = 0;

interface Props {
  chart: string;
  className?: string;
}

export const MermaidDiagram: React.FC<Props> = ({ chart, className }) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const noticeRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let cancelled = false;
    const id = `skyulf-mermaid-${++diagramCounter}`;

    const showNotice = (message: string | null, isError = false) => {
      const el = noticeRef.current;
      if (!el) return;
      el.textContent = message ?? '';
      el.classList.toggle('!text-red-500', isError);
      el.classList.toggle('dark:!text-red-400', isError);
    };

    showNotice('Rendering diagram…');
    loadMermaid()
      .then(async (mermaid) => {
        const { svg } = await mermaid.render(id, chart);
        if (cancelled || !containerRef.current) return;
        const svgEl = svgFromString(svg);
        if (!svgEl) throw new Error('mermaid produced no SVG output');
        containerRef.current.replaceChildren(svgEl);
        showNotice(null);
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        showNotice(
          `Failed to render pipeline diagram: ${err instanceof Error ? err.message : 'unknown error'}`,
          true
        );
      });
    return () => {
      cancelled = true;
    };
  }, [chart]);

  return (
    <div className={className}>
      <div
        ref={noticeRef}
        className="text-sm text-gray-400 dark:text-gray-500 p-4 empty:hidden"
        role="status"
      />
      <div
        ref={containerRef}
        className="flex justify-center overflow-x-auto [&>svg]:max-w-full [&>svg]:h-auto"
      />
    </div>
  );
};
