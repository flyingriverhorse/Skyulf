// C7: debounced canvas-side schema preview.
//
// Watches `nodes` + `edges` from `useGraphStore`, debounces 400 ms, then
// POSTs to `/api/pipeline/schema-preview`. Writes the response into the
// store so `CustomNodeWrapper` can paint `↳ N cols` badges and red
// borders on nodes with broken column references.
//
// Skips empty graphs and graphs that are still being assembled (the
// validator never raises, but a no-input graph yields nothing useful).
// Errors are logged and silently dropped — schema prediction is a
// nice-to-have, never block the canvas.

import { useEffect, useRef } from 'react';
import { useGraphStore } from '../store/useGraphStore';
import { convertGraphToPipelineConfig } from '../utils/pipelineConverter';
import { previewPipelineSchema } from '../api/schemaPreview';

const DEBOUNCE_MS = 400;

export const useSchemaPreview = (): void => {
  const nodes = useGraphStore((s) => s.nodes);
  const edges = useGraphStore((s) => s.edges);
  const setPredictedSchemas = useGraphStore((s) => s.setPredictedSchemas);
  const setBrokenSchemaRefs = useGraphStore((s) => s.setBrokenSchemaRefs);

  // Track in-flight request id so a slow response doesn't overwrite a
  // newer one. Plain ref; we never need to re-render on this.
  const requestIdRef = useRef(0);

  useEffect(() => {
    if (nodes.length === 0) {
      setPredictedSchemas({});
      setBrokenSchemaRefs({});
      return;
    }

    const handle = window.setTimeout(() => {
      const myRequestId = ++requestIdRef.current;
      let cancelled = false;

      void (async () => {
        try {
          const config = convertGraphToPipelineConfig(nodes, edges);
          const response = await previewPipelineSchema(config);

          // Drop stale responses.
          if (cancelled || myRequestId !== requestIdRef.current) return;

          setPredictedSchemas(response.predicted_schemas);

          const grouped: Record<
            string,
            Array<{ field: string; column: string; upstream_node_id: string | null }>
          > = {};
          for (const ref of response.broken_references) {
            const list = grouped[ref.node_id] ?? [];
            list.push({
              field: ref.field,
              column: ref.column,
              upstream_node_id: ref.upstream_node_id,
            });
            grouped[ref.node_id] = list;
          }
          setBrokenSchemaRefs(grouped);
        } catch (err) {
          // Schema preview is best-effort — never noisy in the UI.
          if (!cancelled) {
            console.debug('[schema-preview] API call failed', err);
          }
        }
      })();

      return () => {
        cancelled = true;
      };
    }, DEBOUNCE_MS);

    return () => window.clearTimeout(handle);
  }, [nodes, edges, setPredictedSchemas, setBrokenSchemaRefs]);
};
