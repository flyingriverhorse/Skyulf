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

import { useEffect } from 'react';
import { useGraphStore } from '../store/useGraphStore';
import { convertGraphToPipelineConfig } from '../utils/pipelineConverter';
import { previewPipelineSchema } from '../api/schemaPreview';

const DEBOUNCE_MS = 400;

export const useSchemaPreview = (): void => {
  const nodes = useGraphStore((s) => s.nodes);
  const edges = useGraphStore((s) => s.edges);
  const setPredictedSchemas = useGraphStore((s) => s.setPredictedSchemas);
  const setBrokenSchemaRefs = useGraphStore((s) => s.setBrokenSchemaRefs);

  useEffect(() => {
    if (nodes.length === 0) {
      setPredictedSchemas({});
      setBrokenSchemaRefs({});
      return;
    }

    const controller = new AbortController();
    const handle = window.setTimeout(() => {
      void (async () => {
        try {
          const config = convertGraphToPipelineConfig(nodes, edges);
          const response = await previewPipelineSchema(config, controller.signal);

          // Cleanup invalidates this graph immediately, even during the next debounce.
          if (controller.signal.aborted) return;

          // A degraded response (missing keys) must never land in the
          // store: CustomNodeWrapper indexes these maps per node id and
          // would crash on `undefined`.
          setPredictedSchemas(response.predicted_schemas ?? {});

          const grouped: Record<
            string,
            Array<{ field: string; column: string; upstream_node_id: string | null }>
          > = {};
          for (const ref of response.broken_references ?? []) {
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
          if (!controller.signal.aborted) {
            console.debug('[schema-preview] API call failed', err);
          }
        }
      })();
    }, DEBOUNCE_MS);

    return () => {
      window.clearTimeout(handle);
      controller.abort();
    };
  }, [nodes, edges, setPredictedSchemas, setBrokenSchemaRefs]);
};
