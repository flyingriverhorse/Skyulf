# Split Edge Rendering Regression – Postmortem

## What Went Wrong

Connecting split-aware nodes created edge entries, but the polylines never rendered. React Flow attempts to measure handle positions via the DOM when `updateNodeInternals` runs; however, we were adding edges before the destination node had mounted its new split handles. Because no re-measure happened after those handles appeared, React Flow kept stale `NaN` coordinates and dropped the edge visuals.

### Telltale Signs
- Edges existed in state and split propagation logs looked correct.
- Split handles showed up on the downstream node, yet edges stayed invisible.
- Console errors surfaced once we tried to call `instance.updateNodeInternals`, which isn’t part of the public API in `react-flow-renderer@10`.

## What Fixed It

1. **Move to the supported API** – switched to `useUpdateNodeInternals` and wrapped the canvas in `ReactFlowProvider`, so handle updates go through the renderer’s store instead of the removed instance method.
2. **Delay the refresh** – queued the updates on two `requestAnimationFrame` passes to ensure React has committed the new handles before the recalculation fires.
3. **Broaden triggers** – anytime split metadata or a connection changes we queue `updateNodeInternals` for the affected nodes, guaranteeing React Flow remeasures whenever handles appear or disappear.

With those adjustments, React Flow receives valid coordinates the moment split handles materialize, and the edges render immediately.

**Date resolved:** October 22, 2025
