import { useMemo } from 'react';
import type { Node, Edge } from '@xyflow/react';
import {
  EXECUTION_MODE_AWARE_TYPES,
  AUTO_PARALLEL_TYPES,
  isParallelExecution,
} from '../types/executionMode';

// Terminals = mode-aware modeling nodes plus auto-parallel inspectors.
const TERMINAL_TYPES = new Set([...EXECUTION_MODE_AWARE_TYPES, ...AUTO_PARALLEL_TYPES]);

/** Generate n evenly-spaced, high-saturation HSL colors with a golden-angle offset for variety. */
export function generateBranchColors(count: number): string[] {
  const colors: string[] = [];
  const goldenAngle = 137.508; // degrees — maximises hue separation
  for (let i = 0; i < count; i++) {
    const hue = (i * goldenAngle) % 360;
    colors.push(`hsl(${Math.round(hue)}, 80%, 65%)`);
  }
  return colors;
}

/** Convert snake_case model id to a readable name: "random_forest_classifier" → "Random Forest" */
function prettifyModelType(modelType: string): string {
  return modelType
    .replace(/_classifier$|_regressor$/, '')
    .split('_')
    .map(w => w.charAt(0).toUpperCase() + w.slice(1))
    .join(' ');
}

export interface BranchEdgeInfo {
  color: string;
  /** Label only set on the terminal edge (edge entering the training node). */
  label: string | null;
  /** True when this edge feeds multiple training nodes (shared infrastructure). */
  shared: boolean;
}

/**
 * Assigns branch colors to edges based on which training/tuning terminal
 * they belong to. When a terminal has multiple inputs each input path is
 * treated as a separate branch (mirroring backend partition logic).
 *
 * Shared edges (belonging to multiple branches) get the first branch's
 * color + the `shared` flag.
 *
 * The branch label is only attached to the edge entering the terminal
 * node — not every edge in the branch.
 */
export function useBranchColors(nodes: Node[], edges: Edge[]): Map<string, BranchEdgeInfo> {
  return useMemo(() => {
    const colorMap = new Map<string, BranchEdgeInfo>();

    // Find connected training/tuning/preview-leaf nodes
    const terminals = nodes.filter(
      n => TERMINAL_TYPES.has(n.data.definitionType as string) && edges.some(e => e.target === n.id)
    );

    // Also treat raw data leaves (nodes with no downstream consumers) as
    // pseudo-terminals so dangling preprocessing chains get their own
    // branch color + label on the canvas — matching the Run Preview tabs
    // which now include preview-only branches alongside training branches.
    const consumed = new Set<string>();
    for (const e of edges) consumed.add(e.source);
    const knownTerminalIds = new Set(terminals.map(t => t.id));
    for (const n of nodes) {
      if (knownTerminalIds.has(n.id)) continue;
      if (consumed.has(n.id)) continue;
      // Skip nodes with no incoming edges (orphaned dataset/source nodes
      // would otherwise become "branches" of size zero).
      if (!edges.some(e => e.target === n.id)) continue;
      // Skip the data_preview node itself — it already lives in
      // TERMINAL_TYPES via AUTO_PARALLEL_TYPES if relevant.
      if (n.data.definitionType === 'data_preview') continue;
      terminals.push(n);
    }

    // Sort terminals to match the BFS topological order that pipelineConverter.ts
    // and the backend's partition_parallel_pipeline both use (BFS forward from
    // source/dataset nodes). Without this, the canvas assigns "Path A" to
    // whichever terminal React Flow stores first (insertion order), while the
    // backend assigns it to the first terminal in BFS order — causing the canvas
    // edge labels and Preview Results tab contents to disagree.
    {
      const fwdAdj = new Map<string, string[]>();
      for (const edge of edges) {
        const list = fwdAdj.get(edge.source) ?? [];
        list.push(edge.target);
        fwdAdj.set(edge.source, list);
      }
      // Source nodes = nodes with no incoming edges (dataset roots).
      const srcIds = nodes.filter(n => !edges.some(e => e.target === n.id)).map(n => n.id);
      const topoOrder = new Map<string, number>();
      const bfsQ: string[] = [...srcIds];
      const bfsSeen = new Set<string>(srcIds);
      while (bfsQ.length > 0) {
        const nid = bfsQ.shift()!;
        if (!topoOrder.has(nid)) topoOrder.set(nid, topoOrder.size);
        for (const child of fwdAdj.get(nid) ?? []) {
          if (!bfsSeen.has(child)) { bfsSeen.add(child); bfsQ.push(child); }
        }
      }
      terminals.sort((a, b) => (topoOrder.get(a.id) ?? 999999) - (topoOrder.get(b.id) ?? 999999));
    }

    if (terminals.length === 0) return colorMap;

    // Build adjacency: target → source edges (for BFS backwards)
    const incomingMap = new Map<string, Edge[]>();
    for (const edge of edges) {
      const list = incomingMap.get(edge.target) || [];
      list.push(edge);
      incomingMap.set(edge.target, list);
    }

    // Build branches: one per terminal by default (merge mode).
    // Only split a multi-input terminal into per-input branches when the
    // node is explicitly set to execution_mode === 'parallel' or is an
    // auto-parallel terminal type. Each branch carries its own per-terminal
    // index so the Path letter restarts at A for every terminal—keeping the
    // canvas edge labels aligned with the per-terminal tab bars (e.g. the
    // Data Preview node's tabs always read Path A / Path B / …).
    //
    // FTS y-pass-through: FeatureTargetSplit's "y" output handle carries the
    // TARGET LABEL, not a separate preprocessing experiment. When a user
    // connects FTS-y → AT directly while also connecting TTS → AT (where
    // TTS descends from FTS), counting FTS as a second source would
    // incorrectly split one experiment into two color branches.
    // Solution: exclude y-pass-through edges from source counting and
    // parallel detection. Post-BFS, assign them to the branch whose root
    // is a descendant of that FTS node.
    const isYPassthrough = (edge: Edge): boolean => {
      if (edge.sourceHandle !== 'y') return false;
      const srcNode = nodes.find(n => n.id === edge.source);
      return (srcNode?.data.definitionType as string | undefined) === 'feature_target_split';
    };

    interface BranchDef {
      terminal: Node;
      inputEdge: Edge | null;
      allTerminalEdges: Edge[];
      yHelperEdges: Edge[];  // FTS-y pass-through edges belonging to this terminal
      localIndex: number;
    }
    const branches: BranchDef[] = [];
    for (const terminal of terminals) {
      const terminalIncoming = incomingMap.get(terminal.id) || [];
      if (terminalIncoming.length === 0) continue;

      // Split helpers from real branch edges
      const mainEdges  = terminalIncoming.filter(e => !isYPassthrough(e));
      const yHelperEdges = terminalIncoming.filter(e => isYPassthrough(e));

      const sourceCount = new Set(mainEdges.map(e => e.source)).size;
      const isParallel = isParallelExecution(terminal.data, sourceCount);
      if (isParallel && sourceCount > 1) {
        // Parallel mode: one branch per unique SOURCE NODE (not per edge).
        // Multi-handle splitters (TTS train/test/val) emit several edges from
        // the same source — dedup to one branch per source.
        // Y-pass-throughs are excluded above and assigned post-BFS so that
        // e.g. FTS-y → AT does NOT count as a separate parallel experiment.
        const seenSrc = new Set<string>();
        let localIdx = 0;
        for (const edge of mainEdges) {
          if (seenSrc.has(edge.source)) continue;
          seenSrc.add(edge.source);
          const groupEdges = mainEdges.filter(e => e.source === edge.source);
          branches.push({
            terminal,
            inputEdge: edge,
            allTerminalEdges: groupEdges,
            yHelperEdges,
            localIndex: localIdx,
          });
          localIdx++;
        }
      } else {
        // Merge mode (default): all inputs funnel into one branch.
        // BFS from the terminal naturally visits all incoming edges including
        // y-helper edges, so no special handling needed here.
        branches.push({ terminal, inputEdge: null, allTerminalEdges: [], yHelperEdges, localIndex: 0 });
      }
    }

    // Need 2+ branches for coloring to matter
    if (branches.length < 2) return colorMap;

    const colors = generateBranchColors(branches.length);

    // Disambiguate terminals that share the same model type (e.g. two
    // Advanced Training nodes both running XGBoost). Without this, every
    // branch ends up labeled "Path A · Xgboost" / "Path B · Xgboost" with
    // no way to tell which terminal each path actually feeds. We assign a
    // 1-based suffix (#1, #2, …) per model-type group, but only when a
    // collision exists — single-terminal-per-model canvases stay clean.
    const terminalsByModel = new Map<string, string[]>();
    for (const t of terminals) {
      const mt = (t.data.model_type as string | undefined) ?? '';
      const list = terminalsByModel.get(mt) ?? [];
      list.push(t.id);
      terminalsByModel.set(mt, list);
    }
    const terminalSuffix = new Map<string, string>();
    for (const [, ids] of terminalsByModel) {
      if (ids.length < 2) continue;
      ids.forEach((id, idx) => {
        terminalSuffix.set(id, `#${idx + 1}`);
      });
    }

    // Collect the terminal-entering edge ids so we can tag them with labels
    const terminalEdgeIds = new Set<string>();

    // For each branch, BFS backwards to collect ancestor edges
    const branchEdgeSets: Set<string>[] = [];
    const branchLabels: string[] = [];
    for (let i = 0; i < branches.length; i++) {
      const { terminal, inputEdge, allTerminalEdges, yHelperEdges, localIndex } = branches[i]!;
      const modelType = terminal.data.model_type as string | undefined;
      const modelName = modelType ? prettifyModelType(modelType) : '';
      // Path letter strategy:
      //   * Single terminal w/ multiple parallel inputs → per-terminal
      //     localIndex (so its tabs read Path A / Path B / …).
      //   * Multiple terminals (training + dangling preview branches, etc)
      //     → use the global branch index so canvas labels line up with
      //     the global tab order in Preview Results (Path A, B, C, …).
      const isMultiTerminal = terminals.length > 1;
      const letterIndex = isMultiTerminal ? i : localIndex;
      const pathLetter = String.fromCharCode(65 + letterIndex);
      // For preview-only branches, suffix the upstream source node label
      // so users can tell which path each tab corresponds to.
      let suffix = modelName;
      if (!suffix && inputEdge) {
        const sourceNode = nodes.find(n => n.id === inputEdge.source);
        const data = (sourceNode?.data ?? {}) as Record<string, unknown>;
        suffix = (data.label as string)
          || (data.title as string)
          || (typeof data.definitionType === 'string'
              ? (data.definitionType as string).replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
              : '');
      }
      // Merge-mode branches (no specific input edge) and non-training
      // terminals: fall back to the terminal node's own friendly name so
      // dangling preview branches get a meaningful label like
      // "Path B · Standard Scaler" instead of just "Path B".
      if (!suffix) {
        const data = (terminal.data ?? {}) as Record<string, unknown>;
        suffix = (data.label as string)
          || (data.title as string)
          || (typeof data.definitionType === 'string'
              ? (data.definitionType as string)
                  .replace(/([a-z])([A-Z])/g, '$1 $2')
                  .replace(/_/g, ' ')
                  .replace(/\b\w/g, c => c.toUpperCase())
              : '');
      }
      // Append a #N counter when multiple terminals share the same model
      // type so e.g. two XGBoost training nodes render as "Path A · Xgboost
      // #1" and "Path A · Xgboost #2" instead of two identical labels.
      const dupSuffix = terminalSuffix.get(terminal.id);
      if (dupSuffix && suffix) {
        suffix = `${suffix} ${dupSuffix}`;
      }
      const label = suffix ? `Path ${pathLetter} · ${suffix}` : `Path ${pathLetter}`;
      branchLabels.push(label);

      const visited = new Set<string>();
      const branchEdges = new Set<string>();

      if (inputEdge) {
        // Parallel branch: BFS from all edges of this source group at the terminal.
        const terminalEdges = allTerminalEdges.length > 0 ? allTerminalEdges : [inputEdge];
        // Only the first (representative) edge shows the Path label.
        terminalEdgeIds.add(terminalEdges[0]!.id);
        for (const e of terminalEdges) branchEdges.add(e.id);
        // BFS backwards from all group source nodes.
        const queue = terminalEdges.map(e => e.source);
        while (queue.length > 0) {
          const nodeId = queue.shift()!;
          if (visited.has(nodeId)) continue;
          visited.add(nodeId);
          const incoming = incomingMap.get(nodeId) || [];
          for (const edge of incoming) {
            branchEdges.add(edge.id);
            queue.push(edge.source);
          }
        }
        // Assign FTS y-pass-through edges: if this branch's BFS visited the
        // FTS node that produced the y-edge, color it the same as this branch.
        // This way FTS-y → AT renders the same color as TTS → AT when TTS
        // descends from that same FTS (they're cooperative inputs to one model).
        for (const yEdge of yHelperEdges) {
          if (visited.has(yEdge.source)) {
            branchEdges.add(yEdge.id);
          }
        }
      } else {
        // Merge branch: BFS from the terminal node (all its inputs)
        const terminalIncoming = incomingMap.get(terminal.id) || [];
        if (terminalIncoming.length > 0) {
          terminalEdgeIds.add(terminalIncoming[0]!.id);
        }
        const queue = [terminal.id];
        while (queue.length > 0) {
          const nodeId = queue.shift()!;
          if (visited.has(nodeId)) continue;
          visited.add(nodeId);
          const incoming = incomingMap.get(nodeId) || [];
          for (const edge of incoming) {
            branchEdges.add(edge.id);
            queue.push(edge.source);
          }
        }
      }
      branchEdgeSets.push(branchEdges);
    }

    // Count how many branches each edge belongs to
    const edgeBranchCount = new Map<string, number>();
    const edgeFirstBranch = new Map<string, number>();
    for (let i = 0; i < branchEdgeSets.length; i++) {
      for (const edgeId of branchEdgeSets[i]!) {
        edgeBranchCount.set(edgeId, (edgeBranchCount.get(edgeId) || 0) + 1);
        if (!edgeFirstBranch.has(edgeId)) {
          edgeFirstBranch.set(edgeId, i);
        }
      }
    }

    // Assign colors — multi-branch edges get first branch color + shared flag
    for (const [edgeId, count] of edgeBranchCount) {
      const branchIdx = edgeFirstBranch.get(edgeId)!;
      colorMap.set(edgeId, {
        color: colors[branchIdx]!,
        label: terminalEdgeIds.has(edgeId) ? (branchLabels[branchIdx] ?? null) : null,
        shared: count > 1,
      });
    }

    return colorMap;
  }, [nodes, edges]);
}
