// Used by NodeSettingsModal to fetch dataset snapshots for preview and catalog-aware nodes.
import { useCallback, useEffect, useRef, useState } from 'react';
import {
  fetchPipelinePreview,
  type PipelinePreviewRequest,
} from '../../../api';
import { type PreviewState } from '../nodes/dataset/DataSnapshotSection';
import type { CatalogFlagMap } from './useCatalogFlags';

type GraphSnapshot = {
  nodes: any[];
  edges: any[];
} | null;

type UsePipelinePreviewArgs = {
  shouldFetchPreview: boolean;
  sourceId?: string | null;
  canTriggerPreview: boolean;
  graphSnapshot: GraphSnapshot;
  catalogFlags: CatalogFlagMap;
  targetNodeId: string | null;
  previewSignature?: string | null;
  skipPreview?: boolean;
  requestPreviewRows?: boolean;
  includeSignals?: boolean;
};

type UsePipelinePreviewResult = {
  previewState: PreviewState;
  refreshPreview: () => void;
};

const idleState: PreviewState = { status: 'idle', data: null, error: null };
const DEFAULT_PREVIEW_SAMPLE_SIZE = 1000; // Increased from 200 to 1000 for better class representation

export const usePipelinePreview = ({
  shouldFetchPreview,
  sourceId,
  canTriggerPreview,
  graphSnapshot,
  catalogFlags,
  targetNodeId,
  previewSignature,
  skipPreview = false,
  requestPreviewRows,
  includeSignals,
}: UsePipelinePreviewArgs): UsePipelinePreviewResult => {
  const { isPreviewNode } = catalogFlags;
  const [previewState, setPreviewState] = useState<PreviewState>(idleState);
  const [requestId, setRequestId] = useState(0);
  const lastSignatureRef = useRef<string | null>(null);
  const lastRequestRef = useRef<number>(-1);

  const refreshPreview = useCallback(() => {
    setRequestId((previous) => previous + 1);
  }, []);

  useEffect(() => {
    if (!shouldFetchPreview) {
      setPreviewState(idleState);
      lastSignatureRef.current = null;
      lastRequestRef.current = -1;
      return;
    }

    if (skipPreview) {
      return;
    }

    if (!sourceId) {
      setPreviewState(
        isPreviewNode
          ? { status: 'error', data: null, error: 'Select a dataset to generate previews.' }
          : idleState,
      );
      return;
    }

    if (!canTriggerPreview) {
      setPreviewState(
        isPreviewNode
          ? {
              status: 'error',
              data: null,
              error: 'Add nodes to the canvas before generating a snapshot.',
            }
          : idleState,
      );
      return;
    }

    if (!graphSnapshot) {
      return;
    }

    const fallbackSignature = (() => {
      const nodeIds = Array.isArray(graphSnapshot.nodes)
        ? graphSnapshot.nodes
            .map((entry: any) => (entry && typeof entry.id === 'string' ? entry.id : null))
            .filter((id: string | null): id is string => Boolean(id))
            .sort()
        : [];
      const edgePairs = Array.isArray(graphSnapshot.edges)
        ? graphSnapshot.edges
            .map((edge: any) => {
              const source = edge && typeof edge.source === 'string' ? edge.source.trim() : '';
              const target = edge && typeof edge.target === 'string' ? edge.target.trim() : '';
              return source && target ? `${source}->${target}` : null;
            })
            .filter((value: string | null): value is string => Boolean(value))
            .sort()
        : [];
      return JSON.stringify({
        sourceId: sourceId ?? null,
        targetNodeId: targetNodeId ?? null,
        nodeIds,
        edgePairs,
      });
    })();

    const effectiveSignature = previewSignature ?? fallbackSignature;
    const signatureChanged = effectiveSignature !== lastSignatureRef.current;
    const forcedRefresh = lastRequestRef.current !== requestId;
    if (!forcedRefresh && !signatureChanged) {
      return;
    }

    lastSignatureRef.current = effectiveSignature;
    lastRequestRef.current = requestId;

    let isMounted = true;
    setPreviewState((previous) => ({ status: 'loading', data: previous.data ?? null, error: null }));

    const wantRows = requestPreviewRows ?? isPreviewNode;
    const wantSignals = includeSignals ?? wantRows;

    const previewRequest: PipelinePreviewRequest = {
      dataset_source_id: sourceId,
      graph: {
        nodes: graphSnapshot.nodes,
        edges: graphSnapshot.edges,
      },
      target_node_id: targetNodeId ?? undefined,
      include_preview_rows: wantRows,
      include_signals: wantSignals,
    };

    if (wantRows) {
      previewRequest.sample_size = DEFAULT_PREVIEW_SAMPLE_SIZE;
    }

    fetchPipelinePreview(previewRequest)
      .then((response) => {
        if (!isMounted) {
          return;
        }
        setPreviewState({ status: 'success', data: response, error: null });
      })
      .catch((error: any) => {
        if (!isMounted) {
          return;
        }
        setPreviewState(
          isPreviewNode
            ? {
                status: 'error',
                data: null,
                error: error?.message ?? 'Unable to generate preview.',
              }
            : idleState,
        );
      });

    return () => {
      isMounted = false;
    };
  }, [
    canTriggerPreview,
    graphSnapshot,
    isPreviewNode,
    skipPreview,
    previewSignature,
    requestId,
    shouldFetchPreview,
    sourceId,
    targetNodeId,
  ]);

  return { previewState, refreshPreview };
};
