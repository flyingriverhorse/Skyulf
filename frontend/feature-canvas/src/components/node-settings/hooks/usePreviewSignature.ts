import { useMemo } from 'react';
import { stableStringify } from '../utils/configParsers';

type UsePreviewSignatureArgs = {
  nodeId: string;
  sourceId?: string | null;
  configState: Record<string, any>;
  upstreamNodeIds: string[];
  graphNodes: any[];
};

export const usePreviewSignature = ({
  nodeId,
  sourceId,
  configState,
  upstreamNodeIds,
  graphNodes,
}: UsePreviewSignatureArgs) => {
  const upstreamConfigFingerprints = useMemo(() => {
    if (!upstreamNodeIds.length) {
      return {} as Record<string, any>;
    }
    const map: Record<string, any> = {};
    upstreamNodeIds.forEach((identifier) => {
      const match = graphNodes.find((entry: any) => entry && typeof entry.id === 'string' && entry.id === identifier);
      if (!match) {
        return;
      }
      const configPayload = match?.data?.config ?? null;
      map[identifier] = configPayload;
    });
    return map;
  }, [graphNodes, upstreamNodeIds]);

  const previewSignature = useMemo(() => {
    return stableStringify({
      sourceId: sourceId ?? null,
      nodeId: nodeId || null,
      config: configState,
      upstreamIds: upstreamNodeIds,
      upstreamConfig: upstreamConfigFingerprints,
    });
  }, [configState, nodeId, sourceId, upstreamConfigFingerprints, upstreamNodeIds]);

  return {
    upstreamConfigFingerprints,
    previewSignature,
  };
};
