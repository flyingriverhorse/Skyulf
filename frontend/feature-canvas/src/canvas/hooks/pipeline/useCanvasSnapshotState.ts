import { useCallback, useRef, useState, type Dispatch, type MutableRefObject, type SetStateAction } from 'react';
import type { Edge, Node } from 'react-flow-renderer';
import type { SaveFeedback } from '../../types/feedback';

type Snapshot = {
  nodes: Node[];
  edges: Edge[];
};

type UseCanvasSnapshotStateOptions = {
  setIsDirty: Dispatch<SetStateAction<boolean>>;
  setSaveFeedback: Dispatch<SetStateAction<SaveFeedback | null>>;
  isHydratingRef: MutableRefObject<boolean>;
};

type HydratedSnapshotResult = {
  hasCustomNodes: boolean;
  hasEdges: boolean;
};

type UseCanvasSnapshotStateResult = {
  graphSnapshotRef: MutableRefObject<Snapshot>;
  canClearCanvas: boolean;
  handleGraphChange: (nodes: Node[], edges: Edge[]) => void;
  applyHydratedSnapshot: (nodes?: Node[], edges?: Edge[]) => HydratedSnapshotResult;
};

const cloneGraphSnapshot = (nodes?: Node[], edges?: Edge[]): Snapshot => ({
  nodes: JSON.parse(JSON.stringify(nodes ?? [])),
  edges: JSON.parse(JSON.stringify(edges ?? [])),
});

const hasCustomNodes = (nodes?: Node[]): boolean =>
  Array.isArray(nodes) ? nodes.some((node) => node?.id && node.id !== 'dataset-source') : false;

const hasEdgesPresent = (edges?: Edge[]): boolean => (Array.isArray(edges) ? edges.length > 0 : false);

export const useCanvasSnapshotState = ({
  setIsDirty,
  setSaveFeedback,
  isHydratingRef,
}: UseCanvasSnapshotStateOptions): UseCanvasSnapshotStateResult => {
  const graphSnapshotRef = useRef<Snapshot>({ nodes: [], edges: [] });
  const [canClearCanvas, setCanClearCanvas] = useState(false);

  const applyHydratedSnapshot = useCallback((nodes?: Node[], edges?: Edge[]) => {
    const snapshot = cloneGraphSnapshot(nodes, edges);
    graphSnapshotRef.current = snapshot;
    const hasCustom = hasCustomNodes(snapshot.nodes);
    const hasEdges = hasEdgesPresent(snapshot.edges);
    setCanClearCanvas(hasCustom || hasEdges);
    return { hasCustomNodes: hasCustom, hasEdges };
  }, []);

  const handleGraphChange = useCallback(
    (nodes: Node[], edges: Edge[]) => {
      const nextSnapshot = cloneGraphSnapshot(nodes, edges);
      const previousSnapshot = graphSnapshotRef.current;
      const isSameSnapshot =
        JSON.stringify(previousSnapshot.nodes ?? []) === JSON.stringify(nextSnapshot.nodes ?? []) &&
        JSON.stringify(previousSnapshot.edges ?? []) === JSON.stringify(nextSnapshot.edges ?? []);

      graphSnapshotRef.current = nextSnapshot;

      const hasCustom = hasCustomNodes(nextSnapshot.nodes);
      const hasEdges = hasEdgesPresent(nextSnapshot.edges);
      setCanClearCanvas(hasCustom || hasEdges);

      if (isHydratingRef.current || isSameSnapshot) {
        return;
      }

      setIsDirty(true);
      setSaveFeedback(null);
    },
    [isHydratingRef, setIsDirty, setSaveFeedback]
  );

  return {
    graphSnapshotRef,
    canClearCanvas,
    handleGraphChange,
    applyHydratedSnapshot,
  };
};
