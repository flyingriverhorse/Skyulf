import { useCallback, useRef, useEffect } from 'react';
import { Node, Edge, useReactFlow } from '@xyflow/react';
import { v4 as uuidv4 } from 'uuid';
import { useGraphStore } from '../store/useGraphStore';

interface ClipboardData {
  nodes: Node[];
  edges: Edge[];
}

const PASTE_OFFSET = 50; // px offset per paste

/**
 * Keyboard-driven copy/paste for canvas nodes.
 *
 * Ctrl+C  — copies selected nodes and their internal edges.
 * Ctrl+V  — pastes copies with an incremental position offset.
 */
export function useClipboard() {
  const clipboardRef = useRef<ClipboardData | null>(null);
  const pasteCountRef = useRef(0);
  const { getNodes, getEdges } = useReactFlow();

  const copySelected = useCallback(() => {
    const selectedNodes = getNodes().filter((n) => n.selected);
    if (selectedNodes.length === 0) return;

    const selectedIds = new Set(selectedNodes.map((n) => n.id));

    // Only copy edges where both endpoints are in the selection
    const internalEdges = getEdges().filter(
      (e) => selectedIds.has(e.source) && selectedIds.has(e.target),
    );

    clipboardRef.current = {
      nodes: selectedNodes.map((n) => structuredClone(n)),
      edges: internalEdges.map((e) => structuredClone(e)),
    };
    pasteCountRef.current = 0;
  }, [getNodes, getEdges]);

  const paste = useCallback(() => {
    if (!clipboardRef.current || clipboardRef.current.nodes.length === 0) return;

    pasteCountRef.current += 1;
    const offset = PASTE_OFFSET * pasteCountRef.current;

    // Map old ID → new ID
    const idMap = new Map<string, string>();
    clipboardRef.current.nodes.forEach((n) => {
      const baseName = (n.data?.definitionType as string) || n.type || 'node';
      idMap.set(n.id, `${baseName}-${uuidv4()}`);
    });

    const newNodes: Node[] = clipboardRef.current.nodes.map((n) => ({
      ...n,
      id: idMap.get(n.id)!,
      position: { x: n.position.x + offset, y: n.position.y + offset },
      selected: false,
      data: { ...n.data },
    }));

    const newEdges: Edge[] = clipboardRef.current.edges.map((e) => ({
      ...e,
      id: `e-${uuidv4()}`,
      source: idMap.get(e.source) ?? e.source,
      target: idMap.get(e.target) ?? e.target,
      selected: false,
    }));

    const store = useGraphStore.getState();
    store.setGraph(
      [...store.nodes, ...newNodes],
      [...store.edges, ...newEdges],
    );
  }, []);

  // Global keydown listener
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Skip when user is typing in an input / textarea / contenteditable
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
      if ((e.target as HTMLElement)?.isContentEditable) return;

      const mod = e.ctrlKey || e.metaKey;
      if (!mod) return;

      if (e.key === 'c') {
        copySelected();
      } else if (e.key === 'v') {
        paste();
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [copySelected, paste]);
}
