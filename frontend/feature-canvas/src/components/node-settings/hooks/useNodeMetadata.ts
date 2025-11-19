import { useMemo } from 'react';
import type { Node } from 'react-flow-renderer';

export type MetadataEntry = {
  label: string;
  value: string;
};

export const useNodeMetadata = (node: Node | null | undefined) => {
  const metadata = useMemo<MetadataEntry[]>(() => {
    const entries: MetadataEntry[] = [];

    if (node?.data?.label) {
      entries.push({ label: 'Label', value: String(node.data.label) });
    }

    if (node?.data?.description) {
      entries.push({ label: 'Description', value: String(node.data.description) });
    }

    if (node?.data?.category) {
      entries.push({ label: 'Category', value: String(node.data.category) });
    }

    if (node?.data?.catalogType) {
      entries.push({ label: 'Node type', value: String(node.data.catalogType) });
    }

    if (node?.data?.inputs && Array.isArray(node.data.inputs) && node.data.inputs.length) {
      entries.push({ label: 'Inputs', value: node.data.inputs.join(', ') });
    }

    if (node?.data?.outputs && Array.isArray(node.data.outputs) && node.data.outputs.length) {
      entries.push({ label: 'Outputs', value: node.data.outputs.join(', ') });
    }

    return entries;
  }, [node]);

  const title = useMemo(() => {
    const raw = node?.data?.label ?? node?.id;
    return typeof raw === 'string' && raw.trim() ? raw : `Node ${node?.id ?? ''}`;
  }, [node]);

  return { metadata, title };
};
