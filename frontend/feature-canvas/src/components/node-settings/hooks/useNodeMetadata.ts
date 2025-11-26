import { useMemo } from 'react';
import type { Node } from 'react-flow-renderer';
import type { FeatureNodeCatalogEntry } from '../../../api';

export type MetadataEntry = {
  label: string;
  value: string;
};

export const useNodeMetadata = (
  node: Node | null | undefined,
  catalogEntry?: FeatureNodeCatalogEntry | null,
) => {
  const metadata = useMemo<MetadataEntry[]>(() => {
    const entries: MetadataEntry[] = [];

    const label = node?.data?.label || catalogEntry?.label;
    if (label) {
      entries.push({ label: 'Label', value: String(label) });
    }

    const description = node?.data?.description || catalogEntry?.description;
    if (description) {
      entries.push({ label: 'Description', value: String(description) });
    }

    const category = node?.data?.category || catalogEntry?.category;
    if (category) {
      entries.push({ label: 'Category', value: String(category) });
    }

    if (node?.data?.catalogType) {
      entries.push({ label: 'Node type', value: String(node.data.catalogType) });
    }

    const inputs = node?.data?.inputs || catalogEntry?.inputs;
    if (inputs && Array.isArray(inputs) && inputs.length) {
      entries.push({ label: 'Inputs', value: inputs.join(', ') });
    }

    const outputs = node?.data?.outputs || catalogEntry?.outputs;
    if (outputs && Array.isArray(outputs) && outputs.length) {
      entries.push({ label: 'Outputs', value: outputs.join(', ') });
    }

    return entries;
  }, [node, catalogEntry]);

  const title = useMemo(() => {
    const raw = node?.data?.label ?? catalogEntry?.label ?? node?.id;
    return typeof raw === 'string' && raw.trim() ? raw : `Node ${node?.id ?? ''}`;
  }, [node, catalogEntry]);

  return { metadata, title };
};
