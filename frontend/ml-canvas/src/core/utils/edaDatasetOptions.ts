import type { Dataset } from '../types/api';

interface EdaDatasetOption {
  value: string;
  label: string;
}

function getDatasetLabel(dataset: Dataset): string {
  const name = dataset.name.trim();
  return name || `Dataset ${dataset.id}`;
}

function getDatasetSuffix(dataset: Dataset): string {
  const id = String(dataset.id);
  return id.length <= 6 ? id : id.slice(-6);
}

/**
 * Build EDA dataset-select options and disambiguate duplicate names with a stable suffix.
 */
export function buildEdaDatasetOptions(datasets: readonly Dataset[]): EdaDatasetOption[] {
  const labels = datasets.map((dataset) => getDatasetLabel(dataset));
  const counts = new Map<string, number>();

  for (const label of labels) {
    counts.set(label, (counts.get(label) ?? 0) + 1);
  }

  return datasets.map((dataset, index) => {
    const label = labels[index]!;
    if ((counts.get(label) ?? 0) > 1) {
      return {
        value: dataset.id,
        label: `${label} (${getDatasetSuffix(dataset)})`,
      };
    }

    return {
      value: dataset.id,
      label,
    };
  });
}
