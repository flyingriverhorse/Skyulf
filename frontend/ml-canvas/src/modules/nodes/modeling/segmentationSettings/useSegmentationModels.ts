import { useEffect, useRef, useState } from 'react';
import { jobsApi } from '../../../../core/api/jobs';
import { type RegistryItem, registryApi } from '../../../../core/api/registry';
import type { HyperparameterDef } from '../components/types';
import type { SegmentationConfig } from '../SegmentationSettings';

/** Load the clustering catalog and model defaults for this settings lifetime. */
export function useSegmentationModels(config: SegmentationConfig, onChange: (config: SegmentationConfig) => void, nodeId?: string) {
  const [hyperparameters, setHyperparameters] = useState<HyperparameterDef[]>([]);
  const [isLoadingDefs, setIsLoadingDefs] = useState(false);
  const [availableModels, setAvailableModels] = useState<RegistryItem[]>([]);
  const [isLoadingModels, setIsLoadingModels] = useState(false);
  const selectedModelItem = availableModels.find(m => m.id === config.model_type);
  const requiresScaling = selectedModelItem?.tags?.includes('requires_scaling');
  const latest = useRef({ config, onChange });
  latest.current = { config, onChange };
  const modelType = config.model_type;

  // Fetch clustering-tagged algorithms only.
  useEffect(() => {
    let cancelled = false;
    const fetchModels = async () => {
      setIsLoadingModels(true);
      try {
        const nodes = await registryApi.getAllNodes();
        if (cancelled) return;
        const models = nodes.filter(n => {
          const isModeling = n.category === 'Model' || n.category === 'Modeling';
          const isClustering = n.tags?.includes('clustering') ?? false;
          return isModeling && isClustering;
        });
        setAvailableModels(models);
      } catch (error) {
        if (cancelled) return;
        console.error('Failed to fetch clustering models:', error);
        setAvailableModels([
          { id: 'kmeans', name: 'K-Means', category: 'Modeling', description: '', params: {}, tags: ['clustering'] },
        ]);
      } finally {
        if (!cancelled) setIsLoadingModels(false);
      }
    };
    fetchModels();
    return () => { cancelled = true; };
  }, []);

  // Fetch hyperparameter definitions and always seed defaults — Segmentation
  // has no "Customize" toggle, so params must be ready to show/edit as soon
  // as the model type is known.
  useEffect(() => {
    let cancelled = false;
    setHyperparameters([]);
    setIsLoadingDefs(Boolean(modelType));
    if (modelType) {
      jobsApi.getHyperparameters(modelType)
        .then((defs) => {
          if (cancelled) return;
          const definitions = defs as HyperparameterDef[];
          setHyperparameters(definitions);
          const current = latest.current;
          if (Object.keys(current.config.hyperparameters).length === 0) {
            const defaults: Record<string, unknown> = {};
            definitions.forEach(p => {
              defaults[p.name] = p.default;
            });
            current.onChange({ ...current.config, hyperparameters: defaults });
          }
        })
        .catch(error => { if (!cancelled) console.error(error); })
        .finally(() => { if (!cancelled) setIsLoadingDefs(false); });
    }
    return () => { cancelled = true; };
  }, [modelType, nodeId]);

  const changeModel = (modelType: string) => {
    onChange({ ...config, model_type: modelType, hyperparameters: {} });
  };

  return { hyperparameters, isLoadingDefs, availableModels, isLoadingModels, selectedModelItem, requiresScaling, changeModel };
}
