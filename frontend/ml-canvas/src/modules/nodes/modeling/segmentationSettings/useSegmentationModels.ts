import { useEffect, useRef, useState } from 'react';
import { jobsApi } from '../../../../core/api/jobs';
import { type RegistryItem, registryApi } from '../../../../core/api/registry';
import type { HyperparameterDef } from '../components/types';
import type { SegmentationConfig } from '../SegmentationSettings';

/** Load the clustering catalog and model defaults for this settings lifetime. */
export function useSegmentationModels(config: SegmentationConfig, onChange: (config: SegmentationConfig) => void) {
  const [hyperparameters, setHyperparameters] = useState<HyperparameterDef[]>([]);
  const [isLoadingDefs, setIsLoadingDefs] = useState(false);
  const [availableModels, setAvailableModels] = useState<RegistryItem[]>([]);
  const [isLoadingModels, setIsLoadingModels] = useState(false);
  const selectedModelItem = availableModels.find(m => m.id === config.model_type);
  const requiresScaling = selectedModelItem?.tags?.includes('requires_scaling');

  // Fetch clustering-tagged algorithms only.
  useEffect(() => {
    const fetchModels = async () => {
      setIsLoadingModels(true);
      try {
        const nodes = await registryApi.getAllNodes();
        const models = nodes.filter(n => {
          const isModeling = n.category === 'Model' || n.category === 'Modeling';
          const isClustering = n.tags?.includes('clustering') ?? false;
          return isModeling && isClustering;
        });
        setAvailableModels(models);
      } catch (error) {
        console.error('Failed to fetch clustering models:', error);
        setAvailableModels([
          { id: 'kmeans', name: 'K-Means', category: 'Modeling', description: '', params: {}, tags: ['clustering'] },
        ]);
      } finally {
        setIsLoadingModels(false);
      }
    };
    fetchModels();
  }, []);

  const keepCustomizationOpen = useRef(false);

  // Fetch hyperparameter definitions and always seed defaults — Segmentation
  // has no "Customize" toggle, so params must be ready to show/edit as soon
  // as the model type is known.
  useEffect(() => {
    if (config.model_type) {
      setIsLoadingDefs(true);
      jobsApi.getHyperparameters(config.model_type)
        .then((defs) => {
          const definitions = defs as HyperparameterDef[];
          setHyperparameters(definitions);

          if (keepCustomizationOpen.current || Object.keys(config.hyperparameters).length === 0) {
            const defaults: Record<string, unknown> = {};
            definitions.forEach(p => {
              defaults[p.name] = p.default;
            });
            onChange({ ...config, hyperparameters: defaults });
            keepCustomizationOpen.current = false;
          }
        })
        .catch(console.error)
        .finally(() => { setIsLoadingDefs(false); });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [config.model_type]);

  const changeModel = (modelType: string) => {
    if (Object.keys(config.hyperparameters).length > 0) {
      keepCustomizationOpen.current = true;
    }
    onChange({ ...config, model_type: modelType, hyperparameters: {} });
  };

  return { hyperparameters, isLoadingDefs, availableModels, isLoadingModels, selectedModelItem, requiresScaling, changeModel };
}
