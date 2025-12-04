import React, { useState } from 'react';
import { Play, Save, Loader2, FolderOpen, BrainCircuit } from 'lucide-react';
import { useGraphStore } from '../../core/store/useGraphStore';
import { runPipelinePreview, savePipeline, fetchPipeline, submitTrainingJob } from '../../core/api/client';

export const Toolbar: React.FC = () => {
  const nodes = useGraphStore((state) => state.nodes);
  const edges = useGraphStore((state) => state.edges);
  const setExecutionResult = useGraphStore((state) => state.setExecutionResult);
  const setGraph = useGraphStore((state) => state.setGraph);
  
  const [isRunning, setIsRunning] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [isTraining, setIsTraining] = useState(false);

  const getPipelinePayload = () => ({
    nodes: nodes.map(n => ({
      id: n.id,
      type: n.type,
      position: n.position,
      data: {
        ...n.data,
        catalogType: n.data.catalogType || n.data.definitionType
      }
    })),
    edges: edges.map(e => ({
      id: e.id,
      source: e.source,
      target: e.target,
      sourceHandle: e.sourceHandle,
      targetHandle: e.targetHandle,
      type: e.type,
      data: e.data
    }))
  });

  const handleSave = async () => {
    const datasetNode = nodes.find(n => n.data.definitionType === 'dataset_node');
    const datasetId = datasetNode?.data.datasetId as string;

    if (!datasetId) {
      alert('No dataset node found! Cannot save pipeline without a dataset context.');
      return;
    }

    setIsSaving(true);
    try {
      await savePipeline(datasetId, {
        name: 'My Pipeline', // TODO: Add UI for naming
        description: 'Saved from V2 Canvas',
        graph: getPipelinePayload()
      });
      alert('Pipeline saved successfully!');
    } catch (error) {
      console.error('Save failed:', error);
      alert('Failed to save pipeline.');
    } finally {
      setIsSaving(false);
    }
  };

  const handleLoad = async () => {
    const datasetNode = nodes.find(n => n.data.definitionType === 'dataset_node');
    const datasetId = datasetNode?.data.datasetId as string;

    if (!datasetId) {
      alert('No dataset node found! Cannot load pipeline without a dataset context.');
      return;
    }

    if (!confirm('Loading a pipeline will overwrite your current work. Continue?')) {
      return;
    }

    setIsLoading(true);
    try {
      const pipeline = await fetchPipeline(datasetId);
      if (pipeline && pipeline.graph) {
        setGraph(pipeline.graph.nodes, pipeline.graph.edges);
        alert('Pipeline loaded successfully!');
      } else {
        alert('No saved pipeline found for this dataset.');
      }
    } catch (error) {
      console.error('Load failed:', error);
      alert('Failed to load pipeline.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleRun = async () => {
    // Find dataset node
    const datasetNode = nodes.find(n => n.data.definitionType === 'dataset_node');
    const datasetId = datasetNode?.data.datasetId as string;

    if (!datasetId) {
      alert('No dataset node found!');
      return;
    }

    setIsRunning(true);
    setExecutionResult(null); // Clear previous results
    
    try {
      const result = await runPipelinePreview({
        dataset_source_id: datasetId,
        graph: getPipelinePayload()
      });
      
      setExecutionResult(result);
    } catch (error) {
      console.error('Pipeline failed:', error);
      alert('Pipeline execution failed.');
    } finally {
      setIsRunning(false);
    }
  };

  const handleTrain = async () => {
    const datasetNode = nodes.find(n => n.data.definitionType === 'dataset_node');
    const datasetId = datasetNode?.data.datasetId as string;
    const trainingNode = nodes.find(n => n.data.definitionType === 'train_model_draft');

    if (!datasetId) {
      alert('No dataset node found!');
      return;
    }

    if (!trainingNode) {
      alert('No training node found! Add a "Model Training" node to train a model.');
      return;
    }

    setIsTraining(true);
    try {
      // 1. Save the pipeline first (required for training job)
      const savedPipeline = await savePipeline(datasetId, {
        name: 'Training Pipeline',
        description: 'Auto-saved before training',
        graph: getPipelinePayload()
      });

      // 2. Submit training job
      await submitTrainingJob({
        dataset_source_id: datasetId,
        pipeline_id: savedPipeline.id.toString(),
        node_id: trainingNode.id,
        model_types: [trainingNode.data.modelType as string],
        hyperparameters: trainingNode.data.hyperparameters,
        graph: getPipelinePayload(),
        run_training: true
      });

      alert('Training job submitted successfully! Check the jobs dashboard.');
    } catch (error) {
      console.error('Training failed:', error);
      alert('Failed to submit training job.');
    } finally {
      setIsTraining(false);
    }
  };

  return (
    <div className="absolute top-4 right-4 z-10 flex gap-2">
      <button 
        onClick={handleLoad}
        disabled={isLoading || isRunning || isTraining}
        className="flex items-center gap-2 px-4 py-2 bg-background border rounded-md shadow-sm hover:bg-accent transition-colors disabled:opacity-50"
      >
        {isLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : <FolderOpen className="w-4 h-4" />}
        <span className="text-sm font-medium">{isLoading ? 'Loading...' : 'Load'}</span>
      </button>
      <button 
        onClick={handleSave}
        disabled={isSaving || isRunning || isTraining}
        className="flex items-center gap-2 px-4 py-2 bg-background border rounded-md shadow-sm hover:bg-accent transition-colors disabled:opacity-50"
      >
        {isSaving ? <Loader2 className="w-4 h-4 animate-spin" /> : <Save className="w-4 h-4" />}
        <span className="text-sm font-medium">{isSaving ? 'Saving...' : 'Save'}</span>
      </button>
      <button 
        onClick={handleRun}
        disabled={isRunning || isTraining}
        className="flex items-center gap-2 px-4 py-2 bg-primary text-primary-foreground rounded-md shadow-sm hover:bg-primary/90 transition-colors disabled:opacity-50"
      >
        {isRunning ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
        <span className="text-sm font-medium">{isRunning ? 'Running...' : 'Run Preview'}</span>
      </button>
      <button 
        onClick={handleTrain}
        disabled={isRunning || isTraining}
        className="flex items-center gap-2 px-4 py-2 bg-green-600 text-white rounded-md shadow-sm hover:bg-green-700 transition-colors disabled:opacity-50"
      >
        {isTraining ? <Loader2 className="w-4 h-4 animate-spin" /> : <BrainCircuit className="w-4 h-4" />}
        <span className="text-sm font-medium">{isTraining ? 'Submitting...' : 'Train Model'}</span>
      </button>
    </div>
  );
};
