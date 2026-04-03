import React, { useState, useMemo } from 'react';
import type { Node, Edge } from '@xyflow/react';
import { Play, Save, Loader2, FolderOpen, History, Rocket } from 'lucide-react';
import { useGraphStore } from '../../core/store/useGraphStore';
import { useJobStore } from '../../core/store/useJobStore';
import { runPipelinePreview, savePipeline, fetchPipeline } from '../../core/api/client';
import { convertGraphToPipelineConfig } from '../../core/utils/pipelineConverter';
import { jobsApi } from '../../core/api/jobs';

const TRAINING_TYPES = new Set(['basic_training', 'advanced_tuning']);

export const Toolbar: React.FC = () => {
  const nodes = useGraphStore((state) => state.nodes);
  const edges = useGraphStore((state) => state.edges);
  const setExecutionResult = useGraphStore((state) => state.setExecutionResult);
  const setGraph = useGraphStore((state) => state.setGraph);
  
  const { toggleDrawer } = useJobStore();
  
  const [isRunning, setIsRunning] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [isRunningAll, setIsRunningAll] = useState(false);

  const hasMultipleBranches = useMemo(() => {
    const trainingNodes = nodes.filter(
      n => TRAINING_TYPES.has(n.data.definitionType as string) && edges.some(e => e.target === n.id)
    );
    if (trainingNodes.length < 2) return false;

    // Check that at least two training nodes are on separate branches
    // by comparing their immediate parent sets
    const parentSets = trainingNodes.map(tn =>
      new Set(edges.filter(e => e.target === tn.id).map(e => e.source))
    );
    // If any two training nodes have completely different parents, they're separate
    for (let i = 0; i < parentSets.length; i++) {
      for (let j = i + 1; j < parentSets.length; j++) {
        const overlap = [...parentSets[i]].some(p => parentSets[j].has(p));
        if (!overlap) return true;
      }
    }
    // Even with shared parents, 2+ connected training nodes = separate branches
    return trainingNodes.length >= 2;
  }, [nodes, edges]);

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
        description: 'Saved from Canvas',
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
      if (pipeline) {
        const graphNodes: Node[] = Array.isArray(pipeline.graph.nodes) ? (pipeline.graph.nodes as Node[]) : [];
        const graphEdges: Edge[] = Array.isArray(pipeline.graph.edges) ? (pipeline.graph.edges as Edge[]) : [];
        setGraph(graphNodes, graphEdges);
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

  const handleRunAll = async () => {
    const datasetNode = nodes.find(n => n.data.definitionType === 'dataset_node');
    const datasetId = datasetNode?.data.datasetId as string;

    if (!datasetId) {
      alert('No dataset node found!');
      return;
    }

    setIsRunningAll(true);
    try {
      const pipelineConfig = convertGraphToPipelineConfig(nodes, edges);
      const response = await jobsApi.runPipeline({
        ...pipelineConfig,
        job_type: 'basic_training',
      });
      const count = response.job_ids?.length || 1;
      alert(`${count} experiment${count > 1 ? 's' : ''} queued!`);
      toggleDrawer();
    } catch (error) {
      console.error('Run All failed:', error);
      alert('Failed to run experiments. Check console for details.');
    } finally {
      setIsRunningAll(false);
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
      // Use API
      const pipelineConfig = convertGraphToPipelineConfig(nodes, edges);
      const result = await runPipelinePreview(pipelineConfig);
      
      setExecutionResult(result);
    } catch (error) {
      console.error('Pipeline failed:', error);
      alert('Pipeline execution failed. Check console for details.');
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <div className="absolute top-4 right-4 z-10 flex gap-2">
      <button 
        onClick={() => toggleDrawer()}
        className="flex items-center gap-2 px-4 py-2 bg-background border rounded-md shadow-sm hover:bg-accent transition-colors"
      >
        <History className="w-4 h-4" />
        <span className="text-sm font-medium">Jobs</span>
      </button>
      <button 
        onClick={() => { void handleLoad(); }}
        disabled={isLoading || isRunning}
        className="flex items-center gap-2 px-4 py-2 bg-background border rounded-md shadow-sm hover:bg-accent transition-colors disabled:opacity-50"
      >
        {isLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : <FolderOpen className="w-4 h-4" />}
        <span className="text-sm font-medium">{isLoading ? 'Loading...' : 'Load'}</span>
      </button>
      <button 
        onClick={() => { void handleSave(); }}
        disabled={isSaving || isRunning}
        className="flex items-center gap-2 px-4 py-2 bg-background border rounded-md shadow-sm hover:bg-accent transition-colors disabled:opacity-50"
      >
        {isSaving ? <Loader2 className="w-4 h-4 animate-spin" /> : <Save className="w-4 h-4" />}
        <span className="text-sm font-medium">{isSaving ? 'Saving...' : 'Save'}</span>
      </button>
      {hasMultipleBranches && (
        <button
          onClick={() => { void handleRunAll(); }}
          disabled={isRunningAll || isRunning}
          className="flex items-center gap-2 px-4 py-2 text-white bg-amber-600 rounded-md shadow-sm hover:bg-amber-700 transition-colors disabled:opacity-50"
        >
          {isRunningAll ? <Loader2 className="w-4 h-4 animate-spin" /> : <Rocket className="w-4 h-4" />}
          <span className="text-sm font-medium">{isRunningAll ? 'Queuing...' : 'Run All Experiments'}</span>
        </button>
      )}
      <button 
        onClick={() => { void handleRun(); }}
        disabled={isRunning}
        className="flex items-center gap-2 px-4 py-2 text-white rounded-md shadow-sm transition-all disabled:opacity-50"
        style={{ background: 'var(--main-gradient)' }}
      >
        {isRunning ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
        <span className="text-sm font-medium">{isRunning ? 'Running...' : 'Run Preview'}</span>
      </button>
    </div>
  );
};
