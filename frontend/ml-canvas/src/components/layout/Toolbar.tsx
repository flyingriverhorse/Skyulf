import React, { useState, useMemo } from 'react';
import type { Node, Edge } from '@xyflow/react';
import { Play, Save, Loader2, FolderOpen, History, Rocket, Wand2, HelpCircle, Merge, GitFork, X, CheckCircle2, XCircle } from 'lucide-react';
import { useGraphStore } from '../../core/store/useGraphStore';
import { useJobStore } from '../../core/store/useJobStore';
import { runPipelinePreview, savePipeline, fetchPipeline } from '../../core/api/client';
import { convertGraphToPipelineConfig } from '../../core/utils/pipelineConverter';
import { autoLayoutGraph } from '../../core/utils/autoLayout';
import { jobsApi } from '../../core/api/jobs';

const TRAINING_TYPES = new Set(['basic_training', 'advanced_tuning']);

export const Toolbar: React.FC = () => {
  const nodes = useGraphStore((state) => state.nodes);
  const edges = useGraphStore((state) => state.edges);
  const setExecutionResult = useGraphStore((state) => state.setExecutionResult);
  const setGraph = useGraphStore((state) => state.setGraph);
  
  const { toggleDrawer, setActiveParallelRun, startPolling } = useJobStore();
  
  const [isRunning, setIsRunning] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [isRunningAll, setIsRunningAll] = useState(false);
  const [showLegend, setShowLegend] = useState(false);

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
        const overlap = [...parentSets[i]!].some(p => parentSets[j]!.has(p));
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
      // Exclude Data Preview nodes from "Run All Experiments" -- preview is
      // an inspection-only sink and should never be queued as a training
      // experiment. Backend treats data_preview as a terminal in
      // partition_parallel_pipeline, so leaving it in would spawn a bogus
      // "preview" job alongside the real training runs.
      const previewIds = new Set(
        nodes.filter(n => n.data.definitionType === 'data_preview').map(n => n.id),
      );
      const filteredNodes = nodes.filter(n => !previewIds.has(n.id));
      const filteredEdges = edges.filter(
        e => !previewIds.has(e.source) && !previewIds.has(e.target),
      );
      const pipelineConfig = convertGraphToPipelineConfig(filteredNodes, filteredEdges);
      const response = await jobsApi.runPipeline({
        ...pipelineConfig,
        job_type: 'basic_training',
      });
      const count = response.job_ids?.length || 1;
      if (response.job_ids?.length > 1) {
        setActiveParallelRun({ jobIds: response.job_ids, startedAt: new Date().toISOString() });
        startPolling();
      }
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
      // Exclude Data Preview nodes from the toolbar's "Run Preview". Data
      // Preview is an inspection sink with its own "Run Preview" button
      // inside the node settings panel; including it here would queue a
      // redundant preview job alongside the pipeline preview.
      const previewIds = new Set(
        nodes.filter(n => n.data.definitionType === 'data_preview').map(n => n.id),
      );
      const filteredNodes = nodes.filter(n => !previewIds.has(n.id));
      const filteredEdges = edges.filter(
        e => !previewIds.has(e.source) && !previewIds.has(e.target),
      );
      const pipelineConfig = convertGraphToPipelineConfig(filteredNodes, filteredEdges);
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
        onClick={() => setShowLegend(v => !v)}
        title="Show node badge legend"
        className="flex items-center justify-center w-10 h-10 bg-background border rounded-md shadow-sm hover:bg-accent transition-colors"
      >
        <HelpCircle className="w-4 h-4" />
      </button>
      {showLegend && (
        <div className="absolute top-12 right-0 mt-2 w-96 p-4 bg-background border rounded-md shadow-lg text-sm max-h-[80vh] overflow-y-auto">
          <div className="flex items-center justify-between mb-3">
            <h3 className="font-semibold">Canvas Legend</h3>
            <button
              onClick={() => setShowLegend(false)}
              className="p-1 rounded hover:bg-accent"
              aria-label="Close legend"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          </div>

          <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">Node Badges</div>
          <ul className="space-y-3 mb-4">
            <li className="flex items-start gap-3">
              <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-blue-500/15 text-blue-400 text-[10px] font-semibold shrink-0 mt-0.5">
                <Merge size={10} />2
              </span>
              <div>
                <div className="font-medium">Safe merge</div>
                <div className="text-xs text-muted-foreground">Multiple inputs combined cleanly (no overlapping columns).</div>
              </div>
            </li>
            <li className="flex items-start gap-3">
              <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-amber-500/20 text-amber-600 dark:text-amber-400 ring-1 ring-amber-500/40 text-[10px] font-semibold shrink-0 mt-0.5">
                <Merge size={10} />2
              </span>
              <div>
                <div className="font-medium">Risky merge</div>
                <div className="text-xs text-muted-foreground">Inputs share columns &mdash; one branch wins (overwrite). Check Results banner; tweak strategy in properties.</div>
              </div>
            </li>
            <li className="flex items-start gap-3">
              <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-amber-500/15 text-amber-500 text-[10px] font-semibold shrink-0 mt-0.5">
                <GitFork size={10} />2
              </span>
              <div>
                <div className="font-medium">Parallel experiments</div>
                <div className="text-xs text-muted-foreground">Training/tuning node runs each upstream branch as a separate experiment (no merge).</div>
              </div>
            </li>
            <li className="flex items-start gap-3">
              <span className="inline-flex items-center justify-center w-[22px] h-[22px] rounded-full bg-green-50 text-green-700 border border-green-200 dark:bg-green-900/30 dark:text-green-400 dark:border-green-900 shrink-0 mt-0.5">
                <CheckCircle2 size={10} />
              </span>
              <div>
                <div className="font-medium">Success</div>
                <div className="text-xs text-muted-foreground">Node ran successfully in the last preview / run.</div>
              </div>
            </li>
            <li className="flex items-start gap-3">
              <span className="inline-flex items-center justify-center w-[22px] h-[22px] rounded-full bg-red-50 text-red-700 border border-red-200 dark:bg-red-900/30 dark:text-red-400 dark:border-red-900 shrink-0 mt-0.5">
                <XCircle size={10} />
              </span>
              <div>
                <div className="font-medium">Failed</div>
                <div className="text-xs text-muted-foreground">Node errored. Click it and open the Results panel for the traceback.</div>
              </div>
            </li>
          </ul>

          <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">Edges</div>
          <ul className="space-y-3">
            <li className="flex items-start gap-3">
              <svg width="44" height="12" className="shrink-0 mt-1">
                <line x1="0" y1="6" x2="44" y2="6" stroke="#6366f1" strokeWidth="2" strokeDasharray="8 6" />
              </svg>
              <div>
                <div className="font-medium">Standard edge</div>
                <div className="text-xs text-muted-foreground">Animated dashed indigo line. Default flow from source to target.</div>
              </div>
            </li>
            <li className="flex items-start gap-3">
              <svg width="44" height="12" className="shrink-0 mt-1">
                <line x1="0" y1="6" x2="44" y2="6" stroke="hsl(0, 80%, 65%)" strokeWidth="2" strokeDasharray="8 6" />
              </svg>
              <div>
                <div className="font-medium">Branch-colored edge</div>
                <div className="text-xs text-muted-foreground">Dashed line in a per-branch HSL color (auto-generated, one hue per training/tuning terminal). Appears once 2+ training nodes form parallel branches.</div>
              </div>
            </li>
            <li className="flex items-start gap-3">
              <svg width="44" height="12" className="shrink-0 mt-1">
                <line x1="0" y1="6" x2="44" y2="6" stroke="hsl(0, 80%, 65%)" strokeWidth="2" strokeDasharray="6 4" opacity="0.7" />
              </svg>
              <div>
                <div className="font-medium">Shared branch edge</div>
                <div className="text-xs text-muted-foreground">Same per-branch color but tighter dashes and faded &mdash; this upstream edge feeds more than one parallel experiment.</div>
              </div>
            </li>
            <li className="flex items-start gap-3">
              <svg width="44" height="12" className="shrink-0 mt-1">
                <line x1="0" y1="6" x2="44" y2="6" stroke="#f59e0b" strokeWidth="4" strokeDasharray="8 6" />
              </svg>
              <div>
                <div className="font-medium">Winning merge edge</div>
                <div className="text-xs text-muted-foreground">After a preview run, the branch whose values survived an overlapping-column merge is rendered thicker in amber with a &quot;WINS MERGE&quot; label.</div>
              </div>
            </li>
          </ul>
        </div>
      )}
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
      <button
        onClick={() => {
          // Tidy up multi-branch canvases via dagre topological layout.
          const { nodes: laidOut, edges: keptEdges } = autoLayoutGraph(nodes, edges);
          setGraph(laidOut, keptEdges);
        }}
        disabled={isRunning || nodes.length === 0}
        title="Auto-arrange nodes left-to-right by data flow"
        className="flex items-center gap-2 px-4 py-2 bg-background border rounded-md shadow-sm hover:bg-accent transition-colors disabled:opacity-50"
      >
        <Wand2 className="w-4 h-4" />
        <span className="text-sm font-medium">Tidy</span>
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
