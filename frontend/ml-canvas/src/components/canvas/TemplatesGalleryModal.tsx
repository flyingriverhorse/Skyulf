import React from 'react';
import { ModalShell } from '../shared/ModalShell';
import { PIPELINE_TEMPLATES, buildGraphFromTemplate, type PipelineTemplate } from '../../core/templates/pipelineTemplates';
import { useGraphStore } from '../../core/store/useGraphStore';
import { toast } from '../../core/toast';

interface Props {
  isOpen: boolean;
  onClose: () => void;
}

/**
 * Templates Gallery (L3). Lets a user materialise one of a few curated
 * starter pipelines onto the canvas with one click — replaces the
 * "drop 8 nodes correctly" cold start. Loaded graph still needs the
 * user to bind their dataset and pick a target column before running.
 */
export const TemplatesGalleryModal: React.FC<Props> = ({ isOpen, onClose }) => {
  const setGraph = useGraphStore((s) => s.setGraph);
  const currentNodeCount = useGraphStore((s) => s.nodes.length);

  const handlePick = (template: PipelineTemplate): void => {
    if (currentNodeCount > 0) {
      // Don't silently nuke in-progress work. The empty-state entry
      // path can never trigger this, but the Toolbar entry can.
      const ok = window.confirm(
        `Loading "${template.name}" will replace your current canvas. Continue?`,
      );
      if (!ok) return;
    }
    const { nodes, edges } = buildGraphFromTemplate(template);
    if (nodes.length === 0) {
      toast.error('Template could not be loaded', 'No matching node types in registry.');
      return;
    }
    setGraph(nodes, edges);
    toast.success(`Loaded "${template.name}"`, 'Bind your dataset, then Run All.');
    onClose();
  };

  return (
    <ModalShell
      isOpen={isOpen}
      onClose={onClose}
      title="Start from a template"
      size="4xl"
    >
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3 p-1">
        {PIPELINE_TEMPLATES.map((tpl) => {
          const Icon = tpl.icon;
          return (
            <button
              key={tpl.id}
              type="button"
              onClick={() => handlePick(tpl)}
              data-testid={`template-card-${tpl.id}`}
              className="text-left rounded-lg border border-border bg-card hover:bg-accent/40 hover:border-primary/40 transition-colors p-4 flex flex-col gap-2 focus:outline-none focus:ring-2 focus:ring-primary"
            >
              <div className="flex items-center gap-2">
                <span className="p-1.5 rounded bg-primary/10 text-primary">
                  <Icon className="w-4 h-4" />
                </span>
                <span className="font-medium text-sm">{tpl.name}</span>
              </div>
              <p className="text-xs text-muted-foreground leading-relaxed">
                {tpl.description}
              </p>
              <div className="text-[10px] uppercase tracking-wide text-muted-foreground/70 mt-auto pt-1">
                {tpl.nodes.length} nodes · {tpl.category}
              </div>
            </button>
          );
        })}
      </div>
    </ModalShell>
  );
};
