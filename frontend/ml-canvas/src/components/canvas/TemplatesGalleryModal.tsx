import { useState } from 'react';
import { ArrowRight } from 'lucide-react';
import { ModalShell } from '../shared/ModalShell';
import { PIPELINE_TEMPLATES, buildGraphFromTemplate, type PipelineTemplate, type TemplateBinding } from '../../core/templates/pipelineTemplates';
import { templateStories } from '../../core/templates/templateStories';
import { useGraphStore } from '../../core/store/useGraphStore';
import { useConfirm } from '../shared';
import { toast } from '../../core/toast';
import { TemplateSetup } from './templates/TemplateSetup';

interface Props { isOpen: boolean; onClose: () => void }

/** Unmount setup on close so a new visit never inherits stale selections. */
export function TemplatesGalleryModal({ isOpen, onClose }: Props) {
  return <ModalShell isOpen={isOpen} onClose={onClose} title="Start from a template" size="5xl">
    {isOpen && <TemplateGallery onClose={onClose} />}
  </ModalShell>;
}

/** Present outcomes before asking users to configure a workflow. */
function TemplateGallery({ onClose }: { onClose: () => void }) {
  const [selected, setSelected] = useState<PipelineTemplate | null>(null);
  const [opening, setOpening] = useState(false);
  const confirm = useConfirm();

  async function openTemplate(template: PipelineTemplate, binding: TemplateBinding) {
    setOpening(true);
    try {
      if (useGraphStore.getState().nodes.length > 0) {
        const accepted = await confirm({ title: 'Replace current canvas?',
          message: `Loading "${template.name}" will replace your current canvas. Continue?`,
          confirmLabel: 'Load template', variant: 'danger' });
        if (!accepted) return;
      }
      const { nodes, edges } = buildGraphFromTemplate(template, binding);
      if (nodes.length !== template.nodes.length) {
        toast.error('Template could not be loaded', 'A required node type is unavailable.');
        return;
      }
      useGraphStore.getState().setGraph(nodes, edges);
      toast.success(`Loaded "${template.name}"`, 'Review the node settings, then Run All.');
      onClose();
    } finally { setOpening(false); }
  }

  if (selected) return <TemplateSetup template={selected} opening={opening}
    onBack={() => setSelected(null)} onOpen={binding => { void openTemplate(selected, binding); }} />;

  return <div className="space-y-5 p-4 sm:p-6">
    <div className="max-w-2xl space-y-2">
      <p className="text-xs font-semibold uppercase tracking-[0.18em] text-primary">Your next experiment</p>
      <h3 className="text-2xl font-semibold tracking-tight">What would you like to discover?</h3>
      <p className="text-sm text-muted-foreground">Choose a starting point. Connect your data, choose what to predict, and make the workflow your own.</p>
    </div>
    <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
      {PIPELINE_TEMPLATES.map((template, index) => {
        const story = templateStories[template.id]!;
        const Icon = template.icon;
        return <button key={template.id} type="button" data-testid={`template-card-${template.id}`}
          onClick={() => setSelected(template)}
          className={`group flex flex-col gap-3 rounded-xl border p-4 text-left transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary ${index === 0 ? 'border-primary/35 bg-primary/5 md:col-span-2' : 'border-border bg-card hover:border-primary/40 hover:bg-accent/30'}`}>
          <div className="flex w-full items-center justify-between gap-3">
            <span className="rounded-lg bg-primary/10 p-2 text-primary"><Icon className="h-5 w-5" aria-hidden="true" /></span>
            <span className="text-xs text-muted-foreground">{index === 0 ? 'A good place to start' : template.name}</span>
          </div>
          <div><h4 className="text-lg font-semibold tracking-tight">{story.title}</h4>
            <p className="mt-1 text-sm text-muted-foreground">{story.example}</p></div>
          <div className="flex w-full items-center justify-between gap-3 border-t border-border/60 pt-3 text-xs">
            <span className="text-muted-foreground">{template.nodes.length} connected steps · {template.category}</span>
            <span className="flex items-center gap-2 font-medium text-primary">Set up <ArrowRight className="h-4 w-4" aria-hidden="true" /></span>
          </div>
        </button>;
      })}
    </div>
    <p className="text-xs text-muted-foreground">All five workflows are editable. Training starts only when you run the pipeline.</p>
  </div>;
}
