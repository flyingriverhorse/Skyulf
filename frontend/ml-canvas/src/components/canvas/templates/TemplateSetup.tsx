import { useId, useState } from 'react';
import { ArrowLeft, ArrowRight, Upload } from 'lucide-react';
import { useUsableDatasets } from '../../../core/hooks/useDatasets';
import { useDatasetSchema } from '../../../core/hooks/useDatasetSchema';
import type { PipelineTemplate, TemplateBinding } from '../../../core/templates/pipelineTemplates';
import { templateStories } from '../../../core/templates/templateStories';
import { FileUpload } from '../../../modules/nodes/data/FileUpload';
import { TemplateColumnFields } from './TemplateColumnFields';

interface Props {
  template: PipelineTemplate; opening: boolean; onBack: () => void;
  onOpen: (binding: TemplateBinding) => void;
}

/** Require current schema and explicit supervised/text choices before opening. */
function canOpen(category: PipelineTemplate['category'], schema: ReturnType<typeof useDatasetSchema>, sourceId: string, target: string, text: string) {
  if (!sourceId || schema.isError || schema.isLoading || !schema.data) return false;
  const names = Object.values(schema.data.columns).map(column => column.name);
  if (category !== 'clustering' && !names.includes(target)) return false;
  if (category === 'text') return text !== target && names.includes(text);
  return true;
}

/** Bind a starter to real data before materialising the canvas. */
export function TemplateSetup({ template, opening, onBack, onOpen }: Props) {
  const id = useId();
  const [source, setSource] = useState({ datasetId: '', datasetName: '' });
  const [target, setTarget] = useState('');
  const [text, setText] = useState('');
  const [uploading, setUploading] = useState(false);
  const datasets = useUsableDatasets();
  const schema = useDatasetSchema(source.datasetId);
  const story = templateStories[template.id]!;
  const ready = canOpen(template.category, schema, source.datasetId, target, text);

  function selectSource(datasetId: string, datasetName: string) {
    setSource({ datasetId, datasetName });
    setTarget(''); setText(''); setUploading(false);
  }

  return <div className="space-y-5 p-4 sm:p-6">
    <button type="button" onClick={onBack} disabled={opening} className="flex items-center gap-2 rounded text-sm text-muted-foreground hover:text-foreground focus-visible:ring-2 focus-visible:ring-primary">
      <ArrowLeft className="h-4 w-4" aria-hidden="true" /> All templates
    </button>
    <div className="grid gap-6 lg:grid-cols-[1fr_1.1fr]">
      <section className="space-y-5 rounded-xl border border-primary/20 bg-primary/5 p-5">
        <div><p className="text-xs font-medium uppercase tracking-widest text-primary">{template.name}</p>
          <h3 className="mt-2 text-2xl font-semibold tracking-tight">{story.title}</h3>
          <p className="mt-2 text-sm text-muted-foreground">{story.example}</p></div>
        <ol aria-label="Workflow preview" className="space-y-3">
          {story.steps.map((step, index) => <li key={step} className="flex items-center gap-3 text-sm">
            <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full border border-primary/25 text-xs font-semibold text-primary">{index + 1}</span>{step}
          </li>)}
        </ol>
        <div className="border-t border-primary/15 pt-4"><h4 className="text-sm font-semibold">What you will get</h4>
          <p className="mt-1 text-sm text-muted-foreground">{story.outcome}</p></div>
      </section>
      <section className="space-y-4">
        <div><h3 className="text-lg font-semibold">Connect your data</h3><p className="mt-1 text-sm text-muted-foreground">{story.needs}</p></div>
        {uploading ? <FileUpload onUploadComplete={selectSource} onCancel={() => setUploading(false)} /> : <>
          <div className="space-y-2">
            <label htmlFor={`${id}-dataset`} className="block text-sm font-medium">Dataset</label>
            <select id={`${id}-dataset`} value={source.datasetId} disabled={datasets.isLoading || opening}
              className="w-full rounded-md border bg-background p-2.5 text-sm"
              onChange={event => {
                const dataset = datasets.data?.find(item => String(item.id) === event.target.value);
                selectSource(event.target.value, dataset?.name ?? '');
              }}>
              <option value="">{datasets.isLoading ? 'Loading datasets…' : 'Choose a dataset'}</option>
              {datasets.data?.map(dataset => <option key={dataset.id} value={String(dataset.id)}>{dataset.name}</option>)}
              {source.datasetId && !datasets.data?.some(dataset => String(dataset.id) === source.datasetId) && <option value={source.datasetId}>{source.datasetName}</option>}
            </select>
            <DatasetListStatus query={datasets} />
            <button type="button" disabled={opening} onClick={() => setUploading(true)} className="flex items-center gap-2 rounded text-sm font-medium text-primary focus-visible:ring-2 focus-visible:ring-primary">
              <Upload className="h-4 w-4" aria-hidden="true" /> Upload a dataset
            </button>
          </div>
          {source.datasetId && <TemplateColumnFields id={id} category={template.category} schema={schema}
            target={target} text={text} disabled={opening} onTarget={value => { setTarget(value); if (value === text) setText(''); }} onText={setText} />}
          <div className="border-t pt-4">
            <button type="button" disabled={!ready || opening} onClick={() => onOpen({ ...source, targetColumn: target, textColumn: text })}
              className="flex w-full items-center justify-center gap-2 rounded-lg bg-primary px-4 py-3 text-sm font-semibold text-primary-foreground hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-40">
              {opening ? 'Opening…' : 'Open in Canvas'} <ArrowRight className="h-4 w-4" aria-hidden="true" />
            </button>
            <p className="mt-2 text-xs text-muted-foreground">Review feature selection and model settings in Canvas before running.</p>
          </div>
        </>}
      </section>
    </div>
  </div>;
}

/** Keep loading, empty and failed dataset lists distinct. */
function DatasetListStatus({ query }: { query: ReturnType<typeof useUsableDatasets> }) {
  if (query.isError) return <p role="alert" className="text-sm text-destructive">Could not load datasets. <button type="button" className="underline" onClick={() => { void query.refetch(); }}>Retry</button></p>;
  if (query.isLoading) return null;
  if (!query.data?.length) return <p className="text-sm text-muted-foreground">Upload your first dataset to start this workflow.</p>;
  return null;
}
