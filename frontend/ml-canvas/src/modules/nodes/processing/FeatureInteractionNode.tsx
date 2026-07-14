import { NodeDefinition } from '../../../core/types/nodes';
import { Shuffle, ChevronDown, ChevronRight, Info } from 'lucide-react';
import { useUpstreamData } from '../../../core/hooks/useUpstreamData';
import { useDatasetSchema } from '../../../core/hooks/useDatasetSchema';
import { useUpstreamDroppedColumns } from '../../../core/hooks/useUpstreamDroppedColumns';
import { clickableProps } from '../../../core/utils/a11y';
import { ColumnMultiSelect } from '../shared/ColumnMultiSelect';
import { useIsWideContainer } from '../../../core/hooks/useIsWideContainer';

interface FeatureInteractionConfig {
  columns: string[];
  degree: 2 | 3 | 4;
  interaction_only: boolean;
  include_bias: boolean;
  isExpanded?: boolean;
}

export const FeatureInteractionNode: NodeDefinition = {
  type: 'FeatureInteractionNode',
  label: 'Feature Interaction',
  description: 'Generate 2-way/3-way multiplicative interaction features between numeric columns.',
  category: 'Preprocessing',
  icon: Shuffle,
  inputs: [{ id: 'in', label: 'Input Dataset', type: 'dataset' }],
  outputs: [{ id: 'out', label: 'Transformed Data', type: 'dataset' }],

  getDefaultConfig: () => ({
    columns: [],
    degree: 2,
    interaction_only: true,
    include_bias: false,
    isExpanded: true
  }),

  bodyPreview: (config: { columns?: string[]; degree?: number }) => {
    const cols = config.columns?.length ?? 0;
    const deg = config.degree ?? 2;
    if (cols === 0) return `degree=${deg}`;
    return `degree=${deg} \u00b7 ${cols} ${cols === 1 ? 'col' : 'cols'}`;
  },

  settings: function FeatureInteractionSettings({ config, onChange, nodeId }) {
    const upstreamData = useUpstreamData(nodeId || '');
    const datasetId = upstreamData.find((d) => d.datasetId)?.datasetId as string | undefined;
    const { data: schema } = useDatasetSchema(datasetId);
    const droppedUpstream = useUpstreamDroppedColumns(nodeId);

    // Interactions only make sense between numeric columns.
    const numericColumns = schema ? Object.values(schema.columns)
      .filter((col) => ['int', 'float', 'number', 'double', 'long'].some(t => col.dtype.toLowerCase().includes(t)))
      .filter((col) => !droppedUpstream.has(col.name))
      .map((col) => col.name) : [];

    const updateConfig = (updates: Partial<FeatureInteractionConfig>) => {
      onChange({ ...config, ...updates });
    };

    const toggleExpand = () => {
      updateConfig({ isExpanded: !config.isExpanded });
    };

    // Responsive layout: switch to a 2-column layout once the panel is wider than 450px.
    const [containerRef, isWide] = useIsWideContainer();

    return (
      <div ref={containerRef} className="space-y-2 w-full">
        <div className="border rounded-md bg-card">
          <div
            className="flex items-center justify-between p-2 cursor-pointer hover:bg-accent/50 transition-colors"
            {...clickableProps(toggleExpand)}
          >
            <div className="flex items-center gap-2">
              <Shuffle size={14} className="text-primary" />
              <span className="text-sm font-medium">Configuration</span>
            </div>
            {config.isExpanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
          </div>

          {config.isExpanded && (
            <div className={`p-3 border-t gap-4 ${isWide ? 'grid grid-cols-2 items-start' : 'space-y-4'}`}>

              <ColumnMultiSelect
                label="Input Columns (Numeric)"
                columns={numericColumns}
                selected={config.columns || []}
                onChange={(cols) => updateConfig({ columns: cols })}
                variant="compact"
                showFooterCount={true}
              />

              <div className="space-y-3">
                <div className="space-y-1.5">
                  <span className="text-xs font-medium text-muted-foreground flex items-center gap-1">
                    Degree
                    <div className="group relative">
                      <Info size={10} className="cursor-help" />
                      <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-1 hidden group-hover:block w-52 p-2 bg-popover text-popover-foreground text-[10px] rounded border shadow-lg z-50">
                        2 for pairwise interactions (e.g. a*b), 3 for three-way (a*b*c), 4 for four-way (a*b*c*d). Higher degrees grow combinatorially — use sparingly.
                      </div>
                    </div>
                  </span>
                  <select
                    className="w-full px-2 py-1.5 text-xs border rounded bg-background"
                    value={config.degree || 2}
                    onChange={(e) => {
                      const parsed = parseInt(e.target.value);
                      const degree: 2 | 3 | 4 = parsed === 4 ? 4 : parsed === 3 ? 3 : 2;
                      updateConfig({ degree });
                    }}
                  >
                    <option value={2}>2 (pairwise)</option>
                    <option value={3}>3 (three-way)</option>
                    <option value={4}>4 (four-way)</option>
                  </select>
                </div>

                <div className="space-y-2 pt-1">
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      className="rounded border-muted"
                      checked={config.interaction_only ?? true}
                      onChange={(e) => updateConfig({ interaction_only: e.target.checked })}
                    />
                    <span className="text-xs">Interaction Only</span>
                  </label>
                  <p className="text-[10px] text-muted-foreground pl-5">
                    If true (default), skips self-products (e.g. a*a); only distinct-column combinations are generated.
                  </p>

                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      className="rounded border-muted"
                      checked={config.include_bias || false}
                      onChange={(e) => updateConfig({ include_bias: e.target.checked })}
                    />
                    <span className="text-xs">Include Bias</span>
                  </label>
                  <p className="text-[10px] text-muted-foreground pl-5">
                    Adds a constant column of 1.0 (named &quot;interaction_bias&quot;), useful for some linear models.
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    );
  },

  validate: (data) => {
    if (!data.columns || data.columns.length === 0) {
      return { isValid: false, message: 'Select at least one input column.' };
    }
    if (data.columns.length < (data.degree || 2)) {
      return { isValid: false, message: `Select at least ${data.degree || 2} columns for degree ${data.degree || 2} interactions.` };
    }
    if (![2, 3, 4].includes(data.degree)) {
      return { isValid: false, message: 'Degree must be 2, 3, or 4.' };
    }
    return { isValid: true };
  }
};
