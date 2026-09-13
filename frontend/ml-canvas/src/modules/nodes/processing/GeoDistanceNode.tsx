import { lazy, Suspense } from 'react';
import { MapPin } from 'lucide-react';
import type { NodeDefinition } from '../../../core/types/nodes';
import { COORDINATES, type GeoDistanceConfig } from './geodistance/types';

export type { GeoDistanceConfig } from './geodistance/types';

const GeoDistanceSettings = lazy(() => import('./geodistance/GeoDistanceSettings'));

export const GeoDistanceNode: NodeDefinition<GeoDistanceConfig> = {
  type: 'GeoDistance',
  label: 'Geo Distance',
  description: 'Calculate the distance between two latitude/longitude points in each row.',
  category: 'Preprocessing',
  icon: MapPin,
  inputs: [{ id: 'in', label: 'Input Dataset', type: 'dataset' }],
  outputs: [{ id: 'out', label: 'Transformed Data', type: 'dataset' }],
  getDefaultConfig: () => ({
    lat1_col: '', lon1_col: '', lat2_col: '', lon2_col: '',
    method: 'haversine', unit: 'km', output_column: '',
  }),
  settings: function LazyGeoDistanceSettings(props) {
    return <Suspense fallback={<p role="status" className="p-4 text-xs text-muted-foreground">Loading settings...</p>}>
      <GeoDistanceSettings {...props} />
    </Suspense>;
  },
  bodyPreview: config => `${config.method === 'euclidean' ? 'Local' : 'Haversine'} · ${config.output_column || `geo_distance_${config.unit ?? 'km'}`}`,
  validate: config => {
    for (const { field, label } of COORDINATES) {
      if (typeof config[field] !== 'string' || !config[field].trim()) {
        return { isValid: false, field, message: `Select a numeric column for ${label.toLowerCase()}.` };
      }
    }
    if (!['haversine', 'euclidean'].includes(config.method)) {
      return { isValid: false, field: 'method', message: 'Choose Haversine or Euclidean distance.' };
    }
    if (!['km', 'mi'].includes(config.unit)) {
      return { isValid: false, field: 'unit', message: 'Choose kilometres or miles.' };
    }
    if (config.output_column !== undefined && typeof config.output_column !== 'string') {
      return { isValid: false, field: 'output_column', message: 'Enter a column name or leave it blank for an automatic name.' };
    }
    return { isValid: true };
  },
};
