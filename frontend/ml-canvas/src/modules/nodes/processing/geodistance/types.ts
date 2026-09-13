export interface GeoDistanceConfig {
  lat1_col: string;
  lon1_col: string;
  lat2_col: string;
  lon2_col: string;
  method: 'haversine' | 'euclidean';
  unit: 'km' | 'mi';
  output_column: string;
}

export const COORDINATES = [
  { field: 'lat1_col', label: 'Point 1 latitude' },
  { field: 'lon1_col', label: 'Point 1 longitude' },
  { field: 'lat2_col', label: 'Point 2 latitude' },
  { field: 'lon2_col', label: 'Point 2 longitude' },
] as const;
