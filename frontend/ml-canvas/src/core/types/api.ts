export interface Dataset {
  id: string;
  name: string;
  description?: string;
  created_at: string;
  rows?: number;
  columns?: number;
  size_bytes?: number;
  format?: string;
}
