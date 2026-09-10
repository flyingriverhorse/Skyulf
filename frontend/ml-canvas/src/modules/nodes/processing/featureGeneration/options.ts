import { Calculator, Calendar, Percent, GitCompare, FunctionSquare } from 'lucide-react';

export const OPERATION_TYPES = [
  { value: 'arithmetic', label: 'Arithmetic', icon: Calculator },
  { value: 'datetime_extract', label: 'Date Extraction', icon: Calendar },
  { value: 'ratio', label: 'Ratio', icon: Percent },
  { value: 'similarity', label: 'Similarity', icon: GitCompare },
  { value: 'group_agg', label: 'Group Agg', icon: FunctionSquare },
];

export const ARITHMETIC_METHODS = ['add', 'subtract', 'multiply', 'divide'];
export const DATE_METHODS = ['year', 'quarter', 'month', 'month_name', 'week', 'day', 'day_name', 'weekday', 'is_weekend', 'hour', 'minute', 'second', 'season', 'time_of_day'];

// Tooltip descriptions shown on hover for each datetime feature.
// Both pandas and polars paths produce identical semantics.
export const DATE_METHOD_META: Record<string, { type: string; desc: string }> = {
  year:       { type: 'int',    desc: 'Calendar year (e.g. 2024)' },
  quarter:    { type: 'int',    desc: 'Quarter of year — 1 to 4' },
  month:      { type: 'int',    desc: 'Month number — 1 (Jan) to 12 (Dec)' },
  month_name: { type: 'string', desc: 'Full month name — "January" … "December"' },
  week:       { type: 'int',    desc: 'ISO week number — 1 to 53' },
  day:        { type: 'int',    desc: 'Day of month — 1 to 31' },
  day_name:   { type: 'string', desc: 'Day of week name — "Monday" … "Sunday"' },
  weekday:    { type: 'int',    desc: 'Day of week — 0 (Mon) to 6 (Sun)' },
  is_weekend: { type: 'int',    desc: 'Weekend flag — 1 if Sat/Sun, else 0' },
  hour:       { type: 'int',    desc: 'Hour of day — 0 to 23' },
  minute:     { type: 'int',    desc: 'Minute of hour — 0 to 59' },
  second:     { type: 'int',    desc: 'Second of minute — 0 to 59' },
  season:     { type: 'string', desc: 'Meteorological season — "Winter", "Spring", "Summer", "Autumn"' },
  time_of_day:{ type: 'string', desc: 'Hour bucket — "Morning" (5-11), "Afternoon" (12-16), "Evening" (17-20), "Night" (21-4)' },
};
export const SIMILARITY_METHODS = ['ratio', 'token_sort_ratio', 'token_set_ratio'];
export const GROUP_AGG_METHODS = ['mean', 'sum', 'count', 'min', 'max', 'std', 'median'];
