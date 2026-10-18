import type { Node } from '@xyflow/react';
import { expect, it } from 'vitest';
import { preprocessingConverters } from './preprocessing';

/** API requests must preserve explicit local-calendar options and older omitted timezone settings. */
it.each([undefined, 'Europe/Istanbul'])('preserves date settings with timezone %s', (timezone) => {
  const params = { method: 'date', columns: ['date'], features: ['second', 'hour_sin', 'hour_cos'], ...(timezone ? { timezone } : {}) };
  const node: Node = { id: 'calendar', position: { x: 0, y: 0 }, data: { definitionType: 'TimeSeriesNode', ...params } };
  expect(preprocessingConverters.get('TimeSeriesNode')!(node)).toEqual({ stepType: 'DateFeatures', params });
});
