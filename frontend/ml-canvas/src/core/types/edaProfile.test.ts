import type { ComponentProps } from 'react';
import { describe, expectTypeOf, it } from 'vitest';
import type { CausalTab } from '../../components/eda/tabs/CausalTab';
import type { ClusteringTab } from '../../components/eda/tabs/ClusteringTab';
import type { CorrelationsTab } from '../../components/eda/tabs/CorrelationsTab';
import type { TargetAnalysisTab } from '../../components/eda/tabs/TargetAnalysisTab';
import type { TimeSeriesTab } from '../../components/eda/tabs/TimeSeriesTab';
import type { EDAHistoryEntry, EDAReport } from '../api/eda';
import type { EDAProfile } from './edaProfile';

describe('EDA chart contracts', () => {
  it('checks report props and analysis fields instead of accepting arbitrary shapes', () => {
    /** Chart consumers must reject malformed nested payloads at compile time. */
    expectTypeOf<ComponentProps<typeof CausalTab>['profile']>().not.toBeAny();
    expectTypeOf<ComponentProps<typeof ClusteringTab>['profile']>().not.toBeAny();
    expectTypeOf<ComponentProps<typeof CorrelationsTab>['profile']>().not.toBeAny();
    expectTypeOf<ComponentProps<typeof TargetAnalysisTab>['profile']>().not.toBeAny();
    expectTypeOf<ComponentProps<typeof TimeSeriesTab>['profile']>().not.toBeAny();
    expectTypeOf<ComponentProps<typeof TargetAnalysisTab>['history']>().toEqualTypeOf<EDAHistoryEntry[]>();
    expectTypeOf<ComponentProps<typeof TargetAnalysisTab>['report']>().toEqualTypeOf<EDAReport | null>();
    expectTypeOf<EDAProfile['timeseries']>().not.toBeUnknown();
    expectTypeOf<EDAProfile['clustering']>().not.toBeUnknown();
    expectTypeOf<EDAProfile['target_interactions']>().not.toBeUnknown();
  });
});
