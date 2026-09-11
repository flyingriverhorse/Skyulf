import { EDASidebar } from '../../components/eda/EDASidebar';
import { DashboardTab } from '../../components/eda/tabs/DashboardTab';
import { InsightsTab } from '../../components/eda/tabs/InsightsTab';
import { PIIReviewTab } from '../../components/eda/tabs/PIIReviewTab';
import { PCATab } from '../../components/eda/tabs/PCATab';
import { GeospatialTab } from '../../components/eda/tabs/GeospatialTab';
import { TargetAnalysisTab } from '../../components/eda/tabs/TargetAnalysisTab';
import { TimeSeriesTab } from '../../components/eda/tabs/TimeSeriesTab';
import { VariablesTab } from '../../components/eda/tabs/VariablesTab';
import { BivariateTab } from '../../components/eda/tabs/BivariateTab';
import { OutliersTab } from '../../components/eda/tabs/OutliersTab';
import { CorrelationsTab } from '../../components/eda/tabs/CorrelationsTab';
import { SampleDataTab } from '../../components/eda/tabs/SampleDataTab';
import { CausalTab } from '../../components/eda/tabs/CausalTab';
import { RuleDiscoveryTab } from '../../components/eda/tabs/RuleDiscoveryTab';
import { DecompositionTab } from '../../components/eda/tabs/DecompositionTab';
import { downloadChart } from '../../core/utils/chartUtils';
import type { EDAReport } from '../../core/api/eda';
import type { EDAProfile } from '../../core/types/edaProfile';
import type { EdaPageModel } from './useEdaPageController';

type ProfileProps = EdaPageModel & { profile: EDAProfile; report: EDAReport; };
type ProfileTabProps = ProfileProps & { allColumns: string[]; };

export function EdaProfileContent(props: ProfileProps) {
  const {
    report,
    profile,
    activeTab,
    setActiveTab,
    filtersDraft,
    filtersApplied,
    filtersDirty,
    excludedColsDraft,
    excludedDirty,
    analyzing,
    handleAddFilter,
    handleRemoveFilter,
    handleResetFilters,
    handleApplyFilters,
    handleToggleExclude,
    handleApplyExcluded
  } = props;
  const allColumns = report?.profile_data?.columns ? Object.keys(report.profile_data.columns) : [];

  return (
    <div className="flex h-full w-full overflow-hidden bg-white dark:bg-gray-900">
      <EDASidebar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        profile={profile}
        filtersDraft={filtersDraft}
        filtersApplied={filtersApplied}
        filtersDirty={filtersDirty}
        columns={allColumns}
        excludedCols={excludedColsDraft}
        excludedDirty={excludedDirty}
        analyzing={analyzing}
        onAddFilter={handleAddFilter}
        onRemoveFilter={handleRemoveFilter}
        onResetFilters={handleResetFilters}
        onApplyFilters={handleApplyFilters}
        onToggleExclude={handleToggleExclude}
        onApplyExcluded={handleApplyExcluded}
      />

      <div className="flex-1 overflow-y-auto p-6 pt-4 bg-gray-50 dark:bg-gray-900/50">
        <ProfileTab {...props} allColumns={allColumns} />
      </div>
    </div>
  );
}

/** The selected tab keeps its original data guard and prop contract. */
const PROFILE_TABS = {
  dashboard: (props: ProfileTabProps) => {
    const { profile, handleToggleExclude, excludedColsDraft } = props;
    return (
      <DashboardTab
        profile={profile}
        onToggleExclude={handleToggleExclude}
        excludedCols={excludedColsDraft}
      />
    );
  },
  insights: (props: ProfileTabProps) => {
    const { profile } = props;
    return (
      <InsightsTab profile={profile} />
    );
  },
  pii: (props: ProfileTabProps) => {
    const { profile } = props;
    return (
      <PIIReviewTab alerts={profile.alerts ?? []} />
    );
  },
  pca: (props: ProfileTabProps) => {
    const { profile, scatter, setScatter } = props;
    return (
      <PCATab
        profile={profile}
        isPCA3D={scatter.isPCA3D}
        setIsPCA3D={(v) => setScatter({ isPCA3D: v })}
        downloadChart={downloadChart}
      />
    );
  },
  geospatial: (props: ProfileTabProps) => {
    const { profile } = props;
    return !!profile.geospatial && (
      <GeospatialTab profile={profile} />
    );
  },
  target: (props: ProfileTabProps) => {
    const { profile, history, loading, loadSpecificReport, report } = props;
    return !!profile.target_col && !!profile.target_correlations && (
      <TargetAnalysisTab
        profile={profile}
        downloadChart={downloadChart}
        history={history}
        loading={loading}
        loadSpecificReport={loadSpecificReport}
        report={report}
      />
    );
  },
  timeseries: (props: ProfileTabProps) => {
    const { profile } = props;
    return !!profile.timeseries && (
      <TimeSeriesTab
        profile={profile}
        downloadChart={downloadChart}
      />
    );
  },
  variables: (props: ProfileTabProps) => {
    const { profile, handleToggleExclude, handleAddFilter } = props;
    return (
      <VariablesTab
        profile={profile}
        handleToggleExclude={handleToggleExclude}
        handleAddFilter={handleAddFilter}
      />
    );
  },
  bivariate: (props: ProfileTabProps) => {
    const { profile, scatter, setScatter } = props;
    return (
      <BivariateTab
        profile={profile}
        downloadChart={downloadChart}
        scatterX={scatter.x}
        setScatterX={(v) => setScatter({ x: v })}
        scatterY={scatter.y}
        setScatterY={(v) => setScatter({ y: v })}
        scatterZ={scatter.z}
        setScatterZ={(v) => setScatter({ z: v })}
        scatterColor={scatter.color}
        setScatterColor={(v) => setScatter({ color: v })}
        is3D={scatter.is3D}
        setIs3D={(v) => setScatter({ is3D: v })}
      />
    );
  },
  outliers: (props: ProfileTabProps) => {
    const { profile } = props;
    return profile.outliers && (
      <OutliersTab profile={profile} />
    );
  },
  correlations: (props: ProfileTabProps) => {
    const { profile } = props;
    return (!!profile.correlations || !!profile.correlations_with_target || !!profile.causal_target_exclusion_reason) && (
      <CorrelationsTab
        profile={profile}
      />
    );
  },
  causal: (props: ProfileTabProps) => {
    const { profile } = props;
    return <CausalTab profile={profile} />;
  },
  rules: (props: ProfileTabProps) => {
    const { profile } = props;
    return !!profile.rule_tree && (
      <RuleDiscoveryTab profile={profile} />
    );
  },
  decomposition: (props: ProfileTabProps) => {
    const { selectedDataset, allColumns, filtersApplied } = props;
    return selectedDataset && (
      <DecompositionTab
        datasetId={selectedDataset}
        columns={allColumns}
        initialFilters={filtersApplied}
      />
    );
  },
  sample: (props: ProfileTabProps) => {
    const { profile, excludedColsDraft, handleToggleExclude } = props;
    return profile.sample_data && (
      <SampleDataTab
        profile={profile}
        excludedCols={excludedColsDraft}
        handleToggleExclude={handleToggleExclude}
      />
    );
  },
};

function ProfileTab(props: ProfileTabProps) {
  const tab = props.activeTab;
  if (!Object.prototype.hasOwnProperty.call(PROFILE_TABS, tab)) return null;
  const Tab = PROFILE_TABS[tab as keyof typeof PROFILE_TABS];
  return <Tab {...props} />;
}
