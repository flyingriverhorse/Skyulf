import { AlertCircle, RefreshCw } from 'lucide-react';
import { JobsHistoryModal } from '../../components/eda/JobsHistoryModal';
import { EDAService } from '../../core/api/eda';
import { useEDAStore } from '../../core/store/useEDAStore';
import { edaKeys } from '../../core/hooks/useEdaJobs';
import type { EdaPageModel } from './useEdaPageController';

export function EdaPageFeedback(props: EdaPageModel) {
  const { selectionUnavailable, selectedDataset, analyzeMutation, analyzing, runAnalysis } = props;
  return <>
    {/* A deep link can name a dataset EDA can't analyse (still ingesting, or
          no usable columns). Say so instead of letting the <select> fall back
          to an unrelated option while the queries target the requested id. */}
    {selectionUnavailable && (
      <div
        role="alert"
        className="flex flex-wrap items-center gap-3 border-b border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-900 dark:border-amber-900 dark:bg-amber-950/40 dark:text-amber-200"
      >
        <AlertCircle className="h-4 w-4 shrink-0" />
        <span className="flex-1 min-w-[12rem]">
          Dataset #{selectedDataset} isn&apos;t available for analysis — it may still be
          processing or have no usable columns. Pick another dataset above to continue.
        </span>
      </div>
    )}

    {/* A rejected submission never reaches the report poller, so surface it
          here with the inputs still intact rather than failing silently. */}
    {analyzeMutation.isError && (
      <div
        role="alert"
        className="flex flex-wrap items-center gap-3 border-b border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800 dark:border-red-900 dark:bg-red-950/40 dark:text-red-200"
      >
        <AlertCircle className="h-4 w-4 shrink-0" />
        <span className="flex-1 min-w-[12rem]">
          Could not start the analysis: {analyzeMutation.error instanceof Error
            ? analyzeMutation.error.message
            : 'the request was rejected.'}{' '}
          Your dataset, target, and filter choices were kept.
        </span>
        <button
          onClick={() => {
            const last = analyzeMutation.variables;
            runAnalysis(last?.excluded, last?.filters);
          }}
          disabled={analyzing}
          className="flex items-center rounded-md border border-red-300 px-3 py-1.5 font-medium hover:bg-red-100 disabled:opacity-50 dark:border-red-800 dark:hover:bg-red-900/40"
        >
          <RefreshCw className="mr-2 h-4 w-4" />
          Try again
        </button>
        <button
          onClick={() => analyzeMutation.reset()}
          className="rounded-md px-2 py-1.5 font-medium underline hover:no-underline"
        >
          Dismiss
        </button>
      </div>
    )}
  </>;
}

export function EdaHistoryModal(props: EdaPageModel) {
  const { showHistoryModal, setShowHistoryModal, history, selectedDataset, queryClient } = props;
  return <>
    {/* Jobs History Modal */}
    <JobsHistoryModal
      isOpen={showHistoryModal}
      onClose={() => setShowHistoryModal(false)}
      history={history}
      datasetId={selectedDataset ?? null}
      onRefresh={() =>
        queryClient.invalidateQueries({ queryKey: edaKeys.history(selectedDataset ?? null) })
      }
      onFetchReport={async (id) => {
        const r = await EDAService.getReport(id);
        // EDAReport.id is optional in the API type; here we know the route
        // returned a real report so the id is always present.
        return { ...r, id: r.id ?? id };
      }}
      onSelect={(selectedReport) => {
        if (selectedDataset) {
          queryClient.setQueryData(edaKeys.report(selectedDataset), selectedReport);
        }
        const serverExcluded = Array.isArray(selectedReport.profile_data?.excluded_columns)
          ? selectedReport.profile_data.excluded_columns
          : [];
        useEDAStore.getState().setExcludedApplied(serverExcluded);
        useEDAStore.getState().setExcludedDraft(serverExcluded);
      }}
    />
  </>;
}
