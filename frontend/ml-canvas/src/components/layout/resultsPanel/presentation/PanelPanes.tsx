import type { ComponentProps } from 'react';
import { BranchTabs } from '../BranchTabs';
import { SplitTabs } from '../SplitTabs';
import { ResultsTable } from '../ResultsTable';
import { MergeWarningsBanner } from '../MergeWarningsBanner';
import { ValidationBanner, RunErrorBanner } from './IssueBanners';

/** Which pane of the results panel is showing. */
export type ResultsPane = 'data' | 'issues' | 'steps';

interface DataPaneProps {
  hasResults: boolean;
  branchTabs: ComponentProps<typeof BranchTabs>;
  splitTabs: ComponentProps<typeof SplitTabs>;
  table: ComponentProps<typeof ResultsTable>;
}

/** Preserve the backend's branch and split order around the active table. */
export function DataPane({ hasResults, branchTabs, splitTabs, table }: DataPaneProps) {
  return (
    <div className="flex-1 overflow-hidden flex flex-col">
      {hasResults && branchTabs.branchLabels.length > 0 && (
        <BranchTabs {...branchTabs} />
      )}
      {hasResults && splitTabs.tabNames.length > 1 && (
        <SplitTabs {...splitTabs} />
      )}
      {hasResults ? (
        <ResultsTable {...table} />
      ) : (
        <p className="p-4 text-sm text-muted-foreground">
          Run a preview to see the resulting rows here.
        </p>
      )}
    </div>
  );
}

interface IssuesPaneProps extends ComponentProps<typeof ValidationBanner> {
  lastRunError: string | null;
  hasResults: boolean;
  issueCount: number;
  advisories: ComponentProps<typeof MergeWarningsBanner>;
}

/** Keep run errors, validation focus and branch advisories in the issues pane. */
export function IssuesPane({
  validationIssues,
  validationHeadingId,
  openIssue,
  lastRunError,
  hasResults,
  issueCount,
  advisories,
}: IssuesPaneProps) {
  return (
    <div className="flex-1 overflow-y-auto">
      <ValidationBanner validationIssues={validationIssues} validationHeadingId={validationHeadingId} openIssue={openIssue} />
      <RunErrorBanner lastRunError={lastRunError} />
      {hasResults && advisories.mergeWarnings.length > 0 && <MergeWarningsBanner {...advisories} />}
      {issueCount === 0 && (
        <p className="p-4 text-sm text-muted-foreground">
          No validation issues, run errors, or merge advisories.
        </p>
      )}
    </div>
  );
}

interface StepsPaneProps {
  applied_steps: string[];
  status: string | undefined;
  nodeLabelMap: Record<string, string>;
}

/** Failed runs retain their step count but do not show successful-step pills. */
export function StepsPane({ applied_steps, status, nodeLabelMap }: StepsPaneProps) {
  return (
    <div className="flex-1 overflow-y-auto p-3">
      {applied_steps.length > 0 && status !== 'failed' ? (
        <div className="flex flex-wrap gap-2">
          {applied_steps.map((step: string) => (
            <span
              key={step}
              className="text-xs text-blue-800 dark:text-blue-200 bg-blue-100 dark:bg-blue-900/40 px-2 py-1 rounded border border-blue-200 dark:border-blue-800"
            >
              {nodeLabelMap[step] ?? step}
            </span>
          ))}
        </div>
      ) : (
        <p className="text-sm text-muted-foreground">
          No steps ran. Run a preview to see which nodes executed.
        </p>
      )}
    </div>
  );
}

interface PaneTabsProps {
  activePane: ResultsPane;
  setPane: (pane: ResultsPane) => void;
  issueCount: number;
  stepCount: number;
}

interface PanelContentProps extends PaneTabsProps {
  data: DataPaneProps;
  issues: IssuesPaneProps;
  steps: StepsPaneProps;
}

/** Mount only the selected pane; its durable state belongs to ResultsPanel. */
export function PanelContent({ activePane, setPane, issueCount, stepCount, data, issues, steps }: PanelContentProps) {
  return (
    <div className="flex-1 overflow-hidden flex flex-col">
      <PaneTabs activePane={activePane} setPane={setPane} issueCount={issueCount} stepCount={stepCount} />
      {activePane === 'data' && <DataPane {...data} />}
      {activePane === 'issues' && <IssuesPane {...issues} />}
      {activePane === 'steps' && <StepsPane {...steps} />}
    </div>
  );
}

/** Keep the stable data, issues and steps navigation order. */
export function PaneTabs({ activePane, setPane, issueCount, stepCount }: PaneTabsProps) {
  const paneTabs: { id: ResultsPane; label: string; count?: number }[] = [
    { id: 'data', label: 'Data' },
    { id: 'issues', label: 'Issues', count: issueCount },
    { id: 'steps', label: 'Steps', count: stepCount },
  ];
  return (
    <div className="flex items-center gap-1 px-2 border-b bg-muted/5 shrink-0" role="tablist">
      {paneTabs.map((tab) => (
        <button
          key={tab.id}
          type="button"
          role="tab"
          aria-selected={activePane === tab.id}
          onClick={() => setPane(tab.id)}
          className={`px-3 py-1.5 text-xs font-medium border-b-2 -mb-px transition-colors ${
            activePane === tab.id
              ? 'border-primary text-foreground'
              : 'border-transparent text-muted-foreground hover:text-foreground'
          }`}
        >
          {tab.label}
          {tab.count ? (
            <span className="ml-1.5 text-[10px] px-1 py-0.5 rounded bg-muted text-muted-foreground">
              {tab.count}
            </span>
          ) : null}
        </button>
      ))}
    </div>
  );
}
