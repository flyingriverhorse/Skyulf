import React from 'react';
import * as Tabs from '@radix-ui/react-tabs';
import {
  ArrowRight,
  GitFork,
  Merge,
  Split,
  Rows3,
  Play,
  History,
  ShieldCheck,
  AlertTriangle,
  Tag,
} from 'lucide-react';
import { ModalShell } from '../shared/ModalShell';
import { PreprocessingPlacementGuide } from './PreprocessingPlacementGuide';
import { SplitMergeGuide } from './SplitMergeGuide';

interface HelpGuideModalProps {
  isOpen: boolean;
  onClose: () => void;
  /** Initial section on each open; omitted callers retain the pipeline basics. */
  initialTab?: 'basics' | 'leakage';
}

interface SectionProps {
  icon: React.ReactNode;
  title: string;
  children: React.ReactNode;
}

const Section: React.FC<SectionProps> = ({ icon, title, children }) => (
  <section className="p-5 border-b border-slate-100 dark:border-slate-800 last:border-b-0">
    <h3 className="flex items-center gap-2 text-sm font-semibold text-slate-900 dark:text-slate-100 mb-2">
      <span className="flex items-center justify-center w-6 h-6 rounded-md bg-indigo-500/10 text-indigo-600 dark:text-indigo-400 shrink-0">
        {icon}
      </span>
      {title}
    </h3>
    <div className="text-sm text-slate-600 dark:text-slate-300 space-y-2 pl-8">{children}</div>
  </section>
);

/**
 * Plain-language guide to canvas concepts (linear chains, branches, merges,
 * the post-Split trap, Score Advisory) so new users learn the mental model
 * instead of discovering it from failed runs.
 */
export const HelpGuideModal: React.FC<HelpGuideModalProps> = ({ isOpen, onClose, initialTab = 'basics' }) => (
  <ModalShell
    isOpen={isOpen}
    onClose={onClose}
    title="How pipelines work"
    size="5xl"
    className="h-[calc(100dvh-2rem)] !max-h-[calc(100dvh-2rem)] sm:h-[calc(100dvh-4rem)] sm:!max-h-[calc(100dvh-4rem)]"
  >
    {isOpen && <Tabs.Root key={initialTab} defaultValue={initialTab} className="flex h-full min-h-0 flex-col">
      <Tabs.List aria-label="Pipeline help sections" className="sticky top-0 z-20 grid shrink-0 grid-cols-3 gap-1 border-b border-slate-200 bg-white px-2 dark:border-slate-700 dark:bg-slate-900 sm:px-5">
        <Tabs.Trigger
          value="basics"
          className="min-w-0 whitespace-normal break-words border-b-2 border-transparent px-2 py-3 text-xs font-medium text-slate-500 focus-ring data-[state=active]:border-indigo-500 data-[state=active]:text-indigo-600 dark:text-slate-400 dark:data-[state=active]:text-indigo-400 sm:px-3 sm:text-sm"
        >
          Pipeline Basics
        </Tabs.Trigger>
        <Tabs.Trigger
          value="leakage"
          className="min-w-0 whitespace-normal break-words border-b-2 border-transparent px-2 py-3 text-xs font-medium text-slate-500 focus-ring data-[state=active]:border-indigo-500 data-[state=active]:text-indigo-600 dark:text-slate-400 dark:data-[state=active]:text-indigo-400 sm:px-3 sm:text-sm"
        >
          Preprocessing &amp; Leakage
        </Tabs.Trigger>
        <Tabs.Trigger
          value="split-merge"
          className="min-w-0 whitespace-normal break-words border-b-2 border-transparent px-2 py-3 text-xs font-medium text-slate-500 focus-ring data-[state=active]:border-indigo-500 data-[state=active]:text-indigo-600 dark:text-slate-400 dark:data-[state=active]:text-indigo-400 sm:px-3 sm:text-sm"
        >
          Split &amp; Merge
        </Tabs.Trigger>
      </Tabs.List>
      <div className="min-h-0 flex-1 overflow-y-auto">
      <Tabs.Content value="basics" className="outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-indigo-500">
      <Section icon={<ArrowRight className="w-3.5 h-3.5" />} title="Linear chain — the basics">
        <p>
          Data flows left to right, one node at a time: a Loader reads a dataset, transform
          nodes clean or encode it, and a Train / Tune node at the end fits the model. Each
          node passes its output to the next.
        </p>
      </Section>

      <Section icon={<GitFork className="w-3.5 h-3.5" />} title="Branches — one input, many paths">
        <p>
          Connect one node to two or more nodes and the flow forks into parallel branches.
          There are two common uses:
        </p>
        <ul className="list-disc pl-5 space-y-1">
          <li>
            <span className="font-medium">Compare, don&apos;t merge</span> — each branch ends
            in its own Train / Tune node. Run them as separate experiments with{' '}
            <span className="font-medium">Run All Experiments</span>.
          </li>
          <li>
            <span className="font-medium">Divide and merge</span> — each branch transforms a
            different set of columns, then a downstream node takes several inputs and merges
            the results back together.
          </li>
        </ul>
      </Section>

      <Section icon={<Merge className="w-3.5 h-3.5" />} title="Merging — which branch wins?">
        <p>
          Multiple wires into the same node join branches column-wise; no separate Merge node
          is needed. Different-named columns are retained. If a shared ancestor identifies
          exactly one branch that changed an overlapping column, that branch wins. Otherwise,
          conflicting versions follow the node&apos;s{' '}
          <span className="font-medium">Merge Strategy</span>: first wins or last wins
          (the default). For sibling branches, first/last follows saved incoming-edge order,
          not their position on the canvas. The Split &amp; Merge tab shows examples.
        </p>
      </Section>

      <Section icon={<Split className="w-3.5 h-3.5" />} title="After a Split node — order decides">
        <p>
          Train, validation and test partitions merge separately, and{' '}
          <span className="font-medium">the configured merge strategy still applies</span>.
          Column ownership generally lacks a shared-ancestor baseline after a row split, so a
          branch passing through an original column can overwrite another branch&apos;s encoding
          or scaling. Prefer distinct feature outputs and choose the strategy intentionally.
          X/y merges retain the first branch&apos;s target; all branches need the same aligned y.
        </p>
      </Section>

      <Section icon={<Rows3 className="w-3.5 h-3.5" />} title="Row alignment — branches must stay in step">
        <p>
          Merging lines up branches row by row, so every branch must return the same rows in
          the same order. Steps that drop rows (like drop-missing-rows or filters) are blocked
          inside branches before the run starts.
        </p>
      </Section>

      <Section icon={<Play className="w-3.5 h-3.5" />} title="Preview vs running experiments">
        <p>
          <span className="font-medium">Preview data</span> (Ctrl+Enter) executes the whole
          graph right away and shows the resulting rows in the Preview Results panel at the
          bottom &mdash; the fastest way to check that your columns and shapes come out right.
        </p>
        <p>
          <span className="font-medium">Run All Experiments</span> submits the graph as real
          training jobs that run in the background &mdash; one job per branch when you have
          several. Track their progress in the Jobs list that opens automatically.
        </p>
      </Section>

      <Section icon={<History className="w-3.5 h-3.5" />} title="Where your results live">
        <ul className="list-disc pl-5 space-y-1">
          <li>
            <span className="font-medium">Preview Results</span> (bottom panel): the rows your
            last preview produced, plus validation issues and the steps that ran. Close it with
            the X in its header when you&apos;re done.
          </li>
          <li>
            <span className="font-medium">Jobs</span> (clock icon, top-right of the canvas):
            every training / tuning run with its status, metrics, and artifacts. Click a job
            for the full detail view.
          </li>
          <li>
            <span className="font-medium">Experiments</span> tab: compare finished runs side by
            side with charts.
          </li>
        </ul>
      </Section>

      <Section icon={<ShieldCheck className="w-3.5 h-3.5" />} title="Can you trust the scores?">
        <p>
          Job details carry two verification tiles. The{' '}
          <span className="font-medium">Leakage Gate</span> checks detected preprocessing
          placement violations. The{' '}
          <span className="font-medium">Fold Refit Audit</span> records whether supported
          preprocessing was refit inside cross-validation folds. Read both verdicts and their
          details: green tiles do not prove feature provenance, temporal correctness or entity
          independence. The Preprocessing &amp; Leakage tab explains the placement rules and limits.
        </p>
      </Section>

      <Section icon={<AlertTriangle className="w-3.5 h-3.5" />} title="Score Advisory — the amber tile in Jobs">
        <p>
          If a job&apos;s details show an amber{' '}
          <span className="font-medium">Score Advisory</span> tile, inspect the fallback
          diagnostic. Unsupported fold reconstruction involving learned preprocessing stops
          execution under the default raise policy. Explicit warn/ignore policies can allow
          scores on pre-transformed data, which may be optimistically biased. Prefer a
          supported linear chain or row-aligned branches from a common split joining directly
          into the model, with learned preprocessing after the split.
        </p>
      </Section>

      <Section icon={<Tag className="w-3.5 h-3.5" />} title="Badges and edge colors">
        <p>
          For what the node badges (merge counts, success / failure) and edge colors mean,
          open the legend with the <Tag className="w-3 h-3 inline-block align-text-bottom" />{' '}
          button at the top-left of the canvas. Press{' '}
          <span className="font-medium">?</span> for the full keyboard-shortcut cheat sheet.
        </p>
      </Section>
      </Tabs.Content>
      <Tabs.Content value="leakage" className="outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-indigo-500">
        <PreprocessingPlacementGuide />
      </Tabs.Content>
      <Tabs.Content value="split-merge" className="outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-indigo-500">
        <SplitMergeGuide />
      </Tabs.Content>
      </div>
    </Tabs.Root>}
  </ModalShell>
);
