import React from 'react';
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

interface HelpGuideModalProps {
  isOpen: boolean;
  onClose: () => void;
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
export const HelpGuideModal: React.FC<HelpGuideModalProps> = ({ isOpen, onClose }) => (
  <ModalShell isOpen={isOpen} onClose={onClose} title="How pipelines work" size="2xl">
    <div>
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
          A node with two or more inputs joins branches column-wise. Columns only one branch
          produced are all kept. When two branches produce a column with the{' '}
          <span className="font-medium">same name</span>, the merge node&apos;s{' '}
          <span className="font-medium">Merge Strategy</span> decides whose values survive:
          &quot;last wins&quot; (the default — the last connected branch) or &quot;first
          wins&quot;. After a run, the winning edge is highlighted with a WINS MERGE label.
        </p>
      </Section>

      <Section icon={<Split className="w-3.5 h-3.5" />} title="After a Split node — order decides">
        <p>
          After a Split node the platform can&apos;t track which branch owns which columns, so
          the merge strategy no longer applies: overlapping columns resolve purely by merge
          order — the <span className="font-medium">last connected branch wins every shared
          column</span>. Keep branches after a Split disjoint (each emits different columns) or
          fully numeric, otherwise an earlier branch&apos;s encoding can be silently discarded.
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
          <span className="font-medium">Run Preview</span> (Ctrl+Enter) executes the whole
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
          <span className="font-medium">Leakage Gate</span> confirms the held-out test rows
          never touched the fitting process. The{' '}
          <span className="font-medium">Fold Refit Audit</span> confirms preprocessing
          statistics were refit inside every cross-validation fold &mdash; so nothing learned
          from rows it shouldn&apos;t have seen. Green verdicts on both mean the numbers are
          honest; anything else links to a detail view explaining what happened.
        </p>
      </Section>

      <Section icon={<AlertTriangle className="w-3.5 h-3.5" />} title="Score Advisory — the amber tile in Jobs">
        <p>
          If a job&apos;s details show an amber{' '}
          <span className="font-medium">Score Advisory</span> tile, the graph shape didn&apos;t
          allow per-fold preprocessing refit, so scores were computed on pre-transformed data
          and may be optimistically biased. The usual fix is the divide-and-merge shape above:
          put encoders on branches between the split and the merge.
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
    </div>
  </ModalShell>
);
