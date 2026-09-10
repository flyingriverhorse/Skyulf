import { useState } from 'react';
import { JobInfo, LeakageGateVerdict, RefitAuditVerdict, FoldRefitFallbackCode } from '../../../../core/api/jobs';
import { ModalShell } from '../../../shared';

// Plain-language reasons for the Score Advisory modal, keyed by the stable
// `fold_refit_fallback` codes the engine stamps into the job metrics.
const FALLBACK_REASON_TEXT: Record<FoldRefitFallbackCode, string> = {
  nested_merge:
    'A branch contains its own merge node. Per-fold refit supports exactly one merge straight into the training node.',
  fork_not_splitter:
    'The branches split off at a point that is not a train/test splitter, so the pre-transform rows cannot be reconstructed per fold.',
  learner_before_split:
    'A data-dependent step runs before the last train/test split, so it already learned from held-out rows in the full run — re-fitting it per fold would apply its statistics twice.',
  row_changing_branch_step:
    'A branch step filters or splits rows, so the branches no longer align row-for-row and cannot be re-fit inside each fold.',
  unsupported_graph:
    'The upstream graph is not a shape per-fold refit supports (a linear chain, or one merge of transformer branches after a splitter).',
  payload_reconstruction_failed:
    'Reconstructing the pre-transform rows failed unexpectedly; the run continued on the already-transformed data. Check the job log for the traceback.',
};

export function useJobSafety(job: JobInfo) {
  const [gateModalOpen, setGateModalOpen] = useState(false);
  const [auditModalOpen, setAuditModalOpen] = useState(false);
  const [advisoryModalOpen, setAdvisoryModalOpen] = useState(false);
  // The engine stamps the leakage gate's verdict into metrics at run time;
  // legacy jobs predate the stamp, so the tile is omitted for them.
  const leakageGate = (
    (job.result as Record<string, unknown> | null)?.metrics as Record<string, unknown> | undefined
  )?.leakage_gate as LeakageGateVerdict | undefined;

  // Per-fold refit audit (findings 2026-08-26 §3/B): stamped only when
  // per-fold preprocessing refit was active, so fallback/legacy runs omit it.
  const refitAudit = (
    (job.result as Record<string, unknown> | null)?.metrics as Record<string, unknown> | undefined
  )?.fold_refit_audit as RefitAuditVerdict | undefined;

  // Fallback demand telemetry: stamped only when the run fell back to
  // pre-transformed scoring, so covered/legacy runs omit the tile.
  const refitFallback = (
    (job.result as Record<string, unknown> | null)?.metrics as Record<string, unknown> | undefined
  )?.fold_refit_fallback as FoldRefitFallbackCode | undefined;
  return {
    leakageGate, refitAudit, refitFallback, gateModalOpen, setGateModalOpen,
    auditModalOpen, setAuditModalOpen, advisoryModalOpen, setAdvisoryModalOpen
  };
}

export type JobSafety = ReturnType<typeof useJobSafety>;
export function LeakageGateTile({ safety }: { safety: JobSafety }) {
  const { leakageGate, setGateModalOpen } = safety;
  return (
    <>
      {leakageGate && (
        <button
          type="button"
          onClick={() => setGateModalOpen(true)}
          className="p-4 bg-gray-50 dark:bg-gray-900/50 rounded-lg border border-gray-100 dark:border-gray-700 text-left hover:border-gray-300 dark:hover:border-gray-500 transition-colors"
          title="Show what the leakage gate checked for this run"
        >
          <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">Leakage Gate</div>
          <div className={`font-medium ${leakageGate.status === 'passed'
            ? 'text-green-600 dark:text-green-400'
            : leakageGate.status === 'no_split'
              ? 'text-amber-600 dark:text-amber-400'
              : 'text-red-600 dark:text-red-400'
            }`}>
            {leakageGate.status === 'passed' ? 'Passed' : leakageGate.status === 'no_split' ? 'No split' : 'Warnings'}
          </div>
        </button>
      )}
    </>
  );
}

export function RefitAuditTile({ safety }: { safety: JobSafety }) {
  const { refitAudit, setAuditModalOpen } = safety;
  return (
    <>
      {refitAudit && (
        <button
          type="button"
          onClick={() => setAuditModalOpen(true)}
          className="p-4 bg-gray-50 dark:bg-gray-900/50 rounded-lg border border-gray-100 dark:border-gray-700 text-left hover:border-gray-300 dark:hover:border-gray-500 transition-colors"
          title="Show the per-fold preprocessing audit for this run"
        >
          <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">Fold Refit Audit</div>
          <div className={`font-medium ${refitAudit.isolation_ok === false
            ? 'text-amber-600 dark:text-amber-400'
            : 'text-green-600 dark:text-green-400'
            }`}>
            {refitAudit.isolation_ok === false
              ? 'Isolation warning'
              : refitAudit.train_rows !== undefined
                ? `Isolation verified (${refitAudit.max_fit_rows}/${refitAudit.train_rows})`
                : 'Isolation verified'}
          </div>
        </button>
      )}
    </>
  );
}

export function ScoreAdvisoryTile({ safety }: { safety: JobSafety }) {
  const { refitFallback, setAdvisoryModalOpen } = safety;
  return (
    <>
      {refitFallback && (
        <button
          type="button"
          onClick={() => setAdvisoryModalOpen(true)}
          className="p-4 bg-amber-50 dark:bg-amber-900/10 rounded-lg border border-amber-200 dark:border-amber-800 text-left hover:border-amber-400 dark:hover:border-amber-600 transition-colors"
          title="This run fell back to pre-transformed scoring — show why"
        >
          <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">Score Advisory</div>
          <div className="font-medium text-amber-600 dark:text-amber-400">Scores may be optimistic</div>
        </button>
      )}
    </>
  );
}

function LeakageGateModal({ safety }: { safety: JobSafety }) {
  const { leakageGate, gateModalOpen, setGateModalOpen } = safety;
  return (
    <>
      {leakageGate && (
        <ModalShell
          isOpen={gateModalOpen}
          onClose={() => setGateModalOpen(false)}
          title="Leakage Gate"
          size="lg"
        >
          <div className="p-6 space-y-5 text-sm">
            <GateSummary leakageGate={leakageGate} />

            <GateCheckedNodes leakageGate={leakageGate} />

            {leakageGate.exempted && leakageGate.exempted.length > 0 && (
              <div>
                <h4 className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-2">
                  Allowed before the split (learns nothing in this configuration)
                </h4>
                <ul className="space-y-2">
                  {leakageGate.exempted.map((e) => (
                    <li
                      key={e.node_id}
                      className="p-3 bg-gray-50 dark:bg-gray-900/50 rounded-lg border border-gray-100 dark:border-gray-700"
                    >
                      <span className="font-medium text-gray-800 dark:text-gray-200">{e.step_type}</span>
                      <span className="ml-2 font-mono text-xs text-gray-400">{e.node_id}</span>
                      <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">{e.reason}</p>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {leakageGate.splitters && leakageGate.splitters.length > 0 && (
              <p className="text-xs text-gray-500 dark:text-gray-400">
                Split boundary:{' '}
                {leakageGate.splitters.map((id) => (
                  <code key={id} className="font-mono bg-gray-100 dark:bg-gray-800 px-1 py-0.5 rounded mr-1">{id}</code>
                ))}
              </p>
            )}

            {leakageGate.messages.length > 0 && (
              <div>
                <h4 className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-2">
                  Gate messages
                </h4>
                <ul className="space-y-1">
                  {leakageGate.messages.map((m, i) => (
                    <li key={i} className="text-xs text-gray-600 dark:text-gray-400 whitespace-pre-wrap">{m}</li>
                  ))}
                </ul>
              </div>
            )}

            <GateLegacyNote leakageGate={leakageGate} />
          </div>
        </ModalShell>
      )}
    </>
  );
}

function RefitAuditModal({ safety }: { safety: JobSafety }) {
  const { refitAudit, auditModalOpen, setAuditModalOpen } = safety;
  return (
    <>
      {refitAudit && (
        <ModalShell
          isOpen={auditModalOpen}
          onClose={() => setAuditModalOpen(false)}
          title="Fold Refit Audit"
          size="lg"
        >
          <div className="p-6 space-y-5 text-sm">
            <p className="text-gray-700 dark:text-gray-300">
              {refitAudit.isolation_ok === false
                ? 'A preprocessing fit saw more rows than the train split contains — held-out (validation/test) rows may have entered a fit, so this run\'s CV/tuning scores may be optimistic. Check the job logs for the full audit line.'
                : 'Preprocessing was re-fit inside every CV/tuning fold, and every fit received at most the train-split row count — no held-out row ever entered a preprocessing fit, so the CV/tuning scores are honest estimates.'}
            </p>

            <div>
              <h4 className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-2">
                Measured during this run
              </h4>
              <ul className="space-y-2">
                <li className="flex items-center justify-between gap-3 p-3 bg-gray-50 dark:bg-gray-900/50 rounded-lg border border-gray-100 dark:border-gray-700">
                  <span className="text-gray-700 dark:text-gray-300">Preprocessing fits (one per fold)</span>
                  <span className="font-mono text-xs text-gray-800 dark:text-gray-200">{refitAudit.fit_calls} call(s)</span>
                </li>
                <li className="flex items-center justify-between gap-3 p-3 bg-gray-50 dark:bg-gray-900/50 rounded-lg border border-gray-100 dark:border-gray-700">
                  <span className="text-gray-700 dark:text-gray-300">Largest rows seen by any fit</span>
                  <span className="font-mono text-xs text-gray-800 dark:text-gray-200">
                    {refitAudit.max_fit_rows}
                    {refitAudit.train_rows !== undefined ? ` of ${refitAudit.train_rows} train rows` : ''}
                  </span>
                </li>
                <li className="flex items-center justify-between gap-3 p-3 bg-gray-50 dark:bg-gray-900/50 rounded-lg border border-gray-100 dark:border-gray-700">
                  <span className="text-gray-700 dark:text-gray-300">Transforms (held-out folds)</span>
                  <span className="font-mono text-xs text-gray-800 dark:text-gray-200">{refitAudit.transform_calls} call(s)</span>
                </li>
                <li className="flex items-center justify-between gap-3 p-3 bg-gray-50 dark:bg-gray-900/50 rounded-lg border border-gray-100 dark:border-gray-700">
                  <span className="text-gray-700 dark:text-gray-300">Isolation verdict</span>
                  <span className={`text-xs px-2 py-0.5 rounded-full border whitespace-nowrap ${refitAudit.isolation_ok === false
                    ? 'bg-amber-50 dark:bg-amber-900/20 text-amber-600 dark:text-amber-400 border-amber-200 dark:border-amber-800'
                    : 'bg-green-50 dark:bg-green-900/20 text-green-600 dark:text-green-400 border-green-200 dark:border-green-800'
                    }`}>
                    {refitAudit.isolation_ok === false ? 'held-out rows may have entered a fit' : 'no held-out rows in any fit'}
                  </span>
                </li>
              </ul>
            </div>

            <p className="text-xs text-gray-500 dark:text-gray-400">
              During cross-validation and hyperparameter tuning, preprocessing (imputation,
              scaling, encoding) is re-fit on each fold&apos;s training rows only. A leaked fit
              would receive the train split plus held-out rows — a larger count than shown
              here — and would make the reported scores optimistically biased.
            </p>
          </div>
        </ModalShell>
      )}
    </>
  );
}

function ScoreAdvisoryModal({ safety }: { safety: JobSafety }) {
  const { refitFallback, advisoryModalOpen, setAdvisoryModalOpen } = safety;
  return (
    <>
      {refitFallback && (
        <ModalShell
          isOpen={advisoryModalOpen}
          onClose={() => setAdvisoryModalOpen(false)}
          title="Score Advisory"
          size="lg"
        >
          <div className="p-6 space-y-5 text-sm">
            <p className="text-gray-700 dark:text-gray-300">
              This pipeline&apos;s shape does not support per-fold preprocessing refit, so
              cross-validation and tuning scored candidates on data that was transformed
              using the whole dataset. The reported scores may therefore be optimistically
              biased — treat them as upper bounds, not honest estimates.
            </p>

            <div>
              <h4 className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-2">
                Why this run fell back
              </h4>
              <div className="p-3 bg-amber-50 dark:bg-amber-900/10 rounded-lg border border-amber-200 dark:border-amber-800">
                <span className="text-xs font-mono text-amber-600 dark:text-amber-400">{refitFallback}</span>
                <p className="mt-1 text-gray-700 dark:text-gray-300">
                  {FALLBACK_REASON_TEXT[refitFallback] ??
                    'The upstream graph is not a shape per-fold refit supports.'}
                </p>
              </div>
            </div>

            <p className="text-xs text-gray-500 dark:text-gray-400">
              The model itself was still trained and evaluated normally, and the job log
              carries the full warning line. Restructuring the pipeline into a linear chain
              or a fork-join of transformer branches after the Split node re-enables
              leakage-free per-fold scoring.
            </p>
          </div>
        </ModalShell>
      )}
    </>
  );
}

export function JobSafetyModals({ safety }: { safety: JobSafety }) {
  return (
    <>
      <LeakageGateModal safety={safety} />
      <RefitAuditModal safety={safety} />
      <ScoreAdvisoryModal safety={safety} />
    </>
  );
}

function GateSummary({ leakageGate }: { leakageGate: LeakageGateVerdict }) {
  return (
    <>
      <p className="text-gray-700 dark:text-gray-300">
        {leakageGate.status === 'passed' &&
          'Every node that learns from data was checked against the train/test split — none of them fits on data that still contains test rows, so the evaluation is uncontaminated.'}
        {leakageGate.status === 'no_split' &&
          'This pipeline has no train/test split, so the leakage guarantee does not apply: every fit saw the whole dataset.'}
        {leakageGate.status === 'warnings' &&
          'Nodes that learn from data were found fitting before the train/test split — their statistics saw test rows.'}
      </p>
    </>
  );
}

function GateCheckedNodes({ leakageGate }: { leakageGate: LeakageGateVerdict }) {
  return (
    <>
      {leakageGate.checked && leakageGate.checked.length > 0 && (
        <div>
          <h4 className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-2">
            Data-learning nodes checked
          </h4>
          <ul className="space-y-2">
            {leakageGate.checked.map((c) => (
              <li
                key={c.node_id}
                className="flex items-center justify-between gap-3 p-3 bg-gray-50 dark:bg-gray-900/50 rounded-lg border border-gray-100 dark:border-gray-700"
              >
                <div>
                  <span className="font-medium text-gray-800 dark:text-gray-200">{c.step_type}</span>
                  <span className="ml-2 font-mono text-xs text-gray-400">{c.node_id}</span>
                </div>
                <span className={`text-xs px-2 py-0.5 rounded-full border whitespace-nowrap ${c.violation
                  ? 'bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400 border-red-200 dark:border-red-800'
                  : 'bg-green-50 dark:bg-green-900/20 text-green-600 dark:text-green-400 border-green-200 dark:border-green-800'
                  }`}>
                  {c.violation ? 'fits before the split' : 'runs after the split'}
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </>
  );
}

function GateLegacyNote({ leakageGate }: { leakageGate: LeakageGateVerdict }) {
  return (
    <>
      {!leakageGate.checked && !leakageGate.exempted && !leakageGate.splitters && (
        <p className="text-xs text-gray-400 dark:text-gray-500 italic">
          This job ran before detailed gate reporting was added, so only the verdict and its messages are available.
        </p>
      )}
    </>
  );
}
