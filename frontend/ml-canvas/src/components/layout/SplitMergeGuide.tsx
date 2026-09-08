import { ArrowRight } from 'lucide-react';

const sectionClass = 'p-5 space-y-3 border-b border-slate-100 dark:border-slate-800 last:border-b-0';
const headingClass = 'font-semibold text-slate-900 dark:text-slate-100';
const boxClass = 'rounded-lg border border-slate-200 p-3 dark:border-slate-700';
const stepClass = 'rounded border border-slate-300 bg-white p-2 dark:border-slate-600 dark:bg-slate-900';

/** Visual guide to row boundaries, branch merges and the checks that apply to them. */
export const SplitMergeGuide = () => (
  <div className="text-sm text-slate-600 dark:text-slate-300">
    <section className={sectionClass}>
      <h3 className={headingClass}>Two split nodes, two different jobs</h3>
      <div className="grid gap-3 sm:grid-cols-2">
        <figure className={boxClass}>
          <figcaption className={headingClass}>Train-Test Split</figcaption>
          <p className="mt-1 text-xs">Divides rows into evaluation partitions.</p>
          <div className="mt-3 space-y-1 text-center text-xs">
            <div className="rounded bg-slate-100 p-2 dark:bg-slate-800">All rows: features + target</div>
            <ArrowRight className="mx-auto h-4 w-4 rotate-90 text-slate-400" aria-hidden="true" />
            <div className="flex justify-between gap-2 rounded bg-emerald-50 px-2 py-1.5 text-emerald-800 dark:bg-emerald-900/30 dark:text-emerald-200">
              <span>Train rows</span><span>X + y</span>
            </div>
            <div className="flex justify-between gap-2 rounded bg-slate-100 px-2 py-1.5 dark:bg-slate-800">
              <span>Validation (optional)</span><span>X + y</span>
            </div>
            <div className="flex justify-between gap-2 rounded bg-sky-50 px-2 py-1.5 text-sky-800 dark:bg-sky-900/30 dark:text-sky-200">
              <span>Test rows</span><span>X + y</span>
            </div>
          </div>
          <p className="mt-3 text-xs font-medium">Creates the row boundary. Legacy alias: Split.</p>
        </figure>
        <figure className={boxClass}>
          <figcaption className={headingClass}>Feature-Target Split</figcaption>
          <p className="mt-1 text-xs">Separates columns; keeps the same rows.</p>
          <div className="mt-3 space-y-1 text-center text-xs">
            <div className="rounded bg-slate-100 p-2 dark:bg-slate-800">All rows: features + target</div>
            <ArrowRight className="mx-auto h-4 w-4 rotate-90 text-slate-400" aria-hidden="true" />
            <div className="grid grid-cols-2 gap-1">
              <div className="rounded bg-indigo-50 p-3 text-indigo-800 dark:bg-indigo-900/30 dark:text-indigo-200">
                <strong className="block">Features (X)</strong><span className="mt-2 block">All input rows</span>
              </div>
              <div className="rounded bg-amber-50 p-3 text-amber-800 dark:bg-amber-900/30 dark:text-amber-200">
                <strong className="block">Target (y)</strong><span className="mt-2 block">All input rows</span>
              </div>
            </div>
          </div>
          <p className="mt-3 text-xs font-medium">No rows are held out. Add Train-Test Split.</p>
        </figure>
      </div>
    </section>

    <section className={sectionClass}>
      <h3 className={headingClass}>Where branch outputs meet</h3>
      <p>
        Connecting several wires to the same node merges their columns automatically.
        There is no separate Merge node. Every branch must keep the same rows in the same order.
      </p>
      <figure className="rounded-lg bg-slate-50 p-3 dark:bg-slate-800/50">
        <figcaption className={`mb-3 ${headingClass}`}>Calculate fixed features, then split the rows once</figcaption>
        <div className="grid grid-cols-[minmax(0,1fr)_auto_minmax(0,1.4fr)_auto_minmax(0,1fr)] items-center gap-2 text-center text-xs">
          <div className={stepClass}>One loader</div>
          <ArrowRight className="h-3 w-3" aria-hidden="true" />
          <div className="space-y-2">
            <div className={stepClass}>Date Features<span className="mt-1 block text-slate-500 dark:text-slate-400">month</span></div>
            <div className={stepClass}>Geo Distance<span className="mt-1 block text-slate-500 dark:text-slate-400">distance_km</span></div>
          </div>
          <ArrowRight className="h-3 w-3" aria-hidden="true" />
          <div className={stepClass}>One Train-Test Split</div>
        </div>
        <p className="mt-3 text-xs">
          Both fixed, per-row branches feed the <strong>same</strong> Train-Test Split.
          It merges their columns, then creates the row partitions. This is the first actual
          row split, not Feature-Target Split or the first node on the canvas.
        </p>
        <p className="mt-2 text-xs">
          Put learned steps such as Standard Scaler downstream, followed by the model.
          This shape does not combine independently split datasets or permit learned transforms
          before the split; the downstream chain must support fold reconstruction.
        </p>
      </figure>
      <figure className="rounded-lg bg-slate-50 p-3 dark:bg-slate-800/50">
        <figcaption className={`mb-3 ${headingClass}`}>Split once, transform on branches, connect both directly to one model</figcaption>
        <div className="grid grid-cols-[minmax(0,1fr)_auto_minmax(0,1.4fr)_auto_minmax(0,1fr)] items-center gap-2 text-center text-xs">
          <div className={stepClass}>One Train-Test Split</div>
          <ArrowRight className="h-3 w-3" aria-hidden="true" />
          <div className="space-y-2">
            <div className={stepClass}>One-Hot Encoder<span className="mt-1 block text-slate-500 dark:text-slate-400">city</span></div>
            <div className={stepClass}>Standard Scaler<span className="mt-1 block text-slate-500 dark:text-slate-400">age</span></div>
          </div>
          <ArrowRight className="h-3 w-3" aria-hidden="true" />
          <div className={stepClass}>One model</div>
        </div>
        <div className="mt-3 grid grid-cols-2 gap-2 text-center text-xs">
          <div className="rounded bg-emerald-50 p-2 text-emerald-800 dark:bg-emerald-900/30 dark:text-emerald-200">Train A + Train B<br />One merged train partition</div>
          <div className="rounded bg-sky-50 p-2 text-sky-800 dark:bg-sky-900/30 dark:text-sky-200">Test A + Test B<br />One merged test partition</div>
        </div>
        <p className="mt-3 text-xs">
          Both branches receive the same split. They fit on training rows and apply their fitted
          state to held-out rows. The model merges features within each partition; training and
          test rows stay separate. Validation is merged separately too, when present.
        </p>
        <p className="mt-2 text-xs">
          Keep branches linear and row-aligned, with no extra splitter, nested join or disallowed
          row-changing step. Join directly into the model. Prefer distinct feature outputs:
          passthrough columns can overlap and replace a sibling&apos;s transformed values.
        </p>
      </figure>
      <p className="text-xs">
        For sequential transformations, use a linear chain such as Train-Test Split, Standard
        Scaler, then Model. Merging two branches does not apply their transformations in sequence.
      </p>
    </section>

    <section className={sectionClass}>
      <h3 className={headingClass}>First wins or last wins: which value survives?</h3>
      <p>
        Different-named features are kept from both branches. For conflicting versions of the
        same column, Merge Strategy chooses the winning version. This example uses the same row
        in both branches, and both branches have changed age.
      </p>
      <figure className={boxClass}>
        <figcaption className="mb-3 text-xs font-medium">Example: two versions of age, two distinct additional features</figcaption>
        <div className="grid grid-cols-2 gap-3 text-xs">
          <div className="rounded bg-slate-100 p-3 dark:bg-slate-800">
            <strong className="block">Branch A: first input</strong>
            <div className="mt-2 font-mono">age = 20<br />income = 900</div>
          </div>
          <div className="rounded bg-slate-100 p-3 dark:bg-slate-800">
            <strong className="block">Branch B: last input</strong>
            <div className="mt-2 font-mono">age = 0.4<br />city_code = 2</div>
          </div>
        </div>
        <ArrowRight className="mx-auto my-2 h-4 w-4 rotate-90 text-slate-400" aria-hidden="true" />
        <div className="grid grid-cols-2 gap-3 text-xs">
          <div className="rounded bg-indigo-50 p-3 text-indigo-800 dark:bg-indigo-900/30 dark:text-indigo-200">
            <strong className="block">First wins</strong>
            <div className="mt-2 font-mono"><strong>age = 20</strong><br />income = 900<br />city_code = 2</div>
          </div>
          <div className="rounded bg-amber-50 p-3 text-amber-800 dark:bg-amber-900/30 dark:text-amber-200">
            <strong className="block">Last wins (default)</strong>
            <div className="mt-2 font-mono"><strong>age = 0.4</strong><br />income = 900<br />city_code = 2</div>
          </div>
        </div>
      </figure>
      <div className="rounded bg-slate-50 p-3 text-xs dark:bg-slate-800/50">
        <strong className={headingClass}>Ownership exception</strong>
        <p className="mt-1">
          Shared ancestor shows only A changed age <span aria-hidden="true">&rarr;</span>{' '}
          A&apos;s age wins under either strategy. First/last resolves differing modifications
          or overlap where no ownership baseline is available.
        </p>
      </div>
      <p className="text-xs">
        For sibling branches, first/last means saved incoming-edge order, not top/bottom position
        or completion time. Ancestor inputs are ordered before descendant inputs.
      </p>
      <p className="text-xs">
        The strategy also applies after Train-Test Split, separately to Train, Validation and
        Test. A shared-ancestor ownership baseline is generally unavailable there, so even a
        passthrough branch can overwrite an encoded or scaled column.
      </p>
      <p className="text-xs">
        For X/y inputs, y comes from the first branch; the feature merge strategy does not choose
        the target. Branches must carry the same aligned target. First/last wins does not repair
        mismatched rows or prevent leakage.
      </p>
    </section>

    <section className={sectionClass}>
      <h3 className={headingClass}>Empty columns and target-only operations</h3>
      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs">
          <caption className="mb-2 text-left">An empty selection has different meanings across nodes.</caption>
          <thead className="text-slate-500 dark:text-slate-400">
            <tr><th scope="col" className="pb-2 pr-3">Configuration</th><th scope="col" className="pb-2">Result</th></tr>
          </thead>
          <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
            <tr><th scope="row" className="py-2 pr-3 font-medium">Missing Indicator: omitted / null / []</th><td className="py-2">Discover missing columns; learned.</td></tr>
            <tr><th scope="row" className="py-2 pr-3 font-medium">Custom Binning: explicit columns or []</th><td className="py-2">Fixed selection and configured edges.</td></tr>
            <tr><th scope="row" className="py-2 pr-3 font-medium">Label Encoder: default or []</th><td className="py-2">Target-only or no-op.</td></tr>
            <tr><th scope="row" className="py-2 pr-3 font-medium">Ordinal Encoder: omitted / null vs []</th><td className="py-2">Learn feature categories vs target-only/no-op.</td></tr>
            <tr><th scope="row" className="py-2 pr-3 font-medium">Text vectorizers / tokenizer / embedder</th><td className="py-2">Need explicit nonempty columns; otherwise no-op.</td></tr>
            <tr><th scope="row" className="py-2 pr-3 font-medium">Text Cleaning: omitted / null</th><td className="py-2">Discover text columns; fixed cleaning.</td></tr>
          </tbody>
        </table>
      </div>
      <p className="text-xs">
        Target-only modes need the correct target context. Supported scalers, imputers, binning,
        One-Hot/Dummy/Target/WOE encoders, Power Transformer and IQR/Z-Score/Winsorize/Elliptic
        Envelope treat [] as a no-op. Selectors and resamplers have no general empty-list exemption.
        Use the Preprocessing &amp; Leakage catalog for each node&apos;s rules.
      </p>
    </section>

    <section className={sectionClass}>
      <h3 className={headingClass}>What the audit tiles establish</h3>
      <div className="grid gap-2 text-xs sm:grid-cols-2">
        <div className={boxClass}><strong className={headingClass}>Leakage Gate</strong><p className="mt-1">Checks detected placement violations.</p></div>
        <div className={boxClass}><strong className={headingClass}>Fold Refit Audit</strong><p className="mt-1">Checks preprocessing reconstruction inside folds.</p></div>
      </div>
      <div className="rounded bg-amber-50 p-3 text-xs text-amber-900 dark:bg-amber-900/20 dark:text-amber-200">
        <p className="font-medium">Unsupported reconstruction + learned preprocessing</p>
        <div className="mt-2 grid grid-cols-[auto_1fr] gap-x-3 gap-y-1">
          <strong>raise (default)</strong><span>Execution stops.</span>
          <strong>warn / ignore</strong><span>Fallback may produce optimistic CV scores.</span>
        </div>
      </div>
      <p className="text-xs">
        Green tiles do not prove feature provenance, entity independence or temporal correctness.
        Core-only CV: fit a fresh pipeline on each fold&apos;s raw training rows.
      </p>
    </section>
  </div>
);
