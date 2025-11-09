import React from 'react';
import type { MissingIndicatorInsights } from './missingIndicatorSettings';

type MissingIndicatorInsightsSectionProps = {
  suffix: string;
  insights: MissingIndicatorInsights;
  formatMissingPercentage: (value: number | null) => string;
};

export const MissingIndicatorInsightsSection: React.FC<MissingIndicatorInsightsSectionProps> = ({
  suffix,
  insights,
  formatMissingPercentage,
}) => {
  const { rows, flaggedColumnsInDataset, conflictCount } = insights;
  const hasRows = rows.length > 0;
  const suffixDisplay = suffix || '(no suffix)';
  const flaggedPreview = flaggedColumnsInDataset.slice(0, 4);
  const remainingFlagged = Math.max(flaggedColumnsInDataset.length - flaggedPreview.length, 0);

  return (
    <section className="canvas-modal__section">
      <div className="canvas-modal__section-header">
        <h3>Missing indicator summary</h3>
      </div>
      <p className="canvas-modal__note">
        Generates <strong>{rows.length}</strong> flag column{rows.length === 1 ? '' : 's'} using suffix{' '}
        <strong>{suffixDisplay}</strong>.
      </p>
      {flaggedColumnsInDataset.length > 0 && (
        <p className="canvas-modal__note">
          Detected {flaggedColumnsInDataset.length} existing flag column{flaggedColumnsInDataset.length === 1 ? '' : 's'}{' '}
          with this suffix
          {flaggedPreview.length > 0
            ? ` (examples: ${flaggedPreview.join(', ')}${remainingFlagged > 0 ? ', …' : ''})`
            : ''}.
        </p>
      )}
      {conflictCount > 0 && (
        <p className="canvas-modal__note canvas-modal__note--warning">
          {conflictCount} generated flag column{conflictCount === 1 ? ' has' : 's have'} naming conflicts. Adjust the suffix or remove conflicting columns.
        </p>
      )}
      {!hasRows ? (
        <p className="canvas-modal__note">Select columns below to create missing-value flags.</p>
      ) : (
        <div className="canvas-cast__table-wrapper">
          <table className="canvas-cast__table">
            <thead>
              <tr>
                <th scope="col">Column</th>
                <th scope="col">Missing %</th>
                <th scope="col">Flag column</th>
                <th scope="col">Status</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => {
                const statusParts: string[] = [];
                if (row.conflicts.alreadyExists) {
                  statusParts.push('Already exists');
                }
                if (row.conflicts.duplicateFlag) {
                  statusParts.push('Duplicate name');
                }
                const hasConflict = statusParts.length > 0;
                const statusLabel = hasConflict ? statusParts.join(' · ') : 'Ready';
                const chipClass = hasConflict
                  ? 'canvas-cast__chip canvas-cast__chip--attention'
                  : 'canvas-cast__chip canvas-cast__chip--applied';
                const rowClass = hasConflict ? 'canvas-cast__row canvas-cast__row--attention' : 'canvas-cast__row';
                return (
                  <tr key={`missing-indicator-${row.column}`} className={rowClass}>
                    <th scope="row">{row.column}</th>
                    <td>{formatMissingPercentage(row.missingPercentage)}</td>
                    <td>{row.flagColumn}</td>
                    <td>
                      <span className={chipClass}>{statusLabel}</span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
};
