import React from 'react';

type NodeDetailsSectionProps = {
  metadata: Array<{ label: string; value: string }>;
};

export const NodeDetailsSection: React.FC<NodeDetailsSectionProps> = ({ metadata }) => {
  return (
    <section className="canvas-modal__section">
      <h3>Node details</h3>
      {metadata.length ? (
        <dl className="canvas-modal__metadata">
          {metadata.map((entry, index) => (
            <div key={`${entry.label}-${index}`} className="canvas-modal__metadata-row">
              <dt>{entry.label}</dt>
              <dd>{entry.value}</dd>
            </div>
          ))}
        </dl>
      ) : (
        <p className="canvas-modal__empty">No metadata available for this node.</p>
      )}
    </section>
  );
};
