import React, { useState } from 'react';
import type { DriftAlertDetail, DriftDispositionAction } from '../../core/api/monitoring';
import { ErrorState, LoadingState, ModalShell } from '../../components/shared';
import { AlertIdentity } from './alertDetail/AlertIdentity';
import { AlertEvidence } from './alertDetail/AlertEvidence';
import { AlertDisposition } from './alertDetail/AlertDisposition';
import { prepareEvidenceRows } from './alertDetail/evidence';

interface DriftAlertModalProps {
    alertId: number | null;
    detail: DriftAlertDetail | null;
    loading: boolean;
    error: string | null;
    actionPending: boolean;
    onApplyDisposition: (
        action: DriftDispositionAction,
        actor: string,
        note?: string,
    ) => Promise<unknown>;
    onRetry: () => void;
    onClose: () => void;
    filters: Record<string, string>;
}

/**
 * OPS-003 investigation surface for a single drift alert: identity, severity,
 * the threshold version it was evaluated against, per-feature evidence, links
 * to the related job/model version/deployment, and the acknowledge / resolve
 * / reopen disposition workflow with its full actor/timestamp audit trail.
 */
export const DriftAlertModal: React.FC<DriftAlertModalProps> = ({
    alertId,
    detail,
    loading,
    error,
    actionPending,
    onApplyDisposition,
    onRetry,
    onClose,
    filters,
}) => {
    const [actor, setActor] = useState('');
    const [note, setNote] = useState('');
    const [actionError, setActionError] = useState<string | null>(null);

    const handleAction = async (action: DriftDispositionAction) => {
        if (!actor.trim()) {
            setActionError('Enter your name so the disposition records who made it.');
            return;
        }
        setActionError(null);
        const result = await onApplyDisposition(action, actor.trim(), note.trim() || undefined);
        if (result) setNote('');
    };

    const evidenceRows = prepareEvidenceRows(detail);

    return (
        <ModalShell
            isOpen={alertId !== null}
            onClose={onClose}
            title={alertId !== null ? `Drift alert #${alertId}` : undefined}
            size="3xl"
        >
            {loading && <LoadingState message="Loading drift alert…" />}
            {!loading && error && !detail && <ErrorState error={error} onRetry={onRetry} />}
            {!loading && detail && (
                <div className="p-6 space-y-5">
                    <AlertIdentity detail={detail} filters={filters} />
                    <AlertEvidence
                        alertId={alertId}
                        evaluationStatus={detail.evaluation_status}
                        evidenceRows={evidenceRows}
                    />
                    <AlertDisposition
                        detail={detail}
                        actor={actor}
                        note={note}
                        actionError={actionError}
                        error={error}
                        actionPending={actionPending}
                        setActor={setActor}
                        setNote={setNote}
                        handleAction={handleAction}
                    />
                </div>
            )}
        </ModalShell>
    );
};
