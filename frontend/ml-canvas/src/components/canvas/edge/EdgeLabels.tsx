import { EdgeLabelRenderer } from '@xyflow/react';
import { X } from 'lucide-react';
import { ConnectionHoverCard } from '../ConnectionHoverCard';
import { LeakageIssuePopover } from '../LeakageIssuePopover';
import { useCanvasLeakageFeedback } from '../../../core/contexts/CanvasLeakageContext';
import type { EdgePoint } from './edgeGeometry';
import type { EdgeBranch } from './edgeAppearance';
import type { EdgeInteraction } from './useEdgeInteraction';

interface EndpointLabels { sourceLabel: string; targetLabel: string }

/** Keep winner context above the branch label and warning controls. */
function MergeWinnerBadge({ branch, deletePoint, badgeY }: {
  branch: EdgeBranch; deletePoint: EdgePoint; badgeY: number;
}) {
  const { isMergeWinner, branchLabel } = branch;
  if (!isMergeWinner) return null;
  return (
    <div
      style={{
        position: 'absolute',
        transform: `translate(-50%, -100%) translate(${deletePoint.x}px,${badgeY - (branchLabel ? 40 : 16)}px)`,
        fontSize: 9,
        fontWeight: 700,
        letterSpacing: '0.06em',
        color: '#f59e0b',
        backgroundColor: 'hsl(var(--background) / 0.95)',
        border: '1px solid #f59e0b80',
        borderRadius: 4,
        padding: '1px 6px',
        pointerEvents: 'none',
        whiteSpace: 'nowrap',
        lineHeight: '14px',
        textTransform: 'uppercase',
      }}
      title="This branch won the merge tiebreak on overlapping columns."
    >
      Wins merge
    </div>
  );
}

/** Show a colored branch label only when both text and color are available. */
function BranchBadge({ branch, deletePoint, badgeY }: {
  branch: EdgeBranch; deletePoint: EdgePoint; badgeY: number;
}) {
  const { branchLabel, branchColor } = branch;
  if (!branchLabel || !branchColor) return null;
  return (
    <div
      style={{
        position: 'absolute',
        transform: `translate(-50%, -100%) translate(${deletePoint.x}px,${badgeY - 16}px)`,
        fontSize: 10,
        fontWeight: 600,
        letterSpacing: '0.02em',
        color: branchColor,
        backgroundColor: 'hsl(var(--background) / 0.9)',
        border: `1px solid ${branchColor}50`,
        borderRadius: 6,
        padding: '2px 8px',
        pointerEvents: 'none',
        whiteSpace: 'nowrap',
        lineHeight: '16px',
        boxShadow: `0 0 6px ${branchColor}30`,
      }}
    >
      {branchLabel}
    </div>
  );
}

/** Removal stays keyboard accessible even while hover controls are invisible. */
function RemoveConnectionControl({ interaction, labels, deletePoint }: {
  interaction: EdgeInteraction; labels: EndpointLabels; deletePoint: EdgePoint;
}) {
  const { canDelete, showControls, showTooltip, tooltipId, enter, leave, setButtonFocused, onEdgeClick } = interaction;
  const { sourceLabel, targetLabel } = labels;
  if (!canDelete) return null;
  return (
    <div
      style={{
        position: 'absolute',
        transform: `translate(-50%, -50%) translate(${deletePoint.x}px,${deletePoint.y}px)`,
        fontSize: 12,
        pointerEvents: showControls ? 'all' : 'none',
      }}
      className="nodrag nopan"
      onMouseEnter={enter}
      onMouseLeave={leave}
    >
      <button
        type="button"
        className="w-6 h-6 bg-background border border-border text-muted-foreground rounded-full flex items-center justify-center hover:bg-destructive hover:text-destructive-foreground transition-colors shadow-sm focus-ring"
        style={{ opacity: showControls ? 1 : 0 }}
        onFocus={() => setButtonFocused(true)}
        onBlur={() => setButtonFocused(false)}
        onClick={onEdgeClick}
        title="Remove Connection"
        aria-label={`Remove connection from ${sourceLabel} to ${targetLabel}`}
        aria-describedby={showTooltip ? tooltipId : undefined}
      >
        <X size={10} />
      </button>
    </div>
  );
}

/** Labels share the edge portal and measured coordinates without changing SVG geometry. */
export function EdgeLabels({ id, interaction, branch, labels, deletePoint, warningPoint }: {
  id: string; interaction: EdgeInteraction; branch: EdgeBranch; labels: EndpointLabels;
  deletePoint: EdgePoint; warningPoint: EdgePoint;
}) {
  const { edgeIssues, openGuide } = useCanvasLeakageFeedback();
  const leakageIssues = edgeIssues[id] ?? [];
  const { showTooltip, tooltipId, enter, leave } = interaction;
  const { sourceLabel, targetLabel } = labels;
  const badgeY = leakageIssues.length > 0 ? Math.min(deletePoint.y, warningPoint.y) : deletePoint.y;
  return <EdgeLabelRenderer>
    {showTooltip && <ConnectionHoverCard id={tooltipId} x={deletePoint.x} y={deletePoint.y}
      sourceLabel={sourceLabel} targetLabel={targetLabel} onEnter={enter} onLeave={leave} />}
    <MergeWinnerBadge branch={branch} deletePoint={deletePoint} badgeY={badgeY} />
    <BranchBadge branch={branch} deletePoint={deletePoint} badgeY={badgeY} />
    <RemoveConnectionControl interaction={interaction} labels={labels} deletePoint={deletePoint} />
    {leakageIssues.length > 0 && <div
      style={{
        position: 'absolute',
        transform: `translate(-50%, -50%) translate(${warningPoint.x}px,${warningPoint.y}px)`,
        pointerEvents: 'all',
      }}
      className="nodrag nopan"
    >
      <LeakageIssuePopover issues={leakageIssues} subject={`${sourceLabel} to ${targetLabel}`} openGuide={openGuide} />
    </div>}
  </EdgeLabelRenderer>;
}
