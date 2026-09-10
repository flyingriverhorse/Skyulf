import { useEffect, useId, useRef, useState, type MouseEvent } from 'react';
import { useReactFlow, type EdgeProps } from '@xyflow/react';
import { getReadOnlyMode } from '../../../core/hooks/useReadOnlyMode';

/** Keep SVG hover and detached label controls active across the same grace period. */
function useControlVisibility(props: EdgeProps) {
  const [hovered, setHovered] = useState(false);
  const [buttonFocused, setButtonFocused] = useState(false);
  const hoverTimer = useRef<ReturnType<typeof setTimeout>>();
  const enter = () => { clearTimeout(hoverTimer.current); setHovered(true); };
  const leave = () => { hoverTimer.current = setTimeout(() => setHovered(false), 150); };
  useEffect(() => () => clearTimeout(hoverTimer.current), []);
  const focused = props.data?.isFocused === true;
  const showTooltip = hovered || buttonFocused || focused;
  const showControls = hovered || buttonFocused || props.selected || focused;
  return { enter, leave, setButtonFocused, showTooltip, showControls };
}

/** Read mutation permission again at click time, including after a render. */
export function useEdgeInteraction(props: EdgeProps) {
  const { deleteElements } = useReactFlow();
  const visibility = useControlVisibility(props);
  const tooltipId = useId();
  const canDelete = props.deletable && props.data?.readOnly !== true;
  const onEdgeClick = (event: MouseEvent) => {
    event.stopPropagation();
    if (!canDelete || getReadOnlyMode()) return;
    // Keep keyboard navigation in the canvas after its delete button disappears.
    event.currentTarget.closest('.react-flow')?.parentElement?.focus({ preventScroll: true });
    void deleteElements({ edges: [{ id: props.id }] });
  };
  return { ...visibility, tooltipId, canDelete, onEdgeClick };
}

export type EdgeInteraction = ReturnType<typeof useEdgeInteraction>;
