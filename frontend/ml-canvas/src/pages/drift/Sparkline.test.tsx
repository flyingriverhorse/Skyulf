import { render, screen } from '@testing-library/react';
import { expect, it } from 'vitest';
import { Sparkline } from './Sparkline';

it('leaves gaps at missing measurements and uses the effective PSI threshold', () => {
    // A line across absent checks invents a trend; a fixed threshold invents a verdict.
    const { container, rerender } = render(<Sparkline values={[0.1, 0.2, null, 0.48]} threshold={0.5} />);
    expect(container.querySelector('polyline')?.getAttribute('points')?.split(' ')).toHaveLength(2);
    expect(container.querySelector('circle:last-child')).toHaveAttribute('fill', '#22c55e');
    expect(screen.getByRole('img')).toHaveAccessibleName(/1 missing.*latest PSI 0.48.*threshold 0.5/i);
    rerender(<Sparkline values={[0.1, 0.2, null, 0.48]} threshold={0.3} />);
    expect(container.querySelector('circle:last-child')).toHaveAttribute('fill', '#ef4444');
});

it('marks a missing latest value as unavailable instead of coloring an older value', () => {
    // An old observation must not become today's reassuring green endpoint.
    const { container } = render(<Sparkline values={[0.2, 0.3, null]} threshold={0.5} />);
    expect(screen.getByRole('img')).toHaveAccessibleName(/latest PSI unavailable/i);
    expect(container.querySelector('circle[cx="64"]')).toBeNull();
});

it('distinguishes a falling trend from an endpoint still above threshold', () => {
    // Improving PSI can still require action at the currently selected threshold.
    const { container } = render(<Sparkline values={[0.8, 0.6, 0.48]} threshold={0.2} />);
    expect(container.querySelector('circle:last-child')).toHaveAttribute('fill', '#ef4444');
    expect(container.querySelector('polyline')).toHaveAttribute('stroke', '#3b82f6');
});
