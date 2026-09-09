import React from 'react';
import { useInferenceController } from './inference/useInferenceController';
import { DeploymentHeader } from './inference/InferenceHeader';
import { InferenceInputPanel } from './inference/InferenceInputPanel';
import { InferenceResultsPanel } from './inference/InferenceResultsPanel';

export { useSavedThresholdInfo } from './inference/useSavedThresholdInfo';

export const InferencePage: React.FC = () => {
    const controller = useInferenceController();
    return (
        <div className="h-full flex flex-col bg-gray-50 dark:bg-gray-900 p-4 sm:p-6 overflow-y-auto lg:overflow-hidden">
            {/* Deployment status lives in the header rather than a half-height card:
                it's a one-line fact, and the space it used to occupy is what the
                results panel needs to stay readable on short/narrow viewports. */}
            <DeploymentHeader controller={controller} />

            {/* Only the desktop two-column layout is height-constrained. Stacked on
                narrower viewports the panels keep a usable minimum height and the
                page scrolls instead of compressing both to a few pixels. */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 sm:gap-6 lg:flex-1 lg:min-h-0">
                {/* Left column: input editor */}
                <InferenceInputPanel controller={controller} />

                {/* Right column: results (deployment status now lives in the page header) */}
                <InferenceResultsPanel controller={controller} />
            </div>
        </div>
    );
};
