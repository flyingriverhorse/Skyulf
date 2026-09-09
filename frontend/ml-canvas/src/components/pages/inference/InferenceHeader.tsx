import {
    AlertCircle,
    Box,
    CheckCircle,
    Power,
    Zap
} from 'lucide-react';
import { Link } from 'react-router-dom';
import type { InferenceController } from './useInferenceController';

/** Show the current deployment and its registry or undeploy action. */
export function DeploymentHeader({ controller }: { controller: Pick<InferenceController, 'activeDeployment' | 'handleDeactivate'> }) {
    const { activeDeployment, handleDeactivate } = controller;
    return (<header className="shrink-0 mb-4 flex flex-wrap items-center justify-between gap-x-4 gap-y-2">
        <h1 className="text-xl sm:text-2xl font-semibold text-gray-800 dark:text-gray-100 flex items-center gap-2">
            <Zap className="w-6 h-6 text-blue-500 shrink-0" />
            Testing Model Inference
        </h1>
        {activeDeployment ? (
            <div className="flex items-center gap-2 flex-wrap text-xs min-w-0">
                <span className="inline-flex items-center gap-1.5 px-2 py-1 rounded-full bg-green-50 dark:bg-green-900/30 text-green-700 dark:text-green-400 border border-green-200 dark:border-green-800 font-medium">
                    <CheckCircle className="w-3.5 h-3.5 shrink-0" /> Active
                </span>
                <span className="inline-flex items-center gap-1.5 px-2 py-1 rounded-full bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 border border-gray-200 dark:border-gray-700 min-w-0">
                    <Box className="w-3.5 h-3.5 text-blue-500 shrink-0" />
                    <span className="truncate max-w-[10rem]">{activeDeployment.model_type}</span>
                </span>
                <span
                    className="hidden sm:inline-flex items-center px-2 py-1 rounded-full bg-white dark:bg-gray-800 text-gray-500 dark:text-gray-400 border border-gray-200 dark:border-gray-700 font-mono min-w-0"
                    title={`Job ${activeDeployment.job_id} · deployed ${new Date(activeDeployment.created_at).toLocaleString()}`}
                >
                    <span className="truncate max-w-[8rem]">{activeDeployment.job_id}</span>
                </span>
                <button
                    onClick={() => void handleDeactivate()}
                    className="inline-flex items-center gap-1 px-2 py-1 rounded-full bg-red-50 text-red-600 border border-red-200 hover:bg-red-100 dark:bg-red-900/30 dark:text-red-400 dark:border-red-800 dark:hover:bg-red-900/50 transition-colors"
                    title="Undeploy model"
                >
                    <Power className="w-3.5 h-3.5" /> Undeploy
                </button>
            </div>
        ) : (
            <div className="flex items-center gap-2 flex-wrap text-xs">
                <span className="inline-flex items-center gap-1.5 text-gray-500 dark:text-gray-400 italic">
                    <AlertCircle className="w-4 h-4 shrink-0" />
                    No model is currently deployed.
                </span>
                <Link
                    to="/registry"
                    className="inline-flex items-center gap-1.5 px-2 py-1 rounded-full bg-blue-50 text-blue-600 border border-blue-200 hover:bg-blue-100 dark:bg-blue-900/30 dark:text-blue-400 dark:border-blue-800 dark:hover:bg-blue-900/50 transition-colors"
                >
                    <Box className="w-3.5 h-3.5" /> Browse Model Registry
                </Link>
            </div>
        )}
    </header>);
}
