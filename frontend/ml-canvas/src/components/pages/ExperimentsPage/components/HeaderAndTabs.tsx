import React from 'react';

export type ExperimentsView = 'charts' | 'table' | 'evaluation' | 'importance' | 'shap' | 'diff' | 'segmentation';

interface HeaderProps {
  datasets: { id: string; name: string }[];
  selectedDatasetId: string;
  setSelectedDatasetId: (v: string) => void;
  filterType: 'all' | 'classification' | 'regression' | 'text_classification' | 'segmentation' | 'ensemble';
  setFilterType: (v: 'all' | 'classification' | 'regression' | 'text_classification' | 'segmentation' | 'ensemble') => void;
}

export const ExperimentsHeader: React.FC<HeaderProps> = ({
  datasets,
  selectedDatasetId,
  setSelectedDatasetId,
  filterType,
  setFilterType,
}) => (
  <div className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 p-4 flex justify-between items-center">
    <div>
      <h1 className="text-xl font-semibold text-gray-800 dark:text-gray-100">Experiments & Comparison</h1>
      <p className="text-sm text-gray-500 dark:text-gray-400">Compare metrics and parameters across multiple runs</p>
    </div>
    <div className="flex gap-2">
      <select
        className="bg-gray-50 dark:bg-gray-900 border border-gray-300 dark:border-gray-600 text-gray-900 dark:text-gray-100 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block p-2.5"
        value={selectedDatasetId}
        onChange={(e) => { setSelectedDatasetId(e.target.value); }}
      >
        <option value="all">All Datasets</option>
        {datasets.map(ds => (
          <option key={ds.id} value={ds.id}>{ds.name}</option>
        ))}
      </select>
      <select
        className="bg-gray-50 dark:bg-gray-900 border border-gray-300 dark:border-gray-600 text-gray-900 dark:text-gray-100 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block p-2.5"
        value={filterType}
        onChange={(e) => { setFilterType(e.target.value as 'all' | 'classification' | 'regression' | 'text_classification' | 'segmentation' | 'ensemble'); }}
      >
        <option value="all">All Experiments</option>
        <option value="classification">Classification</option>
        <option value="regression">Regression</option>
        <option value="text_classification">Text Classification</option>
        <option value="segmentation">Segmentation</option>
        <option value="ensemble">Ensemble</option>
      </select>
    </div>
  </div>
);

interface TabsProps {
  activeView: ExperimentsView;
  setActiveView: (v: ExperimentsView) => void;
  hasFeatureImportances: boolean;
  hasShapSummary: boolean;
  hasSegmentation: boolean;
}

const TAB_BASE = 'px-4 py-2 text-sm font-medium border-b-2 transition-colors';
const tabClass = (active: boolean) => `${TAB_BASE} ${
  active
    ? 'border-blue-500 text-blue-600 dark:text-blue-400'
    : 'border-transparent text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200'
}`;

export const ViewTabs: React.FC<TabsProps> = ({ activeView, setActiveView, hasFeatureImportances, hasShapSummary, hasSegmentation }) => (
  <div className="flex border-b border-gray-200 dark:border-gray-700">
    <button className={tabClass(activeView === 'charts')} onClick={() => { setActiveView('charts'); }}>
      Visual Comparison
    </button>
    <button className={tabClass(activeView === 'table')} onClick={() => { setActiveView('table'); }}>
      Detailed Metrics & Params
    </button>
    <button className={tabClass(activeView === 'evaluation')} onClick={() => { setActiveView('evaluation'); }}>
      Model Evaluation
    </button>
    <button
      className={tabClass(activeView === 'diff')}
      onClick={() => { setActiveView('diff'); }}
      data-testid="experiments-tab-diff"
    >
      Pipeline Diff
    </button>
    {hasFeatureImportances && (
      <button className={tabClass(activeView === 'importance')} onClick={() => { setActiveView('importance'); }}>
        Feature Importance
      </button>
    )}
    {hasShapSummary && (
      <button className={tabClass(activeView === 'shap')} onClick={() => { setActiveView('shap'); }}>
        SHAP Explainability
      </button>
    )}
    {hasSegmentation && (
      <button
        className={tabClass(activeView === 'segmentation')}
        onClick={() => { setActiveView('segmentation'); }}
        data-testid="experiments-tab-segmentation"
      >
        Segmentation
      </button>
    )}
  </div>
);
