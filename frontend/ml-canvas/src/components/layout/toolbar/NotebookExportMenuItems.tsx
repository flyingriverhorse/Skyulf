import React from 'react';
import { BookOpen } from 'lucide-react';

interface NotebookExportMenuItemsProps {
  currentDatasetId: string | null | undefined;
  onExportNotebook: (mode: 'compact' | 'full') => void | Promise<void>;
  /** Called before dispatching the export so the caller can close its own open menu (compact overflow vs. desktop export dropdown use separate state). */
  onBeforeSelect: () => void;
}

/**
 * "Notebook (compact)" / "Notebook (full)" menu items, gated behind having a
 * bound dataset. Previously duplicated verbatim between the Toolbar's
 * compact overflow menu and its desktop export dropdown.
 */
export const NotebookExportMenuItems: React.FC<NotebookExportMenuItemsProps> = ({
  currentDatasetId,
  onExportNotebook,
  onBeforeSelect,
}) => {
  if (!currentDatasetId) return null;

  return (
    <>
      <div className="border-t my-1" />
      <button
        role="menuitem"
        onClick={() => { onBeforeSelect(); void onExportNotebook('compact'); }}
        className="w-full flex items-center gap-2 text-left px-3 py-2 text-sm hover:bg-accent"
      >
        <BookOpen className="w-4 h-4" /> Notebook (compact)
      </button>
      <button
        role="menuitem"
        onClick={() => { onBeforeSelect(); void onExportNotebook('full'); }}
        className="w-full flex items-center gap-2 text-left px-3 py-2 text-sm hover:bg-accent"
      >
        <BookOpen className="w-4 h-4" /> Notebook (full)
      </button>
    </>
  );
};
