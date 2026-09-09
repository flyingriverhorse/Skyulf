import type { FC } from 'react';
import { useEdaPageController } from './edaPage/useEdaPageController';
import { EdaPageHeader, RecentTargets } from './edaPage/EdaPageHeader';
import { EdaPageFeedback, EdaHistoryModal } from './edaPage/EdaPageFeedback';
import { EdaReportContent } from './edaPage/EdaReportContent';

export const EDAPage: FC = () => {
  const model = useEdaPageController();
  return (
    <div className="flex flex-col h-full w-full overflow-hidden bg-white dark:bg-slate-950">
      <EdaPageHeader {...model} />
      <RecentTargets {...model} />
      <EdaPageFeedback {...model} />
      <div className="flex-1 overflow-hidden relative">
        <EdaReportContent {...model} />
      </div>
      <EdaHistoryModal {...model} />
    </div>
  );
};
