import { useCallback, useState } from 'react';

type UseSidepanelToggleResult = {
  isSidepanelExpanded: boolean;
  handleToggleSidepanel: () => void;
};

export const useSidepanelToggle = (initiallyExpanded = true): UseSidepanelToggleResult => {
  const [isSidepanelExpanded, setIsSidepanelExpanded] = useState(initiallyExpanded);

  const handleToggleSidepanel = useCallback(() => {
    setIsSidepanelExpanded((previous) => !previous);
  }, []);

  return {
    isSidepanelExpanded,
    handleToggleSidepanel,
  };
};
