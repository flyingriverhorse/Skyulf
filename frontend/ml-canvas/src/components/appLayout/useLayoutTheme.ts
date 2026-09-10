import { useEffect, useState } from 'react';
import { applyTheme } from '../../core/theme/applyTheme';

/** Keep the shell theme synchronized with its pre-paint value and user toggles. */
export function useLayoutTheme() {
  const [isDarkMode, setIsDarkMode] = useState<boolean>(() => {
    if (typeof document === 'undefined') return false;
    return document.documentElement.classList.contains('dark');
  });

  useEffect(() => {
    // Inline script in index.html already applied the right class before mount;
    // this effect only keeps state in sync if something else mutated the class.
    setIsDarkMode(document.documentElement.classList.contains('dark'));
  }, []);

  const toggleTheme = () => {
    const newMode = !isDarkMode;
    setIsDarkMode(newMode);
    applyTheme(newMode);
  };

  return { isDarkMode, toggleTheme };
}
