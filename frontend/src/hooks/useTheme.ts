import { useEffect, useState } from 'react';

const THEME_KEY = 'pss-theme';

export function useTheme() {
  const [theme, setTheme] = useState<'light' | 'dark'>(() => {
    // Check localStorage first
    const stored = localStorage.getItem(THEME_KEY);
    if (stored === 'dark' || stored === 'light') {
      return stored;
    }
    
    // Check system preference
    if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
      return 'dark';
    }
    
    return 'light';
  });

  useEffect(() => {
    const root = document.documentElement;
    if (theme === 'dark') {
      root.classList.add('dark');
    } else {
      root.classList.remove('dark');
    }
    localStorage.setItem(THEME_KEY, theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme((prev) => (prev === 'dark' ? 'light' : 'dark'));
  };

  return { theme, toggleTheme, setTheme };
}

