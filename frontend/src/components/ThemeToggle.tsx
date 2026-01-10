import { Moon, Sun } from 'lucide-react';
import { useTheme } from '@/hooks/useTheme';

export function ThemeToggle() {
  const { theme, toggleTheme } = useTheme();

  return (
    <button
      onClick={toggleTheme}
      className="fixed bottom-6 right-6 w-16 h-16 rounded-full bg-brandDark text-white dark:bg-white dark:text-brandDark shadow-[0_20px_35px_rgba(0,0,0,0.35)] flex items-center justify-center transition-all duration-200 hover:scale-105 focus-visible:outline focus-visible:outline-4 focus-visible:outline-brandAccent/60 z-50"
      aria-label={theme === 'dark' ? 'Activate light mode' : 'Activate dark mode'}
      type="button"
    >
      {theme === 'dark' ? (
        <Sun className="w-6 h-6" />
      ) : (
        <Moon className="w-6 h-6" />
      )}
    </button>
  );
}

