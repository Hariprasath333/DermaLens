import { useEffect, useState } from "react";
import type { ThemeMode } from "../types/lesioniq";

export function useTheme() {
  const [theme, setTheme] = useState<ThemeMode>(() => {
    const stored = window.localStorage.getItem("lesioniq-theme");
    return stored === "light" || stored === "dark" ? stored : "dark";
  });

  useEffect(() => {
    document.documentElement.classList.toggle("dark", theme === "dark");
    document.documentElement.dataset.theme = theme;
    window.localStorage.setItem("lesioniq-theme", theme);
  }, [theme]);

  return {
    theme,
    setTheme,
    toggleTheme: () => setTheme((current) => (current === "dark" ? "light" : "dark"))
  };
}

