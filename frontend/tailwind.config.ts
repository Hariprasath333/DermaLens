import type { Config } from "tailwindcss";

export default {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        clinical: {
          ink: "rgb(var(--color-ink-rgb) / <alpha-value>)",
          muted: "rgb(var(--color-muted-rgb) / <alpha-value>)",
          line: "var(--border-subtle)",
          canvas: "rgb(var(--color-canvas-rgb) / <alpha-value>)",
          surface: "var(--surface)",
          raised: "var(--surface-strong)",
          soft: "var(--surface-soft)",
          accent: "rgb(var(--color-accent-rgb) / <alpha-value>)",
          accentHover: "rgb(var(--color-accent-hover-rgb) / <alpha-value>)",
          accentSoft: "rgb(var(--color-accent-soft-rgb) / <alpha-value>)",
          stone: "rgb(var(--color-stone-rgb) / <alpha-value>)",
          clay: "rgb(var(--color-clay-rgb) / <alpha-value>)",
          warning: "rgb(var(--color-warning-rgb) / <alpha-value>)",
          danger: "rgb(var(--color-danger-rgb) / <alpha-value>)",
          success: "rgb(var(--color-success-rgb) / <alpha-value>)"
        }
      },
      boxShadow: {
        clinical: "var(--shadow-panel)",
        insetline: "inset 0 0 0 1px var(--border-subtle)"
      },
      borderRadius: {
        clinical: "16px"
      },
      fontFamily: {
        sans: [
          "Inter",
          "ui-sans-serif",
          "system-ui",
          "-apple-system",
          "BlinkMacSystemFont",
          "Segoe UI",
          "sans-serif"
        ]
      }
    }
  },
  plugins: []
} satisfies Config;
