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
          // Rare Primary Accent: Smoked Juniper Pine (no blue, no purple)
          accent: "rgb(var(--color-accent-rgb) / <alpha-value>)",
          accentHover: "rgb(var(--color-accent-hover-rgb) / <alpha-value>)",
          accentSoft: "rgb(var(--color-accent-soft-rgb) / <alpha-value>)",
          stone: "rgb(var(--color-muted-rgb) / <alpha-value>)",
          clay: "var(--border-subtle)",
          // Rare Clinical Urgency Accent: Raw Terracotta
          warning: "rgb(var(--color-alert-rgb) / <alpha-value>)",
          danger: "rgb(var(--color-alert-rgb) / <alpha-value>)",
          success: "rgb(var(--color-accent-rgb) / <alpha-value>)",
        }
      },
      boxShadow: {
        clinical: "var(--shadow-panel)",
        hairline: "0 1px 2px 0 rgba(0, 0, 0, 0.04)",
        insetline: "inset 0 0 0 1px var(--border-subtle)"
      },
      borderRadius: {
        clinical: "6px"
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
        ],
        mono: [
          "ui-monospace",
          "SFMono-Regular",
          "Menlo",
          "Monaco",
          "Consolas",
          "Liberation Mono",
          "Courier New",
          "monospace"
        ]
      }
    }
  },
  plugins: []
} satisfies Config;
