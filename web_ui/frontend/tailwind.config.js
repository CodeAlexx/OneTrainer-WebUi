/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        dark: {
          bg: '#000000',
          surface: '#000000',
          card: '#0a0a0a',
          border: '#1a1a1a',
          hover: '#111111',
        },
        primary: {
          DEFAULT: '#3d3d3d',
          hover: '#4a4a4a',
          light: '#5a5a5a',
        },
        accent: {
          DEFAULT: '#3d3d3d',
          purple: '#8b5cf6',
          cyan: '#22d3ee',
        },
        success: '#3fb950',
        warning: '#d29922',
        danger: '#f85149',
        muted: '#8b949e',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'Consolas', 'monospace'],
      },
    },
  },
  plugins: [],
}
