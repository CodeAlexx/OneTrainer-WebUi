/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        dark: {
          bg: '#0f0f0f',
          surface: '#1a1a1a',
          border: '#2a2a2a',
          hover: '#333333',
        },
        primary: {
          DEFAULT: '#f97316',
          hover: '#ea580c',
        },
        success: '#22c55e',
        danger: '#ef4444',
        muted: '#71717a',
      },
    },
  },
  plugins: [],
};
