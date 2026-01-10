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
        brandAccent: '#667eea',
        brandAccentAlt: '#764ba2',
        brandDark: '#0c1228',
        brandDarker: '#080f1f',
        brandSurface: '#101734',
        brandSurfaceLight: '#f8f9ff',
      },
    },
  },
  plugins: [],
}

