/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: "#003153",      // IIT Bombay Prussian Blue
        secondary: "#A6A6A6",    // IIT Bombay Silver
        dark: "#1e293b",         // Soft text color
        darker: "#f8fafc",       // Light background
        panel: "#ffffff",        // White panels
      },
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
      }
    },
  },
  plugins: [],
}
