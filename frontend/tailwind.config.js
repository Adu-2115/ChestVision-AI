/** @type {import('tailwindcss').Config} */
// Colour, type and shape tokens: light shell, one deep-teal accent, dark image viewer.
// Contrast ratios for these values are recorded in the project design notes.
module.exports = {
  content: ["./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      colors: {
        app: '#F3F6F5',
        card: '#FFFFFF',
        line: '#DCE5E2',        // decorative card edges only
        control: '#5F7A73',     // borders of inputs, dropzone, outline buttons
        ink: { DEFAULT: '#17231F', 2: '#43554F', muted: '#4A5B56' },
        accent: { DEFAULT: '#0F6B63', hover: '#0A4D47', tint: '#DDEEEA' },
        flag: { bg: '#FBE6DF', text: '#8A2B14', bar: '#B8431F' },
        caution: { bg: '#FDF1DC', text: '#7A4A06', bar: '#B7791F', chip: '#8A4B08' },
        neutral: { bg: '#E6EEEC', text: '#35504A', bar: '#5C7C74' },
        viewer: { bg: '#0F1619', panel: '#1A2327', text: '#E6EEEC', muted: '#9FB1B7', focus: '#7FD1C7' },
      },
      fontFamily: {
        sans: ['"Instrument Sans"', 'system-ui', '-apple-system', '"Segoe UI"', 'Roboto', 'sans-serif'],
        serif: ['Newsreader', 'Georgia', 'serif'],
        mono: ['"JetBrains Mono"', 'ui-monospace', 'Consolas', 'monospace'],
      },
      borderRadius: { card: '16px', control: '10px' },
    },
  },
  plugins: [],
}
