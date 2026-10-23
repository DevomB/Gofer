// Tailwind build for web/site.css — run scripts/build-site.sh.
module.exports = {
  content: { relative: true, files: ['./index.html', './site.js'] },
  theme: {
    extend: {
      colors: {
        bg: '#07070D',
        surface: '#0D0F19',
        surface2: '#151827',
        line: 'rgba(139,124,246,0.14)',
        ink: { DEFAULT: '#EAEAF4', dim: '#9DA2BC', mute: '#636880' },
        violet: { DEFAULT: '#8B7CF6', bright: '#A99CFF', deep: '#5B4FC0' },
        gold: { DEFAULT: '#D6A85C', bright: '#E8C285', deep: '#8A6A38' },
      },
      fontFamily: {
        display: ['Fraunces', 'Georgia', 'serif'],
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'ui-monospace', 'monospace'],
      },
    },
  },
};
