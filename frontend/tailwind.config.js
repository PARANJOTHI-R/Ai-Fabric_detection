/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Light theme palette
        background: '#F5F7FA',
        surface: '#FFFFFF',
        'surface-2': '#F0F4F8',
        border: '#E2E8F0',
        primary: '#2563EB',
        'primary-light': '#EFF6FF',
        secondary: '#64748B',
        muted: '#94A3B8',
        dark: '#1E293B',
        charcoal: '#1E293B',
        danger: '#DC2626',
        'danger-light': '#FEF2F2',
        success: '#16A34A',
        'success-light': '#F0FDF4',
        warning: '#D97706',
        'warning-light': '#FFFBEB',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
      boxShadow: {
        card: '0 1px 3px 0 rgb(0 0 0 / 0.08), 0 1px 2px -1px rgb(0 0 0 / 0.08)',
        'card-lg': '0 4px 6px -1px rgb(0 0 0 / 0.07), 0 2px 4px -2px rgb(0 0 0 / 0.07)',
        focus: '0 0 0 3px rgb(37 99 235 / 0.15)',
      },
      animation: {
        'fade-in': 'fadeIn 0.3s ease-out',
        'slide-up': 'slideUp 0.3s ease-out',
        'pulse-soft': 'pulseSoft 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      },
      keyframes: {
        fadeIn: { from: { opacity: '0' }, to: { opacity: '1' } },
        slideUp: { from: { opacity: '0', transform: 'translateY(8px)' }, to: { opacity: '1', transform: 'translateY(0)' } },
        pulseSoft: { '0%, 100%': { opacity: '1' }, '50%': { opacity: '.6' } },
      }
    },
  },
  plugins: [],
}
