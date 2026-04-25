module.exports = {
  root: true,
  env: { browser: true, es2020: true },
  extends: [
    'eslint:recommended',
    'plugin:@typescript-eslint/recommended',
    'plugin:react/recommended',
    'plugin:react/jsx-runtime',
    'plugin:react-hooks/recommended',
    'plugin:jsx-a11y/recommended',
  ],
  ignorePatterns: ['dist', '.eslintrc.cjs'],
  parser: '@typescript-eslint/parser',
  plugins: ['react-refresh'],
  settings: {
    react: { version: 'detect' },
  },
  rules: {
    'react-refresh/only-export-components': 'off',
    '@typescript-eslint/no-explicit-any': 'off',
    '@typescript-eslint/no-unused-vars': ['error', { argsIgnorePattern: '^_', varsIgnorePattern: '^_' }],
    // Phase C #29 guardrails — promote to errors so the next remount-loop bug fails CI.
    'react/no-unstable-nested-components': ['error', { allowAsProps: true }],
    'react-hooks/exhaustive-deps': 'error',
    // React 17+ JSX transform: prop-types are unused in this codebase (we use TS).
    'react/prop-types': 'off',
    // A11y bulk rules stay as warnings until the #25 sweep lands.
    'jsx-a11y/click-events-have-key-events': 'warn',
    'jsx-a11y/no-static-element-interactions': 'warn',
    'jsx-a11y/label-has-associated-control': 'warn',
    'jsx-a11y/no-autofocus': 'warn',
    'jsx-a11y/no-noninteractive-tabindex': 'warn',
    'react/no-unescaped-entities': 'warn',
  },
}
