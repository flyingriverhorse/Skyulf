const path = require('node:path');
const fs = require('node:fs');
const root = path.resolve(__dirname, '..');
const cwd = path.join(root, 'frontend/ml-canvas');
const { ESLint } = require(path.join(cwd, 'node_modules/eslint'));
const files = [
  'src/components/pages/ExperimentsPage/components/comparisonTable/trainingConfig.ts',
  'src/components/pages/experiments/PipelineDiffView.test.tsx',
  'src/core/utils/operationalContext.ts',
  'src/core/utils/operationalContext/recordParsers.ts',
];
(async () => {
  const eslint = new ESLint({ cwd, overrideConfig: { rules: { complexity: ['warn', 0] } } });
  const result = await eslint.lintFiles(files);
  const lines = result.flatMap(file => {
    const metrics = file.messages.filter(item => item.ruleId === 'complexity').map(item => ({
      line: item.line, column: item.column, label: item.message,
      ccn: Number(item.message.match(/complexity of (\d+)/)[1]),
    }));
    return [path.relative(cwd, file.filePath),
      `functions=${metrics.length}, sum=${metrics.reduce((sum, item) => sum + item.ccn, 0)}, max=${Math.max(...metrics.map(item => item.ccn))}`,
      ...metrics.map(item => `  ${item.line}:${item.column} ${item.label}`)];
  });
  const output = lines.join('\n') + '\n';
  fs.writeFileSync(path.join(__dirname, `security-review-complexity-${process.argv[2] || 'current'}.txt`), output);
  console.log(output);
})();
