const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const { execFileSync } = require('node:child_process');
const ts = require('../frontend/ml-canvas/node_modules/typescript');
const path = 'frontend/ml-canvas/src/components/panels/jobs/jobDetails/JobLogs.tsx';
function load(source) {
  const block = source.slice(source.indexOf('const LOG_HIGHLIGHT_RULES'), source.indexOf('const LogMessageContent'));
  const js = ts.transpileModule(block, { compilerOptions: { target: ts.ScriptTarget.ES2020 } }).outputText;
  return vm.runInNewContext(js + '; ({ rules: LOG_HIGHLIGHT_RULES.slice(9, 12), tokenize: tokenizeLogMessage })');
}
const old = load(execFileSync('git', ['show', 'HEAD:' + path], { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] }));
const current = load(fs.readFileSync(path, 'utf8'));
const names = ['duration', 'percentage', 'float'];
if (process.argv[2] === 'worker') {
  const version = process.argv[3] === 'old' ? old : current;
  const rule = version.rules[Number(process.argv[4])];
  const input = '7'.repeat(Number(process.argv[5])) + '!';
  for (let i = 0; i < 100; i++) rule.re.exec('123!');
  const start = performance.now();
  assert.equal(rule.re.exec(input), null);
  process.stdout.write((performance.now() - start).toFixed(3));
} else {
  for (const version of ['old', 'current']) {
    for (const [i, name] of names.entries()) {
      for (const length of [2000, 4000, 8000, 16000]) {
        let timing;
        try {
          timing = execFileSync(process.execPath, [__filename, 'worker', version, String(i), String(length)], { encoding: 'utf8', timeout: 5000, stdio: ['ignore', 'pipe', 'ignore'] });
        } catch (error) {
          timing = error.code === 'ETIMEDOUT' ? '>5000 (timeout)' : String(error);
        }
        console.log(`${version} ${name} ${length} digits: ${timing} ms`);
      }
    }
  }
  let seed = 12345;
  const random = n => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed % n; };
  const pieces = ['1', '123', '-', '+', '.', 'e', 'E', 'ms', 's', '%', ' ', '\t', '\n', '_', 'x', '=', '"', '\u00a0', '\u2028', '\u0661'];
  const normalize = match => match ? [match.index, match[0]] : null;
  for (let n = 0; n < 100000; n++) {
    let input = '';
    const count = random(25);
    for (let i = 0; i < count; i++) input += pieces[random(pieces.length)];
    for (let i = 0; i < 3; i++) assert.equal(JSON.stringify(normalize(old.rules[i].re.exec(input))), JSON.stringify(normalize(current.rules[i].re.exec(input))), `${names[i]}: ${JSON.stringify(input)}`);
    assert.equal(JSON.stringify(old.tokenize(input)), JSON.stringify(current.tokenize(input)), `tokenize: ${JSON.stringify(input)}`);
  }
  console.log('PASS: 100000 differential messages, all three matches and full tokenizer segments identical.');
}
