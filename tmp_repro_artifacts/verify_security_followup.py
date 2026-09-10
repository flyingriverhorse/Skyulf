"""Check the final scanner follow-up scope, inventory, docs and built imports."""
from pathlib import Path
import re
import subprocess

root = Path(__file__).resolve().parents[1]

def git(*args):
    return subprocess.check_output(['git', '-c', 'core.safecrlf=false', *args], cwd=root, stderr=subprocess.DEVNULL).decode('utf-8')

expected_text = {
    'frontend/ml-canvas/src/components/panels/jobs/jobDetails/JobLogs.tsx',
    'frontend/ml-canvas/src/components/panels/jobs/jobDetails/JobLogs.test.tsx',
    'frontend/ml-canvas/e2e/layout-jobs-segmentation.spec.ts',
    'frontend/ml-canvas/src/components/pages/ExperimentsPage/components/comparisonTable/trainingConfig.ts',
    'frontend/ml-canvas/src/components/pages/ExperimentsPage/components/comparisonTable/trainingConfig.test.ts',
    'frontend/ml-canvas/src/components/pages/experiments/PipelineDiffView.test.tsx',
    'frontend/ml-canvas/src/core/utils/operationalContext.ts',
    'frontend/ml-canvas/src/core/utils/operationalContext.test.ts',
    'frontend/ml-canvas/src/core/utils/operationalContext/recordParsers.ts',
    'initiatives/analysis/frontend_ccn_remaining_2026-09-10.md',
    'initiatives/analysis/frontend_static_analysis_review_2026-09-10.md',
    'initiatives/analysis/opus_core_analysis-open_queue.md',
    'initiatives/analysis/opus_core_analysis-tracker.md', 'changelog/0.8.x.md',
}
changed = set(git('diff', '--name-only').splitlines()) | set(git('ls-files', '--others', '--exclude-standard').splitlines())
scoped = {name for name in changed if not name.startswith('tmp_repro_artifacts/')}
assert {name for name in scoped if not name.startswith('static/ml_canvas/')} == expected_text
assert all(name == 'static/ml_canvas/index.html' or name.startswith('static/ml_canvas/assets/') for name in scoped - expected_text)
for name in expected_text:
    contents = (root / name).read_text(encoding='utf-8')
    assert contents.endswith('\n') and not contents.endswith('\n\n'), name
    assert all(line == line.rstrip() for line in contents.splitlines()), name
old = git('show', 'HEAD:changelog/0.8.x.md')
new = (root / 'changelog/0.8.x.md').read_text(encoding='utf-8')
assert old.split('## v0.8.18', 1)[1] == new.split('## v0.8.18', 1)[1]
old = git('show', 'HEAD:initiatives/analysis/opus_core_analysis-open_queue.md')
new = (root / 'initiatives/analysis/opus_core_analysis-open_queue.md').read_text(encoding='utf-8')
rows = lambda text: [line for line in text.splitlines() if re.match(r'\| OC-\d+ \|', line)]
assert rows(old) == rows(new)
assert sum('\u2b1c open' in line for line in rows(new)) == 63
assert sum('parked' in line for line in rows(new)) == 4
assert old.split('## Planned enhancement', 1)[1] == new.split('## Planned enhancement', 1)[1]
raw = (root / 'tmp_repro_artifacts/job-log-report8.log').read_bytes()
report = raw.decode('utf-16' if raw.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf-8')
expected = []
path = None
for line in report.splitlines():
    if line.startswith('C:'):
        path = line.replace('\\', '/').split('/frontend/ml-canvas/', 1)[1]
    match = re.fullmatch(r'\s+(\d+):(\d+)\s+warning\s+(.+?) has a complexity of (\d+)\. Maximum allowed is 8\s+complexity\s*', line)
    if match:
        expected.append((path, int(match[1]), int(match[2]), int(match[4]), match[3]))
inventory = (root / 'initiatives/analysis/frontend_ccn_remaining_2026-09-10.md').read_text(encoding='utf-8')
actual = []
for match in re.finditer(r'^\| (\d+) \| \[([^\]]+):(\d+)\]\([^\n]+?\) \| (\d+) \| (.+) \|$', inventory, re.M):
    actual.append((match[2], int(match[3]), int(match[4]), int(match[1]), match[5]))
assert len(expected) == 133 and len({item[0] for item in expected}) == 96
assert sorted(expected) == sorted(actual)
assets = root / 'static/ml_canvas/assets'
imports = 0
for asset in assets.glob('*.js'):
    for ref in re.findall(r'''(?:from|import\s*\()\s*["'](\./[^"']+)["']''', asset.read_text(encoding='utf-8')):
        if ref.endswith(('.js', '.css')):
            assert (asset.parent / ref).is_file(), (asset.name, ref)
            imports += 1
index = (root / 'static/ml_canvas/index.html').read_text(encoding='utf-8')
assert 'index-BVvggaqQ.js' in index
for ref in re.findall(r'(?:src|href)="(/assets/[^"?#]+)', index):
    assert (root / 'static/ml_canvas' / ref.lstrip('/')).is_file(), ref
assert not git('diff', '--cached', '--name-only').strip()
print(f'PASS: {len(expected_text)} authored files plus rebuilt assets; exact133/96 inventory; queue63/4 and deferred work unchanged; older releases unchanged; {imports} generated imports resolve; no staged changes.')
