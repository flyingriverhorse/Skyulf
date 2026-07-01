#!/usr/bin/env node
// Syncs frontend/ml-canvas's package.json (and package-lock.json) version
// with the single source of truth: the root pyproject.toml's
// `[project] version = "X.Y.Z"`.
//
// Run manually after bumping pyproject.toml (per the repo's version-bump
// protocol in .github/instructions/quality_checks.instructions.md), or
// wire it into a release script. Exits non-zero if pyproject.toml can't
// be parsed, so it's also safe to use as a CI drift check via `--check`.
import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '../../..');
const pyprojectPath = path.join(repoRoot, 'pyproject.toml');
const packageJsonPath = path.resolve(__dirname, '../package.json');
const packageLockPath = path.resolve(__dirname, '../package-lock.json');

function readPyprojectVersion() {
  const toml = readFileSync(pyprojectPath, 'utf8');
  const match = toml.match(/^version\s*=\s*"([^"]+)"/m);
  if (!match) {
    throw new Error(`Could not find a top-level version in ${pyprojectPath}`);
  }
  return match[1];
}

function updateJsonVersion(filePath, version, mutate) {
  const original = readFileSync(filePath, 'utf8');
  const data = JSON.parse(original);
  const changed = mutate(data, version);
  if (!changed) return false;
  // Preserve the trailing newline npm/tools expect.
  writeFileSync(filePath, `${JSON.stringify(data, null, 2)}\n`);
  return true;
}

const version = readPyprojectVersion();
const checkOnly = process.argv.includes('--check');

let drift = false;

const pkgChanged = updateJsonVersion(packageJsonPath, version, (data, v) => {
  if (data.version === v) return false;
  drift = true;
  if (!checkOnly) data.version = v;
  return !checkOnly;
});

const lockChanged = updateJsonVersion(packageLockPath, version, (data, v) => {
  let touched = false;
  if (data.version !== v) {
    touched = true;
    if (!checkOnly) data.version = v;
  }
  if (data.packages?.['']?.version !== v) {
    touched = true;
    if (!checkOnly) data.packages[''].version = v;
  }
  drift = drift || touched;
  return !checkOnly && touched;
});

if (checkOnly) {
  if (drift) {
    console.error(
      `Version drift: pyproject.toml is ${version}, but package.json/package-lock.json are out of sync. Run "npm run sync-version" to fix.`,
    );
    process.exit(1);
  }
  console.log(`OK — package.json/package-lock.json already match pyproject.toml (${version}).`);
} else {
  if (pkgChanged || lockChanged) {
    console.log(`Synced version to ${version} (from pyproject.toml).`);
  } else {
    console.log(`Already up to date (${version}).`);
  }
}
