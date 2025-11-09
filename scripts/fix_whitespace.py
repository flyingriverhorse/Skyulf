"""Strip trailing whitespace and ensure single final newline for Python files.
Usage: python scripts/fix_whitespace.py
"""
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
py_files = list(root.rglob('*.py'))
# Skip virtualenv and large vendor dirs if present
skip_dirs = {'.venv', 'venv', '__pycache__', 'site-packages'}

fixed = 0
for p in py_files:
    if any(part in skip_dirs for part in p.parts):
        continue
    try:
        text = p.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        # Try with latin-1 fallback
        try:
            text = p.read_text(encoding='latin-1')
        except Exception:
            print(f"Skipping (encoding): {p}")
            continue
    lines = text.splitlines()
    new_lines = [ln.rstrip() for ln in lines]
    # Remove trailing blank lines at file end
    while new_lines and new_lines[-1] == '':
        new_lines.pop()
    new_text = '\n'.join(new_lines) + '\n'
    if new_text != text:
        p.write_text(new_text, encoding='utf-8')
        fixed += 1
print(f"Whitespace fixer: updated {fixed} files")
sys.exit(0)
