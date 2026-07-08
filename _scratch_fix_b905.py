import ast
import json
import subprocess

result = subprocess.run(
    [".venv/bin/ruff", "check", "--select", "B905", "--output-format=json", "."],
    capture_output=True, text=True, cwd="/Users/BH7043/Skyulf"
)
diags = json.loads(result.stdout)

# locations to skip (genuine ambiguous case + notebook file, handled separately / left alone)
SKIP = {
    ("/Users/BH7043/Skyulf/backend/ml_pipeline/_execution/engine/_merge.py", 279),
}

by_file = {}
for d in diags:
    fname = d["filename"]
    row = d["location"]["row"]
    if fname.endswith(".ipynb"):
        continue
    if (fname, row) in SKIP:
        continue
    by_file.setdefault(fname, []).append((row, d["location"]["column"]))

for fname, locs in sorted(by_file.items()):
    src = open(fname, encoding="utf-8").read()
    tree = ast.parse(src)

    # Find all zip(...) Call nodes
    zip_calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "zip":
            zip_calls.append(node)

    edits = []  # (end_lineno, end_col_offset_before_paren, text)
    for row, col in locs:
        # ruff reports column of the zip(...) expression start (1-based)
        match = None
        for node in zip_calls:
            if node.lineno == row and node.col_offset == col - 1:
                match = node
                break
        if match is None:
            # fallback: any zip call starting on this row
            candidates = [n for n in zip_calls if n.lineno == row]
            if len(candidates) == 1:
                match = candidates[0]
        if match is None:
            print(f"WARN: could not resolve zip() call at {fname}:{row}:{col}")
            continue
        # Check if strict already present
        has_strict = any(kw.arg == "strict" for kw in match.keywords)
        if has_strict:
            continue
        # Insert ", strict=True" right before the closing paren -> end_col_offset - 1
        edits.append((match.end_lineno, match.end_col_offset - 1, ", strict=True"))

    if not edits:
        continue

    lines = src.splitlines(keepends=True)
    edits_by_line = {}
    for end_lineno, end_col, text in edits:
        edits_by_line.setdefault(end_lineno, []).append((end_col, text))

    for lineno in sorted(edits_by_line.keys(), reverse=True):
        line = lines[lineno - 1]
        for col, text in sorted(edits_by_line[lineno], key=lambda t: -t[0]):
            line = line[:col] + text + line[col:]
        lines[lineno - 1] = line

    new_src = "".join(lines)
    with open(fname, "w", encoding="utf-8") as f:
        f.write(new_src)
    print(f"Fixed {len(edits)} in {fname}")
