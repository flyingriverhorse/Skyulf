import os
import re
import tomllib
from pathlib import Path

def main():
    # 1. Read version from pyproject.toml
    root_dir = Path(__file__).resolve().parents[2]
    pyproject_path = root_dir / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)
    version = data["project"]["version"] # e.g. "0.6.0"
    
    # 2. Locate corresponding changelog series file
    major_minor = ".".join(version.split(".")[:2]) # e.g. "0.6"
    changelog_file = root_dir / "changelog" / f"{major_minor}.x.md"
    if not changelog_file.exists():
        raise FileNotFoundError(f"Changelog file {changelog_file.relative_to(root_dir)} not found.")
        
    # 3. Extract the release notes block for the version
    content = changelog_file.read_text(encoding="utf-8")
    
    # Find headers matching the version, e.g., "## v0.6.0" or "## v0.6.0 — ..."
    # Handle both CRLF and LF lines.
    content = content.replace("\r\n", "\n")
    header_pattern = re.compile(rf"^##\s+v{re.escape(version)}(?:\s+—\s+(.*))?$", re.MULTILINE)
    match = header_pattern.search(content)
    if not match:
        raise ValueError(f"Could not find changelog header matching '## v{version}' in {changelog_file.name}")
        
    title_suffix = match.group(1) or ""
    title = f"v{version}"
    if title_suffix:
        title += f" — {title_suffix.strip()}"
        
    start_idx = match.end()
    # Find start of the next release block or end of file
    next_header_pattern = re.compile(r"^##\s+v\d", re.MULTILINE)
    next_match = next_header_pattern.search(content, pos=start_idx)
    
    if next_match:
        end_idx = next_match.start()
    else:
        end_idx = len(content)
        
    notes = content[start_idx:end_idx].strip()
    
    # Set step outputs for GitHub Actions
    notes_file = root_dir / "temp_release_notes.md"
    notes_file.write_text(notes, encoding="utf-8")
    
    # Set GITHUB_OUTPUT environment variables
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="utf-8") as out:
            out.write(f"version=v{version}\n")
            out.write(f"title={title}\n")
            out.write(f"notes_file={notes_file.as_posix()}\n")
    else:
        print(f"Version: v{version}")
        print(f"Title: {title}")
        print(f"Notes file written to: {notes_file}")
        print(f"Notes:\n{notes}")

if __name__ == "__main__":
    main()
