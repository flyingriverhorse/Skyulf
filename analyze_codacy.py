import json
from collections import Counter

try:
    with open('codacy_results.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Total Issues: {len(data)}")

    patterns = Counter()
    levels = Counter()
    files = Counter()

    for item in data:
        issue = item.get('Issue', {})
        pattern = issue.get('patternId', {}).get('value', 'Unknown')
        level = issue.get('level', 'Unknown')
        filename = issue.get('filename', 'Unknown')
        
        patterns[pattern] += 1
        levels[level] += 1
        files[filename] += 1

    print("\n--- Issues by Level ---")
    for level, count in levels.most_common():
        print(f"{level}: {count}")

    print("\n--- Top 10 Patterns ---")
    for pattern, count in patterns.most_common(10):
        print(f"{count} - {pattern}")

    print("\n--- Top 10 Files with Issues ---")
    for filename, count in files.most_common(10):
        print(f"{count} - {filename}")

except Exception as e:
    print(f"Error: {e}")
