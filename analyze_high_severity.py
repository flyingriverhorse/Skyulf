import json
from collections import Counter, defaultdict

try:
    with open('codacy_results.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    high_severity_issues = []
    for item in data:
        issue = item.get('Issue', {})
        if issue.get('level') == 'High':
            high_severity_issues.append(issue)

    print(f"Total High Severity Issues: {len(high_severity_issues)}")

    patterns = Counter()
    files = defaultdict(list)

    for issue in high_severity_issues:
        pattern = issue.get('patternId', {}).get('value', 'Unknown')
        filename = issue.get('filename', 'Unknown')
        patterns[pattern] += 1
        files[filename].append(pattern)

    print("\n--- Top High Severity Patterns ---")
    for pattern, count in patterns.most_common(10):
        print(f"{count} - {pattern}")

    print("\n--- Files with High Severity Issues ---")
    for filename, issues in sorted(files.items(), key=lambda x: len(x[1]), reverse=True)[:10]:
        print(f"{filename}: {len(issues)} issues")
        # Print unique patterns for this file
        unique_patterns = set(issues)
        for p in unique_patterns:
            print(f"  - {p}")

except Exception as e:
    print(f"Error: {e}")
