"""Example company snapshot features; enable the group in config/features.yml.

Input: company_id, observed_at, employee_count, and optional unused source columns.
Output: exactly company_id, observed_at, employee_count.
The source must already have one row per company and observation date. The job
rejects duplicate/null keys; this example does not silently deduplicate them.
The job reads the pinned Delta input and writes the returned DataFrame for you.
"""


def compute_features(frame):
    """Select the company snapshot without learning or filling missing values."""
    return frame.select("company_id", "observed_at", "employee_count")
