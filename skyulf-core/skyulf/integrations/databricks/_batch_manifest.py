"""Package-internal publication evidence shared by Spark and local writers."""

import hashlib
import json
from dataclasses import asdict
from typing import Any

from ._contracts import BatchSpec


def batch_manifest(
    spec: BatchSpec,
    source_id: str,
    source: str,
    committed_us: int,
    input_count: int,
    output_count: int,
) -> dict[str, Any]:
    """Fingerprint the complete request and retain snapshot, code and model evidence."""
    request = asdict(spec)
    for key in ("period_start", "period_end", "as_of"):
        request[key] = getattr(spec, key + "_utc").isoformat()
    request.update(source_table_id=source_id, source_table=source)
    payload = json.dumps(request, sort_keys=True, separators=(",", ":"))
    return dict(
        request,
        request_digest=hashlib.sha256(payload.encode()).hexdigest(),
        input_count=input_count,
        output_count=output_count,
        source_committed_us=committed_us,
        source_temporal_contract="delta_snapshot_committed_by_as_of",
    )
