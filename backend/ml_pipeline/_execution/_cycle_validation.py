"""Pre-execution cycle guard for pipeline graphs.

Connect-time checks block cycles one edge at a time on the canvas, but bulk
graph loads (saved projects) can still deliver a cyclic config. Without this
guard the engine executes nodes in list order and a node in the loop dies
late with a cryptic "Artifact not found" error; here we fail fast with a
message naming the loop.
"""

from __future__ import annotations

from backend.ml_pipeline._execution.schemas import NodeConfig


class PipelineCycleError(ValueError):
    """Raised when the pipeline graph contains a cycle."""


def validate_no_cycles(nodes: list[NodeConfig]) -> None:
    """Raise :class:`PipelineCycleError` if ``nodes`` contain a cycle.

    Runs Kahn's algorithm over the ``inputs`` edges; any node that never
    reaches in-degree 0 sits on or downstream of a cycle. The stuck set is
    then pruned to the exact loop members so the error names only the loop.
    """
    known = {node.node_id for node in nodes}
    inputs = {node.node_id: [i for i in node.inputs if i in known] for node in nodes}

    in_degree = {nid: len(ups) for nid, ups in inputs.items()}
    children: dict[str, list[str]] = {nid: [] for nid in known}
    for nid, ups in inputs.items():
        for up in ups:
            children[up].append(nid)

    ready = [nid for nid, deg in in_degree.items() if deg == 0]
    ordered: set[str] = set()
    while ready:
        nid = ready.pop(0)
        ordered.add(nid)
        for child in children[nid]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                ready.append(child)
    if len(ordered) == len(known):
        return

    stuck = known - ordered
    # A stuck node with no stuck successor is merely downstream of the loop,
    # not part of it — prune those so the message targets the loop only.
    while True:
        removable = {nid for nid in stuck if not any(c in stuck for c in children[nid])}
        if not removable:
            break
        stuck -= removable

    loop = _trace_loop(stuck, inputs)
    raise PipelineCycleError(
        "Pipeline contains a cycle: "
        + " -> ".join([*loop, loop[0]])
        + ". Remove one of these connections so the pipeline flows in one direction."
    )


def _trace_loop(stuck: set[str], inputs: dict[str, list[str]]) -> list[str]:
    """Follow inputs inside ``stuck`` until a node repeats, returning the loop."""
    start = next(iter(stuck))
    path = [start]
    position = {start: 0}
    current = start
    while True:
        nxt = next(up for up in inputs[current] if up in stuck)
        if nxt in position:
            return path[position[nxt] :]
        position[nxt] = len(path)
        path.append(nxt)
        current = nxt
