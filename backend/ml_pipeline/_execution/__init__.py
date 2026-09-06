"""Pipeline execution layer: engine, graph topology, gates and job bookkeeping.

Turns a user-built node graph into a run. ``engine`` executes a
:class:`.schemas.PipelineConfig` node-by-node and collects per-node results,
captured warnings and merge advisories; ``graph_utils`` owns the topology work
(execution ordering, parallel-branch and preview partitioning, model-family
resolution); ``_cycle_validation`` and ``_leakage_validation`` are the pre-run
gates that fail fast on a cyclic graph or on preprocessing wired above a
splitter; ``_schema_graph`` and ``_schema_validator`` predict each node's
output schema and check the columns its params reference; ``schemas`` holds the
config/result/job dataclasses shared across the layer; ``jobs``,
``job_manager_base``, ``basic_training_manager``, ``advanced_tuning_manager``
and ``strategies`` persist and drive training and tuning jobs; ``summary`` and
``diagram`` build the node-card one-liner and the Mermaid topology shown on the
canvas and Experiments pages. Nothing is re-exported here — import what you
need by path.
"""
