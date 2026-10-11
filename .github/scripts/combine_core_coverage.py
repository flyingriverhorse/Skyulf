"""Validate base and native Core coverage shards and enforce combined branch coverage."""

from pathlib import Path

from coverage import Coverage, CoverageData


def combine_core_coverage() -> float:
    """Reject unusable shards before merging data and preserve source-relative XML paths."""
    expected = [Path(f".coverage.shard-{index}") for index in (0, 1, 2, 3, "spark", "delta")]
    if set(Path(".").glob(".coverage.shard-*")) != set(expected):
        raise ValueError(
            "Exactly six Core coverage shards are required: 0 through 3, spark, and delta."
        )

    # Coverage.combine warns and skips corrupt inputs. Read and merge explicitly
    # so no unusable shard can silently disappear from the coverage denominator.
    shards = []
    for path in expected:
        if not path.is_file():
            raise ValueError(f"Core coverage shard must be a file: {path}")
        shard = CoverageData(basename=str(path))
        shard.read()
        if not shard.has_arcs() or not shard.measured_files():
            raise ValueError(f"Core coverage shard must contain measured branch data: {path}")
        shards.append(shard)

    coverage = Coverage(data_file=".coverage", source=["skyulf-core/skyulf"], branch=True)
    coverage.erase()
    merged = coverage.get_data()
    for shard in shards:
        merged.update(shard)
    coverage.save()
    coverage.xml_report(outfile="coverage-core.xml")
    return coverage.report(show_missing=True)


def main() -> int:
    """Fail CI unless all shards combine and retain the existing 90 percent floor."""
    total = combine_core_coverage()
    if total < 90:
        print(f"Core combined branch coverage {total:.2f}% is below the required 90%.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
