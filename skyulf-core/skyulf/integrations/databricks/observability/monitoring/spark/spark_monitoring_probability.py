"""Exact distributed probability metrics from score-frequency curves."""

import importlib
import math
from typing import Any

from .spark_monitoring_metrics import spark_column, spark_functions


def _curve(pairs: Any, label: Any, probability: str) -> tuple[float, float, int, int]:
    """Aggregate tied-score ROC and precision-recall integrals on Spark executors."""
    f = spark_functions()
    window = importlib.import_module("pyspark.sql.window").Window
    positive = (spark_column("__sm_truth") == f.lit(label)).cast("long")
    curve = pairs.select(spark_column(probability).alias("score"), positive.alias("positive"))
    curve = curve.groupBy("score").agg(f.sum("positive").alias("p"), f.count("*").alias("n"))
    order = window.orderBy(f.col("score").desc()).rowsBetween(
        window.unboundedPreceding, window.currentRow
    )
    curve = curve.withColumn("tp", f.sum("p").over(order)).withColumn(
        "seen", f.sum("n").over(order)
    )
    summary = curve.agg(
        f.sum("p").alias("positive"),
        f.sum("n").alias("total"),
        f.sum(f.col("p") * f.col("tp") / f.col("seen")).alias("ap"),
        f.sum((f.col("n") - f.col("p")) * (f.col("tp") - f.col("p") / 2)).alias("auc"),
    ).first()
    positives, total = int(summary["positive"]), int(summary["total"])
    negatives = total - positives
    auc = summary["auc"] / (positives * negatives) if positives and negatives else math.nan
    ap = summary["ap"] / positives if positives else 0.0
    return float(auc), float(ap), positives, total


def _log_loss(pairs: Any, classes: tuple, probabilities: tuple) -> float:
    """Use the same double-precision clipping as sklearn saved probability arrays."""
    f = spark_functions()
    terms = [
        f.when(spark_column("__sm_truth") == f.lit(label), spark_column(name)).otherwise(0.0)
        for label, name in zip(classes, probabilities, strict=True)
    ]
    probability = f.greatest(
        f.lit(2.220446049250313e-16), f.least(f.lit(1 - 2.220446049250313e-16), sum(terms))
    )
    return float(pairs.agg(f.avg(-f.log(probability)).alias("loss")).first()["loss"])


def _multiclass(pairs: Any, classes: tuple, probabilities: tuple) -> dict[str, float]:
    """Match one-vs-rest and Hand-Till one-vs-one weighting without observation collection."""
    curves = [
        _curve(pairs, label, name) for label, name in zip(classes, probabilities, strict=True)
    ]
    total = curves[0][3]
    weighted = sum(auc * positives for auc, _, positives, _ in curves) / total
    result = {
        "roc_auc_ovr_weighted": weighted,
        "roc_auc_weighted": weighted,
        "roc_auc_ovr": sum(value[0] for value in curves) / len(classes),
        "pr_auc_weighted": sum(ap * positives for _, ap, positives, _ in curves) / total,
    }
    present = [index for index, value in enumerate(curves) if value[2]]
    return result | _ovo(pairs, classes, probabilities, present)


def _ovo(pairs: Any, classes: tuple, probabilities: tuple, present: list[int]) -> dict[str, float]:
    """Average pairwise class-rank areas with the Core prevalence weighting."""
    comparisons = []
    for offset, first in enumerate(present):
        for second in present[offset + 1 :]:
            subset = pairs.where(spark_column("__sm_truth").isin(classes[first], classes[second]))
            a = _curve(subset, classes[first], probabilities[first])
            b = _curve(subset, classes[second], probabilities[second])
            comparisons.append(((a[0] + b[0]) / 2, a[3]))
    result = {}
    if comparisons:
        result["roc_auc_ovo"] = sum(value for value, _ in comparisons) / len(comparisons)
        result["roc_auc_ovo_weighted"] = sum(value * weight for value, weight in comparisons) / sum(
            weight for _, weight in comparisons
        )
    return result


def probability_values(pairs: Any, classes: tuple, probabilities: tuple) -> dict[str, float]:
    """Measure all Core probability metrics through exact grouped-score integrations."""
    result = {"log_loss": _log_loss(pairs, classes, probabilities)}
    if len(classes) != 2:
        return result | _multiclass(pairs, classes, probabilities)
    auc, ap, _, _ = _curve(pairs, classes[1], probabilities[1])
    result.update(
        dict.fromkeys(
            (
                "roc_auc",
                "roc_auc_weighted",
                "roc_auc_ovr",
                "roc_auc_ovo",
                "roc_auc_ovr_weighted",
                "roc_auc_ovo_weighted",
            ),
            auc,
        )
    )
    result.update({"pr_auc": ap, "pr_auc_weighted": ap})
    return result
