"""Domain inference utilities for Skyulf MLflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

try:  # Optional dependency (scikit-learn pulls joblib, but be defensive)
    import joblib  # type: ignore
except Exception:  # pragma: no cover - joblib missing in minimal envs
    joblib = None

import pickle
import warnings

GEOSPATIAL_LAT_HINTS: Tuple[str, ...] = ("latitude", "lat", "lat_deg", "latitud")
GEOSPATIAL_LON_HINTS: Tuple[str, ...] = ("longitude", "lon", "lng", "long", "lon_deg")
GEOSPATIAL_TEXT_HINTS: Tuple[str, ...] = (
    "address",
    "city",
    "state",
    "country",
    "zip",
    "zipcode",
    "postal",
    "postcode",
    "region",
    "county",
    "province",
    "territory",
    "geometry",
    "geom",
    "geography",
    "geocode",
    "geocoded",
    "coordinate",
    "coordinates",
    "spatial",
    "geospatial",
    "geopoint",
    "geohash",
    "shape",
    "wkt",
)

ENVIRONMENTAL_HINTS: Tuple[str, ...] = (
    "temperature",
    "temp",
    "humidity",
    "precip",
    "precipitation",
    "rain",
    "rainfall",
    "snow",
    "wind",
    "speed",
    "air_quality",
    "aqi",
    "pm2_5",
    "pm10",
    "emission",
    "emissions",
    "pollution",
    "co2",
    "carbon",
    "ghg",
    "water_quality",
    "water",
    "soil",
    "moisture",
    "pressure",
    "climate",
    "weather",
    "sensor",
    "sensors",
    "environment",
    "environmental",
    "solar",
    "radiation",
    "uv",
    "ozone",
    "visibility",
    "dewpoint",
    "dew_point",
    "turbidity",
    "ph",
)

ENVIRONMENTAL_VALUE_HINTS: Tuple[str, ...] = (
    "°c",
    "°f",
    " ppm",
    " ppb",
    "µg/m3",
    "ug/m3",
    "mg/l",
    "mg\\l",
    "aqi",
    "co2",
    "carbon",
    "pm2.5",
    "pm2_5",
    "pm10",
    "humidity",
    "%rh",
    "relative humidity",
    "wind speed",
    "m/s",
    "km/h",
    "knots",
    "solar",
    "uv",
    "ozone",
    "rain",
    "rainfall",
    "precip",
    "hpa",
    "baro",
)

_TEXT_IMAGE_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp")

OBJECT_DETECTION_HINTS: Tuple[str, ...] = (
    "bbox",
    "bounding_box",
    "x_min",
    "xmin",
    "x_max",
    "xmax",
    "y_min",
    "ymin",
    "y_max",
    "ymax",
    "box",
    "mask",
    "segmentation",
)

_DEFAULT_ARTIFACT_LOCATIONS = (
    Path(__file__).resolve().parent / "models" / "domain_analyzer_xgb.pkl",
    Path(__file__).resolve().parents[2] / "skyulf_mlflow" / "eda" / "model_ml" / "domain_analyzer_xgb.pkl",
)


@dataclass
class DomainInferenceResult:
    """Structured response from :class:`DomainAnalyzer`."""

    primary_domain: str
    primary_confidence: float
    domain_scores: Dict[str, float]
    secondary_domains: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    patterns: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    ml_prediction: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary_domain": self.primary_domain,
            "primary_confidence": self.primary_confidence,
            "domain_scores": self.domain_scores,
            "secondary_domains": self.secondary_domains,
            "recommendations": self.recommendations,
            "patterns": self.patterns,
            "metadata": self.metadata,
            "ml_prediction": self.ml_prediction,
        }


def _column_contains_hint(column: str, hints: Sequence[str]) -> bool:
    if not column:
        return False
    lowered = column.lower()
    for hint in hints:
        if hint in lowered:
            return True
    return False


def _text_contains_hint(text: str, hints: Sequence[str]) -> bool:
    lowered = text.lower()
    return any(hint in lowered for hint in hints)


def _count_outliers_iqr(series: pd.Series) -> int:
    if series.empty:
        return 0
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return int(((series < lower) | (series > upper)).sum())


class DomainAnalyzer:
    """Rule-based (with optional ML) domain inference for tabular datasets."""

    def __init__(
        self,
        *,
        enable_ml_classifier: bool = True,
        ml_model_path: Optional[Union[str, Path]] = None,
        additional_model_paths: Optional[Iterable[Union[str, Path]]] = None,
        ml_min_confidence: float = 0.7,
    ) -> None:
        self.enable_ml_classifier = enable_ml_classifier
        self.ml_min_confidence = ml_min_confidence
        self.ml_classifier: Optional[Any] = None
        self.ml_label_encoder: Optional[Any] = None
        self.ml_feature_names: List[str] = []

        candidate_paths: List[Path] = []
        if ml_model_path:
            candidate_paths.append(Path(ml_model_path))
        for path in additional_model_paths or []:
            candidate_paths.append(Path(path))
        candidate_paths.extend(_DEFAULT_ARTIFACT_LOCATIONS)
        self._candidate_model_paths = candidate_paths

        if self.enable_ml_classifier:
            self._load_ml_classifier()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def analyze_dataset_domain(
        self,
        columns: Optional[Sequence[str]],
        sample_data: Union[pd.DataFrame, Sequence[Dict[str, Any]], Sequence[Sequence[Any]]],
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DomainInferenceResult:
        """Infer likely business domain for the provided sample."""

        df = self._coerce_sample(columns, sample_data)
        column_patterns = self._analyze_column_patterns(list(df.columns))
        data_patterns = self._analyze_data_patterns(df)

        domain_scores: Dict[str, float] = {
            "healthcare": 0.0,
            "finance": 0.0,
            "retail": 0.0,
            "real_estate": 0.0,
            "tech": 0.0,
            "marketing": 0.0,
            "time_series": 0.0,
            "nlp": 0.0,
            "computer_vision": 0.0,
            "fraud": 0.0,
            "geospatial": 0.0,
            "environmental": 0.0,
            "general": 1.0,
        }

        fraud_indicators = 0.0
        geospatial_indicators = 0.0

        for pattern, score in column_patterns.items():
            if pattern == "price":
                domain_scores["finance"] += score
                domain_scores["retail"] += score * 0.7
                domain_scores["real_estate"] += score * 0.5
            elif pattern == "medical":
                domain_scores["healthcare"] += score * 2
            elif pattern == "user":
                domain_scores["tech"] += score
                domain_scores["marketing"] += score * 0.8
            elif pattern == "tech":
                domain_scores["tech"] += score * 1.5
            elif pattern == "marketing":
                domain_scores["marketing"] += score * 3
            elif pattern == "fraud":
                domain_scores["fraud"] += score * 3
                fraud_indicators += score
            elif pattern == "location":
                domain_scores["real_estate"] += score
                domain_scores["marketing"] += score * 0.4
                domain_scores["geospatial"] += score * 1.5
                geospatial_indicators += score
            elif pattern == "geospatial":
                domain_scores["geospatial"] += score * 2.5
                geospatial_indicators += score
            elif pattern == "environmental":
                domain_scores["environmental"] += score * 2.0
                domain_scores["time_series"] += score * 0.3
                domain_scores["geospatial"] += score * 0.2
            elif pattern == "text":
                domain_scores["nlp"] += score
            elif pattern == "time":
                domain_scores["time_series"] += score * 0.5
            elif pattern == "computer_vision":
                domain_scores["computer_vision"] += score * 2.0
            elif pattern == "object_detection":
                domain_scores["computer_vision"] += score * 2.5

        if fraud_indicators >= 2:
            domain_scores["fraud"] += 3.0
        if geospatial_indicators >= 2:
            domain_scores["geospatial"] += 3.0

        if data_patterns:
            if data_patterns.get("has_time_series"):
                domain_scores["time_series"] += 2.0
            if data_patterns.get("has_text_columns"):
                domain_scores["nlp"] += 1.0
            if data_patterns.get("has_numeric_target"):
                domain_scores["finance"] += 1.0
                domain_scores["real_estate"] += 1.0
            if data_patterns.get("has_geospatial_coordinates"):
                domain_scores["geospatial"] += 2.5
            if data_patterns.get("has_geospatial_text"):
                domain_scores["geospatial"] += 1.0
            if data_patterns.get("has_image_data"):
                domain_scores["computer_vision"] += 2.5
                domain_scores["tech"] += 0.5
            if data_patterns.get("has_object_detection_annotations"):
                domain_scores["computer_vision"] += 2.0
            if data_patterns.get("has_environmental_signals"):
                domain_scores["environmental"] += 2.5
                if data_patterns.get("has_time_series"):
                    domain_scores["time_series"] += 0.5
            if data_patterns.get("has_binary_target"):
                domain_scores["fraud"] += 1.5
            if data_patterns.get("has_mixed_features"):
                domain_scores["fraud"] += 0.5

        ml_prediction = self._ml_predict_domain(df)
        if ml_prediction:
            label = ml_prediction.get("domain")
            confidence = float(ml_prediction.get("confidence", 0))
            if label:
                base_sum = max(sum(domain_scores.values()), 1.0)
                domain_scores[label] = max(domain_scores.get(label, 0.0), confidence * base_sum)

        sorted_domains = sorted(domain_scores.items(), key=lambda item: item[1], reverse=True)
        primary_domain, primary_score = sorted_domains[0]
        total_score = max(sum(domain_scores.values()), 1.0)
        primary_confidence = min(primary_score / total_score, 0.95)

        secondary_domains: List[Dict[str, Any]] = []
        for domain, score in sorted_domains[1:]:
            if score >= max(primary_score * 0.1, 0.5) and len(secondary_domains) < 2:
                secondary_domains.append(
                    {
                        "domain": domain,
                        "confidence": min(score / total_score, 0.95),
                    }
                )

        recommendations = self._generate_domain_recommendations(primary_domain, domain_scores, data_patterns)

        result = DomainInferenceResult(
            primary_domain=primary_domain,
            primary_confidence=primary_confidence,
            domain_scores=domain_scores,
            secondary_domains=secondary_domains,
            recommendations=recommendations,
            patterns={
                "column_patterns": column_patterns,
                "data_patterns": data_patterns,
            },
            metadata=metadata or {},
            ml_prediction=ml_prediction,
        )

        if ml_prediction and ml_prediction.get("confidence", 0) >= self.ml_min_confidence:
            if ml_prediction.get("domain") != result.primary_domain:
                result.secondary_domains.insert(
                    0,
                    {
                        "domain": result.primary_domain,
                        "confidence": result.primary_confidence,
                    },
                )
                result.primary_domain = str(ml_prediction["domain"])
                result.primary_confidence = float(ml_prediction["confidence"])
            else:
                result.primary_confidence = max(
                    result.primary_confidence,
                    float(ml_prediction.get("confidence", 0)),
                )

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _coerce_sample(
        self,
        columns: Optional[Sequence[str]],
        sample_data: Union[pd.DataFrame, Sequence[Dict[str, Any]], Sequence[Sequence[Any]]],
    ) -> pd.DataFrame:
        if isinstance(sample_data, pd.DataFrame):
            df = sample_data.copy()
            if columns is not None and list(df.columns) != list(columns):
                df.columns = list(columns)
            return df

        if sample_data is None:
            return pd.DataFrame(columns=list(columns) if columns else [])

        if isinstance(sample_data, Sequence) and not isinstance(sample_data, (str, bytes)):
            if len(sample_data) == 0:  # type: ignore[arg-type]
                return pd.DataFrame(columns=list(columns) if columns else [])
            first = sample_data[0]  # type: ignore[index]
            if isinstance(first, dict):
                return pd.DataFrame(sample_data)
            if columns:
                return pd.DataFrame(sample_data, columns=list(columns))

        return pd.DataFrame(columns=list(columns) if columns else [])

    def _load_ml_classifier(self) -> None:
        for path in self._candidate_model_paths:
            if not path or not path.exists():
                continue
            try:
                artifact = None
                if joblib is not None:
                    artifact = joblib.load(path)
                else:
                    with path.open("rb") as handle:
                        artifact = pickle.load(handle)

                model = label_encoder = feature_names = None
                if isinstance(artifact, dict):
                    model = artifact.get("model")
                    label_encoder = artifact.get("label_encoder")
                    feature_names = artifact.get("feature_names")
                elif isinstance(artifact, (list, tuple)) and len(artifact) >= 3:
                    model, label_encoder, feature_names = artifact[:3]

                if model is None or label_encoder is None or not feature_names:
                    continue
                if not hasattr(model, "predict"):
                    continue

                self.ml_classifier = model
                self.ml_label_encoder = label_encoder
                self.ml_feature_names = list(feature_names)
                return
            except Exception:  # pragma: no cover - artifact may be incompatible
                continue

        self.ml_classifier = None
        self.ml_label_encoder = None
        self.ml_feature_names = []

    def _analyze_column_patterns(self, columns: Sequence[str]) -> Dict[str, float]:
        patterns: Dict[str, float] = {}
        pattern_keywords: Dict[str, Sequence[str]] = {
            "price": ["price", "cost", "amount", "fee", "charge", "value"],
            "time": ["date", "time", "timestamp", "created", "updated"],
            "user": ["user", "customer", "client", "account"],
            "tech": ["event", "click", "session", "device", "browser", "api", "endpoint", "latency", "request", "response", "app", "platform"],
            "medical": ["patient", "diagnosis", "treatment", "medical", "health"],
            "location": ["address", "city", "state", "country", "zip", "location"],
            "id": ["id", "key", "identifier", "uuid"],
            "category": ["type", "category", "class", "group", "segment"],
            "text": ["text", "description", "comment", "review", "message"],
            "marketing": [
                "campaign",
                "impression",
                "click",
                "conversion",
                "ctr",
                "cpc",
                "roas",
                "roi",
                "spend",
                "budget",
                "channel",
                "source",
                "medium",
                "utm",
                "ad",
                "audience",
                "engagement",
                "funnel",
                "attribution",
                "cohort",
                "retention",
                "acquisition",
                "bounce",
                "session",
            ],
            "fraud": [
                "fraud",
                "suspicious",
                "anomaly",
                "risk",
                "flag",
                "alert",
                "transaction",
                "payment",
                "transfer",
                "card",
                "dispute",
                "claim",
                "chargeback",
                "refund",
                "verify",
                "legitimate",
                "genuine",
                "authentic",
                "valid",
                "invalid",
                "label",
            ],
            "environmental": list(ENVIRONMENTAL_HINTS),
            "object_detection": list(OBJECT_DETECTION_HINTS),
            "computer_vision": [
                "image",
                "img",
                "picture",
                "pixel",
                "frame",
                "mask",
                "segmentation",
                "annotation",
                "image_path",
                "image_url",
                "filepath",
                "file_path",
                "sprite",
                "tilename",
                "tile",
                "width",
                "height",
                "channel",
            ],
        }

        for col in columns:
            lowered = str(col).lower()
            for pattern, keywords in pattern_keywords.items():
                if any(keyword in lowered for keyword in keywords):
                    patterns[pattern] = patterns.get(pattern, 0.0) + 1.0
                    break

            if _column_contains_hint(lowered, GEOSPATIAL_LAT_HINTS):
                patterns["geospatial"] = patterns.get("geospatial", 0.0) + 1.0
            if _column_contains_hint(lowered, GEOSPATIAL_LON_HINTS):
                patterns["geospatial"] = patterns.get("geospatial", 0.0) + 1.0
            if _column_contains_hint(lowered, GEOSPATIAL_TEXT_HINTS):
                patterns["location"] = patterns.get("location", 0.0) + 0.5
                patterns["geospatial"] = patterns.get("geospatial", 0.0) + 0.5

        return patterns

    def _analyze_data_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        if df.empty:
            return {}

        patterns: Dict[str, Any] = {
            "has_time_series": False,
            "has_text_columns": False,
            "has_numeric_target": False,
            "has_binary_target": False,
            "has_categorical_features": False,
            "has_numeric_features": False,
            "column_count": len(df.columns),
            "row_count": len(df),
            "numeric_columns": 0,
            "categorical_columns": 0,
            "has_geospatial_coordinates": False,
            "has_geospatial_text": False,
            "geospatial_latitude_columns": [],
            "geospatial_longitude_columns": [],
            "geospatial_text_columns": [],
            "has_image_data": False,
            "image_columns": [],
            "has_object_detection_annotations": False,
            "object_detection_columns": [],
            "has_environmental_signals": False,
            "environmental_columns": [],
        }

        for col in df.columns:
            col_lower = str(col).lower()
            series = df[col]

            if pd.api.types.is_datetime64_any_dtype(series):
                patterns["has_time_series"] = True
                continue

            if pd.api.types.is_object_dtype(series):
                sample = series.dropna().astype(str).head(10)
                if not sample.empty:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=UserWarning, message="Could not infer format")
                        try:
                            pd.to_datetime(sample, errors="raise")
                        except Exception:
                            pass
                        else:
                            patterns["has_time_series"] = True
                            continue

            if pd.api.types.is_object_dtype(series):
                avg_length = series.astype(str).str.len().mean()
                if avg_length > 20:
                    patterns["has_text_columns"] = True
                patterns["categorical_columns"] += 1
                patterns["has_categorical_features"] = True

                if _column_contains_hint(col_lower, GEOSPATIAL_TEXT_HINTS) and col not in patterns["geospatial_text_columns"]:
                    patterns["geospatial_text_columns"].append(col)

                sample_values = series.dropna().astype(str).head(5)
                sample_lower = [value.strip().lower() for value in sample_values]

                geometry_prefixes = (
                    "point",
                    "linestring",
                    "polygon",
                    "multipoint",
                    "multilinestring",
                    "multipolygon",
                    "geometrycollection",
                )
                if any(value.startswith(prefix) for value in (val.upper() for val in sample_values) for prefix in geometry_prefixes):
                    if col not in patterns["geospatial_text_columns"]:
                        patterns["geospatial_text_columns"].append(col)
                elif any(value.strip().startswith("{") and '"type"' in value and '"coordinates"' in value for value in sample_values):
                    if col not in patterns["geospatial_text_columns"]:
                        patterns["geospatial_text_columns"].append(col)

                if any(lower.startswith("data:image") for lower in sample_lower) or any(lower.endswith(ext) for lower in sample_lower for ext in _TEXT_IMAGE_EXTENSIONS):
                    patterns["has_image_data"] = True
                    if col not in patterns["image_columns"]:
                        patterns["image_columns"].append(col)
                elif any("<img" in lower or "image/" in lower for lower in sample_lower):
                    patterns["has_image_data"] = True
                    if col not in patterns["image_columns"]:
                        patterns["image_columns"].append(col)

                if any(hint in col_lower for hint in OBJECT_DETECTION_HINTS):
                    patterns["has_object_detection_annotations"] = True
                    if col not in patterns["object_detection_columns"]:
                        patterns["object_detection_columns"].append(col)
                    patterns["has_image_data"] = True

                if _column_contains_hint(col_lower, ENVIRONMENTAL_HINTS) and col not in patterns["environmental_columns"]:
                    patterns["environmental_columns"].append(col)
                    patterns["has_environmental_signals"] = True
                elif any(_text_contains_hint(lower, ENVIRONMENTAL_VALUE_HINTS) for lower in sample_lower):
                    if col not in patterns["environmental_columns"]:
                        patterns["environmental_columns"].append(col)
                    patterns["has_environmental_signals"] = True

                unique_values = series.nunique(dropna=True)
                if unique_values == 2 and col_lower in {"target", "fraud", "is_fraud", "label", "class", "y"}:
                    patterns["has_binary_target"] = True

            elif pd.api.types.is_numeric_dtype(series):
                patterns["numeric_columns"] += 1
                patterns["has_numeric_features"] = True

                if col_lower in {"target", "price", "value", "amount", "score"}:
                    patterns["has_numeric_target"] = True

                unique_values = series.nunique(dropna=True)
                if unique_values == 2 and col_lower in {"target", "fraud", "is_fraud", "label", "class", "y"}:
                    patterns["has_binary_target"] = True

                if _column_contains_hint(col_lower, GEOSPATIAL_LAT_HINTS) and col not in patterns["geospatial_latitude_columns"]:
                    patterns["geospatial_latitude_columns"].append(col)
                if _column_contains_hint(col_lower, GEOSPATIAL_LON_HINTS) and col not in patterns["geospatial_longitude_columns"]:
                    patterns["geospatial_longitude_columns"].append(col)

                if any(keyword in col_lower for keyword in ["image", "pixel", "frame", "bbox", "mask"]):
                    patterns["has_image_data"] = True
                    if col not in patterns["image_columns"]:
                        patterns["image_columns"].append(col)

                if any(hint in col_lower for hint in OBJECT_DETECTION_HINTS):
                    patterns["has_object_detection_annotations"] = True
                    if col not in patterns["object_detection_columns"]:
                        patterns["object_detection_columns"].append(col)
                    patterns["has_image_data"] = True

                if _column_contains_hint(col_lower, ENVIRONMENTAL_HINTS):
                    if col not in patterns["environmental_columns"]:
                        patterns["environmental_columns"].append(col)
                    patterns["has_environmental_signals"] = True

        if patterns["geospatial_latitude_columns"] and patterns["geospatial_longitude_columns"]:
            patterns["has_geospatial_coordinates"] = True
        if patterns["geospatial_text_columns"]:
            patterns["has_geospatial_text"] = True
        if patterns["image_columns"]:
            patterns["has_image_data"] = True
        if patterns["object_detection_columns"]:
            patterns["has_object_detection_annotations"] = True
        if patterns["environmental_columns"]:
            patterns["has_environmental_signals"] = True

        patterns["has_missing_values"] = bool(df.isna().any().any())

        numeric_df = df.select_dtypes(include=[np.number])
        outlier_count = 0
        for col in numeric_df.columns:
            outlier_count += _count_outliers_iqr(numeric_df[col].dropna())
        patterns["has_outliers"] = outlier_count > max(int(len(df) * 0.05), 0)

        patterns["has_mixed_features"] = bool(patterns["has_categorical_features"] and patterns["has_numeric_features"])

        return patterns

    def _generate_domain_recommendations(
        self,
        primary_domain: str,
        domain_scores: Dict[str, float],
        data_patterns: Dict[str, Any],
    ) -> List[str]:
        recommendations: List[str] = []

        if primary_domain == "finance":
            recommendations.extend(
                [
                    "Analyze price trends and volatility",
                    "Check for outliers in financial metrics",
                    "Examine correlations between financial indicators",
                    "Consider time-based analysis for trends",
                ]
            )
        elif primary_domain == "healthcare":
            recommendations.extend(
                [
                    "Examine patient demographics and outcomes",
                    "Check for missing values in medical records",
                    "Analyze treatment effectiveness patterns",
                    "Consider privacy implications in visualizations",
                ]
            )
        elif primary_domain == "retail":
            recommendations.extend(
                [
                    "Analyze customer purchase patterns",
                    "Examine seasonal trends in sales",
                    "Check product performance metrics",
                    "Identify customer segmentation opportunities",
                ]
            )
        elif primary_domain == "real_estate":
            recommendations.extend(
                [
                    "Analyze price distributions by location",
                    "Examine property features vs. price correlations",
                    "Check for geographical clustering patterns",
                    "Consider market trend analysis",
                ]
            )
        elif primary_domain == "time_series":
            recommendations.extend(
                [
                    "Plot time-based trends and seasonality",
                    "Check for autocorrelation patterns",
                    "Examine moving averages and trends",
                    "Consider forecasting analysis",
                ]
            )
        elif primary_domain == "nlp":
            recommendations.extend(
                [
                    "Analyze text length distributions",
                    "Examine sentiment patterns if applicable",
                    "Check for common words/phrases",
                    "Consider text preprocessing needs",
                ]
            )
        elif primary_domain == "marketing":
            recommendations.extend(
                [
                    "Analyze campaign performance metrics",
                    "Examine customer engagement patterns",
                    "Check conversion funnel analysis",
                    "Consider A/B testing results",
                ]
            )
        elif primary_domain == "tech":
            recommendations.extend(
                [
                    "Review product usage funnels and drop-off points",
                    "Monitor latency, error rates, and API performance over time",
                    "Segment events by platform or device to uncover stability issues",
                    "Correlate feature adoption with retention or engagement metrics",
                ]
            )
        elif primary_domain == "computer_vision":
            recommendations.extend(
                [
                    "Visualize sample images alongside labels to verify data quality",
                    "Check class balance and annotation coverage across image categories",
                    "Inspect image dimensions, channels, and augmentation readiness",
                    "Evaluate label consistency before training",
                ]
            )
            if data_patterns.get("has_object_detection_annotations"):
                recommendations.extend(
                    [
                        "Plot bounding boxes on sample images to validate annotation alignment",
                        "Check distribution of box sizes and aspect ratios for augmentation planning",
                    ]
                )
        elif primary_domain == "fraud":
            recommendations.extend(
                [
                    "Inspect class imbalance and consider resampling or class weighting",
                    "Review high-risk rule triggers and investigation outcomes",
                    "Analyze temporal or geographic spikes in suspicious activity",
                    "Validate feature leakage and ensure robust cross-validation setup",
                ]
            )
        elif primary_domain == "geospatial":
            recommendations.extend(
                [
                    "Visualize latitude and longitude points on interactive maps",
                    "Validate coordinate reference system consistency",
                    "Run geospatial proximity or clustering analyses",
                    "Overlay contextual boundaries to compare metrics",
                ]
            )
        elif primary_domain == "environmental":
            recommendations.extend(
                [
                    "Plot environmental metrics over time to assess trends",
                    "Monitor threshold breaches for regulatory compliance",
                    "Correlate environmental readings with external factors",
                    "Validate sensor calibration and handle missing telemetry",
                ]
            )
        else:
            recommendations.extend(
                [
                    "Start with basic exploratory data analysis",
                    "Check data quality and missing values",
                    "Examine variable distributions",
                    "Identify correlations between variables",
                ]
            )

        if data_patterns.get("has_missing_values"):
            recommendations.insert(0, "Address missing values before analysis")
        if data_patterns.get("has_outliers"):
            recommendations.append("Investigate and handle outliers appropriately")
        if data_patterns.get("has_geospatial_coordinates"):
            recommendations.append("Leverage geospatial analytics for coordinate columns")
        if data_patterns.get("has_geospatial_text"):
            recommendations.append("Geocode address fields to unlock geospatial insights")
        if data_patterns.get("has_image_data"):
            recommendations.append("Audit sample images to confirm annotation quality")
        if data_patterns.get("has_object_detection_annotations"):
            recommendations.append("Review bounding box coverage and label consistency")
        if data_patterns.get("has_environmental_signals"):
            recommendations.append("Track environmental KPIs and set alerts for extremes")

        return recommendations[:6]

    def _extract_ml_schema_signature(self, df: pd.DataFrame, max_sample: int = 5000) -> Dict[str, Any]:
        if df.empty:
            return {}

        sampled = df.sample(min(len(df), max_sample), random_state=42)
        features: Dict[str, Any] = {}
        n_rows, n_cols = sampled.shape
        features["n_rows"] = n_rows
        features["n_cols"] = n_cols

        dtype_counts = sampled.dtypes.astype(str).value_counts().to_dict()
        for dtype, count in dtype_counts.items():
            features[f"dtype_{dtype}"] = count / max(n_cols, 1)

        colnames = " ".join(sampled.columns.str.lower())
        for token in [
            "id",
            "date",
            "time",
            "amount",
            "price",
            "score",
            "code",
            "age",
            "lat",
            "long",
            "desc",
            "text",
        ]:
            features[f"col_has_{token}"] = int(token in colnames)

        numeric_cols = sampled.select_dtypes(include=[np.number])
        if not numeric_cols.empty:
            features["num_mean_mean"] = float(numeric_cols.mean().mean())
            features["num_mean_std"] = float(numeric_cols.std(ddof=0).mean())
            features["num_skew_mean"] = float(numeric_cols.skew().mean())
            features["num_missing_rate"] = float(numeric_cols.isna().mean().mean())

        categorical_cols = sampled.select_dtypes(include=["object", "category"])
        if not categorical_cols.empty:
            cardinalities = [categorical_cols[col].nunique(dropna=False) for col in categorical_cols.columns]
            features["cat_mean_cardinality"] = float(np.mean(cardinalities))
            features["cat_max_cardinality"] = float(np.max(cardinalities))
            features["cat_missing_rate"] = float(categorical_cols.isna().mean().mean())

        return features

    def _ml_predict_domain(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        if (
            not self.enable_ml_classifier
            or self.ml_classifier is None
            or self.ml_label_encoder is None
            or not self.ml_feature_names
            or df.empty
        ):
            return None

        try:
            signature = self._extract_ml_schema_signature(df)
            if not signature:
                return None

            feature_frame = pd.DataFrame([signature], columns=self.ml_feature_names).fillna(0)

            if hasattr(self.ml_classifier, "predict_proba"):
                proba = self.ml_classifier.predict_proba(feature_frame)[0]
                class_index = int(np.argmax(proba))
                confidence = float(proba[class_index])
            else:
                prediction = self.ml_classifier.predict(feature_frame)[0]
                class_index = int(prediction)
                confidence = 1.0
                proba = None

            label = self.ml_label_encoder.inverse_transform([class_index])[0]

            probability_map: Dict[str, float] = {}
            if proba is not None:
                indices = np.arange(len(proba))
                labels = self.ml_label_encoder.inverse_transform(indices)
                probability_map = {label_name: float(score) for label_name, score in zip(labels, proba)}

            return {
                "domain": label,
                "confidence": confidence,
                "probabilities": probability_map,
            }
        except Exception:
            return None


def infer_domain(
    df: pd.DataFrame,
    *,
    analyzer: Optional[DomainAnalyzer] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> DomainInferenceResult:
    """Convenience helper to run :class:`DomainAnalyzer` on a DataFrame."""

    analyzer = analyzer or DomainAnalyzer(enable_ml_classifier=False)
    return analyzer.analyze_dataset_domain(df.columns.tolist(), df, metadata=metadata)
