"""XGBoost estimator integration preserving Skyulf's target-label contract."""

import json
from typing import Any

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import check_classification_targets
from xgboost import XGBClassifier as NativeXGBClassifier  # ty: ignore[unresolved-import]


class XGBClassifier(NativeXGBClassifier):
    """Keep original target labels on the cloneable sklearn-facing classifier.

    Native XGBoost requires consecutive integer labels and checks its own
    ``classes_`` during fit. A native training instance keeps that contract
    separate from our label view, which pruning callbacks need during fit.
    The fitted native state is then retained for prediction and pickling.
    """

    def fit(
        self,
        X: Any,
        y: Any,
        *,
        sample_weight: Any = None,
        eval_set: Any = None,
        **fit_params: Any,
    ) -> "XGBClassifier":
        """Encode training and evaluation targets with this fit's class axis."""
        check_classification_targets(y)
        self._label_encoder = LabelEncoder().fit(y)
        encoded_y = self._label_encoder.transform(y)
        encoded_eval = (
            [(features, self._label_encoder.transform(labels)) for features, labels in eval_set]
            if eval_set is not None
            else None
        )
        self.n_classes_ = len(self._label_encoder.classes_)
        native = NativeXGBClassifier(**self.get_params(deep=False))
        native.fit(
            X,
            encoded_y,
            sample_weight=sample_weight,
            eval_set=encoded_eval,
            **fit_params,
        )
        classes = self._label_encoder.classes_
        native.get_booster().set_attr(
            skyulf_label_encoder=json.dumps(
                {"classes": classes.tolist(), "dtype": classes.dtype.str}
            )
        )
        # Preserve the configured objective when native fit selects a
        # multiclass objective, so a later fit can learn a binary target.
        objective = self.objective
        self.__dict__.update(vars(native))
        self.objective = objective
        return self

    @property
    def classes_(self) -> np.ndarray:
        """Expose the original labels in native probability-column order."""
        encoder = getattr(self, "_label_encoder", None)
        return encoder.classes_ if encoder is not None else super().classes_

    def predict(self, X: Any, *, output_margin: bool = False, **kwargs: Any) -> np.ndarray:
        """Decode class indices while preserving explicitly requested raw margins."""
        predictions = super().predict(X, output_margin=output_margin, **kwargs)
        encoder = getattr(self, "_label_encoder", None)
        if output_margin or encoder is None:
            return predictions
        return encoder.inverse_transform(predictions)

    def load_model(self, fname: Any) -> None:
        """Restore saved label metadata, retaining numeric labels for legacy boosters."""
        super().load_model(fname)
        self.__dict__.pop("_label_encoder", None)
        metadata = self.get_booster().attr("skyulf_label_encoder")
        if metadata is not None:
            state = json.loads(metadata)
            encoder = LabelEncoder()
            encoder.classes_ = np.asarray(state["classes"], dtype=state["dtype"])
            if encoder.classes_.shape != (self.n_classes_,):
                raise ValueError("Saved XGBoost labels do not match the model's class count.")
            self._label_encoder = encoder
