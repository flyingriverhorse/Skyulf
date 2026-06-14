"""Naive Bayes classification models.

Provides MultinomialNB and BernoulliNB as Calculator/Applier pairs registered
in the NodeRegistry.  Both are well-suited for text classification pipelines
when paired with a vectorization node (CountVectorizer or TfidfVectorizer).

Note: MultinomialNB requires **non-negative** features (raw counts or TF-IDF
scores work; TF-IDF with ``sublinear_tf=True`` still gives non-negative values).
BernoulliNB works with binary or boolean feature matrices as well as counts.
"""

from sklearn.naive_bayes import BernoulliNB, MultinomialNB

from ..core.meta.decorators import node_meta
from ..registry import NodeRegistry
from .sklearn_wrapper import SklearnApplier, SklearnCalculator

# ── Multinomial NB ────────────────────────────────────────────────────────────


class MultinomialNBApplier(SklearnApplier):
    """Multinomial Naive Bayes Applier."""


@NodeRegistry.register("multinomial_nb", MultinomialNBApplier)
@node_meta(
    id="multinomial_nb",
    name="Multinomial Naive Bayes (counts / text)",
    category="Modeling",
    description=(
        "Naive Bayes classifier for multinomially-distributed features "
        "(e.g. token counts or TF-IDF). "
        "Requires non-negative input features."
    ),
    params={"alpha": 1.0, "fit_prior": True},
    tags=["text", "nlp", "classification", "naive_bayes"],
)
class MultinomialNBCalculator(SklearnCalculator):
    """Multinomial Naive Bayes Calculator."""

    def __init__(self):
        super().__init__(
            model_class=MultinomialNB,
            default_params={"alpha": 1.0, "fit_prior": True},
            problem_type="classification",
        )

    @property
    def problem_type(self) -> str:
        return "classification"


# ── Bernoulli NB ──────────────────────────────────────────────────────────────


class BernoulliNBApplier(SklearnApplier):
    """Bernoulli Naive Bayes Applier."""


@NodeRegistry.register("bernoulli_nb", BernoulliNBApplier)
@node_meta(
    id="bernoulli_nb",
    name="Bernoulli Naive Bayes (binary / text)",
    category="Modeling",
    description=(
        "Naive Bayes classifier designed for binary/boolean features. "
        "Each feature is treated as a binary indicator of a token's presence. "
        "Also works with continuous features via a binarization threshold."
    ),
    params={"alpha": 1.0, "binarize": 0.0, "fit_prior": True},
    tags=["text", "nlp", "classification", "naive_bayes"],
)
class BernoulliNBCalculator(SklearnCalculator):
    """Bernoulli Naive Bayes Calculator."""

    def __init__(self):
        super().__init__(
            model_class=BernoulliNB,
            default_params={"alpha": 1.0, "binarize": 0.0, "fit_prior": True},
            problem_type="classification",
        )

    @property
    def problem_type(self) -> str:
        return "classification"
