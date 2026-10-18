"""Execute project guide examples and resolve public documentation links offline."""

import io
import json
import pickle
import re
import shutil
import subprocess
import sys
import unicodedata
from pathlib import Path
from urllib.parse import unquote, urlsplit

import numpy as np
import pandas as pd
import pytest

yaml = pytest.importorskip("yaml")
pl = pytest.importorskip("polars")

from skyulf import SkyulfPipeline
from skyulf.data.dataset import SplitDataset
from skyulf.inference.fitted_pipeline import load_pipeline, predict_pipeline
from skyulf.inference.pipeline_scoring import score_pipeline
from skyulf.inference.project_scoring import run_project_scoring
from skyulf.integrations.databricks.projects.project import load_project_workflow
from skyulf.integrations.databricks.scoring.batch.frame_batch import fit_workflow
from skyulf.integrations.databricks.training.fitting.candidate import (
    TrainingSpec,
    apply_pre_split_step,
    candidate_config,
    split_labeled_snapshot,
)
from skyulf.integrations.databricks.training.tuning.cv import CVSpec
from skyulf.registry import NodeRegistry

ROOT = Path(__file__).resolve().parents[4]
TEMPLATE = ROOT / "skyulf-core/templates/databricks/template/{{.project_name}}"
GUIDES = ROOT / "docs/user_guide"


CATALOG_PAGES = sorted(GUIDES.glob("preprocessing_recipes_*.md"))


def _catalog_mappings():
    """Read the hub index: registered ID -> (family page, card anchor)."""
    text = (GUIDES / "preprocessing_recipes.md").read_text(encoding="utf-8")
    return re.findall(
        r"^\| (?:<span id=\"[a-z_0-9-]+\"></span>)?([A-Za-z_][A-Za-z_0-9]*) \| "
        r"\[[^]]+\]\((preprocessing_recipes_[a-z_]+\.md)#([a-z_0-9-]+)\) \|$",
        text,
        re.MULTILINE,
    )


def _catalog_card_list():
    """List each family page's `## ` card bodies as (page, heading anchor, body)."""
    cards = []
    for page in CATALOG_PAGES:
        text = page.read_text(encoding="utf-8")
        for card in re.split(r"(?m)^## ", text)[1:]:
            anchor = next(iter(_anchors(f"## {card.splitlines()[0]}")))
            cards.append((page.name, anchor, card))
    return cards


def _catalog_cards():
    """Index card bodies by (page, heading anchor)."""
    return {(page, anchor): card for page, anchor, card in _catalog_card_list()}


def test_catalog_maps_every_registered_transformer_to_a_real_card():
    """New or renamed preprocessing operations must update the user-facing catalog."""
    mappings = _catalog_mappings()
    assert {name for name, _, _ in mappings} == set(NodeRegistry.list_transformers())
    assert len(mappings) == len({name for name, _, _ in mappings})
    cards = _catalog_card_list()
    # Each target must be a real card heading, not a hub stub span.
    assert {(page, anchor) for _, page, anchor in mappings} <= {(p, a) for p, a, _ in cards}
    # Every card exists exactly once across the hub and split pages.
    hub = (GUIDES / "preprocessing_recipes.md").read_text(encoding="utf-8")
    hub_headings = {a for h in re.findall(r"^#{1,6} .+$", hub, re.M) for a in _anchors(h)}
    assert not hub_headings & {anchor for _, _, anchor in mappings}
    for _, _, anchor in mappings:
        owners = [page for page, card, _ in cards if card == anchor]
        assert len(owners) == 1, f"{anchor} appears on {owners}"


def test_catalog_documents_each_metadata_parameter():
    """Metadata changes must update the corresponding step's parameter table."""
    cards = {
        key: set(re.findall(r"`([^`]+)`", "\n".join(re.findall(r"^\| (.*?) \|", body, re.M))))
        for key, body in _catalog_cards().items()
    }
    metadata = NodeRegistry.get_all_metadata()
    missing = {}
    for name, page, anchor in _catalog_mappings():
        params = set(metadata[name].get("params", {}))
        absent = params - cards[(page, anchor)]
        if absent:
            missing[name] = sorted(absent)
    assert not missing, f"Registry parameters absent from their card tables: {missing}"


def _anchors(document):
    """Resolve the standard Markdown heading slugs and explicit HTML anchors."""
    anchors = set(re.findall(r'\bid=["\']([^"\']+)', document))
    for heading in re.findall(r"^#{1,6} (.+)$", document, re.MULTILINE):
        plain = unicodedata.normalize("NFKD", heading).encode("ascii", "ignore").decode()
        plain = re.sub(r"[^\w\s-]", "", plain).strip().lower()
        anchors.add(re.sub(r"[-\s]+", "-", plain))
    return anchors


def test_template_public_links_resolve_to_documented_pages_and_anchors():
    """Generated offline guides must never point at nonexistent public source pages."""
    sources = [ROOT / "README.md", *TEMPLATE.rglob("*.md"), *TEMPLATE.rglob("*.md.tmpl")]
    checked = []
    for source in sources:
        for url in re.findall(
            r'https://www\.skyulf\.com/manual/[^\s)>"\]]+', source.read_text(encoding="utf-8")
        ):
            parsed = urlsplit(url)
            relative = unquote(parsed.path).removeprefix("/manual/")
            page = ROOT / "docs" / (relative.removesuffix(".html") + ".md")
            assert page.is_file(), f"{source}: {url} has no source page"
            if parsed.fragment:
                assert unquote(parsed.fragment) in _anchors(page.read_text(encoding="utf-8")), (
                    f"{source}: missing anchor in {url}"
                )
            checked.append(url)
    assert checked


def test_databricks_topic_links_resolve_after_splitting_guides():
    """Moving a guide must preserve reachable files and sections from its readers."""
    sources = [
        *GUIDES.glob("databricks_*.md"),
        *GUIDES.glob("preprocessing_recipes*.md"),
        GUIDES / "preprocessing_context.md",
        GUIDES / "preprocessing_yaml_walkthrough.md",
    ]
    checked = []
    for source in sources:
        document = source.read_text(encoding="utf-8")
        document = re.sub(r"```.*?```", "", document, flags=re.DOTALL)
        for target in re.findall(r"\]\(([^\s)]+)\)", document):
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc:
                continue
            page = source.parent / unquote(parsed.path) if parsed.path else source
            assert page.is_file(), f"{source.name}: missing {target}"
            if parsed.fragment and page.suffix == ".md":
                assert unquote(parsed.fragment) in _anchors(page.read_text(encoding="utf-8")), (
                    f"{source.name}: missing anchor in {target}"
                )
            checked.append(target)
    assert checked


SHARED_HEADINGS = {
    # Every task page follows the same five-part outline, so these repeat by design.
    "when you need it": "per-page outline",
    "what to edit": "per-page outline",
    "the command": "per-page outline",
    "what you should see": "per-page outline",
    "common errors and fixes": "per-page outline",
}


def test_databricks_guides_give_each_topic_heading_one_owner_page():
    """A topic written on two pages drifts apart; one page owns it and the others link."""
    owners = {}
    for page in sorted(GUIDES.glob("databricks_*.md")):
        document = re.sub(r"```.*?```", "", page.read_text(encoding="utf-8"), flags=re.DOTALL)
        for heading in re.findall(r"^#{2,3} (.+)$", document, re.MULTILINE):
            key = " ".join(re.sub(r"[^a-z0-9 ]", " ", heading.lower()).split())
            owners.setdefault(key, set()).add(page.name)
    repeated = {
        key: sorted(pages)
        for key, pages in owners.items()
        if len(pages) > 1 and key not in SHARED_HEADINGS
    }
    assert not repeated, (
        f"Merge into one owner page and link, or whitelist with a reason: {repeated}"
    )


def _blocks(page, language):
    """Read executable declarations directly from the published guide source."""
    return re.findall(rf"```{language}\n(.*?)```", page.read_text(encoding="utf-8"), re.DOTALL)


def _project(tmp_path, pre_split, preprocessing):
    """Use the shipped project Python with literal phase files from the guide."""
    features = tmp_path / "src/features"
    shutil.copytree(TEMPLATE / "src/features", features)
    config = tmp_path / "config"
    config.mkdir()
    shutil.copyfile(TEMPLATE / "config/scoring.yml", config / "scoring.yml")
    for filename, value in (("pre_split.yml", pre_split), ("preprocessing.yml", preprocessing)):
        (config / filename).write_text(yaml.safe_dump(value), encoding="utf-8")
    return features


def test_catalog_customer_example_loads_and_replays_saved_predictions(tmp_path):
    """The complete customer tutorial must resolve its target filter and unseen inputs."""
    page = GUIDES / "preprocessing_recipes.md"
    document = page.read_text(encoding="utf-8")
    assert document.index("skip_target_pre_split_steps: true") < document.index(
        "python src/tools/preview.py --action train"
    )
    blocks = [yaml.safe_load(block) for block in _blocks(page, "yaml")]
    training = blocks[0]
    pre_split = next(
        item for item in blocks if isinstance(item, dict) and "eligible" in item.get("recipes", {})
    )
    preprocessing = next(
        item for item in blocks if isinstance(item, dict) and "customer" in item.get("recipes", {})
    )
    features = _project(tmp_path, pre_split, preprocessing)
    scoring = next(item for item in blocks if "skip_target_pre_split_steps" in item)
    (tmp_path / "config/scoring.yml").write_text(yaml.safe_dump(scoring), encoding="utf-8")
    defaults = training["defaults"]
    config = load_project_workflow(
        {
            "engine": "pandas",
            "target_column": defaults["target_column"],
            "input_columns": defaults["input_columns"],
            "pipeline": {"preprocessing": [], "modeling": training["models"]["main"]["model"]},
        },
        features,
        pre_split_recipe=defaults["pre_split_recipe"],
        preprocessing_recipe=defaults["preprocessing_recipe"],
    )
    data = pd.read_csv(io.StringIO(_blocks(page, "csv")[0]))
    pipeline = SkyulfPipeline(config["pipeline"])
    pipeline.fit(data.drop(columns=["customer_id"]), target_column=defaults["target_column"])
    query = pd.DataFrame({"age": [40.0], "income": [np.nan], "tenure": [3.0], "segment": ["NEW"]})
    saved = tmp_path / "customer.pkl"
    saved.write_bytes(pickle.dumps(pipeline))
    restored = pickle.loads(saved.read_bytes())
    actual = restored.predict(query)
    np.testing.assert_array_equal(pipeline.predict(query), actual)
    assert len(actual) == 1


def test_walkthrough_sentinel_recipe_preserves_excluded_row_and_reason(tmp_path):
    """The tutorial must exclude the sentinel before prediction without dropping its row."""
    page = GUIDES / "databricks_bundle_walkthrough.md"
    blocks = [yaml.safe_load(block) for block in _blocks(page, "yaml")]
    pre_split = next(item for item in blocks if "known_income" in item.get("recipes", {}))
    preprocessing = next(item for item in blocks if "claim_inputs" in item.get("recipes", {}))
    features = _project(tmp_path, pre_split, preprocessing)
    config = load_project_workflow(
        {
            "engine": "pandas",
            "target_column": "claim_amount",
            "input_columns": ["income", "age"],
            "pipeline": {"preprocessing": [], "modeling": {"type": "linear_regression"}},
        },
        features,
        pre_split_recipe="known_income",
        preprocessing_recipe="claim_inputs",
    )["pipeline"]
    pipeline = SkyulfPipeline(config)
    pipeline.fit(
        pd.DataFrame(
            {
                "income": [10.0, 20.0, 30.0, 40.0],
                "age": [30.0, 40.0, 50.0, 60.0],
                "claim_amount": [1.0, 2.0, 3.0, 4.0],
            }
        ),
        target_column="claim_amount",
    )
    received = []

    def predict(frame):
        """Retain the accepted input to prove the sentinel never reaches the model."""
        received.extend(frame.income.tolist())
        return pd.DataFrame({"prediction": pipeline.predict(frame)})

    result = run_project_scoring(
        pd.DataFrame({"income": [-999.0, 10.0], "age": [40.0, np.nan]}),
        predict,
        source=config["project_python_source"],
        config=config["project_scoring"],
        row_keys=[],
        prediction_dtypes={"prediction": "float64"},
    )
    assert received == [10.0]
    assert result.scoring_status.tolist() == ["excluded", "predicted"]
    assert pd.isna(result.prediction.iloc[0])
    assert result.exclusion_reason.iloc[0] == "pre_split:known_income"
    assert np.isfinite(result.prediction.iloc[1])


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_scoring_guide_yaml_fits_and_replays_its_saved_rules(tmp_path, engine):
    """The published YAML must preserve exclusions and bands after editable files change."""
    page = GUIDES / "databricks_scoring.md"
    empty = {"version": 1, "recipes": {"default": []}}
    features = _project(tmp_path, empty, empty)
    (tmp_path / "config/scoring.yml").write_text(
        _example(page, "customer-scoring", "yaml"), encoding="utf-8"
    )
    workflow = load_project_workflow(
        {
            "engine": engine,
            "target_column": "churn",
            "input_columns": ["age"],
            "pipeline": {"preprocessing": [], "modeling": {"type": "logistic_regression"}},
        },
        features,
    )
    train = pd.DataFrame(
        {"age": [20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0], "churn": [0] * 4 + [1] * 4}
    )
    native = pl.from_pandas(train) if engine == "polars" else train
    artifact = fit_workflow(
        workflow["pipeline"],
        SplitDataset(train=native, test=native[:0]),
        target_column="churn",
        artifact_path=tmp_path / "artifact",
        max_rows=100,
        max_bytes=100_000,
    )
    query = pd.DataFrame({"age": [35.0, -1.0, np.nan, 60.0]})
    result = score_pipeline(query, artifact)
    assert result.scoring_status.tolist() == ["predicted", "excluded", "excluded", "predicted"]
    assert result.exclusion_reason.iloc[1:3].tolist() == ["outside_range:age", "missing:age"]
    assert result.prediction.isna().tolist() == [False, True, True, False]
    probability = result.probability_1.take([0, 3]).astype(float)
    expected_bands = np.asarray(["low", "medium", "high"])[
        np.searchsorted([0.3, 0.7], probability, side="right")
    ]
    assert result.risk_band.take([0, 3]).tolist() == expected_bands.tolist()
    assert result.risk_band.iloc[1:3].isna().all()
    _replay_after_project_edit(tmp_path, features, query, result)
    assert artifact.manifest.project_source_sha256


def _example(page, name, language):
    """Copy a complete marked declaration without reconstructing its implementation."""
    matches = re.findall(
        rf"<!-- example:{name} -->\s*```{language}\n(.*?)```",
        page.read_text(encoding="utf-8"),
        re.DOTALL,
    )
    assert len(matches) == 1, f"{page.name}: missing or duplicate {name} example"
    return matches[0]


def test_yaml_walkthrough_inventory_reaches_every_registered_step():
    """The entry guide must keep new operations reachable without duplicating parameter cards."""
    page = GUIDES / "preprocessing_yaml_walkthrough.md"
    inventory = page.read_text(encoding="utf-8").split("<!-- yaml-catalog:start -->")[1]
    inventory = inventory.split("<!-- yaml-catalog:end -->")[0]
    names = re.findall(r"`([^`]+)`", inventory)
    assert len(names) == len(set(names))
    assert set(names) == set(NodeRegistry.list_transformers())
    assert set(re.findall(r"\]\(([^)]+)\)", inventory)) == {path.name for path in CATALOG_PAGES}


def _yaml_walkthrough_project(tmp_path, engine, recipe):
    """Load the literal complete files with unmodified generated feature modules."""
    page = GUIDES / "preprocessing_yaml_walkthrough.md"
    pre_split = yaml.safe_load(_example(page, "yaml-customer-pre-split", "yaml"))
    preprocessing = yaml.safe_load(_example(page, "yaml-customer-preprocessing", "yaml"))
    training = yaml.safe_load(_example(page, "yaml-customer-training", "yaml"))
    features = _project(tmp_path, pre_split, preprocessing)
    defaults = training["defaults"]
    defaults["preprocessing_recipe"] = recipe
    (tmp_path / "config/training.yml").write_text(yaml.safe_dump(training), encoding="utf-8")
    workflow = load_project_workflow(
        {
            "engine": engine,
            "target_column": defaults["target_column"],
            "input_columns": defaults["input_columns"],
            "pipeline": {
                "engine": engine,
                "preprocessing": [],
                "modeling": {},
            },
        },
        features,
    )
    assert workflow["pipeline"]["project_scoring"]["pre_split"]["skipped_target_steps"] == []
    return workflow, features


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("recipe", ["customer", "fixed_categories", "isolation", "local_density"])
def test_yaml_walkthrough_recipes_fit_save_reload_and_preserve_scoring_rows(
    tmp_path, engine, recipe
):
    """Copied YAML must train and retain raw-query exclusions after the project is edited."""
    page = GUIDES / "preprocessing_yaml_walkthrough.md"
    workflow, features = _yaml_walkthrough_project(tmp_path, engine, recipe)
    data = pd.read_csv(io.StringIO(_example(page, "yaml-customer-data", "csv")))
    spec = TrainingSpec(
        table="workspace.example.customers",
        version=0,
        record_key_columns=("customer_id",),
        input_columns=("age", "income", "tenure", "segment"),
        target_column="churn",
        max_rows=100,
        max_bytes=100_000,
        pre_split_steps=tuple(workflow["pre_split_steps"]),
        stratify=True,
    )
    train, test, unavailable = split_labeled_snapshot(data, spec, engine=engine)
    assert unavailable == 0
    assert test.attrs["survivor_rows"] == 31
    assert test.attrs["pre_split_filter_counts"][-1]["excluded_rows"] == 1
    # Match actual candidate training: keep raw surviving features and save fixed normalization.
    config = candidate_config(
        spec,
        workflow["pipeline"],
        engine=engine,
        cv=CVSpec(),
        metric="heldout_accuracy",
        min_improvement=0.0,
        champion_version=None,
        quality_threshold=None,
        risk_category=None,
    )
    dataset = SplitDataset(
        train=pl.from_pandas(train) if engine == "polars" else train,
        test=pl.from_pandas(test) if engine == "polars" else test,
    )
    artifact = fit_workflow(
        config,
        dataset,
        target_column="churn",
        artifact_path=tmp_path / "artifact",
        max_rows=100,
        max_bytes=100_000,
    )
    query = pd.read_csv(io.StringIO(_example(page, "yaml-customer-query", "csv")))
    query = query.drop(columns="customer_id")
    original = query.copy(deep=True)
    expected = score_pipeline(query, artifact)
    missing = query.iloc[[4]].copy()
    missing["income"] = np.nan
    pd.testing.assert_frame_equal(
        score_pipeline(query.iloc[[4]], artifact), score_pipeline(missing, artifact)
    )
    pd.testing.assert_frame_equal(query, original)
    if engine == "polars":
        pd.testing.assert_frame_equal(score_pipeline(pl.from_pandas(query), artifact), expected)
    assert len(expected) == len(query)
    assert expected.scoring_status.iloc[2] == "excluded"
    assert expected.exclusion_reason.iloc[2] == "pre_split:age_range"
    assert pd.isna(expected.prediction.iloc[2])
    if recipe in ("customer", "fixed_categories"):
        assert expected.scoring_status.tolist() == [
            "predicted",
            "predicted",
            "excluded",
            "predicted",
            "predicted",
        ]
    else:
        assert expected.scoring_status.iloc[3] == "excluded"
        assert expected.exclusion_reason.iloc[3] == "outlier:unusual_measurements"
        assert pd.isna(expected.prediction.iloc[3])
    _assert_yaml_recipe_transformed_values(artifact, query, train, engine, recipe)
    for module in features.rglob("*.py"):
        assert (
            module.read_bytes()
            == (TEMPLATE / "src/features" / module.relative_to(features)).read_bytes()
        )
    restored = load_pipeline(tmp_path / "artifact")
    pd.testing.assert_frame_equal(score_pipeline(query, restored), expected)
    _replay_after_project_edit(tmp_path, features, query, expected)
    assert artifact.manifest.project_source_sha256 == restored.manifest.project_source_sha256


def _assert_yaml_recipe_transformed_values(artifact, query, train, engine, recipe):
    """Pin real output effects rather than accepting an unchanged or empty pipeline."""
    if recipe not in ("customer", "fixed_categories"):
        return
    # These rows survive the fixed age check and exercise the saved numeric/category state.
    probe = query.iloc[:2]
    native = pl.from_pandas(probe) if engine == "polars" else probe
    transformed = artifact.pipeline.feature_engineer.transform(native)
    frame = transformed.to_pandas() if isinstance(transformed, pl.DataFrame) else transformed
    if recipe == "fixed_categories":
        names = ["segment_a", "segment_b", "segment_c"]
        assert frame.columns.tolist() == ["age", "income", "tenure", *names]
        np.testing.assert_array_equal(frame[names], [[1, 0, 0], [0, 0, 0]])
    else:
        np.testing.assert_allclose(np.linalg.norm(frame[["age", "income", "tenure"]], axis=1), 1)
        counts = train.segment.value_counts()
        np.testing.assert_allclose(
            frame.segment, [counts["A"] / len(train), counts[counts < 3].sum() / len(train)]
        )
    assert np.isfinite(frame.to_numpy()).all()


def _custom_workflow(features, defaults):
    """Resolve the guide's selected recipe through the actual captured-project loader."""
    return load_project_workflow(
        {
            "engine": "pandas",
            "target_column": defaults["target_column"],
            "input_columns": defaults["input_columns"],
            "pipeline": {"preprocessing": [], "modeling": {"type": "linear_regression"}},
        },
        features,
        pre_split_recipe=defaults["pre_split_recipe"],
        preprocessing_recipe=defaults["preprocessing_recipe"],
    )


def _fit_documented_workflow(workflow, train, artifact_path):
    """Fit and save with the same bounded native workflow used by Bundle training."""
    return fit_workflow(
        workflow["pipeline"],
        SplitDataset(train=train, test=train.iloc[:0]),
        target_column="claim_amount",
        artifact_path=artifact_path,
        max_rows=100,
        max_bytes=100_000,
    )


def _replay_after_project_edit(tmp_path, features, query, expected):
    """A fresh interpreter must score using saved source after editable files change."""
    for module in features.rglob("*.py"):
        module.write_text(
            "raise RuntimeError('current project must not execute')\n", encoding="utf-8"
        )
    for declaration in (tmp_path / "config").glob("*.yml"):
        declaration.write_text("version: 999\n", encoding="utf-8")
    query.to_json(tmp_path / "query.json", orient="table")
    source = """
import sys
import pandas as pd
from skyulf.inference.fitted_pipeline import load_pipeline
from skyulf.inference.pipeline_scoring import score_pipeline
artifact = load_pipeline(sys.argv[1])
query = pd.read_json(sys.argv[2], orient="table")
print(score_pipeline(query, artifact).to_json(orient="split"))
"""
    completed = subprocess.run(
        [sys.executable, "-c", source, str(tmp_path / "artifact"), str(tmp_path / "query.json")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    actual = pd.DataFrame(**json.loads(completed.stdout)).astype(expected.dtypes.to_dict())
    pd.testing.assert_frame_equal(actual, expected, atol=1e-8)


def test_custom_clipping_selection_learns_and_reloads_captured_source(tmp_path):
    """The literal custom selector must learn training bounds and reuse them after saving."""
    page = GUIDES / "databricks_custom_steps.md"
    declaration = yaml.safe_load(_example(page, "custom-clipping-recipe", "yaml"))
    features = _project(tmp_path, {"version": 1, "recipes": {"default": []}}, declaration)
    (features / "income_rules.py").write_text(
        _example(page, "custom-clipping-code", "python"), encoding="utf-8"
    )
    workflow = _custom_workflow(
        features,
        {
            "input_columns": ["income"],
            "target_column": "claim_amount",
            "pre_split_recipe": "default",
            "preprocessing_recipe": "custom_clip",
        },
    )
    train = pd.DataFrame({"income": [10.0, 20.0, 30.0, 40.0], "claim_amount": [1.0, 2.0, 3.0, 4.0]})
    artifact = _fit_documented_workflow(workflow, train, tmp_path / "artifact")
    restored = load_pipeline(tmp_path / "artifact")
    query = pd.DataFrame({"income": [-1000.0, 1000.0]})
    transformed = restored.pipeline.feature_engineer.transform(query)
    np.testing.assert_allclose(transformed["income"], [10.3, 39.7])
    expected = predict_pipeline(query, artifact)
    _replay_after_project_edit(tmp_path, features, query, expected)
    assert artifact.manifest.project_source_sha256 == restored.manifest.project_source_sha256


@pytest.mark.parametrize("with_target", [False, True], ids=["feature-only", "target-filter"])
def test_custom_filter_examples_keep_survivors_and_freeze_scoring_policy(tmp_path, with_target):
    """Custom filters must preserve rows and require an explicit target-skip choice."""
    page = GUIDES / "databricks_custom_pre_split.md"
    declaration = yaml.safe_load(_example(page, "custom-filter-recipes", "yaml"))
    features = _project(
        tmp_path, declaration, {"version": 1, "recipes": {"default": [], "none": []}}
    )
    (features / "eligibility_rules.py").write_text(
        _example(page, "custom-filter-code", "python"), encoding="utf-8"
    )
    defaults = yaml.safe_load(_example(page, "custom-filter-selection", "yaml"))["defaults"]
    if with_target:
        defaults["pre_split_recipe"] = "adult_labeled"
        with pytest.raises(ValueError, match="reads target 'claim_amount'"):
            _custom_workflow(features, defaults)
        (tmp_path / "config/scoring.yml").write_text(
            _example(page, "custom-filter-target-skip", "yaml"), encoding="utf-8"
        )
    workflow = _custom_workflow(features, defaults)
    policy = workflow["pipeline"]["project_scoring"]["pre_split"]
    assert policy["skipped_target_steps"] == (["known_claim_amount"] if with_target else [])
    train = pd.read_csv(io.StringIO(_example(page, "custom-filter-training", "csv")))
    if with_target:
        train.loc[train["record_id"] == 104, "claim_amount"] = np.nan
    survivors = train
    for step in workflow["pre_split_steps"]:
        survivors = apply_pre_split_step(
            survivors, step, keys=["record_id"], target_column="claim_amount"
        )
    kept = [102, 103, 105] if with_target else [102, 103, 104, 105]
    assert isinstance(survivors, pd.DataFrame)
    pd.testing.assert_frame_equal(
        survivors.reset_index(drop=True), train[train.record_id.isin(kept)].reset_index(drop=True)
    )
    artifact = _fit_documented_workflow(
        workflow, survivors.drop(columns=["record_id"]), tmp_path / "artifact"
    )
    query = pd.read_csv(io.StringIO(_example(page, "custom-filter-scoring", "csv")))
    expected = score_pipeline(query, artifact)
    assert expected.scoring_status.tolist() == ["excluded", "predicted", "predicted", "excluded"]
    assert expected.prediction.isna().tolist() == [True, False, False, True]
    assert [expected.exclusion_reason.iloc[i] for i in (0, 3)] == ["pre_split:adult_only"] * 2
    _replay_after_project_edit(tmp_path, features, query, expected)
    assert artifact.manifest.project_source_sha256
