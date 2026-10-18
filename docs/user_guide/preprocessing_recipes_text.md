# Text and embeddings

[Back to the preprocessing catalog](preprocessing_recipes.md)

These steps tokenize text, build count, TF-IDF or hashed term features, or add
sentence embeddings.

YAML blocks are list fragments to paste under a selected recipe unless a card
marks them as standalone Core configuration. Shared column, row and runtime rules
are in [read the step cards](preprocessing_recipes.md#read-the-step-cards).

These steps use `config/preprocessing.yml`, not `config/pre_split.yml`. `columns` is an explicit
list default `[]`; absent, empty or wholly absent names give no-op state, not text auto-detection.
The target is excluded. Count/TF-IDF default to `column_mode: joined`: selected columns become
one document per row with spaces and a source-name prefix joined by `_`. Opt into
`per_column` for independent vocabularies and source prefixes. Hashing and embeddings remain
joined; Hashing has no learned vocabulary or per-column vocabulary option. Tokenizer processes each source
separately. Null text becomes empty text. Text outputs are attached to the original engine; sklearn
operations can cross through bounded NumPy/pandas data. Output-name collisions raise.

## tokenizer

**Purpose and effect:** Produce normalized tokens and optional token counts; Blue sky becomes blue
sky with count 2.

**File:** `config/preprocessing.yml`.

| Parameter | Type | Default | Allowed values | What it changes |
| --- | --- | --- | --- | --- |
| `columns` | list[string] | [] | Existing column names. | Select explicitly; no automatic detection. |
| `analyzer` | string | word | word, char, char_wb | Choose tokenization unit. |
| `lowercase` | boolean | true | true, false | Lowercase before tokenization. |
| `stop_words` | string/list[string]/null | null | english, custom word list, null | Remove stop words. |
| `ngram_range` | pair[integer] | [1, 1] | Positive ordered [min, max]. | Choose token sequence lengths. |
| `drop_original` | boolean | false | true, false | Remove selected source text columns. |
| `add_token_count` | boolean | false | true, false | Add an integer token count per source. |

**Minimal YAML:**

```yaml
- name: tokenizer
  transformer: tokenizer
  params: {columns: [description], analyzer: word, add_token_count: true}
```

**Common mistake:** The token output is a space-joined string, not a list; default word tokenization
excludes one-character words.

**Details:** Saves rules only. Adds string `<source>__tokens` (space-joined tokens, not a list)
and optionally integer `<source>__token_count`. `"Blue sky"` becomes
`"blue sky"` with count 2; null becomes empty tokens/count 0. Word tokenization
uses sklearn's default token pattern, so one-character words are normally
excluded. Context row for text input; non-text string rendering may be
engine/context-dependent. Dependencies: base/sklearn, no NLP model download.

## count_vectorizer

**Purpose and effect:** Learn a token vocabulary and append dense counts; blue blue contributes
count 2 for blue.

**File:** `config/preprocessing.yml`.

| Parameter | Type | Default | Allowed values | What it changes |
| --- | --- | --- | --- | --- |
| `columns` | list[string] | [] | Existing column names. | Select explicitly; no automatic detection. |
| `column_mode` | string | joined | joined, per_column | Learn one shared vocabulary or a separate vocabulary per source. |
| `min_df` | integer/number | 1 | Integer count or share in [0, 1]. | Remove tokens rarer than this training frequency. |
| `max_df` | integer/number | 1.0 | Integer count or share in [0, 1]. | Remove tokens more common than this training frequency. |
| `max_features` | integer/null | null | Positive; falsy 0 is treated as unset. | Limit vocabulary size. |
| `ngram_range` | pair[integer] | [1, 1] | Positive ordered [min, max]. | Choose token sequence lengths. |
| `lowercase` | boolean | true | true, false | Lowercase before tokenization. |
| `stop_words` | string/list[string]/null | null | english, custom word list, null | Remove stop words. |
| `binary` | boolean | false | true, false | Clip each count to presence/absence. |
| `drop_original` | boolean | false | true, false | Remove selected source text columns. |

**Minimal YAML:**

```yaml
- name: count_vectorizer
  transformer: count_vectorizer
  params: {columns: [description], max_features: 1000, drop_original: true}
```

**Common mistake:** max_df: 1 is a count, while max_df: 1.0 is a share; choose the intended YAML
type.

**Details:** Shares must be in [0,1], counts nonnegative and ordered thresholds viable for
the training corpus. Saves vocabulary and fitted vectorizer. Adds dense
`<prefix>__count__<token>` counts (binary clips presence to 1); `"blue blue"`
has count 2 for blue. Unknown tokens add nothing. Empty/stopword-only training
vocabulary in joined mode produces no useful vectorizer (inspect the warning/state).
In `per_column` mode each source learns its own vocabulary, emits
`<source>__count__<token>` columns, and applies `max_features` independently. N-grams
cannot cross sources. If any selected source has no usable vocabulary, fitting raises
with its name. Missing fitted sources also raise at inference. Null text is an empty
document; empty apply preserves learned width. Context row for text inputs. Base/sklearn.

## tfidf_vectorizer

**Purpose and effect:** Weight token counts by saved inverse document frequencies to reduce
common-word influence.

**File:** `config/preprocessing.yml`.

| Parameter | Type | Default | Allowed values | What it changes |
| --- | --- | --- | --- | --- |
| `columns` | list[string] | [] | Existing column names. | Select explicitly; no automatic detection. |
| `column_mode` | string | joined | joined, per_column | Learn shared vocabulary/IDF or independent vocabulary/IDF per source. |
| `min_df` | integer/number | 1 | Integer count or share in [0, 1]. | Remove tokens rarer than this training frequency. |
| `max_df` | integer/number | 1.0 | Integer count or share in [0, 1]. | Remove tokens more common than this training frequency. |
| `max_features` | integer/null | null | Positive; falsy 0 is treated as unset. | Limit vocabulary size. |
| `ngram_range` | pair[integer] | [1, 1] | Positive ordered [min, max]. | Choose token sequence lengths. |
| `lowercase` | boolean | true | true, false | Lowercase before tokenization. |
| `stop_words` | string/list[string]/null | null | english, custom word list, null | Remove stop words. |
| `sublinear_tf` | boolean | false | true, false | Use logarithmic term-frequency scaling. |
| `drop_original` | boolean | false | true, false | Remove selected source text columns. |

**Minimal YAML:**

```yaml
- name: tfidf_vectorizer
  transformer: tfidf_vectorizer
  params: {columns: [description], max_features: 1000, sublinear_tf: true, drop_original: true}
```

**Common mistake:** binary, norm, use_idf and smooth_idf are not exposed by this wrapper.

**Details:** Saves vocabulary, inverse-document frequencies
and vectorizer. Adds dense floating `<prefix>__tfidf__<token>` values; frequent
words across the training corpus receive less inverse-document weight. Unknown
tokens are ignored; empty text is the zero vector. Normalization follows the
wrapped sklearn default L2; `norm`, use_idf and smooth_idf are not exposed here.
With `per_column`, each source has independent IDF statistics and normalization,
emits `<source>__tfidf__<token>` columns, and gets its own `max_features` limit.
Empty-vocabulary/empty-apply behavior follows count_vectorizer. Context row for
text inputs; base/sklearn.

## hashing_vectorizer

**Purpose and effect:** Append fixed-width numeric hash vectors without learning a vocabulary.

**File:** `config/preprocessing.yml`.

| Parameter | Type | Default | Allowed values | What it changes |
| --- | --- | --- | --- | --- |
| `columns` | list[string] | [] | Existing column names. | Select explicitly; no automatic detection. |
| `n_features` | integer | 1024 | Positive. | Choose dense hash-vector width. |
| `norm` | string/null | l2 | l2, l1, null | Normalize each output vector. |
| `alternate_sign` | boolean | true | true, false | Use signed hashing contributions. |
| `lowercase` | boolean | true | true, false | Lowercase before tokenization. |
| `stop_words` | string/list[string]/null | null | english, custom word list, null | Remove stop words. |
| `drop_original` | boolean | false | true, false | Remove selected source text columns. |

**Minimal YAML:**

```yaml
- name: hashing_vectorizer
  transformer: hashing_vectorizer
  params: {columns: [description], n_features: 1024, norm: l2, drop_original: true}
```

**Common mistake:** The final output is dense; a very large n_features value consumes dense memory.

**Details:** No public ngram_range or vocabulary field.
Saves fixed configuration/vectorizer, not corpus statistics. Adds dense float
`<prefix>__hash__0` through `n_features-1`; new words hash into the same width
and collisions are expected. Empty documents produce zero vectors. Large widths
consume dense memory even though sklearn's intermediate result is sparse.
Context row for text inputs; base/sklearn.

## sentence_embedder

**Purpose and effect:** Append pretrained semantic vectors; similar text should have similar
representations.

**File:** `config/preprocessing.yml`.

| Parameter | Type | Default | Allowed values | What it changes |
| --- | --- | --- | --- | --- |
| `columns` | list[string] | [] | Existing column names. | Select explicitly; no automatic detection. |
| `model_name` | string | all-MiniLM-L6-v2 | Pretrained sentence-transformers model; empty uses default. | Choose the captured encoder. |
| `normalize` | boolean | true | true, false | Normalize embeddings. |
| `drop_original` | boolean | false | true, false | Remove selected source text columns. |

**Minimal YAML:**

```yaml
- name: sentence_embedder
  transformer: sentence_embedder
  params: {columns: [description], model_name: all-MiniLM-L6-v2, normalize: true, drop_original: true}
```

**Common mistake:** Local embedding success does not establish Spark-worker or REST admission; the
packaged snapshot and dependencies matter.

**Details:** Appends float `<prefix>__emb__<index>` features whose width comes from
the selected pretrained encoder. Semantically similar text should have similar
vectors; exact values depend on that captured model. No task-specific fine-tuning
occurs. Fit may download the model on a cold cache and saves its weights,
tokenizer/settings, SHA256 and exact dependency versions; modern apply restores
the saved CPU snapshot rather than fetching changed model weights. Snapshot
size is bounded (256 MiB) in addition to model-package budgets. Load only trusted
producer artifacts because native PyTorch serialization is executable. Null text
is encoded as empty text, not automatically a zero vector. Context is the
pretrained encoder's supported local apply contract; floating tolerances and
dependency pins matter. Requires optional sentence-transformers/PyTorch stack;
base installation alone is insufficient. No Spark-worker/REST certification
is implied by successful local embedding.
