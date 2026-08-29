// Regression guard for the core/backend diagram format: mermaid rejects
// UNQUOTED parens/brackets inside `[...]` labels — real canvas node ids like
// `3-a415-6ee3f5cf0b4d (DropMissingColumns)` once crashed the Experiments
// tab with "Parse error ... got 'PS'". These tests run the REAL mermaid
// parser (no mock) against the quoted-label format build_mermaid_diagram
// emits, so any future format drift that mermaid can't parse fails here.

import mermaid from 'mermaid';
import { beforeAll, describe, expect, it } from 'vitest';

import { svgFromString } from './MermaidDiagram';

beforeAll(() => {
  mermaid.initialize({ startOnLoad: false, securityLevel: 'strict' });
});

// Mirrors what backend `_execution/diagram.py` persists for a real job:
// human display names with a transformer subtitle, a `<br/>` runtime-summary
// detail line per node, and the humanized model stadium node.
const BACKEND_STYLE_DIAGRAM = `flowchart TD
    data["Input Data"]
    pp0["Encoding (label_encoder)<br/>3 categories"]
    data --> pp0
    pp1["scale[subset] (standard_scaler)<br/>columns: a, b"]
    pp0 --> pp1
    model(["Logistic Regression<br/>acc 0.87 · f1 0.84"])
    pp1 --> model`;

describe('core mermaid format parses', () => {
  it('accepts the quoted-label flowchart the core builder emits', async () => {
    expect(await mermaid.parse(BACKEND_STYLE_DIAGRAM)).toBeTruthy();
  });

  it('accepts a preprocessing-only diagram (no model node)', async () => {
    const chart = `flowchart TD
    data["Input Data"]
    pp0["impute (simple_imputer)"]
    data --> pp0`;
    expect(await mermaid.parse(chart)).toBeTruthy();
  });

  it('rejects the old unquoted format (documents why quoting is mandatory)', async () => {
    const broken = 'flowchart TD\n    pp0[node_1 (DropMissingColumns)]';
    await expect(mermaid.parse(broken)).rejects.toThrow();
  });
});

describe('svgFromString mounts mermaid output', () => {
  it('mounts an SVG whose foreignObject label uses HTML-only tags', () => {
    // Regression: mermaid serializes htmlLabels with an UN-closed <br> inside
    // <p>, which image/svg+xml parsing rejects ("tag mismatch: br ... and p")
    // — the browser then rendered the <parsererror> document in the tab.
    const svg =
      '<svg xmlns="http://www.w3.org/2000/svg" data-testid="m">' +
      '<foreignObject><body xmlns="http://www.w3.org/1999/xhtml">' +
      '<p>Encoding (label_encoder)<br>3 categories</p>' +
      '</body></foreignObject></svg>';
    const el = svgFromString(svg);
    expect(el).not.toBeNull();
    expect(el?.tagName.toLowerCase()).toBe('svg');
  });

  it('returns null when the markup contains no svg', () => {
    expect(svgFromString('<div>no diagram here</div>')).toBeNull();
  });
});
