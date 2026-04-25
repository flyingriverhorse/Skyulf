import { describe, it, expect, beforeAll } from 'vitest';
import { registry } from './NodeRegistry';
import { initializeRegistry } from './init';

// Smoke loop: every registered node's `getDefaultConfig()` must satisfy
// its own `validate()`. This catches regressions where someone adds a
// new required field to the validator but forgets to update the
// default config — a class of bugs that has bitten us before.
describe('NodeRegistry — every node\'s default config validates', () => {
  beforeAll(() => {
    initializeRegistry();
  });

  it('initializeRegistry registers at least 20 nodes', () => {
    expect(registry.getAll().length).toBeGreaterThanOrEqual(20);
  });

  it.each(registry.getAll().map((n) => [n.type, n] as const))(
    '%s: getDefaultConfig() passes validate()',
    (_type, definition) => {
      // Some nodes (dataset_node) legitimately fail validation until the
      // user picks a dataset; those are the only allowed exceptions and
      // their validators must still return a string error message
      // rather than crash.
      const ALLOWED_INVALID_DEFAULTS = new Set(['dataset_node']);

      const cfg = definition.getDefaultConfig();
      const result = definition.validate(cfg);

      if (ALLOWED_INVALID_DEFAULTS.has(definition.type)) {
        if (!result.isValid) {
          expect(typeof result.message).toBe('string');
          expect(result.message?.length ?? 0).toBeGreaterThan(0);
        }
        return;
      }

      // Defaults must validate cleanly. If this fails, either widen
      // the default config or relax the validator — don't add to the
      // allow-list silently.
      expect(result.isValid, result.message ?? 'no error message').toBe(true);
    },
  );
});
