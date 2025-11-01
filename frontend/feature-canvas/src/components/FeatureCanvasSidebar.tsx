import React, { useEffect, useMemo, useState } from 'react';
import type { FeatureNodeCatalogEntry } from '../api';

interface FeatureCanvasSidebarProps {
  nodes: FeatureNodeCatalogEntry[];
  onAddNode?: (node: FeatureNodeCatalogEntry) => void;
}

const formatList = (items?: string[]) => {
  if (!items || !items.length) {
    return '—';
  }
  return items.join(', ');
};

export const FeatureCanvasSidebar: React.FC<FeatureCanvasSidebarProps> = ({ nodes, onAddNode }) => {
  const [query, setQuery] = useState('');
  const [expandedCategories, setExpandedCategories] = useState<string[]>([]);

  const normalizedQuery = query.trim().toLowerCase();

  const filteredNodes = useMemo(() => {
    if (!normalizedQuery) {
      return nodes;
    }

    return nodes.filter((node) => {
      const haystacks: Array<string | undefined | null> = [
        node.label,
        node.type,
        node.description,
      ];

      if (node.category) {
        haystacks.push(node.category);
      }

      if (node.tags && node.tags.length) {
        haystacks.push(node.tags.join(' '));
      }

      return haystacks.some((value) => value?.toLowerCase().includes(normalizedQuery));
    });
  }, [nodes, normalizedQuery]);

  const groupedNodes = useMemo(() => {
    const groups = new Map<string, FeatureNodeCatalogEntry[]>();
    filteredNodes.forEach((node) => {
      const category = node.category?.trim() || 'Other';
      const collection = groups.get(category) ?? [];
      collection.push(node);
      groups.set(category, collection);
    });

    return Array.from(groups.entries())
      .sort(([categoryA], [categoryB]) => categoryA.localeCompare(categoryB))
      .map(([category, entries]) => ({
        category,
        entries: entries.sort((a, b) => (a.label || a.type).localeCompare(b.label || b.type)),
      }));
  }, [filteredNodes]);

  useEffect(() => {
    if (!normalizedQuery) {
      setExpandedCategories([]);
      return;
    }

    setExpandedCategories((previous) => {
      const next = new Set(previous);
      let changed = false;
      groupedNodes.forEach(({ category }) => {
        if (!next.has(category)) {
          next.add(category);
          changed = true;
        }
      });
      return changed ? Array.from(next) : previous;
    });
  }, [groupedNodes, normalizedQuery]);

  const toggleCategory = (category: string) => {
    setExpandedCategories((previous) => {
      const next = new Set(previous);
      if (next.has(category)) {
        next.delete(category);
      } else {
        next.add(category);
      }
      return Array.from(next);
    });
  };

  const totalVisible = filteredNodes.length;
  const hasMatches = totalVisible > 0;

  return (
    <div className="feature-canvas-sidebar" role="presentation">
      <header className="feature-canvas-sidebar__header">
  <h3>Available steps</h3>
        <div className="feature-canvas-sidebar__search">
          <input
            type="search"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search by name, tag, or output"
            className="feature-canvas-sidebar__search-input"
            aria-label="Search available steps"
          />
          {query && (
            <button
              type="button"
              className="feature-canvas-sidebar__search-clear"
              onClick={() => setQuery('')}
              aria-label="Clear search"
            >
              ×
            </button>
          )}
        </div>
        <span className="feature-canvas-sidebar__results">
          Showing {totalVisible} {totalVisible === 1 ? 'step' : 'steps'}
          {normalizedQuery ? ` matching “${query.trim()}”` : ''}
        </span>
      </header>
      {hasMatches ? (
        <div className="feature-canvas-sidebar__sections">
          {groupedNodes.map(({ category, entries }) => {
            const isExpanded = expandedCategories.includes(category);
            return (
              <section key={category} className="feature-canvas-sidebar__section">
                <button
                  type="button"
                  className="feature-canvas-sidebar__section-toggle"
                  onClick={() => toggleCategory(category)}
                  aria-expanded={isExpanded}
                >
                  <span className="feature-canvas-sidebar__section-label">
                    {category.charAt(0).toUpperCase() + category.slice(1)}
                  </span>
                  <span className="feature-canvas-sidebar__section-meta">
                    {entries.length}
                    <span className="feature-canvas-sidebar__section-arrow" data-expanded={isExpanded}>
                      ▾
                    </span>
                  </span>
                </button>
                {isExpanded && (
                  <div className="feature-canvas-sidebar__group">
                    {entries.map((node) => (
                      <button
                        key={node.type}
                        className="feature-canvas-sidebar__card"
                        type="button"
                        onClick={() => onAddNode?.(node)}
                      >
                        <div className="feature-canvas-sidebar__card-body">
                          <span className="feature-canvas-sidebar__card-title">{node.label || node.type}</span>
                          {node.description && (
                            <span className="feature-canvas-sidebar__card-description">{node.description}</span>
                          )}
                          <div className="feature-canvas-sidebar__card-tags">
                            {node.inputs?.length ? (
                              <span className="feature-canvas-sidebar__chip" title={`Inputs: ${formatList(node.inputs)}`}>
                                In: {formatList(node.inputs)}
                              </span>
                            ) : null}
                            {node.outputs?.length ? (
                              <span className="feature-canvas-sidebar__chip" title={`Outputs: ${formatList(node.outputs)}`}>
                                Out: {formatList(node.outputs)}
                              </span>
                            ) : null}
                            {node.parameters && node.parameters.length > 0 ? (
                              <span className="feature-canvas-sidebar__chip">
                                {node.parameters.length} param{node.parameters.length > 1 ? 's' : ''}
                              </span>
                            ) : null}
                          </div>
                        </div>
                      </button>
                    ))}
                  </div>
                )}
              </section>
            );
          })}
        </div>
      ) : (
        <div className="feature-canvas-sidebar__empty">
          <p>No steps match “{query.trim()}”. Try adjusting your search or browse by category.</p>
        </div>
      )}
    </div>
  );
};
