import { createContext, useContext } from 'react';
import type { CanvasLeakageIssue } from '../types/leakage';

export interface CanvasLeakageFeedback {
  nodeIssues: Record<string, CanvasLeakageIssue[]>;
  edgeIssues: Record<string, CanvasLeakageIssue[]>;
  openGuide: () => void;
}

/** Keep presentation state outside graph data so clipboard and persistence stay plain. */
export const CanvasLeakageContext = createContext<CanvasLeakageFeedback>({
  nodeIssues: {},
  edgeIssues: {},
  openGuide: () => {},
});

export const useCanvasLeakageFeedback = (): CanvasLeakageFeedback => useContext(CanvasLeakageContext);
