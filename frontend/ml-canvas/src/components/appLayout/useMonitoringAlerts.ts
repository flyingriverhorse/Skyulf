import { useEffect, useState } from 'react';
import { monitoringApi } from '../../core/api/monitoring';

/** Fetch drift once and poll unresolved errors while the current route is eligible. */
export function useMonitoringAlerts(pathname: string) {
  const [driftAlert, setDriftAlert] = useState(false);
  const [errorAlert, setErrorAlert] = useState(false);

  useEffect(() => {
    monitoringApi.getDriftStatus()
      // OPS-003: the badge reflects alerts still needing triage (new/reopened
      // criticals), not just "some drift happened somewhere, ever".
      .then(s => setDriftAlert(s.unacknowledged_critical > 0 || s.has_drift))
      .catch(() => { });
  }, []);

  useEffect(() => {
    // Drives the red dot on the "Errors" nav link. We don't need a live
    // counter — the dot just signals "there's something to look at", so
    // a 5-minute poll is plenty. Skip entirely when the user is already
    // on /errors (they're seeing the live list) or when the tab is
    // hidden (saves a request per inactive tab per cycle).
    const check = () =>
      monitoringApi.getUnresolvedCount()
        .then(n => setErrorAlert(n > 0))
        .catch(() => { });
    const tick = () => {
      if (document.hidden) return;
      if (pathname === '/errors') return;
      check();
    };
    tick();
    const id = setInterval(tick, 300_000);
    return () => clearInterval(id);
  }, [pathname]);

  return { driftAlert, errorAlert };
}
