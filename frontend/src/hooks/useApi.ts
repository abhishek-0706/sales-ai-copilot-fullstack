import { useEffect, useState, type DependencyList } from "react";

export interface ApiState<T> {
  data?: T;
  error?: Error;
  loading: boolean;
}

/** Runs `load` when deps change; aborts the in-flight request on change/unmount. Pass null to skip. */
export function useApi<T>(load: ((signal: AbortSignal) => Promise<T>) | null, deps: DependencyList): ApiState<T> {
  const [state, setState] = useState<ApiState<T>>({ loading: load !== null });

  useEffect(() => {
    if (!load) return;
    const ctrl = new AbortController();
    setState({ loading: true });
    load(ctrl.signal)
      .then((data) => setState({ data, loading: false }))
      .catch((error: Error) => {
        if (!ctrl.signal.aborted) setState({ error, loading: false });
      });
    return () => ctrl.abort();
  }, deps);

  return state;
}
