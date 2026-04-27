/**
 * Realtime job-event socket (singleton).
 *
 * Replaces the per-component HTTP polling for job status. The backend
 * publishes small "invalidator" events on `/ws/jobs`; consumers
 * subscribe with a handler and decide what to refetch.
 *
 * The socket is process-wide on the page: one connection per browser
 * tab, shared by every subscriber. If the connection drops we
 * exponential-backoff reconnect; while disconnected, callers are
 * expected to fall back to their existing poll loop.
 */

export interface JobEvent {
    event: 'status' | 'progress' | 'created' | 'deleted';
    job_id: string;
    status?: string;
    progress?: number;
    current_step?: string;
}

interface Envelope {
    channel: string;
    data: JobEvent;
}

type Listener = (event: JobEvent) => void;
type StatusListener = (connected: boolean) => void;

class JobEventsSocket {
    private socket: WebSocket | null = null;
    private listeners = new Set<Listener>();
    private statusListeners = new Set<StatusListener>();
    private retryDelay = 1000;
    private retryTimer: ReturnType<typeof setTimeout> | null = null;
    private explicitlyClosed = false;
    private refCount = 0;

    private wsUrl(): string {
        const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        return `${proto}//${window.location.host}/ws/jobs`;
    }

    private notifyStatus(connected: boolean): void {
        for (const fn of this.statusListeners) {
            try {
                fn(connected);
            } catch {
                // listener bugs must not break others
            }
        }
    }

    private connect(): void {
        if (this.socket || this.explicitlyClosed) return;
        let ws: WebSocket;
        try {
            ws = new WebSocket(this.wsUrl());
        } catch {
            this.scheduleReconnect();
            return;
        }
        this.socket = ws;

        ws.onopen = () => {
            this.retryDelay = 1000;
            this.notifyStatus(true);
        };

        ws.onmessage = (msg) => {
            try {
                const env = JSON.parse(msg.data) as Envelope;
                if (env.channel !== 'jobs' || !env.data) return;
                for (const fn of this.listeners) {
                    try {
                        fn(env.data);
                    } catch {
                        // swallow listener errors
                    }
                }
            } catch {
                // ignore non-JSON frames
            }
        };

        const handleClose = () => {
            if (this.socket === ws) this.socket = null;
            this.notifyStatus(false);
            if (!this.explicitlyClosed) this.scheduleReconnect();
        };
        ws.onerror = handleClose;
        ws.onclose = handleClose;
    }

    private scheduleReconnect(): void {
        if (this.retryTimer || this.explicitlyClosed) return;
        const delay = this.retryDelay;
        this.retryDelay = Math.min(this.retryDelay * 2, 30_000);
        this.retryTimer = setTimeout(() => {
            this.retryTimer = null;
            this.connect();
        }, delay);
    }

    /** Subscribe to job events. Returns an unsubscribe function. */
    subscribe(fn: Listener): () => void {
        this.listeners.add(fn);
        this.refCount += 1;
        if (this.refCount === 1) {
            this.explicitlyClosed = false;
            this.connect();
        }
        return () => {
            this.listeners.delete(fn);
            this.refCount = Math.max(0, this.refCount - 1);
            if (this.refCount === 0) this.close();
        };
    }

    /** Subscribe to connection status changes. */
    onStatus(fn: StatusListener): () => void {
        this.statusListeners.add(fn);
        // Fire current state synchronously so callers can initialize.
        try {
            fn(this.socket?.readyState === WebSocket.OPEN);
        } catch {
            // ignore
        }
        return () => {
            this.statusListeners.delete(fn);
        };
    }

    private close(): void {
        this.explicitlyClosed = true;
        if (this.retryTimer) {
            clearTimeout(this.retryTimer);
            this.retryTimer = null;
        }
        if (this.socket) {
            try {
                this.socket.close();
            } catch {
                // ignore
            }
            this.socket = null;
        }
        this.notifyStatus(false);
    }
}

export const jobEventsSocket = new JobEventsSocket();
