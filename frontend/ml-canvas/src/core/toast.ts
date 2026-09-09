import { useNotificationsStore } from './store/useNotificationsStore';

/** Preserve the shared message API while delivering feedback only to the navbar bell. */
function notify(level: string, message: string, description?: string): void {
  useNotificationsStore.getState().addAppMessage(level, description ? `${message}\n${description}` : message);
}

export const toast = {
  success: (message: string, description?: string): void => {
    notify('success', message, description);
  },
  error: (message: string, description?: string): void => {
    notify('error', message, description);
  },
  info: (message: string, description?: string): void => {
    notify('info', message, description);
  },
  warning: (message: string, description?: string): void => {
    notify('warning', message, description);
  },
};
