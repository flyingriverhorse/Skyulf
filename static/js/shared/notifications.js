/**
 * Shared notifications & loading utilities
 * ----------------------------------------
 * Provides toast notifications, stacked alerts, and loading overlay helpers
 * for any page in the platform (data ingestion, EDA, admin, etc.).
 */
(function (global) {
    'use strict';

    // Ensure namespaces exist for backwards compatibility with the data ingestion module
    global.DI = global.DI || {};
    global.DI.utilities = global.DI.utilities || {};
    global.DI.utilities.notifications = global.DI.utilities.notifications || {};

    const namespace = global.DI.utilities.notifications;
    let notificationPollingHandle = null;

    const NOTIFICATION_ICONS = {
        success: 'fas fa-check-circle',
        error: 'fas fa-exclamation-circle',
        info: 'fas fa-info-circle',
        warning: 'fas fa-exclamation-triangle'
    };

    function resolveElementByIds(ids) {
        for (const id of ids) {
            const element = document.getElementById(id);
            if (element) return element;
        }
        return null;
    }

    function resolveLoadingOverlay() {
        return resolveElementByIds(['loading-overlay', 'loadingOverlay']);
    }

    function resolveLoadingTitle() {
        return resolveElementByIds(['loading-title', 'loadingTitle']);
    }

    function resolveLoadingMessage() {
        return resolveElementByIds(['loading-message', 'loadingMessage']);
    }

    /**
     * Toggle visibility of content sections with smooth animations
     * @param {string} contentId - ID of the content element to toggle
     * @param {string} toggleId - ID of the toggle button element
     */
    function toggleSection(contentId, toggleId) {
        const content = document.getElementById(contentId);
        const toggle = document.getElementById(toggleId);

        if (!content || !toggle) {
            console.warn(`toggleSection: Elements not found - content: ${contentId}, toggle: ${toggleId}`);
            return;
        }

        const icon = toggle.querySelector('i');
        const isHidden = content.style.display === 'none' || content.style.display === '';

        if (isHidden) {
            content.style.display = 'block';
            content.style.opacity = '0';
            content.style.transform = 'translateY(-10px)';
            requestAnimationFrame(() => {
                content.style.transition = 'opacity 0.25s ease, transform 0.25s ease';
                content.style.opacity = '1';
                content.style.transform = 'translateY(0)';
            });
            if (icon) icon.className = 'fas fa-chevron-up';
        } else {
            content.style.transition = 'opacity 0.25s ease, transform 0.25s ease';
            content.style.opacity = '0';
            content.style.transform = 'translateY(-10px)';
            setTimeout(() => {
                content.style.display = 'none';
                if (icon) icon.className = 'fas fa-chevron-down';
            }, 250);
        }
    }

    // ---------- Time Utilities ----------
    function getTimeAgo(date) {
        const now = new Date();
        const diffInSeconds = Math.floor((now - date) / 1000);
        if (diffInSeconds < 60) return 'just now';
        if (diffInSeconds < 3600) return `${Math.floor(diffInSeconds / 60)} minutes ago`;
        if (diffInSeconds < 86400) return `${Math.floor(diffInSeconds / 3600)} hours ago`;
        return `${Math.floor(diffInSeconds / 86400)} days ago`;
    }

    // ---------- Toast Notifications ----------
    function showNotification(message, type = 'info') {
        document.querySelectorAll('.toast-notification').forEach((n) => n.remove());

        const notification = document.createElement('div');
        notification.className = `toast-notification toast-${type}`;
        notification.setAttribute('role', 'status');
        notification.setAttribute('aria-live', 'polite');

        const iconClass = NOTIFICATION_ICONS[type] || NOTIFICATION_ICONS.info;
        notification.innerHTML = `
            <i class="${iconClass}"></i>
            <div class="toast-body">
                <span class="toast-title">${type.charAt(0).toUpperCase() + type.slice(1)}</span>
                <span class="toast-text">${message}</span>
            </div>
        `;

        document.body.appendChild(notification);

        setTimeout(() => {
            notification.style.animation = 'slideOutDown 0.28s ease forwards';
            setTimeout(() => notification.remove(), 280);
        }, 5000);
    }

    // ---------- Loading Overlay ----------
    function showLoading(show, title = 'Loading...', message = '', allowBackgroundInteraction = false) {
        let overlay = resolveLoadingOverlay();

        if (show) {
            if (!overlay) {
                overlay = document.createElement('div');
                overlay.id = 'loading-overlay';
                overlay.className = 'loading-overlay';
                overlay.innerHTML = `
                    <div class="loading-card" role="status" aria-live="polite">
                        <div class="loading-spinner"></div>
                        <h5 id="loading-title">${title}</h5>
                        <p class="text-muted" id="loading-message">${message}</p>
                        <div class="loading-timer"><span id="timer-display">00:00</span></div>
                    </div>
                `;
                document.body.appendChild(overlay);
            }

            const loadingTitle = resolveLoadingTitle();
            const loadingMessage = resolveLoadingMessage();

            if (loadingTitle) loadingTitle.textContent = title;
            if (loadingMessage) loadingMessage.textContent = message;

            overlay.style.display = 'flex';
            overlay.style.pointerEvents = allowBackgroundInteraction ? 'none' : 'auto';
        } else if (overlay) {
            overlay.style.display = 'none';
            overlay.style.pointerEvents = 'auto';
        }
    }

    // ---------- Theme Toggle ----------
    function toggleTheme() {
        const isDark = document.body.classList.toggle('dark-mode');
        const icon = document.getElementById('theme-icon') || document.getElementById('themeToggleNavIcon');
        if (icon) {
            icon.className = isDark ? 'fas fa-sun' : 'fas fa-moon';
        }
        localStorage.setItem('eda-theme', isDark ? 'dark' : 'light');
        localStorage.setItem('darkMode', isDark.toString());
    }

    window.toggleTheme = toggleTheme;

    // ---------- Scroll Helper ----------
    function scrollToTop() {
        window.scrollTo({
            top: 0,
            behavior: 'smooth'
        });
    }

    // ---------- Server Notification Queue ----------
    function checkForNotifications() {
        fetch('/data/api/notifications?unread_only=true&mark_read=true')
            .then((r) => r.json())
            .then((result) => {
                if (result.success && Array.isArray(result.notifications) && result.notifications.length) {
                    displayNotifications(result.notifications);
                }
            })
            .catch((err) => console.error('Error checking notifications:', err));
    }

    const notifQueue = [];
    let notifProcessing = false;

    function enqueueServerNotifications(notifications) {
        notifications.forEach((n) => {
            if (!n.read) notifQueue.push(n);
        });
        processNotificationQueue();
    }

    function processNotificationQueue() {
        if (notifProcessing) return;
        if (!notifQueue.length) return;

        const overlay = resolveLoadingOverlay();
        const isLoadingVisible = overlay && overlay.style && overlay.style.display !== 'none';
        const activeToasts = document.querySelectorAll('.toast-notification, .notification-toast, .notification-container .alert').length;
        if (isLoadingVisible || activeToasts > 0) {
            setTimeout(processNotificationQueue, 800);
            return;
        }

        const notification = notifQueue.shift();
        if (!notification) return;
        notifProcessing = true;

        let container = document.getElementById('notificationContainer');
        if (!container) {
            container = document.createElement('div');
            container.id = 'notificationContainer';
            container.className = 'notification-container';
            document.body.appendChild(container);
        }

        const element = document.createElement('div');
        const typeClass = notification.type === 'warning' ? 'alert-warning' : (notification.type === 'success' ? 'alert-success' : (notification.type === 'error' ? 'alert-danger' : 'alert-info'));
        const icon = notification.type === 'warning' ? 'exclamation-triangle' : (notification.type === 'success' ? 'check-circle' : (notification.type === 'error' ? 'exclamation-circle' : 'info-circle'));
        const timeAgo = getTimeAgo(new Date(notification.timestamp));

        element.className = `alert ${typeClass} alert-dismissible fade show`;
        element.innerHTML = `
            <div class="d-flex align-items-start">
                <i class="fas fa-${icon} me-2 mt-1"></i>
                <div class="flex-grow-1">
                    <div class="fw-bold">${notification.title || 'Data Source Notification'}</div>
                    <div>${notification.message}</div>
                    <small class="text-muted">${timeAgo}</small>
                </div>
                <button type="button" class="btn-close" aria-label="Close"></button>
            </div>
        `;

        const closeButton = element.querySelector('.btn-close');
        if (closeButton) {
            closeButton.addEventListener('click', () => dismissNotification(notification.id, element));
        }

        container.appendChild(element);

        const visibleDuration = 10000;
        setTimeout(() => {
            if (element.parentNode) {
                element.style.animation = 'slideOutDown 0.28s ease forwards';
                setTimeout(() => {
                    dismissNotification(notification.id, element);
                    notifProcessing = false;
                    setTimeout(processNotificationQueue, 200);
                }, 280);
            } else {
                notifProcessing = false;
                setTimeout(processNotificationQueue, 200);
            }
        }, visibleDuration);
    }

    function displayNotifications(notifications) {
        enqueueServerNotifications(notifications);
    }

    function dismissNotification(notificationId, element) {
        fetch(`/data/api/notifications/${notificationId}/read`, { method: 'POST' }).catch(() => {});
        if (!element) {
            notifProcessing = false;
            return;
        }
        element.classList.remove('show');
        element.style.animation = 'slideOutDown 0.28s ease forwards';
        setTimeout(() => {
            if (element.parentNode) element.parentNode.removeChild(element);
            notifProcessing = false;
            setTimeout(processNotificationQueue, 150);
        }, 280);
    }

    function startNotificationPolling(options = {}) {
        const { intervalMs = 60000, immediate = true } = options;

        if (notificationPollingHandle) {
            clearInterval(notificationPollingHandle);
        }

        if (immediate) {
            try {
                checkForNotifications();
            } catch (error) {
                console.error('Notification polling (immediate) failed:', error);
            }
        }

        notificationPollingHandle = window.setInterval(() => {
            try {
                checkForNotifications();
            } catch (error) {
                console.error('Notification polling failed:', error);
            }
        }, Math.max(15000, Number(intervalMs) || 60000));

        return notificationPollingHandle;
    }

    function stopNotificationPolling() {
        if (notificationPollingHandle) {
            clearInterval(notificationPollingHandle);
            notificationPollingHandle = null;
        }
    }

    // Export to namespace and global scope for backwards compatibility
    namespace.showNotification = showNotification;
    namespace.showLoading = showLoading;
    namespace.toggleTheme = toggleTheme;
    namespace.scrollToTop = scrollToTop;
    namespace.getTimeAgo = getTimeAgo;
    namespace.checkForNotifications = checkForNotifications;
    namespace.displayNotifications = displayNotifications;
    namespace.dismissNotification = dismissNotification;
    namespace.toggleSection = toggleSection;
    namespace.startNotificationPolling = startNotificationPolling;
    namespace.stopNotificationPolling = stopNotificationPolling;

    global.showNotification = showNotification;
    global.showLoading = showLoading;
    global.toggleTheme = toggleTheme;
    global.scrollToTop = scrollToTop;
    global.getTimeAgo = getTimeAgo;
    global.checkForNotifications = checkForNotifications;
    global.displayNotifications = displayNotifications;
    global.dismissNotification = dismissNotification;
    global.toggleSection = toggleSection;
    global.startNotificationPolling = startNotificationPolling;
    global.stopNotificationPolling = stopNotificationPolling;
})(window);
