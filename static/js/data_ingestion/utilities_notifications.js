/**
 * @deprecated This file has moved to /static/js/shared/notifications.js.
 * Include that script directly instead of this legacy shim.
 */
(function(global) {
'use strict';

if (global.__DI_SHARED_NOTIFICATIONS_SHIM__) {
    return;
}
global.__DI_SHARED_NOTIFICATIONS_SHIM__ = true;

if (typeof global.showNotification === 'function') {
    return;
}

var script = document.createElement('script');
script.src = '/static/js/shared/notifications.js';
script.async = false;
script.onload = function() {
    console.warn('[notifications] Loaded shared notifications module via legacy shim. Please update references to /static/js/shared/notifications.js.');
};
script.onerror = function() {
    console.error('[notifications] Failed to load shared notifications module.');
};

document.head.appendChild(script);
})(window);
