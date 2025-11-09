// Loading System Debug Test
// Run this in the browser console to test loading functionality

console.log('=== LOADING SYSTEM DEBUG TEST ===');

// Test 1: Check if global namespace exists
console.log('1. Global DI namespace:', typeof global !== 'undefined' && global.DI ? '✅ Available' : '❌ Missing');

// Test 2: Check if utilities.notifications exists
console.log('2. Utilities.notifications:', typeof global !== 'undefined' && global.DI && global.DI.utilities && global.DI.utilities.notifications ? '✅ Available' : '❌ Missing');

// Test 3: Check if showLoading function exists
console.log('3. showLoading function:', typeof global !== 'undefined' && global.DI && global.DI.utilities && global.DI.utilities.notifications && typeof global.DI.utilities.notifications.showLoading === 'function' ? '✅ Available' : '❌ Missing');

// Test 4: Check if loading overlay element exists in DOM
const loadingOverlay = document.getElementById('loading-overlay');
console.log('4. Loading overlay element:', loadingOverlay ? '✅ Found' : '❌ Missing');

if (loadingOverlay) {
    console.log('   - Current display style:', loadingOverlay.style.display || getComputedStyle(loadingOverlay).display);
    console.log('   - Element HTML:', loadingOverlay.outerHTML.substring(0, 200) + '...');
}

// Test 5: Check loading title and message elements
const loadingTitle = document.getElementById('loading-title');
const loadingMessage = document.getElementById('loading-message');
console.log('5. Loading title element:', loadingTitle ? '✅ Found' : '❌ Missing');
console.log('6. Loading message element:', loadingMessage ? '✅ Found' : '❌ Missing');

// Test 7: Try to show loading
console.log('\n=== TESTING SHOW LOADING ===');
try {
    if (global && global.DI && global.DI.utilities && global.DI.utilities.notifications) {
        global.DI.utilities.notifications.showLoading(true, 'Debug Test', 'Testing loading overlay...');
        console.log('✅ showLoading(true) executed without errors');
        
        // Wait 2 seconds then hide
        setTimeout(() => {
            console.log('\n=== TESTING HIDE LOADING ===');
            global.DI.utilities.notifications.showLoading(false);
            console.log('✅ showLoading(false) executed without errors');
        }, 2000);
    } else {
        console.log('❌ Cannot test - global namespace not available');
    }
} catch (error) {
    console.log('❌ Error testing showLoading:', error);
}

// Test 8: Check for common issues
console.log('\n=== CHECKING FOR COMMON ISSUES ===');

// Check if there are multiple loading overlays
const allLoadingElements = document.querySelectorAll('[id*="loading"]');
console.log('8. Elements with "loading" in ID:', allLoadingElements.length);
allLoadingElements.forEach((el, i) => {
    console.log(`   ${i+1}. #${el.id} - display: ${getComputedStyle(el).display}`);
});

// Check for JavaScript errors in console
console.log('9. JavaScript errors: Check console for any red error messages');

// Check CSS
const loadingOverlayCSS = getComputedStyle(loadingOverlay);
if (loadingOverlay) {
    console.log('10. CSS z-index:', loadingOverlayCSS.zIndex);
    console.log('11. CSS position:', loadingOverlayCSS.position);
    console.log('12. CSS background:', loadingOverlayCSS.background);
}

console.log('\n=== DEBUG TEST COMPLETE ===');
console.log('If loading is stuck, check the specific error messages above.');
