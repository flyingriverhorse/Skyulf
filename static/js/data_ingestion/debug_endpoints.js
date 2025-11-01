/**
 * Simple debug test for data ingestion endpoints
 * Run this in browser console to test if endpoints are working
 */

async function testDataIngestionEndpoints() {
    console.log('🔍 Testing Data Ingestion Endpoints...');
    
    // Test basic endpoints
    const tests = [
        {
            name: 'List Sources',
            url: '/data/api/sources',
            method: 'GET'
        }
    ];
    
    for (const test of tests) {
        try {
            console.log(`📡 Testing ${test.name}: ${test.url}`);
            const response = await fetch(test.url, {
                method: test.method,
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('access_token') || ''}`,
                    'Content-Type': 'application/json'
                }
            });
            
            if (response.ok) {
                const data = await response.json();
                console.log(`✅ ${test.name}: SUCCESS`, data);
            } else {
                console.log(`❌ ${test.name}: HTTP ${response.status}`, await response.text());
            }
        } catch (error) {
            console.log(`❌ ${test.name}: ERROR`, error);
        }
    }
    
    console.log('🏁 Data Ingestion Endpoint Tests Complete');
}

// Auto-run if in browser console
if (typeof window !== 'undefined') {
    console.log('🚀 Run testDataIngestionEndpoints() to test the endpoints');
}