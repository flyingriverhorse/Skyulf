#!/usr/bin/env python3
"""
Compare admin routes between main.py and core/admin/routes.py
"""

def analyze_admin_routes():
    # Routes in main.py (with @app decorator)
    main_py_routes = [
        "GET /admin/api/data-ingestion/stats",
        "GET /admin/api/data-ingestion/sources", 
        "DELETE /admin/api/data-ingestion/source/{source_id}",
        "GET /admin/api/users/stats",
        "GET /admin/api/users",
        "POST /admin/api/users",
        "PATCH /admin/api/users/{user_id}",
        "DELETE /admin/api/users/{user_id}",
        "GET /api/admin/system/settings",  # Note: Different pattern!
        "GET /admin/api/system/info",
        "GET /admin/api/system/feature-flags",
        "GET /data/api/ingestion/admin/app-log",  # Note: Different pattern!
        "GET /admin/api/database/status",
        "GET /admin/api/registries/list/sqlite",
        "GET /admin/api/registries/list/postgres", 
        "GET /admin/api/database/compare",
        "POST /admin/api/database/migrate"
    ]
    
    # Routes in admin_router (with @admin_router, prefix="/admin" is added)
    # So @admin_router.get("/api/users") becomes /admin/api/users
    admin_router_routes = [
        "GET /admin/api/data-ingestion/stats",
        "GET /admin/api/data-ingestion/sources",
        "DELETE /admin/api/data-ingestion/source/{source_id}",
        "POST /admin/api/data-ingestion/sources/bulk-delete",  # EXTRA!
        "GET /admin/api/users/stats", 
        "GET /admin/api/users",
        "POST /admin/api/users",
        "PATCH /admin/api/users/{user_id}",
        "DELETE /admin/api/users/{user_id}",
        "GET /admin/api/system/info",
        "GET /admin/api/system/health",  # EXTRA!
        "GET /admin/api/system/feature-flags",
        "GET /admin/debug/logs/tail",  # EXTRA!
        "GET /admin/ping",  # EXTRA!
        "GET /admin/api/database/status",
        "POST /admin/api/database/migrate",
        "GET /admin/api/database/contents",  # EXTRA!
        "POST /admin/api/database/test-data",  # EXTRA!
        "GET /admin/users",  # HTML route - EXTRA!
        "GET /admin/dashboard",  # HTML route - EXTRA!
        "GET /admin"  # HTML route - EXTRA!
    ]
    
    print("=== ADMIN ROUTES COMPARISON ===\n")
    
    print("📋 ROUTES IN MAIN.PY:")
    for route in sorted(main_py_routes):
        print(f"  {route}")
    
    print(f"\n📋 ROUTES IN ADMIN_ROUTER:")
    for route in sorted(admin_router_routes):
        print(f"  {route}")
    
    print(f"\n🔍 ANALYSIS:")
    print(f"  Main.py routes: {len(main_py_routes)}")
    print(f"  Admin_router routes: {len(admin_router_routes)}")
    
    # Find duplicates
    main_set = set(main_py_routes)
    admin_set = set(admin_router_routes)
    
    duplicates = main_set & admin_set
    only_in_main = main_set - admin_set
    only_in_admin = admin_set - main_set
    
    print(f"\n❌ DUPLICATES (exist in both):")
    for route in sorted(duplicates):
        print(f"  {route}")
    
    print(f"\n⚠️  ONLY IN MAIN.PY:")
    for route in sorted(only_in_main):
        print(f"  {route}")
        
    print(f"\n✅ ONLY IN ADMIN_ROUTER:")
    for route in sorted(only_in_admin):
        print(f"  {route}")
    
    print(f"\n📊 SUMMARY:")
    print(f"  Duplicates: {len(duplicates)}")
    print(f"  Main.py unique: {len(only_in_main)}")
    print(f"  Admin_router unique: {len(only_in_admin)}")
    
    print(f"\n💡 RECOMMENDATION:")
    print(f"  - Remove {len(duplicates)} duplicate routes from main.py")
    print(f"  - Move {len(only_in_main)} unique main.py routes to admin_router")
    print(f"  - Keep admin_router routes (better organized with auth)")

if __name__ == "__main__":
    analyze_admin_routes()