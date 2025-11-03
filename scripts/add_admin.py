import asyncio


async def main():
    try:
        from core.database.engine import init_db, get_async_session, close_db
        from core.database.repository import get_user_repository
        from core.auth.auth_core import get_password_hash
    except Exception as exc:
        print("Import failed:", exc)
        return

    await init_db()

    username = "temp_admin"
    email = "temp_admin@example.local"
    password = "TempPass123!"

    try:
        async for session in get_async_session():
            repo = get_user_repository(session)

            existing = await repo.get_by_username(username)
            if existing:
                print(f"User '{username}' already exists (id={existing.id}). No action taken.")
                break

            admin_data = {
                "username": username,
                "email": email,
                "password_hash": get_password_hash(password),
                "full_name": "Temporary Admin",
                "is_active": True,
                "is_admin": True,
                "is_verified": True,
            }

            created = await repo.create(admin_data)
            print(f"Created user: id={created.id} username={created.username} email={created.email}")
            break
    finally:
        await close_db()


if __name__ == "__main__":
    asyncio.run(main())
