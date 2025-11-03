import asyncio
from sqlalchemy import select


async def main():
    try:
        # Import here so errors show clearly if dependencies are missing
        from core.database.engine import init_db, get_async_session, close_db
        from core.database.models import User
    except Exception as exc:
        print("Failed to import database modules:", exc)
        return

    try:
        await init_db()
    except Exception as exc:
        print("Failed to initialize DB (check DATABASE_URL and dependencies):", exc)
        return

    try:
        async for session in get_async_session():
            result = await session.execute(select(User))
            users = result.scalars().all()

            if not users:
                print("No users found in database.")
            else:
                print(f"Found {len(users)} user(s):")
                for u in users:
                    try:
                        print(f"id={u.id} username={u.username!r} email={u.email!r} is_admin={u.is_admin} is_active={u.is_active}")
                    except Exception:
                        # Fall back to repr
                        print(repr(u))
            break
    finally:
        try:
            await close_db()
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())
