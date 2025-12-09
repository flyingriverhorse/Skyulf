import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text
from core.database.engine import sync_engine, init_db
import asyncio

async def setup():
    await init_db()

def add_columns():
    # Initialize DB first
    asyncio.run(setup())
    
    from core.config import get_settings
    settings = get_settings()
    print(f"Original DATABASE_URL: {settings.DATABASE_URL}")
    
    # Now sync_engine should be initialized
    from core.database.engine import sync_engine
    
    # Manually recreate sync engine if the default one is broken
    from sqlalchemy import create_engine
    
    sync_url = settings.DATABASE_URL.replace("sqlite+aiosqlite://", "sqlite://")
    print(f"Using sync URL: {sync_url}")
    
    sync_engine = create_engine(sync_url)

    print("Attempting to add progress columns to training_jobs table...")
    with sync_engine.connect() as conn:
        try:
            # Check if column exists first to avoid error spam (optional, but cleaner)
            # But simple ALTER TABLE is fine for this helper script
            conn.execute(text("ALTER TABLE training_jobs ADD COLUMN progress INTEGER DEFAULT 0"))
            print("✅ Added 'progress' column to training_jobs")
        except Exception as e:
            print(f"ℹ️  'progress' column likely exists in training_jobs or error: {e}")
            
        try:
            conn.execute(text("ALTER TABLE training_jobs ADD COLUMN current_step VARCHAR(100)"))
            print("✅ Added 'current_step' column to training_jobs")
        except Exception as e:
            print(f"ℹ️  'current_step' column likely exists in training_jobs or error: {e}")

        # Hyperparameter Tuning Jobs
        try:
            conn.execute(text("ALTER TABLE hyperparameter_tuning_jobs ADD COLUMN progress INTEGER DEFAULT 0"))
            print("✅ Added 'progress' column to hyperparameter_tuning_jobs")
        except Exception as e:
            print(f"ℹ️  'progress' column likely exists in hyperparameter_tuning_jobs or error: {e}")
            
        try:
            conn.execute(text("ALTER TABLE hyperparameter_tuning_jobs ADD COLUMN current_step VARCHAR(100)"))
            print("✅ Added 'current_step' column to hyperparameter_tuning_jobs")
        except Exception as e:
            print(f"ℹ️  'current_step' column likely exists in hyperparameter_tuning_jobs or error: {e}")
        
        conn.commit()
    print("Done.")

if __name__ == "__main__":
    add_columns()
