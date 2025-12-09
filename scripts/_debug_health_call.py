import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
from fastapi.testclient import TestClient
from core.main import app

with TestClient(app) as client:
    res = client.get('/health')
    print('STATUS:', res.status_code)
    print('HEADERS:', res.headers)
    try:
        print('JSON:', res.json())
    except Exception as e:
        print('JSON decode error:', e)
        print('TEXT:', res.text)
