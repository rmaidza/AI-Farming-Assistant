"""
database.py
Supabase client
"""

import os
from pathlib import Path
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

_url: str = os.environ["SUPABASE_URL"]
_key: str = os.environ["SUPABASE_ANON_KEY"]

supabase: Client = create_client(_url, _key)
