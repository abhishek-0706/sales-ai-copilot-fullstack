"""Write the OpenAPI spec to backend/openapi.json (the front end generates its TypeScript types from it)."""
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.main import create_app  # noqa: E402

logging.getLogger("salesai").disabled = True
out = Path(__file__).resolve().parents[1] / "openapi.json"
out.write_text(json.dumps(create_app().openapi(), indent=2) + "\n", encoding="utf-8")
print(f"wrote {out}")
