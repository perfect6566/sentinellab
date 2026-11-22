"""Verify production artifact integrity and application readiness without binding a port."""

from __future__ import annotations

import json

from app import create_app


def main() -> int:
    app = create_app({"TESTING": True})
    response = app.test_client().get("/health")
    payload = response.get_json()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if response.status_code == 200 and payload["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
