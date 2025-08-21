"""Email utility functions shared across email processing modules."""
import hashlib
import json
from typing import Any, Dict


def compute_header_hash(record: Dict[str, Any]) -> str:
    """Return a deterministic hash of common email headers."""
    from_addr = (record.get("from_addr") or "").lower()
    to_addrs = record.get("to_addrs") or []
    if isinstance(to_addrs, str):
        try:
            to_addrs = json.loads(to_addrs)
        except Exception:
            to_addrs = [to_addrs]
    to_norm = ",".join(sorted(a.lower() for a in to_addrs if a))
    subject = record.get("subject") or ""
    date_utc = record.get("date_utc") or ""
    message_id = record.get("message_id") or ""
    payload = "\n".join([from_addr, to_norm, subject, date_utc, message_id])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
