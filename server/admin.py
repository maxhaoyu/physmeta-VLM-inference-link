"""Offline-safe administration for inference jobs.

Mutating commands require both an explicit job ID and a fresh SQLite backup.
This tool never accepts or prints node/upload credentials or run tokens.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import server as queue


ACTIVE_STATUSES = ("pending", "claimed", "processing", "dead_letter")
REQUEUEABLE_STATUSES = ("claimed", "processing", "failed", "dead_letter")


def backup_database(destination: Path) -> Path:
    destination = destination.resolve()
    if destination.exists():
        raise ValueError(f"Backup already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    source = queue.get_db()
    target = sqlite3.connect(destination)
    try:
        source.backup(target)
        integrity = target.execute("PRAGMA integrity_check").fetchone()
        if not integrity or integrity[0] != "ok":
            raise ValueError(f"Backup integrity check failed: {integrity}")
    finally:
        target.close()
        source.close()
    return destination


def job_details(job_id: str) -> dict | None:
    conn = queue.get_db()
    try:
        row = conn.execute(
            "SELECT id, status, node_id, lease_managed, attempt_count, max_attempts, "
            "agent_version, protocol_version, last_phase, last_progress, error, "
            "created_at, updated_at, heartbeat_at, input_path "
            "FROM inference_jobs WHERE id=?",
            (job_id,),
        ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["input_exists"] = Path(result.pop("input_path")).is_file()
        result["events"] = [
            dict(event)
            for event in conn.execute(
                "SELECT event_type, node_id, details, created_at FROM job_events "
                "WHERE job_id=? ORDER BY id",
                (job_id,),
            )
        ]
        return result
    finally:
        conn.close()


def list_jobs(statuses: tuple[str, ...] = ACTIVE_STATUSES) -> list[dict]:
    placeholders = ",".join("?" for _ in statuses)
    conn = queue.get_db()
    try:
        return [
            dict(row)
            for row in conn.execute(
                "SELECT id, status, node_id, lease_managed, attempt_count, max_attempts, "
                "agent_version, protocol_version, last_phase, last_progress, "
                "created_at, updated_at, heartbeat_at FROM inference_jobs "
                f"WHERE status IN ({placeholders}) ORDER BY created_at",
                statuses,
            )
        ]
    finally:
        conn.close()


def requeue_job(job_id: str, reason: str) -> dict:
    conn = queue.get_db()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT * FROM inference_jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise ValueError(f"Job does not exist: {job_id}")
        if row["status"] not in REQUEUEABLE_STATUSES:
            raise ValueError(f"Job status cannot be requeued: {row['status']}")
        if not Path(row["input_path"]).is_file():
            raise ValueError("Job input file is missing; refusing to requeue")
        next_max_attempts = max(
            int(row["max_attempts"] or queue.MAX_ATTEMPTS),
            int(row["attempt_count"] or 0) + 1,
        )
        conn.execute(
            "UPDATE inference_jobs SET status='pending', node_id=NULL, run_token=NULL, "
            "lease_managed=0, heartbeat_at=NULL, last_phase='manual_requeue', "
            "last_progress=NULL, error=?, max_attempts=?, updated_at=CURRENT_TIMESTAMP "
            "WHERE id=?",
            (f"Manual requeue: {reason}"[:1000], next_max_attempts, job_id),
        )
        queue._event(
            conn,
            job_id,
            "manual_requeue",
            node_id=row["node_id"],
            details={"previous_status": row["status"], "reason": reason[:500]},
        )
        if row["node_id"]:
            conn.execute(
                "UPDATE inference_nodes SET current_job=NULL WHERE node_id=? AND current_job=?",
                (row["node_id"], job_id),
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    details = job_details(job_id)
    assert details is not None
    return details


def _print(payload) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect and rescue inference queue jobs")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List active/stale jobs")
    list_parser.add_argument("--status", action="append", dest="statuses")

    show_parser = subparsers.add_parser("show", help="Show one job and its audit events")
    show_parser.add_argument("job_id")

    requeue_parser = subparsers.add_parser("requeue", help="Back up the DB and requeue one job")
    requeue_parser.add_argument("job_id")
    requeue_parser.add_argument("--reason", required=True)
    requeue_parser.add_argument("--backup", type=Path)
    requeue_parser.add_argument("--execute", action="store_true")

    args = parser.parse_args()
    queue.init_db()

    if args.command == "list":
        _print(list_jobs(tuple(args.statuses or ACTIVE_STATUSES)))
        return 0
    if args.command == "show":
        details = job_details(args.job_id)
        if details is None:
            parser.error(f"Job does not exist: {args.job_id}")
        _print(details)
        return 0

    before = job_details(args.job_id)
    if before is None:
        parser.error(f"Job does not exist: {args.job_id}")
    if not args.execute:
        _print({"dry_run": True, "job": before, "reason": args.reason})
        return 0
    if args.backup is None:
        parser.error("--backup is required with --execute")
    backup = backup_database(args.backup)
    after = requeue_job(args.job_id, args.reason)
    _print({"backup": str(backup), "job": after})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
