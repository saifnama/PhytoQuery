from datetime import datetime, timezone, timedelta

from backend.core.upload_jobs import UploadJobStore


def test_job_store_persists_across_reloads(tmp_path):
    store = UploadJobStore(tmp_path)
    record = {
        "job_id": "job-1",
        "user_id": "sess_a",
        "status": "processing",
        "message": "queued",
        "files": ["paper.pdf"],
        "parser_type": "pymupdf",
        "summaries": None,
        "error": None,
        "created_at": "2026-05-01T00:00:00+00:00",
        "completed_at": None,
    }
    store.create(record)

    reloaded = UploadJobStore(tmp_path)
    assert reloaded.get("job-1") == record


def test_job_store_cleans_up_user_records(tmp_path):
    store = UploadJobStore(tmp_path)
    for job_id, user_id in (("job-1", "sess_a"), ("job-2", "sess_b")):
        store.create(
            {
                "job_id": job_id,
                "user_id": user_id,
                "status": "completed",
                "message": "done",
                "files": [],
                "parser_type": "pymupdf",
                "summaries": None,
                "error": None,
                "created_at": "2026-05-01T00:00:00+00:00",
                "completed_at": "2026-05-01T00:01:00+00:00",
            }
        )

    store.delete_user_jobs("sess_a")

    assert store.get("job-1") is None
    assert store.get("job-2") is not None


def test_job_store_prune_removes_stale_jobs(tmp_path):
    store = UploadJobStore(tmp_path)
    now = datetime.now(timezone.utc)
    old_time = (now - timedelta(days=8)).isoformat()
    recent_time = (now - timedelta(days=1)).isoformat()

    store.create(
        {
            "job_id": "old-job",
            "user_id": "sess_a",
            "status": "completed",
            "message": "done",
            "files": [],
            "parser_type": "pymupdf",
            "summaries": None,
            "error": None,
            "created_at": old_time,
            "completed_at": None,
        }
    )
    store.create(
        {
            "job_id": "recent-job",
            "user_id": "sess_a",
            "status": "completed",
            "message": "done",
            "files": [],
            "parser_type": "pymupdf",
            "summaries": None,
            "error": None,
            "created_at": recent_time,
            "completed_at": None,
        }
    )

    deleted = store.prune(max_age_seconds=604800)  # 7 days
    assert deleted == 1
    assert store.get("old-job") is None
    assert store.get("recent-job") is not None
