"""Entry point: GUI by default, headless CLI with --cli (used for automation
and testing; identical pipeline underneath).
"""

from __future__ import annotations

import argparse
import sys

from .config import APP_DIR, load_config, save_config
from .db import JobStore


def main() -> int:
    parser = argparse.ArgumentParser(prog="ytscribe",
                                     description="Speaker-diarized transcripts from YouTube.")
    parser.add_argument("--cli", metavar="URL", nargs="*",
                        help="run headless: enqueue URL(s) and process the queue")
    parser.add_argument("--retry-failed", action="store_true",
                        help="(CLI) requeue failed jobs before running")
    parser.add_argument("--list", action="store_true", help="(CLI) print queue and exit")
    args = parser.parse_args()

    if args.cli is not None or args.list or args.retry_failed:
        return run_cli(args)
    return run_gui()


def run_cli(args) -> int:
    import builtins
    import functools

    from . import pipeline

    # progress must be visible live when stdout is a pipe (scheduled jobs, logs)
    print = functools.partial(builtins.print, flush=True)  # noqa: A001
    cfg = load_config()
    problems = cfg.validate()
    if problems:
        for p in problems:
            print("config error:", p)
        return 2
    save_config(cfg)  # materialize defaults on first run
    store = JobStore(APP_DIR / "jobs.sqlite3")

    if args.list:
        for j in store.all_jobs():
            print(f"{j['status']:<10} {j['stage']:<11} {j['video_id']}  {j['title'][:60]}")
        return 0
    if args.retry_failed:
        print(f"Requeued {store.requeue_failed()} failed job(s).")

    for url in args.cli or []:
        try:
            pipeline.enqueue(url, cfg, store, print)
        except Exception as exc:
            print(f"ERROR enqueuing {url}: {exc}")
            return 1

    cb = pipeline.Callbacks(
        on_log=print,
        on_job_start=lambda vid, title: print(f"\n=== {title} ({vid}) ==="),
        on_stage=lambda vid, stage: print(f"  -> {stage}"),
        on_job_done=lambda vid, status, err: print(f"  {status.upper()} {err}".rstrip()),
        on_queue_progress=lambda done, total, eta, rtf: print(
            f"  [{done}/{total} done, {rtf:.2f}x realtime, ~{eta / 60:.0f} min left]"),
    )
    runner = pipeline.QueueRunner(cfg, store, cb)
    summary = runner.run()
    print(f"\nQueue finished: {summary['completed']} completed, "
          f"{summary['failed']} failed, {summary['wall_seconds'] / 60:.1f} min.")
    return 0 if summary["failed"] == 0 else 1


def run_gui() -> int:
    from PySide6.QtWidgets import QApplication

    from .ui.main_window import MainWindow
    app = QApplication(sys.argv)
    app.setApplicationName("YTScribe")
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
