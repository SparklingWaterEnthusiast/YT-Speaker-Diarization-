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
    parser.add_argument("--seed", metavar="URL",
                        help="seed voice profiles from a reference transcript "
                             "(requires --reference)")
    parser.add_argument("--reference", metavar="FILE",
                        help="professionally diarized transcript with "
                             "'Name (M:SS-M:SS): text' lines")
    parser.add_argument("--profiles", action="store_true",
                        help="print stored voice profiles and exit")
    args = parser.parse_args()

    if args.seed or args.profiles:
        return run_seed(args)
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


def run_seed(args) -> int:
    """Create voice profiles by aligning a processed video's diarization to a
    professionally diarized reference transcript (DESIGN.md §5.4)."""
    import json
    import time
    from pathlib import Path

    from . import pipeline, voices

    cfg = load_config()
    vdb = voices.VoiceDB(APP_DIR / "voices.sqlite3")

    if args.profiles and not args.seed:
        profiles = vdb.profiles()
        if not profiles:
            print("Voice database is empty.")
        for p in profiles:
            print(f"{p['name']}: {p['confirmed_samples']} confirmed sample(s), "
                  f"created {time.strftime('%Y-%m-%d', time.localtime(p['created_at']))}, "
                  f"updated {time.strftime('%Y-%m-%d', time.localtime(p['updated_at']))}")
        return 0

    if not args.reference:
        print("--seed requires --reference FILE")
        return 2
    ref_path = Path(args.reference)
    if not ref_path.exists():
        print(f"reference file not found: {ref_path}")
        return 2
    ref_turns = voices.parse_reference(ref_path)
    real_names = sorted({t["speaker"] for t in ref_turns
                         if not voices.PLACEHOLDER.match(t["speaker"])})
    if not ref_turns or not real_names:
        print("reference contains no named speaker turns — nothing to seed")
        return 2
    print(f"Reference: {len(ref_turns)} turns, named speakers: {', '.join(real_names)}")

    store = JobStore(APP_DIR / "jobs.sqlite3")
    entries = pipeline.media.enumerate_videos(args.seed, cfg, print)
    if len(entries) != 1:
        print("--seed expects a single-video URL")
        return 2
    entry = entries[0]
    vid = entry["video_id"]
    store.add(vid, entry["url"], entry["title"], entry["duration"],
              entry.get("channel", ""))

    runner = pipeline.QueueRunner(cfg, store, pipeline.Callbacks(
        on_log=print, on_stage=lambda v, s: print(f"  -> {s}")))
    print(f"Ensuring '{entry['title'] or vid}' is fully processed "
          "(cached stages are reused)...")
    runner.run_single(vid)

    vdir = cfg.cache_path / vid
    diar = json.loads((vdir / "diarization.json").read_text(encoding="utf-8"))
    alignment = voices.align_to_reference(diar, ref_turns)
    if not alignment:
        print("No confident alignment between detected speakers and the "
              "reference — no profiles created.")
        return 1
    for label, info in alignment.items():
        print(f"{label} -> {info['name']} "
              f"(overlap {info['overlap']}s, purity {info['purity']:.0%})")
        voices.apply_rename(cfg, vdir, label, info["name"], vdb, print)
    print("\nStored profiles:")
    for p in vdb.profiles():
        print(f"  {p['name']}: {p['confirmed_samples']} confirmed sample(s)")
    return 0


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
