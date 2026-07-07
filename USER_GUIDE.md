# USER_GUIDE.md

## Launching

- **GUI:** `.\run.ps1`
- **Headless:** `.\run.ps1 --cli "<url>"` (same pipeline, prints progress to
  the console; useful for scheduled/overnight jobs)

## The main window

```
┌──────────────────────────────────────────────────────────────┐
│ [ Paste a YouTube URL…                    ] [Add to Queue]   │
├──────────────────────────────────────────────────────────────┤
│ Queue table: Title | Duration | Status | Stage | ✓ | Error   │
├────────────────────────────┬─────────────────────────────────┤
│ Current                    │ Log                             │
│  title, stage              │  timestamped processing log,    │
│  item progress bar         │  warnings, retries              │
│  queue progress, elapsed,  │                                 │
│  remaining, speed          │                                 │
├────────────────────────────┴─────────────────────────────────┤
│ [Start][Pause][Cancel Current][Retry Failed][Clear Finished] │
│ [Open Output Folder][Settings…]                              │
│ GPU: 87%  VRAM: 5.2/8 GB  CPU: 22%  RAM: 12/32 GB  Disk: …   │
└──────────────────────────────────────────────────────────────┘
```

## Typical workflow

1. **Paste a URL** — a single video (`watch?v=`), a playlist
   (`playlist?list=`), or a channel (`youtube.com/@name`). Channels are
   expanded through their Videos tab; expanding a large channel can take a
   minute and happens in the background.
2. **Start.** Videos process one at a time through five stages:
   *acquire → transcribe → diarize → merge → export*. While the GPU works on
   one video, the next video's audio downloads in parallel.
3. **Watch progress.** Per-item progress shows download MB and transcribed
   minutes; queue progress shows videos done, elapsed time, estimated
   remaining time (based on measured speed), and processing speed as a
   multiple of realtime.
4. **Find transcripts** in the output folder (button on the toolbar), one set
   of files per video plus `combined <name>.md/.json` per finished queue.

## Pause / Resume / Cancel

- **Pause** finishes the current stage, then waits. **Resume** continues.
- **Cancel Current** abandons the current video at the next checkpoint
  (mid-download and mid-transcription cancel quickly; a diarization pass
  finishes its current model call first) and moves on.
- **Closing the app** mid-run is safe: every completed stage is cached, and on
  the next launch interrupted jobs return to the queue. Press Start to resume
  where it left off — completed stages are not repeated.

## Failures and retries

A video that fails (network error, removed video, etc.) is marked **failed**
with the error shown in the queue table, and the run continues. Press
**Retry Failed** to requeue all failures — already-completed stages (e.g. a
finished download) are reused, so retries are cheap.

## Interpreting the transcripts

- **Markdown** (`.md`) — the readable one: video header, then
  `**SPEAKER_00** (12:34–13:02):` turns with paragraphs.
- **JSON** — everything: per-word timings, confidences, speaker IDs, video
  metadata. Use this for downstream processing.
- **TXT** — plain text with timestamps.
- **SRT / VTT** — subtitle files with speaker prefixes, usable in video
  players and editors.

Speakers are labeled `SPEAKER_00`, `SPEAKER_01`, … per video until YTScribe
knows their voices (below). Unrecognized voices always keep their neutral
labels — names are never guessed.

## Naming speakers (v0.2)

**Click any completed video in the queue table.** A dropdown lists every
detected speaker with:

- the current label or recognized name (auto-matches show their similarity,
  e.g. `Cliffe Knechtle (SPEAKER_03) · auto 0.86`),
- **▶** — plays a ~5-second sample of that voice,
- a **rename field** — type the real name and press Enter.

Renaming immediately rewrites that video's transcript files (Markdown plus
any other formats present), and stores the voice in the **voice database**
(`Documents\YTScribe\voices.sqlite3`). From then on, every new video is
checked against the stored voices after diarization, and confident matches
are named automatically — no dialogs, nothing interrupts processing. A
mis-recognized speaker can be corrected the same way; manual names always
win over automatic ones.

The more videos you confirm a person in, the more voice samples their
profile holds and the more robust recognition becomes. Profiles can be
listed with `.\run.ps1 --profiles` and seeded in bulk from a professionally
diarized transcript with `--seed <url> --reference <file>` (see
CONFIGURATION.md).

## Processing a whole channel (e.g. 1,400 videos)

1. Set the **cache policy** to `delete_after_video` (default) so disk usage
   stays flat (~50 MB per video peak).
2. Add the channel URL, press Start, leave it running. At roughly 10–15×
   realtime on an RTX 3080 Laptop, ~700 hours of content takes on the order
   of 2–3 days of continuous GPU time.
3. The randomized 8–15 s courtesy delay between downloads keeps YouTube
   happy; the GPU never waits on downloads because of prefetching.
4. If YouTube starts challenging downloads ("confirm you're not a bot"), set
   **Cookies from browser** in Settings — see CONFIGURATION.md.
5. Failures accumulate quietly in the retry queue; press **Retry Failed** at
   the end.
