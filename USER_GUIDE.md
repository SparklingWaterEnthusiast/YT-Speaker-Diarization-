# USER_GUIDE.md — v0.3.0

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
│ [Open Output Folder][Open Audio / Cache][Settings…]           │
│ GPU / VRAM / CPU / RAM / Disk / Temp / Clock / Power          │
└──────────────────────────────────────────────────────────────┘
```

## Queue tabs (v0.2.1)

Queues work like File Explorer tabs: each tab is an independent queue with
its **own subfolder** in the output directory (the original "Main" tab
writes to the output root).

- **＋** creates a new tab (named with the current date/time by default).
- **Double-click a tab** to rename it — the output subfolder is renamed
  with it.
- **Drag tabs** to reorder them; **right-click a tab** for rename, open
  output folder, stop queue, and delete queue (deleting removes only the
  list — transcripts and cached results stay on disk).
- URLs you add always go to the **currently selected tab**.

**Priority processing:** press Start on any tab. If another queue is
already running, the new tab takes over as soon as the current video
finishes, and the older queue resumes automatically when the new tab is
done. That's the way to push one urgent video through while an
866-video channel run is in progress: new tab → paste URL → Start.
The same video may live in several tabs; re-processing it elsewhere is
nearly instant because all pipeline results are cached.

Each tab gets its own combined transcript (in its subfolder) when it
finishes.

## Managing the queue list

The queue table behaves like a file manager:

- **Multi-select** with Ctrl/Shift-click or Ctrl+A.
- **Right-click** for actions on the selection: retry failed *and
  cancelled* videos, reprocess completed ones, move up/down/top/bottom
  (manual queue reordering), open the transcript or output folder, and
  remove entries (Del key works too). Removing never deletes transcripts
  or cached results.
- **Double-click** a completed video to open the speaker editor.
- The toolbar "Retry Failed" button requeues both failed and cancelled
  videos across the current tab.

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

- **Pause** requests a wait at the next checkpoint after the current stage.
  It does not immediately interrupt a GPU call or unload resident models.
  **Resume** continues in the same process.
- **Cancel Current** abandons the current video at the next checkpoint
  (mid-download and mid-transcription cancel quickly; a diarization pass
  finishes its current model call first) and moves on.
- **Closing the app** with work running asks to stop and waits for workers,
  active model calls, prefetch, and cleanup to finish. Completed stage artifacts
  stay on disk; incomplete work returns to the queue. On relaunch, press Start.
  If transcription is cached, only the missing diarization model loads. A new
  process cannot reuse the previous process's GPU objects.
- A second worker using the same cache or job database is rejected. Let the first window/CLI
  finish closing before starting another run.

## Optimization (v0.3.0)

Open **Settings → Optimization**. Opening Settings does not probe CUDA; click
**Probe hardware capabilities** to inspect GPU/VRAM, compute capability,
supported precisions, CPU cores, and RAM. The explicit probe may initialize a
CUDA context, but does not load a model. Unknown information stays unknown.

Choose a preset, click **Apply preset**, then **OK** to save:

| Preset | Behavior |
|---|---|
| Custom | Preserves saved controls; merely selecting a preset does not apply it |
| Safe / Conservative | Sequential large-v3, beam 5; releases the other model before an uncached GPU stage, trading reload time for VRAM |
| Balanced / Baseline | Sequential large-v3, beam 5; keeps loaded models during the run |
| Performance / Optional batch mode | large-v3, beam 5, ASR batch 4; faster in the measured full-video comparison; can increase memory demand and changes decoding behavior |

All presets keep VAD and word timestamps enabled and one ASR worker. A probed
GPU with at most 6 GB selects stage residency even for Balanced/Performance.
CPU-thread and diarization-batch values retain your existing settings. No
preset enables simultaneous GPU jobs. See [CONFIGURATION.md](CONFIGURATION.md)
for each control and the [benchmark report](benchmark/V0.3.0_REPORT.md) for evidence; batch 8 has only
been measured on a 180-second ASR prefix, not the full-video pipeline.

Settings are copied when a run starts. Changes saved during processing apply
to the next run, not the next video in that already-running queue. Existing
cached stages are still reused; changing settings or choosing **Reprocess**
does not automatically invalidate them. Keep word timestamps on for speaker
attribution. If memory is insufficient, the app can reduce the effective batch
and retry within the configured limit; the log shows what happened. Diarization
also applies a temporary PyTorch allocator budget based on physical free VRAM.
Before ASR, unused PyTorch cache is released while live model tensors stay.
No NVIDIA driver setting is changed. Batch 1 and model-loading failures can
still fail the item.

## Resources and diagnostics

The resource strip adds GPU temperature, SM clock, and power where the driver
reports them (`n/a` otherwise). Sustained driver-reported thermal slowdown
produces a warning; the app does not change settings to throttle the GPU.
High GPU utilization is not itself a fault, and high VRAM use alone is not a leak.

Enable **Record diagnostic performance traces** for timing and resource JSONL
files under `<cache_dir>\diagnostics\`; the log prints the filename. Default
sampling is once per second. Profiling is off by default. Traces distinguish
stages/cache hits and include process IDs; driver VRAM includes other processes,
while PyTorch memory counters exclude CTranslate2. Traces may contain local
paths and video identifiers; inspect them before sharing.

## Audio and cache folders

**Open Output Folder** opens the selected queue's transcripts. **Open Audio /
Cache** opens the cache root; right-click a single video to open its own audio
and stage-cache folder. Settings also has **Open** buttons beside both paths.

To keep source audio and `audio.wav`, choose **Retain downloaded and processed
audio until I delete it** in Settings → Downloads → Audio cache policy before
starting the run. This uses the existing per-video cache, not a separate archive.
Deletion policies remove application-owned audio after successful completion;
stage JSON and short speaker samples remain. Failed/cancelled items can retain
audio for retry. Changing retention cannot restore audio already deleted.

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

**Double-click any completed video in the queue table** (or right-click →
"Edit speakers…"). A dropdown lists every detected speaker with:

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

1. Use **delete audio after each completed video** to limit working audio.
   Disk use still grows with retained JSON/samples and failed-item audio;
   WAV size and prefetch also matter. Retain audio only if you need it.
2. Add the channel URL and press Start. Use measured queue progress for an
   estimate; the earlier 10–15×/2–3-day projection was not a validated sustained
   benchmark. Cache-only re-exports no longer inflate the speed estimate.
3. Downloads use an 8–15 s randomized delay by default. Prefetch can hide
   download latency, but slow networks or server delays can still leave the GPU waiting.
4. If YouTube starts challenging downloads ("confirm you're not a bot"), set
   **Cookies from browser** in Settings — see CONFIGURATION.md.
5. Failures accumulate quietly in the retry queue; press **Retry Failed** at
   the end.
