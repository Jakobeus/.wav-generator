# .wav Generator

Chrome extension (Manifest V3) that downloads YouTube audio as a WAV file —
the whole video or a clip between two timestamps.

## Load it

1. `chrome://extensions` → enable **Developer mode**
2. **Load unpacked** → select this folder
3. Open a YouTube video, click the extension icon

## Usage

- **Start / End** accept `15s`, `1m30s`, `1:30`, or bare seconds. Leave End blank for end of video.
- **now** buttons fill the field with the player's current position — scrub the video to mark your clip.
- The clip must be **buffered**: seek near the clip start and let it play a few seconds first. The popup shows what audio is buffered.

## How it works

googlevideo stream URLs are single-use/session-bound — replaying them 403s.
So instead of re-fetching, we capture the bytes the player itself delivers:

| Piece | Role |
|---|---|
| `sniffer.js` | MAIN-world content script (document_start). Hooks `MediaSource.addSourceBuffer` / `SourceBuffer.appendBuffer` and copies every audio segment the player buffers, tagged with its `timestampOffset` → `window.__wavMedia` |
| `popup.js` → `extractPlayerInfo` | Reads video title/duration + audio buffered ranges |
| `popup.js` → `clipAndDownload` | Injected on demand (MAIN world): picks the continuous append run covering the clip, concatenates segments → `decodeAudioData` → sample-accurate slice → 16-bit WAV → `<a download>` |

Trimming happens on the decoded `AudioBuffer`, so clips are sample-accurate
and never lossy — one decode, exact slice out.

## Known limits

- Chrome Web Store won't accept YouTube downloaders — side-load only.
- Only buffered audio is available: whole-video export means letting it buffer through.
- Capture starts at page load — refresh the tab if the extension was just installed/reloaded.
- WAV is uncompressed: ~10 MB/min for stereo 44.1 kHz.
