# .wav Generator

Chrome extension (Manifest V3) that downloads YouTube audio as a WAV file —
the whole video or a clip between two timestamps.

## Load it

1. `chrome://extensions` → enable **Developer mode**
2. **Load unpacked** → select this folder
3. Open a YouTube video, click the extension icon

## Usage

- **Start / End** accept `90`, `1:30`, `1:02:30`, or `1:30.5`. Leave End blank for end of video.
- **now** buttons fill the field with the player's current position — scrub the video to mark your clip.
- The decoded audio is cached for the last video, so cutting several clips back-to-back skips the refetch.

## How it works

| Piece | Role |
|---|---|
| `background.js` | `webRequest` listener records every `videoplayback` request the tab emits — fully signed URLs, no de-ciphering needed |
| `popup.js` | Reads video info via MAIN-world injection, gets best stream URL from the background sniffer |
| `popup.js` → `clipAndDownload` | Injected into the page (ISOLATED world): fetches the stream (same context the player uses — no CORS issues), `decodeAudioData` → sample-accurate slice → 16-bit WAV → `<a download>` |
| `sniffer.js` | Extra fetch/XHR hook as a backup URL source |

Trimming happens on the decoded `AudioBuffer`, so clips are sample-accurate
and never lossy — one decode, exact slice out.

## Known limits

- Chrome Web Store won't accept YouTube downloaders — side-load only.
- Ciphered formats are handled by reusing the signed stream URL the player itself fetched (Resource Timing sniff) — press play once if the popup reports no stream. True DRM content still won't work.
- WAV is uncompressed: ~10 MB/min for stereo 44.1 kHz.
- Long videos mean big decode buffers; only the last one is kept in memory.
