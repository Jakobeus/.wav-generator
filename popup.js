'use strict';

// =========================================================================
// Injected into the YouTube page (chrome.scripting). MUST be self-contained:
// these functions are serialized and cannot see popup.js scope.
// =========================================================================

// Video info + what audio the player has buffered (captured via MSE hooks).
function extractPlayerInfo() {
  const pr = window.ytInitialPlayerResponse || {};
  const vd = pr.videoDetails || {};
  const video = document.querySelector('video');

  // buffered ranges on the media element + total captured segment count
  let buffered = [];
  let chunks = 0;
  let coverage = []; // captured-audio intervals [[s,e]] from the smallest rec
  try {
    const b = video && video.buffered;
    if (b) for (let i = 0; i < b.length; i++) buffered.push([b.start(i), b.end(i)]);
    const media = window.__wavMedia;
    if (media) {
      const recs = [...media.recs.values()].filter(r => r.chunks.length);
      recs.sort((a, z) => a.bytes - z.bytes);
      const rec = recs[0];
      if (rec) {
        for (const r of recs) chunks += r.chunks.length;
        const offs = [...new Set(rec.chunks.map(c => c.off))].sort((a, z) => a - z);
        const gaps = offs.slice(1).map((o, i) => o - offs[i]).sort((a, z) => a - z);
        const med = gaps.length ? gaps[Math.floor(gaps.length / 2)] : 2;
        const thr = Math.max(15, med * 3);
        let cs = offs[0];
        for (let i = 1; i < offs.length; i++) {
          if (offs[i] - offs[i - 1] > thr) { coverage.push([cs, offs[i - 1] + med]); cs = offs[i]; }
        }
        coverage.push([cs, offs[offs.length - 1] + med]);
      }
    }
  } catch (_) {}

  return {
    videoId: vd.videoId || null,
    title: vd.title || document.title.replace(/ - YouTube$/, ''),
    duration: Number(vd.lengthSeconds) || (video ? video.duration : 0),
    currentTime: video ? video.currentTime : 0,
    buffered,
    coverage,
    chunks,
  };
}

// Runs in the page's MAIN world. Assembles captured audio segments that
// cover [start, end) seconds of video time, decodes, trims, encodes WAV.
async function clipAndDownload(start, end, filename) {
  const badge = document.createElement('div');
  badge.style.cssText =
    'position:fixed;top:12px;right:12px;z-index:2147483647;background:#111;color:#0f0;' +
    'padding:6px 10px;font:12px monospace;border-radius:6px;pointer-events:none';
  document.body.appendChild(badge);
  const say = t => { badge.textContent = 'wav: ' + t; };
  try {
    const media = window.__wavMedia;
    const recs = media ? [...media.recs.values()].filter(r => r.chunks.length) : [];
    if (!recs.length) return { error: 'No audio captured yet — press play on the video' };
    // Audio streams are far smaller than video — try smallest first; a
    // video-only stream fails decodeAudioData anyway, so this is how we
    // tell them apart without mime info.
    recs.sort((a, b) => a.bytes - b.bytes);
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    let buf = null, t0 = 0, decodedAny = false, trimmed = null;

    for (const rec of recs) {
      // Split appends into runs continuous in video time (a seek restarts one).
      const runs = [[]];
      for (const c of rec.chunks) {
        const run = runs[runs.length - 1];
        const prev = run[run.length - 1];
        if (prev && (c.off < prev.off - 0.5 || c.off - prev.off > 120)) runs.push([c]);
        else run.push(c);
      }
      // Sort by video-time offset, dedupe re-appends at the same offset.
      runs.forEach(r => r.sort((a, b) => a.off - b.off));
      // Most likely run first: the one whose start offset is just below `start`.
      runs.sort((a, b) => (b[0].off <= start ? b[0].off : -1e9) - (a[0].off <= start ? a[0].off : -1e9));

      for (const run of runs) {
        const seen = new Set();
        const parts = [];
        let total = 0;
        for (const c of run) {
          const k = `${c.off}|${c.data.byteLength}`;
          if (seen.has(k)) continue;
          seen.add(k);
          parts.push(c);
          total += c.data.byteLength;
        }
        const ab = new Uint8Array(total);
        let off = 0;
        for (const c of parts) { ab.set(new Uint8Array(c.data), off); off += c.data.byteLength; }
        say(`decoding ${(total / 1e6).toFixed(1)} MB`);
        try {
          const d = await ctx.decodeAudioData(ab.buffer);
          decodedAny = true;
          const base = run[0].off;
          // take the intersection of requested range and decoded coverage
          if (start < base + d.duration - 0.5 && end > base + 0.5) {
            buf = d; t0 = base;
            trimmed = start < base || end > base + d.duration
              ? [Math.max(start, base), Math.min(end, base + d.duration)]
              : null;
            break;
          }
        } catch (_) {}
      }
      if (buf) break;
    }
    if (!buf) {
      const ranges = [];
      try {
        const b = document.querySelector('video').buffered;
        for (let i = 0; i < b.length; i++) ranges.push(`${b.start(i).toFixed(0)}-${b.end(i).toFixed(0)}s`);
      } catch (_) {}
      return decodedAny
        ? { error: `Clip not fully buffered. Player has ${ranges.join(', ') || 'nothing'} — ` +
                   `seek just before the clip and let it play, then retry.` }
        : { error: `Segments captured (${recs.reduce((n, r) => n + r.chunks.length, 0)}) but won't decode — ` +
                   `init segment was likely missed. Hard-refresh the tab, press play, retry.` };
    }

    say('encoding wav');
    const sr = buf.sampleRate;
    const s = Math.max(0, Math.floor((start - t0) * sr));
    const e = Math.min(buf.length, Math.ceil((end - t0) * sr));
    if (e <= s) return { error: 'Clip range is empty' };
    const ch = Math.min(2, buf.numberOfChannels);
    const frames = e - s;
    const dataLen = frames * ch * 2;
    const wav = new ArrayBuffer(44 + dataLen);
    const dv = new DataView(wav);
    const wstr = (o, t) => { for (let i = 0; i < t.length; i++) dv.setUint8(o + i, t.charCodeAt(i)); };
    wstr(0, 'RIFF'); dv.setUint32(4, 36 + dataLen, true); wstr(8, 'WAVE');
    wstr(12, 'fmt '); dv.setUint32(16, 16, true);
    dv.setUint16(20, 1, true);
    dv.setUint16(22, ch, true);
    dv.setUint32(24, sr, true);
    dv.setUint32(28, sr * ch * 2, true);
    dv.setUint16(32, ch * 2, true);
    dv.setUint16(34, 16, true);
    wstr(36, 'data'); dv.setUint32(40, dataLen, true);
    const chans = [];
    for (let c = 0; c < ch; c++) chans.push(buf.getChannelData(c));
    let p = 44;
    for (let i = 0; i < frames; i++) {
      for (let c = 0; c < ch; c++) {
        const v = Math.max(-1, Math.min(1, chans[c][s + i]));
        dv.setInt16(p, v < 0 ? v * 0x8000 : v * 0x7fff, true);
        p += 2;
      }
    }

    say('saving');
    const blob = new Blob([wav], { type: 'audio/wav' });
    const objUrl = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = objUrl;
    a.download = filename;
    a.click();
    setTimeout(() => URL.revokeObjectURL(objUrl), 60000);
    badge.remove();
    return {
      ok: true,
      saved: [t0 + s / sr, t0 + e / sr],
      note: trimmed ? `trimmed to buffered range ${trimmed[0].toFixed(1)}s–${trimmed[1].toFixed(1)}s` : null,
    };
  } catch (err) {
    badge.remove();
    return { error: String(err && err.message || err) };
  }
}

// =========================================================================
// Popup context below
// =========================================================================

const $ = id => document.getElementById(id);
const status = (msg, err) => { $('status').textContent = msg; $('status').className = err ? 'err' : ''; };

function parseTime(s) {
  s = (s || '').trim().toLowerCase();
  if (!s) return NaN;
  // "1m30s", "2m", "15s", "0.5s"
  const ms = s.match(/^(?:(\d+(?:\.\d+)?)\s*m)?\s*(?:(\d+(?:\.\d+)?)\s*s)?$/);
  if (ms && (ms[1] !== undefined || ms[2] !== undefined))
    return Number(ms[1] || 0) * 60 + Number(ms[2] || 0);
  // "1:30", "1:02:30", or bare seconds
  const parts = s.split(':').map(Number);
  if (parts.length <= 3 && parts.every(n => !isNaN(n)))
    return parts.reduce((acc, p) => acc * 60 + p, 0);
  return NaN;
}

function fmtTime(sec) {
  sec = Math.max(0, Math.floor(sec));
  const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
  const mmss = `${m}:${String(s).padStart(2, '0')}`;
  return h ? `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}` : mmss;
}

const fmtRanges = rs =>
  rs.map(([a, b]) => `${fmtTime(a)}-${fmtTime(b)}`).join(', ');

const sanitize = s => s.replace(/[\\/:*?"<>|]+/g, '_').replace(/\s+/g, ' ').trim().slice(0, 120);

let tab = null;
let info = null;

async function refreshInfo(fillTimes) {
  const [res] = await chrome.scripting.executeScript({
    target: { tabId: tab.id },
    world: 'MAIN',
    func: extractPlayerInfo,
  });
  if (res?.error) throw new Error(res.error.message || 'Injection failed');
  info = res?.result;
  if (!info) throw new Error('Could not read player data');
  if (fillTimes) {
    $('start').value = '0:00';
    $('end').value = '';
  }
  $('title').textContent = info.title;
  $('meta').textContent = info.coverage.length
    ? `${fmtTime(info.duration)} · captured ${fmtRanges(info.coverage)} · ${info.chunks} segments`
    : `${fmtTime(info.duration)} · no audio captured — press play`;
  $('go').disabled = !info.chunks;
  return info;
}

async function currentTime() {
  const [res] = await chrome.scripting.executeScript({
    target: { tabId: tab.id },
    world: 'MAIN',
    func: () => document.querySelector('video')?.currentTime ?? 0,
  });
  return res.result || 0;
}

function defaultFilename(start, end) {
  const base = sanitize(info.title || info.videoId || 'clip');
  const clipped = start > 0 || end < info.duration;
  return clipped ? `${base} [${fmtTime(start)}-${fmtTime(end)}].wav` : `${base}.wav`;
}

async function init() {
  try {
    [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (!tab || !/youtube\.com\/watch/.test(tab.url || '')) {
      status('Open a YouTube watch page first.', true);
      return;
    }
    await refreshInfo(true);
    $('fname').value = defaultFilename(0, info.duration);
  } catch (e) {
    status(e.message, true);
  }

  $('startNow').onclick = async () => { $('start').value = fmtTime(await currentTime()); };
  $('endNow').onclick = async () => { $('end').value = fmtTime(await currentTime()); };

  $('go').onclick = async () => {
    const start = parseTime($('start').value) || 0;
    const endIn = $('end').value.trim();
    const end = endIn ? parseTime(endIn) : info.duration;
    if (isNaN(start) || isNaN(end) || end <= start) {
      status('Invalid start/end times.', true);
      return;
    }
    const filename = $('fname').value.trim() || defaultFilename(start, end);
    $('go').disabled = true;
    status('Working — watch the badge on the page…');
    try {
      const [r] = await chrome.scripting.executeScript({
        target: { tabId: tab.id },
        world: 'MAIN',
        func: clipAndDownload,
        args: [start, end, filename],
      });
      const out = r?.result;
      if (r?.error) status('Error: ' + (r.error.message || r.error), true);
      else if (out?.error) status('Error: ' + out.error, true);
      else if (out?.ok)
        status(`Saved ${fmtTime(out.saved[0])}–${fmtTime(out.saved[1])}` + (out.note ? ` (${out.note})` : ''));
      else status('Unexpected result — check page console.', true);
    } catch (e) {
      status('Error: ' + e.message, true);
    }
    $('go').disabled = false;
  };
}

init();
