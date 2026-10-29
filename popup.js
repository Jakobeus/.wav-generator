'use strict';

// =========================================================================
// Injected into the YouTube page (chrome.scripting). MUST be self-contained:
// these functions are serialized and cannot see popup.js scope.
// =========================================================================

// Reads the player response for the audio-only adaptive format URL.
// Falls back to URLs captured by the sniffer/content hooks + resource timing.
function extractPlayerInfo() {
  const AUDIO_ITAGS = new Set([139, 140, 141, 249, 250, 251, 600]);
  const ITAG_RANK = { 251: 5, 141: 4, 140: 3, 250: 2, 600: 2, 139: 1, 249: 0 };

  const sniffAudioUrl = () => {
    const hooked = Array.isArray(window.__wavUrls) ? window.__wavUrls.slice() : [];
    const timed = performance.getEntriesByType('resource').map(e => e.name);
    const candidates = hooked.concat(timed);
    const hits = candidates
      .filter(n => n.includes('googlevideo.com/videoplayback'))
      .map(n => {
        const itag = Number((n.match(/itag=(\d+)/) || [])[1]);
        return { n, itag, audio: AUDIO_ITAGS.has(itag) || /mime=audio/.test(n) };
      })
      .filter(h => h.audio);
    if (!hits.length) return { url: null, seen: candidates.length };
    hits.sort((a, b) => (ITAG_RANK[b.itag] || 0) - (ITAG_RANK[a.itag] || 0));
    const u = new URL(hits[0].n);
    u.searchParams.delete('range');
    u.searchParams.delete('rn');
    u.searchParams.delete('rbuf');
    return { url: u.toString(), seen: candidates.length };
  };

  const pr = window.ytInitialPlayerResponse || {};
  const vd = pr.videoDetails || {};
  const audio = (pr.streamingData?.adaptiveFormats || [])
    .filter(f => /^audio\//.test(f.mimeType || '') && f.url)
    .sort((a, b) => (b.averageBitrate || b.bitrate || 0) - (a.averageBitrate || a.bitrate || 0));
  const best = audio[0];
  const video = document.querySelector('video');
  const sniffed = best ? null : sniffAudioUrl();
  const sniffedUrl = sniffed ? sniffed.url : null;
  const sniffedMime = sniffedUrl
    ? decodeURIComponent((sniffedUrl.match(/mime=([^&]+)/) || [])[1] || '')
    : null;
  return {
    videoId: vd.videoId || null,
    title: vd.title || document.title.replace(/ - YouTube$/, ''),
    duration: Number(vd.lengthSeconds) || (video ? video.duration : 0),
    currentTime: video ? video.currentTime : 0,
    audioUrl: best ? best.url : sniffedUrl,
    mime: best ? String(best.mimeType).split(';')[0] : sniffedMime,
    bitrate: best ? Math.round(best.averageBitrate || best.bitrate || 0) : null,
    sniffSeen: sniffed ? sniffed.seen : 0,
  };
}

// Runs in the page's MAIN world (identical request context to the player):
// fetch → decode → trim [start,end) sec → encode WAV → save.
async function clipAndDownload(url, rawUrl, start, end, filename) {
  const badge = document.createElement('div');
  badge.style.cssText =
    'position:fixed;top:12px;right:12px;z-index:2147483647;background:#111;color:#0f0;' +
    'padding:6px 10px;font:12px monospace;border-radius:6px;pointer-events:none';
  document.body.appendChild(badge);
  const say = t => { badge.textContent = 'wav: ' + t; };
  try {
    say('fetching 0%');
    let resp = await fetch(url, { credentials: 'include' });
    // If the signature covers the stripped range params, retry the raw URL
    // exactly as the player requested it.
    if (!resp.ok && rawUrl && rawUrl !== url)
      resp = await fetch(rawUrl, { credentials: 'include' });
    if (!resp.ok) return { error: `HTTP ${resp.status} on audio fetch` };
    const total = Number(resp.headers.get('content-length')) || 0;
    const reader = resp.body.getReader();
    const chunks = [];
    let got = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      got += value.length;
      say(total ? `fetching ${Math.round((got / total) * 100)}%` : `fetched ${(got / 1e6).toFixed(1)} MB`);
    }
    const ab = new Uint8Array(got);
    let off = 0;
    for (const c of chunks) { ab.set(c, off); off += c.length; }

    say('decoding');
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const buf = await ctx.decodeAudioData(ab.buffer);
    const sr = buf.sampleRate;
    const s = Math.max(0, Math.floor(start * sr));
    const e = Math.min(buf.length, Math.ceil(end * sr));
    if (e <= s) return { error: 'Clip range is empty' };

    say('encoding wav');
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
    return { ok: true };
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
  // Background webRequest sniffer takes priority — its URL is one the
  // player literally just fetched, so it's guaranteed fresh and signed.
  let bg = null;
  try {
    bg = await chrome.runtime.sendMessage({ type: 'find-stream', videoId: info.videoId, tabId: tab.id });
  } catch (_) {}
  if (bg?.url) {
    info.audioUrl = bg.url;
    info.rawUrl = bg.rawUrl;
    info.mime = bg.mime || info.mime;
  }
  info.sniffSeen = bg ? bg.seen : info.sniffSeen;
  info.sniffTotal = bg ? bg.totalSeen : 0;
  if (fillTimes) {
    $('start').value = '0:00';
    $('end').value = '';
  }
  $('title').textContent = info.title;
  $('meta').textContent = info.audioUrl
    ? `${fmtTime(info.duration)} · ${info.mime || 'stream'}${info.bitrate ? ` @ ${info.bitrate} kbps` : ''}`
    : `No audio stream found (${info.sniffSeen} matched / ${info.sniffTotal} total) — press play, then reopen`;
  $('go').disabled = !info.audioUrl;
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
        args: [info.audioUrl, info.rawUrl, start, end, filename],
      });
      const out = r?.result;
      if (r?.error) status('Error: ' + (r.error.message || r.error), true);
      else if (out?.error) status('Error: ' + out.error, true);
      else if (out?.ok) status('Done — check your downloads.');
      else status('Unexpected result — check page console.', true);
    } catch (e) {
      status('Error: ' + e.message, true);
    }
    $('go').disabled = false;
  };
}

init();
