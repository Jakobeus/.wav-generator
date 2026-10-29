'use strict';

// Service worker: webRequest sniffer only. Sees every request the tab makes
// (page, worker, or service worker context) — page-context fetch/XHR hooks
// can't guarantee coverage.

const seen = []; // {url, tabId, vid, itag, mime, ts}
const AUDIO_ITAGS = new Set([139, 140, 141, 249, 250, 251, 600]);
const COMBINED_ITAGS = new Set([18, 22, 37, 38]); // audio+video progressive
const ITAG_RANK = { 251: 5, 141: 4, 140: 3, 250: 2, 600: 2, 139: 1, 249: 0 };

const RANK = s => {
  if (s.mime && s.mime.startsWith('audio')) return 10 + (ITAG_RANK[s.itag] || 0);
  if (AUDIO_ITAGS.has(s.itag)) return 10 + (ITAG_RANK[s.itag] || 0);
  return COMBINED_ITAGS.has(s.itag) ? 5 : 0;
};

chrome.webRequest.onBeforeRequest.addListener(
  d => {
    try {
      const u = new URL(d.url);
      const itag = Number(u.searchParams.get('itag'));
      const mime = u.searchParams.get('mime') || '';
      let vid = null;
      try { vid = d.documentUrl && new URL(d.documentUrl).searchParams.get('v'); } catch (_) {}
      seen.push({ url: d.url, tabId: d.tabId, vid, itag, mime, ts: d.timeStamp });
      if (seen.length > 200) seen.splice(0, seen.length - 200);
    } catch (_) {}
  },
  { urls: ['*://*.googlevideo.com/videoplayback*'] }
);

function findStream(msg) {
  // Prefer requests attributable to this video/tab; fall back to recent
  // audio-typed requests from any tab (SW-issued requests lack tabId).
  const cands = seen.filter(s =>
    (msg.videoId && s.vid === msg.videoId) || s.tabId === msg.tabId);
  let pool = cands.filter(s => RANK(s) > 0);
  if (!pool.length) pool = seen.filter(s => RANK(s) > 10); // audio-only, any tab
  let pick = null;
  for (const c of pool)
    if (!pick || RANK(c) > RANK(pick) || (RANK(c) === RANK(pick) && c.ts > pick.ts)) pick = c;
  if (!pick) return { url: null, seen: cands.length, totalSeen: seen.length };
  const u = new URL(pick.url);
  ['range', 'rn', 'rbuf'].forEach(p => u.searchParams.delete(p));
  return {
    url: u.toString(),
    rawUrl: pick.url,
    itag: pick.itag,
    mime: pick.mime || u.searchParams.get('mime'),
    seen: cands.length,
    totalSeen: seen.length,
  };
}

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.type === 'find-stream') sendResponse(findStream(msg));
});
