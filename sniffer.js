'use strict';

// MAIN-world content script (document_start). Captures the media bytes the
// player buffers — googlevideo URLs are single-use, but the bytes delivered
// to MediaSource are right here. Handles appends made in workers too, by
// wrapping the Worker constructor and injecting the same hook into workers.

// ---- shared capture logic (also stringified into worker scripts) ----------
// Must be self-contained: it is serialized via .toString().
//
// Players don't bump timestampOffset per segment (fMP4 segments carry their
// own timestamps), so we derive each chunk's video time by diffing
// sb.buffered on 'updateend': the range that grew is where it landed.
function __wavInstallCapture(g) {
  try {
    const SB = g.SourceBuffer;
    if (!SB || !SB.prototype || !SB.prototype.appendBuffer) return;
    if (SB.prototype.__wavHooked) return;
    SB.prototype.__wavHooked = true;

    const state = new WeakMap(); // sb -> {id, pending:[ArrayBuffer], last:[[s,e]], lastOff}
    let next = 0;
    const isWorker = typeof g.document === 'undefined';
    const snap = sb => {
      const b = sb.buffered, r = [];
      for (let i = 0; i < b.length; i++) r.push([b.start(i), b.end(i)]);
      return r;
    };
    const emit = (st, off, data) => {
      if (isWorker) g.postMessage({ __wavChunk: { id: st.id, off, data } }, [data]);
      else g.__wavStore('m' + st.id, off, data);
    };
    const onUpdateEnd = function () {
      const s = state.get(this);
      if (!s || !s.pending.length) return;
      const data = s.pending.shift(); // one append → one updateend
      const cur = snap(this);
      let off = null;
      for (const [a, b] of cur) {
        const prev = s.last.find(([pa, pb]) => a < pb - 0.01 && b > pa + 0.01);
        if (!prev) { off = a; break; }        // brand-new range
        if (b > prev[1] + 0.01) { off = prev[1]; break; } // extended at end
      }
      if (off === null) off = s.lastOff;
      s.lastOff = off;
      s.last = cur;
      emit(s, off, data);
    };
    const orig = SB.prototype.appendBuffer;
    SB.prototype.appendBuffer = function (data) {
      try {
        let s = state.get(this);
        if (!s) {
          s = { id: next++, pending: [], last: snap(this), lastOff: 0 };
          state.set(this, s);
          this.addEventListener('updateend', onUpdateEnd);
        }
        const buf = data instanceof ArrayBuffer
          ? data.slice(0)
          : (data && data.buffer
              ? data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength)
              : null);
        if (buf) s.pending.push(buf);
      } catch (_) {}
      return orig.call(this, data);
    };
  } catch (_) {}
}

if (!window.__wavMedia) {
  // ---- main-thread store -------------------------------------------------
  const media = { recs: new Map() }; // key -> {chunks:[{off,data}], bytes}
  window.__wavMedia = media;
  window.__wavStore = (key, off, data) => {
    let rec = media.recs.get(key);
    if (!rec) { rec = { chunks: [], bytes: 0 }; media.recs.set(key, rec); }
    rec.chunks.push({ off, data });
    rec.bytes += data.byteLength;
    while (rec.bytes > 120e6 && rec.chunks.length > 1) {
      rec.bytes -= rec.chunks[0].data.byteLength;
      rec.chunks.shift();
    }
  };

  __wavInstallCapture(window);

  // ---- worker injection --------------------------------------------------
  let wid = 0;
  const hookSrc = '\n;(' + __wavInstallCapture.toString() + ')(self);\n';
  const OrigWorker = window.Worker;
  window.Worker = function (url, opts) {
    let real = url;
    try {
      const u = typeof url === 'string' ? url : url && url.url;
      const xhr = new XMLHttpRequest();
      xhr.open('GET', u, false); // sync — must be ready before construct returns
      xhr.send();
      real = URL.createObjectURL(new Blob([hookSrc, xhr.responseText], { type: 'text/javascript' }));
    } catch (_) {}
    let w;
    try {
      w = new OrigWorker(real, opts);
    } catch (_) {
      w = new OrigWorker(url, opts); // CSP refused the blob — don't break playback
    }
    const prefix = 'w' + (++wid) + '_';
    w.addEventListener('message', e => {
      const m = e.data;
      if (m && m.__wavChunk)
        window.__wavStore(prefix + m.__wavChunk.id, m.__wavChunk.off, m.__wavChunk.data);
    });
    return w;
  };
  window.Worker.prototype = OrigWorker.prototype;
  window.Worker.toString = () => OrigWorker.toString();

  // ---- fallback: record stream URLs seen in fetch/XHR --------------------
  const urls = [];
  window.__wavUrls = urls;
  const AUDIO = /mime=audio|itag=(139|140|141|249|250|251|600)\b/;
  const record = u => {
    try {
      const s = typeof u === 'string' ? u : u && u.url;
      if (s && s.includes('googlevideo.com/videoplayback') && AUDIO.test(s) && !urls.includes(s))
        urls.push(s);
    } catch (_) {}
  };
  const origFetch = window.fetch;
  window.fetch = function (...args) {
    record(args[0]);
    return origFetch.apply(this, args);
  };
  const origOpen = XMLHttpRequest.prototype.open;
  XMLHttpRequest.prototype.open = function (method, url, ...rest) {
    record(url);
    return origOpen.call(this, method, url, ...rest);
  };
}
