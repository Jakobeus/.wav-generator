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
    const isWorker = typeof g.document === 'undefined';

    // ---- init-segment siphon --------------------------------------------
    // The init append usually lands before our hook exists, but its bytes
    // also pass through fetch/XHR responses. Peek at the first bytes of EVERY
    // response — a media container init starts with 'ftyp' (fMP4) or the EBML
    // magic (WebM). Non-matching clones are cancelled after one chunk so big
    // media responses are never duplicated.
    const looksInit = buf => {
      try {
        const b = new Uint8Array(buf.slice(0, 8));
        return (b[4] === 0x66 && b[5] === 0x74 && b[6] === 0x79 && b[7] === 0x70) || // 'ftyp'
               (b[0] === 0x1a && b[1] === 0x45 && b[2] === 0xdf && b[3] === 0xa3);   // EBML
      } catch (_) { return false; }
    };
    const stashInit = data => {
      if (isWorker) g.postMessage({ __wavInit: { data } }, [data]);
      else g.__wavStoreInit(data);
    };
    if (typeof g.fetch === 'function' && !g.fetch.__wavInitTap) {
      const of = g.fetch;
      const wrapped = function (...a) {
        const p = of.apply(this, a);
        p.then(r => {
          try {
            const body = r.clone().body;
            if (!body || !body.getReader) return;
            const reader = body.getReader();
            reader.read().then(first => {
              const v = first.value;
              if (!v || !looksInit(v.buffer.slice(v.byteOffset, v.byteOffset + v.byteLength))) {
                reader.cancel().catch(() => {});
                return;
              }
              const parts = [v];
              const readAll = () => reader.read().then(({ value: w, done }) => {
                if (w) parts.push(w);
                if (!done) return readAll();
                let n = 0;
                for (const x of parts) n += x.length;
                const all = new Uint8Array(n);
                let o = 0;
                for (const x of parts) { all.set(x, o); o += x.length; }
                stashInit(all.buffer);
              });
              readAll().catch(() => {});
            }).catch(() => {});
          } catch (_) {}
        }).catch(() => {});
        return p;
      };
      wrapped.__wavInitTap = true;
      g.fetch = wrapped;
    }
    const XP = g.XMLHttpRequest && g.XMLHttpRequest.prototype;
    if (XP && XP.send && !XP.__wavInitTap) {
      XP.__wavInitTap = true;
      const oSend = XP.send;
      XP.send = function (...a) {
        this.addEventListener('load', () => {
          try {
            const r = this.response;
            if (r instanceof ArrayBuffer && looksInit(r)) stashInit(r.slice(0));
          } catch (_) {}
        });
        return oSend.apply(this, a);
      };
    }

    // ---- SourceBuffer append capture -------------------------------------
    const SB = g.SourceBuffer;
    if (!SB || !SB.prototype || !SB.prototype.appendBuffer) return;
    if (SB.prototype.__wavHooked) return;
    SB.prototype.__wavHooked = true;

    const state = new WeakMap(); // sb -> {id, pending:[ArrayBuffer], last:[[s,e]], lastOff}
    let next = 0;
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
      let cur;
      try {
        cur = snap(this);
      } catch (_) {
        return; // SourceBuffer removed (SPA nav) — dead sb, drop the chunk
      }
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

window.__wavVer = 9;

if (!window.__wavMedia) {
  // ---- main-thread store -------------------------------------------------
  const media = { recs: new Map(), inits: [], vid: undefined };
  window.__wavMedia = media;
  const dbg = window.__wavDbg = {
    hook: Math.round(performance.now()), // ms after nav-start when this ran
    first: null,                          // ms of first captured append
    ftyp: 0,                              // recs that contain an ftyp/EBML chunk
    wInj: 0, wFail: 0,                    // workers instrumented vs fell back
  };
  const hasInitMagic = data => {
    const b = new Uint8Array(data.slice(0, 8));
    return (b[4] === 0x66 && b[5] === 0x74 && b[6] === 0x79 && b[7] === 0x70)
        || (b[0] === 0x1a && b[1] === 0x45 && b[2] === 0xdf && b[3] === 0xa3);
  };
  const curVid = () => (location.href.match(/[?&]v=([^&]+)/) || [])[1] || '';
  // YouTube is an SPA: 'yt-navigate-finish' can fire SECONDS into playback of
  // the new video (async navigation completion), so clearing blindly here
  // wipes init segments already captured. Tag every rec with its video id and
  // only drop recs belonging to OTHER videos.
  window.addEventListener('yt-navigate-finish', () => {
    const v = curVid();
    for (const [k, r] of media.recs) if (r.vid !== v) media.recs.delete(k);
    media.inits = media.inits.filter(i => i.vid === v);
  });
  window.__wavStoreInit = data => {
    const vid = curVid();
    if (!media.inits.some(i => i.vid === vid && i.data.byteLength === data.byteLength
        && new Uint8Array(i.data.slice(0, 16)).join() === new Uint8Array(data.slice(0, 16)).join()))
      media.inits.push({ data, vid });
  };
  window.__wavStore = (key, off, data) => {
    const vid = curVid();
    if (dbg.first === null) dbg.first = Math.round(performance.now());
    let rec = media.recs.get(key);
    if (!rec) { rec = { chunks: [], bytes: 0, ftyp: false, vid }; media.recs.set(key, rec); }
    if (rec.vid !== vid) { rec.chunks.length = 0; rec.bytes = 0; rec.vid = vid; }
    if (!rec.ftyp && hasInitMagic(data)) { rec.ftyp = true; dbg.ftyp++; }
    rec.chunks.push({ off, data, vid });
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
      if (real !== url) dbg.wInj++;
    } catch (_) {
      dbg.wFail++;
      w = new OrigWorker(url, opts); // CSP refused the blob — don't break playback
    }
    const prefix = 'w' + (++wid) + '_';
    w.addEventListener('message', e => {
      const m = e.data;
      if (m && m.__wavChunk)
        window.__wavStore(prefix + m.__wavChunk.id, m.__wavChunk.off, m.__wavChunk.data);
      else if (m && m.__wavInit)
        window.__wavStoreInit(m.__wavInit.data);
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
