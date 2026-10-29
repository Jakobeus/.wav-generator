'use strict';

// MAIN-world content script (document_start): hooks fetch/XHR before the
// player initializes, recording every googlevideo audio URL it requests.
// The popup reads window.__wavUrls from the page later.

if (!window.__wavUrls) {
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
