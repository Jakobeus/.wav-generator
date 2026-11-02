'use strict';

// ISOLATED world, document_start — this world is guaranteed to run before
// any page script. Fetch our sniffer synchronously and inline it as a
// <script> text node: it executes the moment it's inserted, before the
// parser reaches any of the page's own scripts (a src= tag would load
// async and lose the race against early player init).

try {
  const xhr = new XMLHttpRequest();
  xhr.open('GET', chrome.runtime.getURL('sniffer.js'), false); // sync on purpose
  xhr.send();
  const s = document.createElement('script');
  s.textContent = xhr.responseText;
  (document.documentElement || document.head).appendChild(s);
  s.remove();
} catch (_) {
  // the MAIN-world content_script registration in the manifest still applies
}
