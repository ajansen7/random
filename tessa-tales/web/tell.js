/* Tessa's turn: one page at a time, talk into the snowflake, hear yourself. Then story time. */
(() => {
  const bookId = location.pathname.split('/').filter(Boolean)[1];
  const $ = (id) => document.getElementById(id);
  const { chime, burst, thumb, large, sleep, reduce } = TT;
  const DWELL_MS = 1800;

  let book, voice = {}, items = [], at = 0;
  let flipTimer = null, layer = 'a';

  // ---------- loading ----------
  async function load() {
    $('back').href = `/book/${bookId}`;
    const [b, v] = await Promise.all([fetch(`/api/book/${bookId}`), fetch(`/api/voice/${bookId}`)]);
    book = await b.json(); voice = await v.json();
    document.title = `${book.title} · Tessa's turn`;
    const pages = book.pages || [];
    items = [
      { key: 'cover', kind: 'cover', photo: pages[0] && pages[0].photo, frames: [] },
      ...pages.map((p) => ({ key: p.id, kind: p.type, photo: p.photo, frames: p.type === 'flipbook' ? p.frames : [p.photo] })),
      { key: 'end', kind: 'end', photo: null, frames: [] },
    ];
    const firstEmpty = items.findIndex((it) => !voice[it.key]);
    show(firstEmpty < 0 ? 0 : firstEmpty, false);
    pollIfListening();
  }

  // ---------- showing a page ----------
  function setPic(src, fade = true) {
    const cur = $(`img-${layer}`), next = $(`img-${layer === 'a' ? 'b' : 'a'}`);
    if (!src) { cur.removeAttribute('src'); return; }
    if (cur.getAttribute('src') === src) return;
    if (!fade || reduce) { cur.src = src; return; }
    next.onload = () => { next.classList.add('on'); cur.classList.remove('on'); layer = layer === 'a' ? 'b' : 'a'; };
    next.src = src;
  }

  function show(i, sound = true) {
    at = Math.max(0, Math.min(items.length - 1, i));
    const it = items[at];
    clearInterval(flipTimer);
    $('page').className = `page ${it.kind}`;
    $('cover-title').textContent = it.kind === 'cover' ? book.title : '';
    setPic(it.photo ? large(it.photo) : null);
    if (it.kind === 'flipbook' && it.frames.length > 1) {
      let k = Math.max(0, it.frames.indexOf(it.photo));
      it.frames.forEach((h) => { const im = new Image(); im.src = large(h); });
      flipTimer = setInterval(() => { k = (k + 1) % it.frames.length; setPic(large(it.frames[k])); }, DWELL_MS);
    }
    $('prev').disabled = at === 0;
    $('next').disabled = at === items.length - 1;
    renderWords(); renderDots();
    if (sound) chime('page');
  }

  function renderWords() {
    const it = items[at], v = voice[it.key], w = $('words');
    w.contentEditable = 'false';
    w.classList.toggle('listening', !!v && v.status === 'listening');
    if (!v) w.textContent = '';
    else if (v.status === 'listening') w.textContent = '❄ ❄ ❄';
    else w.textContent = v.words || '♪';
    $('edit').hidden = !v || v.status === 'listening';
    $('hear').hidden = !v;
  }

  function renderDots() {
    $('dots').innerHTML = items.map((it, i) => {
      const icon = it.kind === 'cover' ? '★' : it.kind === 'end' ? '♥' : '❄';
      return `<li><button data-go="${i}" class="${voice[it.key] ? 'has' : ''} ${i === at ? 'here' : ''}" aria-label="Page ${i}">${icon}</button></li>`;
    }).join('');
  }

  $('prev').onclick = () => show(at - 1);
  $('next').onclick = () => show(at + 1);
  $('dots').addEventListener('click', (e) => { const b = e.target.closest('[data-go]'); if (b) show(+b.dataset.go); });

  // ---------- hearing it back ----------
  const player = $('player');
  function play(key) {
    const v = voice[key];
    if (!v) return Promise.resolve();
    player.src = `/audio/${bookId}/${v.audio.split('/').pop()}`;
    return new Promise((res) => {
      player.onended = res; player.onerror = res;
      player.play().catch(res);
    });
  }
  $('hear').onclick = () => { burst($('hear'), '🎵', 6, 70); play(items[at].key); };

  // ---------- recording: hold space, or tap the snowflake ----------
  let stream = null, analyser = null, rec = null, chunks = [], recStart = 0, raf = null;
  const note = (msg) => { $('note').textContent = msg; $('note').classList.add('show'); setTimeout(() => $('note').classList.remove('show'), 4000); };

  async function ensureMic() {
    if (stream) return true;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } });
      const ctx = new (window.AudioContext || window.webkitAudioContext)();
      analyser = ctx.createAnalyser(); analyser.fftSize = 1024;
      ctx.createMediaStreamSource(stream).connect(analyser);
      return true;
    } catch (e) {
      note('The microphone is blocked. Allow it for localhost in the browser, then try again.');
      return false;
    }
  }

  // Tap once and talk: recording stops by itself once she goes quiet.
  const QUIET_STOP_MS = 2200;    // this much quiet after talking ends the recording
  const NOBODY_TALKED_MS = 8000; // nothing said at all: stop and don't keep it
  const LONGEST_MS = 60000;      // a toddler story page never needs more than a minute
  let heard = false;
  function levelLoop() {
    const buf = new Uint8Array(analyser.fftSize);
    let floor = 0.02, lastLoud = performance.now(), calib = [];
    heard = false;
    const tick = () => {
      analyser.getByteTimeDomainData(buf);
      let sum = 0; for (const x of buf) { const d = (x - 128) / 128; sum += d * d; }
      const rms = Math.sqrt(sum / buf.length);
      const now = performance.now(), since = now - recStart;
      if (since < 350) { calib.push(rms); floor = Math.max(0.01, calib.reduce((a, b) => a + b, 0) / calib.length); }
      const loud = since >= 350 && rms > Math.max(0.035, floor * 2.8);
      if (loud) { heard = true; lastLoud = now; }
      const level = Math.min(1, rms * 6);
      const quiet = heard ? Math.min(1, (now - lastLoud) / QUIET_STOP_MS) : 0;
      document.documentElement.style.setProperty('--level', level.toFixed(3));
      document.documentElement.style.setProperty('--hush', quiet.toFixed(3));
      if ((heard && now - lastLoud > QUIET_STOP_MS) || (!heard && since > NOBODY_TALKED_MS) || since > LONGEST_MS) {
        stopRec();
        return;
      }
      raf = requestAnimationFrame(tick);
    };
    tick();
  }

  async function startRec() {
    if (rec || $('edit').dataset.editing) return;
    if (!(await ensureMic())) return;
    player.pause();
    const type = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4'].find((t) => window.MediaRecorder && MediaRecorder.isTypeSupported(t));
    rec = new MediaRecorder(stream, type ? { mimeType: type } : undefined);
    chunks = [];
    rec.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
    rec.start();
    recStart = performance.now();
    $('mic').classList.remove('idle'); $('mic').classList.add('rec'); $('pola').classList.add('rec');
    chime('start');
    levelLoop();
  }

  function stopRec() {
    if (!rec) return;
    const r = rec, key = items[at].key, long = performance.now() - recStart, talked = heard;
    rec = null;
    cancelAnimationFrame(raf);
    document.documentElement.style.setProperty('--level', 0); document.documentElement.style.setProperty('--hush', 0);
    $('mic').classList.remove('rec'); $('mic').classList.add('idle'); $('pola').classList.remove('rec');
    r.onstop = async () => {
      if (long < 700 || !talked) { $('mic').classList.add('wobble'); setTimeout(() => $('mic').classList.remove('wobble'), 450); return; }
      const blob = new Blob(chunks, { type: r.mimeType || 'audio/webm' });
      burst($('mic'), '❄', 14, 160); chime('done');
      try {
        const res = await fetch(`/api/voice/${bookId}/${key}`, { method: 'POST', headers: { 'Content-Type': blob.type }, body: blob });
        if (!res.ok) throw new Error((await res.json()).error || res.status);
        voice[key] = await res.json();
        if (items[at].key === key) { renderWords(); play(key); }
        renderDots(); pollIfListening();
      } catch (e) {
        note("Couldn't save that one. Is the app still running?");
      }
    };
    r.stop();
  }

  $('mic').onclick = () => (rec ? stopRec() : startRec());
  document.addEventListener('keydown', (e) => {
    if (e.target.isContentEditable || !$('story-layer').hidden) return;
    if (e.code === 'Space') { e.preventDefault(); if (!e.repeat) (rec ? stopRec() : startRec()); }
    if (e.key === 'ArrowRight' && !rec) show(at + 1);
    if (e.key === 'ArrowLeft' && !rec) show(at - 1);
  });

  // ---------- Whisper writes the words; check back while it listens ----------
  let pollTimer = null;
  function pollIfListening() {
    clearTimeout(pollTimer);
    if (!Object.values(voice).some((v) => v.status === 'listening')) return;
    pollTimer = setTimeout(async () => {
      try { voice = await (await fetch(`/api/voice/${bookId}`)).json(); renderWords(); } catch (e) {}
      pollIfListening();
    }, 1500);
  }

  // ---------- grown-ups can fix the words ----------
  $('edit').onclick = () => {
    const w = $('words');
    $('edit').dataset.editing = '1';
    w.contentEditable = 'true'; w.classList.remove('listening');
    if (w.textContent === '♪') w.textContent = '';
    w.focus();
    document.getSelection().selectAllChildren(w);
  };
  async function saveWords() {
    const w = $('words'), key = items[at].key;
    delete $('edit').dataset.editing;
    w.contentEditable = 'false';
    try {
      const res = await fetch(`/api/voice/${bookId}/${key}/words`, { method: 'PUT', body: JSON.stringify({ words: w.textContent }) });
      if (res.ok) voice[key] = await res.json();
    } catch (e) {}
    renderWords();
  }
  $('words').addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); $('words').blur(); } });
  $('words').addEventListener('blur', () => { if ($('edit').dataset.editing) saveWords(); });

  // ---------- story time ----------
  let storyRun = 0, sLayer = 'a';
  function sPic(src) {
    const cur = $(`s-img-${sLayer}`), next = $(`s-img-${sLayer === 'a' ? 'b' : 'a'}`);
    if (!src) { cur.classList.remove('on'); next.classList.remove('on'); return; }
    next.onload = () => { next.classList.add('on'); cur.classList.remove('on'); sLayer = sLayer === 'a' ? 'b' : 'a'; };
    next.src = src;
  }
  function frost() {
    if (reduce) return;
    const f = document.createElement('div'); f.className = 'frost'; document.body.appendChild(f);
    setTimeout(() => f.remove(), 1150);
  }

  async function storyTime() {
    const run = ++storyRun;
    const alive = () => run === storyRun;
    clearInterval(flipTimer);
    $('story-layer').hidden = false;
    for (const it of items) {
      if (!alive()) return;
      frost(); await sleep(reduce ? 0 : 450); if (!alive()) return;
      const v = voice[it.key];
      $('s-title').textContent = it.kind === 'cover' ? book.title : it.kind === 'end' ? 'The End ❄' : '';
      $('s-words').classList.remove('show');
      sPic(it.photo ? large(it.photo) : null);
      await sleep(700); if (!alive()) return;
      if (v && v.words && it.kind !== 'cover') { $('s-words').textContent = v.words; $('s-words').classList.add('show'); }
      let k = 0, t = null;
      if (it.frames.length > 1) t = setInterval(() => { k = (k + 1) % it.frames.length; sPic(large(it.frames[k])); }, DWELL_MS);
      const started = performance.now();
      if (v) await play(it.key);
      const minimum = Math.max(v ? 0 : 3500, it.frames.length > 1 ? it.frames.length * DWELL_MS : 0);
      const left = minimum - (performance.now() - started);
      if (left > 0) await sleep(left);
      clearInterval(t);
      if (!alive()) return;
      await sleep(800);
    }
    if (!alive()) return;
    chime('done'); burst($('s-title'), '❄', 18, 260);
    await sleep(2800);
    if (alive()) closeStory();
  }
  function closeStory() {
    storyRun++; player.pause();
    $('story-layer').hidden = true; $('s-words').classList.remove('show');
    show(at, false);
  }
  $('story').onclick = () => { chime('tell'); storyTime(); };
  $('s-close').onclick = closeStory;
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && !$('story-layer').hidden) closeStory(); });

  TT.snow($('snow'), '--frost', .35);
  $('mic').classList.add('idle');
  load();
})();
