/* Tessa's turn: she talks about her pictures, one at a time (tap the snowflake, talk, it stops
   when she goes quiet). Then story time plays the woven book: a narrator line, then her voice. */
(() => {
  const storyId = location.pathname.split('/').filter(Boolean)[1];
  const $ = (id) => document.getElementById(id);
  const { chime, burst, thumb, large, sleep, reduce } = TT;
  const DWELL_MS = 1800;

  let view, story, talk = {}, own = {}, items = [], at = 0;
  let flipTimer = null, layer = 'a';

  // ---------- loading ----------
  async function load() {
    $('back').href = location.hash === '#play' ? '/' : `/story/${storyId}`;
    const [v, t, o] = await Promise.all([fetch(`/api/story/${storyId}`), fetch('/api/talk'), fetch(`/api/voice/${storyId}`)]);
    view = await v.json(); talk = await t.json(); own = await o.json();
    story = view.story;
    document.title = `${story.title} · Tessa's turn`;
    const nope = new Set(story.nope || []);
    const pageFor = (st) => story.pages.find((p) => st.members.includes(p.photo));
    const pictures = view.sieve.stacks
      .filter((s) => s.pile !== 'snowdrift' && !s.members.every((h) => nope.has(h)))
      .filter((s) => s.pile === 'treasure' || pageFor(s))
      .sort((a, b) => a.start.localeCompare(b.start))
      .map((s) => {
        const pg = pageFor(s), hl = s.highlights || [s.cover];
        const said = [s.cover, ...s.members].find((h) => talk[h]);
        return { box: 'talk', key: said || (pg ? pg.photo : s.cover), kind: hl.length >= 3 && s.flipbook ? 'flipbook' : 'photo',
          photo: pg ? pg.photo : s.cover, frames: s.flipbook ? hl : [pg ? pg.photo : s.cover], stack: s };
      });
    const firstPhoto = story.pages[0] ? story.pages[0].photo : (pictures[0] && pictures[0].photo);
    items = [
      ...pictures,
      { box: 'story', key: 'cover', kind: 'cover', photo: firstPhoto, frames: [] },
      { box: 'story', key: 'end', kind: 'end', photo: null, frames: [] },
    ];
    if (location.hash === '#play') { $('start').hidden = false; show(0, false); return; }
    const firstEmpty = items.findIndex((it) => !entryOf(it));
    show(firstEmpty < 0 ? 0 : firstEmpty, false);
    pollIfListening();
  }
  const entryOf = (it) => (it.box === 'talk' ? talk[it.key] : own[it.key]);
  const audioUrl = (it, e) => (it.box === 'talk' ? `/audio/talk/${e.audio}` : `/audio/story/${storyId}/${e.audio}`);
  const clipUrl = (it) => (it.box === 'talk' ? `/api/talk/${it.key}` : `/api/voice/${storyId}/${it.key}`);

  // ---------- showing a picture ----------
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
    $('cover-title').textContent = it.kind === 'cover' ? story.title : '';
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
    const e = entryOf(items[at]), w = $('words');
    w.contentEditable = 'false';
    w.classList.toggle('listening', !!e && e.status === 'listening');
    if (!e) w.textContent = '';
    else if (e.status === 'listening') w.textContent = '❄ ❄ ❄';
    else w.textContent = e.words || '♪';
    $('edit').hidden = !e || e.status === 'listening';
    $('hear').hidden = !e;
  }

  function renderDots() {
    $('dots').innerHTML = items.map((it, i) => {
      const icon = it.kind === 'cover' ? '★' : it.kind === 'end' ? '♥' : '❄';
      return `<li><button data-go="${i}" class="${entryOf(it) ? 'has' : ''} ${i === at ? 'here' : ''}" aria-label="Picture ${i + 1}">${icon}</button></li>`;
    }).join('');
  }

  $('prev').onclick = () => show(at - 1);
  $('next').onclick = () => show(at + 1);
  $('dots').addEventListener('click', (e) => { const b = e.target.closest('[data-go]'); if (b) show(+b.dataset.go); });

  // ---------- hearing it back ----------
  const player = $('player');
  function playClip(it) {
    const e = entryOf(it);
    if (!e) return Promise.resolve();
    player.src = audioUrl(it, e);
    return new Promise((res) => { player.onended = res; player.onerror = res; player.play().catch(res); });
  }
  $('hear').onclick = () => { burst($('hear'), '🎵', 6, 70); playClip(items[at]); };

  // ---------- recording: tap once and talk; it stops when she goes quiet ----------
  const QUIET_STOP_MS = 2200, NOBODY_TALKED_MS = 8000, LONGEST_MS = 60000;
  let stream = null, analyser = null, rec = null, chunks = [], recStart = 0, raf = null, heard = false;
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

  function levelLoop() {
    const buf = new Uint8Array(analyser.fftSize);
    let floor = 0.02, lastLoud = performance.now();
    const calib = [];
    heard = false;
    const tick = () => {
      analyser.getByteTimeDomainData(buf);
      let sum = 0; for (const x of buf) { const d = (x - 128) / 128; sum += d * d; }
      const rms = Math.sqrt(sum / buf.length);
      const now = performance.now(), since = now - recStart;
      if (since < 350) { calib.push(rms); floor = Math.max(0.01, calib.reduce((a, b) => a + b, 0) / calib.length); }
      if (since >= 350 && rms > Math.max(0.035, floor * 2.8)) { heard = true; lastLoud = now; }
      const quiet = heard ? Math.min(1, (now - lastLoud) / QUIET_STOP_MS) : 0;
      document.documentElement.style.setProperty('--level', Math.min(1, rms * 6).toFixed(3));
      document.documentElement.style.setProperty('--hush', quiet.toFixed(3));
      if ((heard && now - lastLoud > QUIET_STOP_MS) || (!heard && since > NOBODY_TALKED_MS) || since > LONGEST_MS) { stopRec(); return; }
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
    const r = rec, it = items[at], long = performance.now() - recStart, talked = heard;
    rec = null;
    cancelAnimationFrame(raf);
    document.documentElement.style.setProperty('--level', 0); document.documentElement.style.setProperty('--hush', 0);
    $('mic').classList.remove('rec'); $('mic').classList.add('idle'); $('pola').classList.remove('rec');
    r.onstop = async () => {
      if (long < 700 || !talked) { $('mic').classList.add('wobble'); setTimeout(() => $('mic').classList.remove('wobble'), 450); return; }
      const blob = new Blob(chunks, { type: r.mimeType || 'audio/webm' });
      burst($('mic'), '❄', 14, 160); chime('done');
      try {
        const res = await fetch(clipUrl(it), { method: 'POST', headers: { 'Content-Type': blob.type }, body: blob });
        if (!res.ok) throw new Error((await res.json()).error || res.status);
        const entry = await res.json();
        if (it.box === 'talk') talk[it.key] = entry; else own[it.key] = entry;
        if (items[at] === it) { renderWords(); playClip(it); }
        renderDots(); pollIfListening();
      } catch (e) {
        note("Couldn't save that one. Is the app still running?");
      }
    };
    r.stop();
  }

  $('mic').onclick = () => (rec ? stopRec() : startRec());
  document.addEventListener('keydown', (e) => {
    if (e.target.isContentEditable || !$('story-layer').hidden || !$('start').hidden) return;
    if (e.code === 'Space') { e.preventDefault(); if (!e.repeat) (rec ? stopRec() : startRec()); }
    if (e.key === 'ArrowRight' && !rec) show(at + 1);
    if (e.key === 'ArrowLeft' && !rec) show(at - 1);
  });

  // ---------- Whisper writes the words; check back while it listens ----------
  let pollTimer = null;
  function pollIfListening() {
    clearTimeout(pollTimer);
    if (![...Object.values(talk), ...Object.values(own)].some((e) => e.status === 'listening')) return;
    pollTimer = setTimeout(async () => {
      try {
        const [t, o] = await Promise.all([fetch('/api/talk'), fetch(`/api/voice/${storyId}`)]);
        talk = await t.json(); own = await o.json(); renderWords();
      } catch (e) {}
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
    const w = $('words'), it = items[at];
    delete $('edit').dataset.editing;
    w.contentEditable = 'false';
    try {
      const res = await fetch(`${clipUrl(it)}/words`, { method: 'PUT', body: JSON.stringify({ words: w.textContent }) });
      if (res.ok) { const e = await res.json(); if (it.box === 'talk') talk[it.key] = e; else own[it.key] = e; }
    } catch (e) {}
    renderWords();
  }
  $('words').addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); $('words').blur(); } });
  $('words').addEventListener('blur', () => { if ($('edit').dataset.editing) saveWords(); });

  // ---------- story time: the woven book, narrator line then her voice ----------
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
  let narrator = null;
  function pickNarrator() {
    const voices = speechSynthesis.getVoices().filter((v) => v.lang.startsWith('en'));
    narrator = ['Samantha', 'Karen', 'Moira', 'Tessa', 'Google US English'].map((n) => voices.find((v) => v.name.includes(n))).find(Boolean) || voices[0] || null;
  }
  if ('speechSynthesis' in window) { pickNarrator(); speechSynthesis.onvoiceschanged = pickNarrator; }
  function speak(text) {
    if (!('speechSynthesis' in window) || !TT.soundOn) return sleep(Math.min(6000, 1200 + text.length * 60));
    return new Promise((res) => {
      const u = new SpeechSynthesisUtterance(text);
      if (narrator) u.voice = narrator;
      u.rate = 0.88; u.pitch = 1.05;
      u.onend = res; u.onerror = res;
      speechSynthesis.speak(u);
      setTimeout(res, 15000);
    });
  }

  function storyPages() {
    // the woven pages; each plays her talk about that picture (any picture in its stack)
    const stackOf = {};
    for (const s of view.sieve.stacks) for (const h of s.members) stackOf[h] = s;
    const pages = story.pages.map((p) => {
      const s = stackOf[p.photo];
      const said = [p.photo, ...(p.frames || []), ...(s ? s.members : [])].find((h) => talk[h]);
      return { box: 'talk', key: said || p.photo, photo: p.photo, frames: p.type === 'flipbook' ? p.frames : [p.photo], bridge: p.bridge };
    });
    const cover = items.find((it) => it.key === 'cover'), end = items.find((it) => it.key === 'end');
    return [{ ...cover, title: story.title }, ...pages, { ...end, title: 'The End ❄' }];
  }

  async function storyTime() {
    const run = ++storyRun, alive = () => run === storyRun;
    clearInterval(flipTimer);
    $('story-layer').hidden = false;
    for (const it of storyPages()) {
      if (!alive()) return;
      frost(); await sleep(reduce ? 0 : 450); if (!alive()) return;
      const e = entryOf(it);
      $('s-title').textContent = it.title || '';
      $('s-words').classList.remove('show'); $('s-bridge').classList.remove('show');
      sPic(it.photo ? large(it.photo) : null);
      await sleep(700); if (!alive()) return;
      let k = 0, t = null;
      if (it.frames.length > 1) t = setInterval(() => { k = (k + 1) % it.frames.length; sPic(large(it.frames[k])); }, DWELL_MS);
      const started = performance.now();
      if (it.bridge) {
        $('s-bridge').textContent = it.bridge; $('s-bridge').classList.add('show');
        await speak(it.bridge); if (!alive()) { clearInterval(t); return; }
        await sleep(400);
      }
      if (e && e.words && !it.title) { $('s-words').textContent = e.words; $('s-words').classList.add('show'); }
      if (e) await playClip(it);
      const minimum = Math.max(e || it.bridge ? 0 : 3500, it.frames.length > 1 ? it.frames.length * DWELL_MS : 0);
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
    if ('speechSynthesis' in window) speechSynthesis.cancel();
    $('story-layer').hidden = true; $('s-words').classList.remove('show'); $('s-bridge').classList.remove('show');
    if (location.hash === '#play') { location.href = '/'; return; }
    show(at, false);
  }
  $('story').onclick = () => { chime('tell'); storyTime(); };
  $('start').onclick = () => { $('start').hidden = true; chime('tell'); storyTime(); };
  $('s-close').onclick = closeStory;
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && !$('story-layer').hidden) closeStory(); });

  TT.snow($('snow'), '--frost', .35);
  $('mic').classList.add('idle');
  load();
})();
