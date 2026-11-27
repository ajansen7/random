/* One story screen. One picture at a time: Tessa talks about it (tap the snowflake), and ❤️ puts it
   in the book, 👎 sends it to Marshmallow's pile. ⊞ shows everything, ✨ weaves, ▶ plays the book. */
(() => {
  const storyId = location.pathname.split('/').filter(Boolean)[1];
  const $ = (id) => document.getElementById(id);
  const { chime, burst, thumb, large, sleep, reduce, esc } = TT;
  const DWELL_MS = 1800;

  let view, story, talk = {}, own = {}, nope = new Set();
  let stacks = {}, stackOf = {}, rescued = new Set();
  let items = [], at = 0;
  let flipTimer = null, layer = 'a';

  // ---------- data ----------
  async function load() {
    $('back').href = '/';
    const [v, t, o] = await Promise.all([fetch(`/api/story/${storyId}`), fetch('/api/talk'), fetch(`/api/voice/${storyId}`)]);
    take(await v.json()); talk = await t.json(); own = await o.json();
    buildItems();
    if (location.hash === '#play') { $('start').hidden = false; show(0, false); return; }
    const firstEmpty = items.findIndex((it) => !entryOf(it));
    show(firstEmpty < 0 ? 0 : firstEmpty, false);
    pollIfListening();
  }
  function take(v) {
    view = v; story = v.story; nope = new Set(story.nope || []);
    stacks = {}; stackOf = {};
    for (const s of v.sieve.stacks) { stacks[s.id] = s; for (const h of s.members) stackOf[h] = s; }
    document.title = `${story.title} · Tessa Tales`;
  }
  const pageFor = (st) => story.pages.find((p) => st.members.includes(p.photo));
  const isNope = (st) => st.members.every((h) => nope.has(h));
  const flipOK = (st) => st.flipbook && (st.highlights || []).length >= 3;

  function buildItems() {
    const keep = items[at] && items[at].stack ? items[at].stack.id : null;
    const pics = view.sieve.stacks
      .filter((s) => !isNope(s) && (s.pile === 'treasure' || pageFor(s) || rescued.has(s.id)))
      .sort((a, b) => a.start.localeCompare(b.start))
      .map((s) => {
        const pg = pageFor(s);
        return { kind: 'pic', box: 'talk', stack: s, frame: Math.max(0, s.members.indexOf(pg ? pg.photo : s.cover)) };
      });
    items = [...pics, { kind: 'cover', box: 'story', key: 'cover' }, { kind: 'end', box: 'story', key: 'end' }];
    const i = keep ? items.findIndex((it) => it.stack && it.stack.id === keep) : -1;
    if (i >= 0) at = i;
  }
  // which picture of a stack is showing, and which key her talk lives under
  const frameSha = (it) => it.stack.members[it.frame];
  function talkKey(it) {
    if (it.box === 'story') return it.key;
    const st = it.stack;
    return [frameSha(it), st.cover, ...st.members].find((h) => talk[h]) || frameSha(it);
  }
  const entryOf = (it) => (it.box === 'talk' ? talk[talkKey(it)] : own[it.key]);
  const audioUrl = (it, e) => (it.box === 'talk' ? `/audio/talk/${e.audio}` : `/audio/story/${storyId}/${e.audio}`);
  const clipUrl = (it) => (it.box === 'talk' ? `/api/talk/${talkKey(it)}` : `/api/voice/${storyId}/${it.key}`);

  let saveTimer = null;
  function save() {
    clearTimeout(saveTimer);
    saveTimer = setTimeout(async () => {
      story.nope = [...nope];
      try {
        const r = await fetch(`/api/story/${storyId}`, { method: 'PUT', body: JSON.stringify(story) });
        if (!r.ok) throw new Error((await r.json()).error || r.status);
      } catch (e) { note("Couldn't save. Is the app still running?"); }
    }, 300);
  }
  const note = (msg) => { $('note').textContent = msg; $('note').classList.add('show'); setTimeout(() => $('note').classList.remove('show'), 4000); };

  // ---------- showing one picture ----------
  function setPic(src, fade = true) {
    const cur = $(`img-${layer}`), next = $(`img-${layer === 'a' ? 'b' : 'a'}`);
    if (!src) { cur.removeAttribute('src'); return; }
    if (cur.getAttribute('src') === src) return;
    if (!fade || reduce) { cur.src = src; return; }
    next.onload = () => { next.classList.add('on'); cur.classList.remove('on'); layer = layer === 'a' ? 'b' : 'a'; };
    next.src = src;
  }

  function show(i, sound = true) {
    if (sound) $('start').hidden = true;
    at = Math.max(0, Math.min(items.length - 1, i));
    const it = items[at];
    clearInterval(flipTimer);
    $('page').className = `page ${it.kind === 'pic' ? (flipOK(it.stack) ? 'flipbook' : 'photo') : it.kind}`;
    $('pola').classList.remove('going');
    $('cover-title').textContent = it.kind === 'cover' ? story.title : '';
    $('cover-title').contentEditable = 'false';
    if (it.kind === 'pic') {
      const st = it.stack;
      if (flipOK(st)) {
        const hl = st.highlights;
        let k = Math.max(0, hl.indexOf(frameSha(it)));
        setPic(large(hl[k]));
        hl.forEach((h) => { const im = new Image(); im.src = large(h); });
        flipTimer = setInterval(() => { k = (k + 1) % hl.length; setPic(large(hl[k])); }, DWELL_MS);
      } else setPic(large(frameSha(it)));
    } else if (it.kind === 'cover') {
      setPic(story.pages[0] ? large(story.pages[0].photo) : (items[0].stack ? large(frameSha(items[0])) : null));
    } else setPic(null);
    $('prev').disabled = at === 0;
    $('next').disabled = at === items.length - 1;
    renderPicture(); renderWords(); renderDots();
    if (sound) chime('page');
  }

  function renderPicture() {
    const it = items[at], pic = it.kind === 'pic';
    const page = pic && pageFor(it.stack);
    $('pola').classList.toggle('loved', !!page);
    $('heart').classList.toggle('on', !!page);
    $('heart').hidden = $('nope').hidden = !pic;
    const multi = pic && !flipOK(it.stack) && it.stack.members.length > 1;
    $('frames').hidden = !multi;
    if (multi) $('frame-n').textContent = `${it.frame + 1} / ${it.stack.members.length}`;
    $('bridge').hidden = !page;
    if (page && document.activeElement !== $('bridge')) $('bridge').value = page.bridge || '';
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
      const loved = it.kind === 'pic' && pageFor(it.stack);
      const icon = it.kind === 'cover' ? '★' : it.kind === 'end' ? '♥' : loved ? '❤' : '❄';
      return `<li><button data-go="${i}" class="${entryOf(it) ? 'has' : ''} ${loved ? 'loved' : ''} ${i === at ? 'here' : ''}" aria-label="Picture ${i + 1}">${icon}</button></li>`;
    }).join('');
  }

  $('prev').onclick = () => show(at - 1);
  $('next').onclick = () => show(at + 1);
  $('dots').addEventListener('click', (e) => { const b = e.target.closest('[data-go]'); if (b) show(+b.dataset.go); });

  // pictures in a stack (not flip-books): step through and pick which one goes in the book
  function stepFrame(d) {
    const it = items[at];
    it.frame = (it.frame + d + it.stack.members.length) % it.stack.members.length;
    const page = pageFor(it.stack);
    if (page && page.type === 'photo') { page.photo = frameSha(it); page.frames = [page.photo]; save(); }
    setPic(large(frameSha(it))); renderPicture(); renderWords();
  }
  $('frame-prev').onclick = () => stepFrame(-1);
  $('frame-next').onclick = () => stepFrame(1);

  // ---------- ❤️ in the book, 👎 to Marshmallow's pile ----------
  function insertInOrder(page) {
    const t = (p) => (stackOf[p.photo] ? stackOf[p.photo].start : '');
    const k = story.pages.findIndex((p) => t(p) > t(page));
    if (k < 0) story.pages.push(page); else story.pages.splice(k, 0, page);
  }
  $('heart').onclick = () => {
    const it = items[at];
    if (it.kind !== 'pic') return;
    const st = it.stack, page = pageFor(st);
    if (page) {
      story.pages = story.pages.filter((p) => p !== page);
      chime('off');
    } else {
      const flip = flipOK(st), photo = flip ? st.cover : frameSha(it);
      insertInOrder({ id: `p-${st.id}-${Date.now().toString(36)}`, type: flip ? 'flipbook' : 'photo', photo,
        frames: flip ? st.highlights.slice() : [photo], from_stack: st.id, bridge: null });
      chime('heart'); burst($('heart'), '❤️');
    }
    save(); renderPicture(); renderDots();
  };

  let undoStack = null, undoTimer = null;
  $('nope').onclick = async () => {
    const it = items[at];
    if (it.kind !== 'pic') return;
    const st = it.stack;
    story.pages = story.pages.filter((p) => !st.members.includes(p.photo));
    st.members.forEach((h) => nope.add(h));
    rescued.delete(st.id);
    chime('nope');
    $('pola').classList.add('going');
    await sleep(reduce ? 0 : 350);
    const was = at;
    items.splice(at, 1);
    at = Math.min(was, items.length - 1);
    save(); show(at, false);
    undoStack = st;
    $('undo').classList.add('show');
    clearTimeout(undoTimer); undoTimer = setTimeout(() => $('undo').classList.remove('show'), 4000);
  };
  $('undo-btn').onclick = () => {
    $('undo').classList.remove('show');
    if (!undoStack) return;
    const st = undoStack; undoStack = null;
    st.members.forEach((h) => nope.delete(h));
    rescued.add(st.id);
    buildItems(); save(); chime('off');
    show(items.findIndex((x) => x.stack && x.stack.id === st.id), false);
  };

  // ---------- the narrator line (grown-ups) ----------
  $('bridge').addEventListener('input', () => {
    const it = items[at], page = it.kind === 'pic' && pageFor(it.stack);
    if (!page) return;
    page.bridge = $('bridge').value.trim().slice(0, 140) || null;
    save();
  });

  // ---------- the story's name: tap the title on the ★ cover ----------
  $('cover-title').onclick = () => {
    if (items[at].kind !== 'cover' || $('cover-title').isContentEditable) return;
    $('cover-title').contentEditable = 'true'; $('cover-title').focus();
    document.getSelection().selectAllChildren($('cover-title'));
  };
  $('cover-title').addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); $('cover-title').blur(); } });
  $('cover-title').addEventListener('blur', () => {
    $('cover-title').contentEditable = 'false';
    const v = $('cover-title').textContent.trim().slice(0, 60);
    if (v && v !== story.title) { story.title = v; document.title = `${v} · Tessa Tales`; save(); }
    else $('cover-title').textContent = story.title;
  });

  // ---------- hearing it back ----------
  const player = $('player');
  function playClip(it) {
    const e = entryOf(it);
    if (!e) return Promise.resolve();
    player.src = audioUrl(it, e);
    return new Promise((res) => { player.onended = res; player.onerror = res; player.play().catch(res); });
  }
  $('hear').onclick = () => { burst($('hear'), '🎵', 6, 70); playClip(items[at]); };

  // ---------- talking: tap once and talk; it stops when she goes quiet ----------
  const QUIET_STOP_MS = 2200, NOBODY_TALKED_MS = 8000, LONGEST_MS = 60000;
  let stream = null, analyser = null, rec = null, chunks = [], recStart = 0, raf = null, heard = false;

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
    $('start').hidden = true;
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
    const r = rec, it = items[at], url = clipUrl(it), key = it.box === 'talk' ? talkKey(it) : it.key;
    const long = performance.now() - recStart, talked = heard;
    rec = null;
    cancelAnimationFrame(raf);
    document.documentElement.style.setProperty('--level', 0); document.documentElement.style.setProperty('--hush', 0);
    $('mic').classList.remove('rec'); $('mic').classList.add('idle'); $('pola').classList.remove('rec');
    r.onstop = async () => {
      if (long < 700 || !talked) { $('mic').classList.add('wobble'); setTimeout(() => $('mic').classList.remove('wobble'), 450); return; }
      const blob = new Blob(chunks, { type: r.mimeType || 'audio/webm' });
      burst($('mic'), '❄', 14, 160); chime('done');
      try {
        const res = await fetch(url, { method: 'POST', headers: { 'Content-Type': blob.type }, body: blob });
        if (!res.ok) throw new Error((await res.json()).error || res.status);
        const entry = await res.json();
        if (it.box === 'talk') talk[key] = entry; else own[key] = entry;
        if (items[at] === it) { renderWords(); playClip(it); }
        renderDots(); pollIfListening();
      } catch (e) { note("Couldn't save that one. Is the app still running?"); }
    };
    r.stop();
  }
  $('mic').onclick = () => (rec ? stopRec() : startRec());

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') $('start').hidden = true;
    if (e.target.isContentEditable || e.target.matches('input, textarea') || !$('story-layer').hidden || $('grid').open || $('weave').open) return;
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

  // ---------- grown-ups can fix her words ----------
  $('edit').onclick = () => {
    const w = $('words');
    $('edit').dataset.editing = '1';
    w.contentEditable = 'true'; w.classList.remove('listening');
    if (w.textContent === '♪') w.textContent = '';
    w.focus();
    document.getSelection().selectAllChildren(w);
  };
  async function saveWords() {
    const w = $('words'), it = items[at], key = it.box === 'talk' ? talkKey(it) : it.key;
    delete $('edit').dataset.editing;
    w.contentEditable = 'false';
    try {
      const res = await fetch(`${clipUrl(it)}/words`, { method: 'PUT', body: JSON.stringify({ words: w.textContent }) });
      if (res.ok) { const e = await res.json(); if (it.box === 'talk') talk[key] = e; else own[key] = e; }
    } catch (e) {}
    renderWords();
  }
  $('words').addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); $('words').blur(); } });
  $('words').addEventListener('blur', () => { if ($('edit').dataset.editing) saveWords(); });

  // ---------- ⊞ all the pictures, the pile, and the book's page order ----------
  function gCard(s) {
    const page = pageFor(s), said = [s.cover, ...s.members].some((h) => talk[h]);
    const shown = page ? page.photo : s.cover;
    return `<button class="g-card ${page ? 'loved' : ''}" data-jump="${s.id}" aria-label="Go to this picture">
      <img src="${thumb(shown)}" alt="" loading="lazy">
      ${page ? '<span class="b h" aria-hidden="true">❤️</span>' : ''}${said ? '<span class="b t" aria-hidden="true">💬</span>' : ''}
      ${s.gold ? '<span class="b g" aria-hidden="true">❄</span>' : ''}${s.members.length > 1 ? `<span class="n">×${flipOK(s) ? s.highlights.length : s.members.length}</span>` : ''}
    </button>`;
  }
  function renderGrid() {
    const shown = new Set(items.filter((it) => it.stack).map((it) => it.stack.id));
    const html = view.sieve.moments.map((m) => {
      const mine = m.stacks.map((id) => stacks[id]).filter((s) => s && shown.has(s.id));
      if (!mine.length) return '';
      return `<h3 class="g-moment">${esc((story.moment_names || {})[m.id] || m.name)}</h3><div class="g-cards">${mine.map(gCard).join('')}</div>`;
    }).join('');
    const pile = view.sieve.stacks.filter((s) => !shown.has(s.id) && (s.pile !== 'treasure' || isNope(s)));
    const pileN = pile.reduce((n, s) => n + s.members.length, 0);
    $('grid-body').innerHTML = html + (pile.length
      ? `<details class="g-pile"><summary>⛄ Marshmallow's pile (${pileN}) ›</summary><div class="g-cards">${pile.map(gCard).join('')}</div></details>` : '');
    renderStrip();
  }
  function renderStrip() {
    const ol = $('pages');
    if (!story.pages.length) { ol.innerHTML = '<li class="empty" aria-label="Empty: tap a heart">❤️</li>'; return; }
    ol.innerHTML = story.pages.map((p, i) => `<li class="page-t" draggable="true" data-page="${p.id}">
      <span class="num">${i + 1}</span>
      <img src="${thumb(p.photo)}" alt="page ${i + 1}" data-jump-photo="${p.photo}">
      <span class="badge" aria-hidden="true">${p.type === 'flipbook' ? '🎞️' : ''}${p.bridge ? '📜' : ''}</span>
      <button class="x" data-remove="${p.id}" aria-label="Take page ${i + 1} out">✕</button>
    </li>`).join('');
  }
  function jumpTo(stackId) {
    const s = stacks[stackId];
    if (!s) return;
    if (isNope(s)) s.members.forEach((h) => nope.delete(h));
    rescued.add(s.id);
    buildItems();
    $('grid').close();
    show(items.findIndex((it) => it.stack && it.stack.id === s.id));
  }
  $('open-grid').onclick = () => { renderGrid(); $('grid').showModal(); };
  $('grid-close').onclick = () => $('grid').close();
  $('grid').addEventListener('click', (e) => {
    if (e.target === $('grid')) return $('grid').close();
    const j = e.target.closest('[data-jump]');
    if (j) return jumpTo(j.dataset.jump);
    const jp = e.target.closest('[data-jump-photo]');
    if (jp && stackOf[jp.dataset.jumpPhoto]) return jumpTo(stackOf[jp.dataset.jumpPhoto].id);
    const rm = e.target.closest('[data-remove]');
    if (rm) { story.pages = story.pages.filter((p) => p.id !== rm.dataset.remove); chime('off'); save(); renderGrid(); renderPicture(); renderDots(); }
  });
  let dragId = null;
  $('pages').addEventListener('dragstart', (e) => {
    const li = e.target.closest('.page-t'); if (!li) return;
    dragId = li.dataset.page; li.classList.add('dragging');
    e.dataTransfer.effectAllowed = 'move'; e.dataTransfer.setData('text/plain', dragId);
  });
  $('pages').addEventListener('dragend', () => { dragId = null; document.querySelectorAll('.page-t').forEach((l) => l.classList.remove('dragging', 'drop-before')); });
  $('pages').addEventListener('dragover', (e) => {
    if (!dragId) return; e.preventDefault();
    document.querySelectorAll('.page-t').forEach((l) => l.classList.remove('drop-before'));
    const li = e.target.closest('.page-t'); if (li && li.dataset.page !== dragId) li.classList.add('drop-before');
  });
  $('pages').addEventListener('drop', (e) => {
    if (!dragId) return; e.preventDefault();
    const li = e.target.closest('.page-t');
    const [moved] = story.pages.splice(story.pages.findIndex((p) => p.id === dragId), 1);
    let to = li ? story.pages.findIndex((p) => p.id === li.dataset.page) : story.pages.length;
    if (to < 0) to = story.pages.length;
    story.pages.splice(to, 0, moved);
    dragId = null; save(); renderStrip();
  });

  // ---------- ✨ weave ----------
  $('open-weave').onclick = () => {
    $('w-hint').value = story.hint || '';
    $('w-note').textContent = story.weave_note || '';
    $('w-error').textContent = '';
    $('w-busy').hidden = true; $('w-go').disabled = false;
    $('w-summary').hidden = true;
    $('w-count').textContent = items.filter((it) => it.kind === 'pic' && entryOf(it)).length;
    $('weave').showModal();
  };
  $('w-close').onclick = () => $('weave').close();

  // what the weave changed, so it's never a mystery
  function renderWeaveSummary() {
    const before = (story.previous_pages || []).map((p) => p.photo), after = story.pages.map((p) => p.photo);
    const stackKey = (h) => (stackOf[h] ? stackOf[h].id : h);
    const b = new Set(before.map(stackKey)), a = new Set(after.map(stackKey));
    const added = after.filter((h) => !b.has(stackKey(h))), dropped = before.filter((h) => !a.has(stackKey(h)));
    const kept = after.filter((h) => b.has(stackKey(h))).map(stackKey);
    const keptBefore = before.map(stackKey).filter((k) => a.has(k));
    const reordered = kept.some((k, i) => k !== keptBefore[i]);
    const bridges = story.pages.filter((p) => p.bridge).length;
    const thumbs = (hs) => hs.slice(0, 12).map((h) => `<img src="${thumb(h)}" alt="">`).join('') + (hs.length > 12 ? ` +${hs.length - 12}` : '');
    const rows = [`<p class="w-row"><b>📖 ${before.length} → ${after.length} pages</b>${reordered ? ' · reordered' : ' · same order'}</p>`];
    if (added.length) rows.push(`<p class="w-row">➕ added ${added.length}<span class="w-thumbs">${thumbs(added)}</span></p>`);
    if (dropped.length) rows.push(`<p class="w-row">➖ left out ${dropped.length}<span class="w-thumbs">${thumbs(dropped)}</span></p>`);
    rows.push(`<p class="w-row">📜 ${bridges} narrator line${bridges === 1 ? '' : 's'}${bridges ? `: <i>${esc(story.pages.find((p) => p.bridge).bridge)}</i>${bridges > 1 ? ' …' : ''}` : ''}</p>`);
    if (story.suggested_title && story.suggested_title.toLowerCase() !== story.title.toLowerCase()) {
      rows.push(`<p class="w-row">🏷️ title idea: <button class="w-title" id="w-use-title">${esc(story.suggested_title)}</button></p>`);
    }
    rows.push('<button class="chunky w-watch" id="w-watch">▶ Watch it</button>');
    $('w-summary').innerHTML = rows.join('');
    $('w-summary').hidden = false;
    const use = $('w-use-title');
    if (use) use.onclick = () => { story.title = story.suggested_title; save(); show(at, false); use.closest('.w-row').remove(); chime('heart'); };
    $('w-watch').onclick = () => { $('weave').close(); $('start').hidden = true; chime('tell'); storyTime(); };
  }
  $('w-go').onclick = async () => {
    $('w-go').disabled = true; $('w-busy').hidden = false; $('w-error').textContent = '';
    try {
      const r = await fetch(`/api/story/${storyId}/weave`, { method: 'POST', body: JSON.stringify({ hint: $('w-hint').value }) });
      const data = await r.json();
      if (!r.ok) throw new Error(data.error || r.status);
      take(data); buildItems(); show(at, false);
      $('w-note').textContent = story.weave_note || '';
      renderWeaveSummary();
      chime('star'); burst($('w-go'), '✨', 14);
    } catch (e) { $('w-error').textContent = String(e.message || e); }
    $('w-go').disabled = false; $('w-busy').hidden = true;
  };

  // ---------- ▶ story time: an open storybook ----------
  // The cover (title + her title recording) opens straight into page 1. Each page: her picture on
  // the left, her words printed on the right. Pages turn by themselves (⏸ to linger), or by hand.
  const sb = { pages: [], i: -1, token: 0, auto: true, busy: false, flip: null, open: false };
  const isSpread = () => innerWidth >= 900 && innerWidth / innerHeight > 1.15;

  function bookPages() {
    const pages = story.pages.map((p) => {
      const s = stackOf[p.photo];
      const key = [p.photo, ...(p.frames || []), ...(s ? s.members : [])].find((h) => talk[h]);
      const entry = key ? talk[key] : null;
      return { kind: 'page', box: 'talk', entry, words: entry && entry.words, photo: p.photo,
        frames: p.type === 'flipbook' ? p.frames : [p.photo], bridge: p.bridge };
    });
    pages.push({ kind: 'end', box: 'story', entry: own.end || null, words: own.end && own.end.words, frames: [] });
    return pages;
  }
  const tilt = (n) => `${((n * 37) % 7) - 3}deg`;
  function leftHTML(p, n) {
    if (p.kind === 'end') return '<div class="inner"><div class="b-snowman">☃️</div></div>';
    return `<div class="inner"><div class="b-photo" style="--tilt:${tilt(n)}"><img src="${large(p.photo)}" alt="">
      <span class="k tl">❄</span><span class="k tr">❄</span><span class="k bl">❄</span><span class="k br">❄</span></div></div>`;
  }
  function wordsHTML(text) {
    const t = text.trim();
    return `<span class="cap">${esc(t[0])}</span>${esc(t.slice(1))}`;
  }
  function rightHTML(p, n) {
    if (p.kind === 'end') {
      return `<div class="inner"><div class="b-end"><h2>The End</h2>${p.words ? `<p class="b-words">${esc(p.words)}</p>` : ''}</div></div>`;
    }
    const body = p.words ? `<p class="b-words">${wordsHTML(p.words)}</p>` : `<div class="b-orn">${p.entry ? '♪' : '❄'}</div>`;
    return `<div class="inner"><div class="b-text">${p.bridge ? `<p class="b-bridge">${esc(p.bridge)}</p>` : ''}${body}</div></div>
      <div class="b-flourish">❄ ❄ ❄</div><div class="b-num">${n + 1}</div>`;
  }
  const singleHTML = (p, n) => `<div class="inner">${leftHTML(p, n).replace(/^<div class="inner">|<\/div>$/g, '')}</div>${rightHTML(p, n)}`;

  function layout() { $('book').classList.toggle('single', !isSpread()); }
  function paint(i) {
    layout();  // the window may have changed shape, even mid-turn
    const p = sb.pages[i];
    if (isSpread()) { $('b-left').innerHTML = leftHTML(p, i); $('b-right').innerHTML = rightHTML(p, i); }
    else $('b-single').innerHTML = singleHTML(p, i);
  }

  // a page turns over the spine: front = what's there now, back = what's coming
  function turn(dir, fromI, toI) {
    const ms = reduce ? 0 : 900 + Math.random() * 450;
    const lift = (Math.random() * 6 - 3).toFixed(1);
    if (!ms) { paint(toI); return Promise.resolve(); }
    TT.swish(ms * 0.8);
    whimsy();
    const spread = isSpread();
    const book = $('book'), rect = book.getBoundingClientRect();
    const leaf = document.createElement('div');
    leaf.className = 'b-leaf';
    const half = spread ? rect.width / 2 : rect.width;
    leaf.style.width = `${half}px`;
    const from = sb.pages[fromI], to = sb.pages[toI];
    let front, back, start, end, origin;
    if (spread && dir > 0) {        // right page lifts and lands on the left
      leaf.style.left = `${half}px`; origin = 'left center';
      front = `<div class="face front b-page b-right">${rightHTML(from, fromI)}</div>`;
      back = `<div class="face back b-page b-left">${leftHTML(to, toI)}</div>`;
      $('b-right').innerHTML = rightHTML(to, toI);
      start = 0; end = -180;
    } else if (spread) {            // left page lifts and lands on the right
      leaf.style.left = '0px'; origin = 'right center';
      front = `<div class="face front b-page b-left">${leftHTML(from, fromI)}</div>`;
      back = `<div class="face back b-page b-right">${rightHTML(to, toI)}</div>`;
      $('b-left').innerHTML = leftHTML(to, toI);
      start = 0; end = 180;
    } else if (dir > 0) {           // one page: it flips away to reveal the next
      leaf.style.left = '0px'; origin = 'left center';
      front = `<div class="face front b-page b-single" style="display:grid;grid-template-rows:1.25fr 1fr">${singleHTML(from, fromI)}</div>`;
      back = '<div class="face back b-page b-single" style="display:block"></div>';
      $('b-single').innerHTML = singleHTML(to, toI);
      start = 0; end = -180;
    } else {                        // one page: the previous one flips back over
      leaf.style.left = '0px'; origin = 'left center';
      front = `<div class="face front b-page b-single" style="display:grid;grid-template-rows:1.25fr 1fr">${singleHTML(to, toI)}</div>`;
      back = '<div class="face back b-page b-single" style="display:block"></div>';
      start = -180; end = 0;
    }
    leaf.style.transformOrigin = origin;
    leaf.innerHTML = front + back + '<div class="shade"></div>';
    book.appendChild(leaf);
    const anim = leaf.animate([
      { transform: `rotateY(${start}deg) rotateX(0deg)` },
      { transform: `rotateY(${(start + end) / 2}deg) rotateX(${lift}deg) translateZ(40px)`, offset: 0.5 },
      { transform: `rotateY(${end}deg) rotateX(0deg)` },
    ], { duration: ms, easing: 'cubic-bezier(.45, .05, .35, 1)', fill: 'forwards' });
    return anim.finished.then(() => { paint(toI); leaf.remove(); });
  }

  // a little surprise on every turn
  function fly(icons, n, from, to, dur = 1500) {
    for (let k = 0; k < n; k++) {
      const s = document.createElement('span'); s.className = 'whim';
      s.textContent = icons[k % icons.length];
      s.style.fontSize = `${1.4 + Math.random() * 1.8}rem`;
      document.body.appendChild(s);
      const a = from(k), b = to(k);
      s.animate([{ transform: `translate(${a[0]}px, ${a[1]}px) rotate(0deg)`, opacity: 1 },
        { transform: `translate(${b[0]}px, ${b[1]}px) rotate(${(Math.random() - .5) * 720}deg)`, opacity: 0 }],
      { duration: dur * (0.7 + Math.random() * 0.6), delay: k * 40, easing: 'cubic-bezier(.3, .2, .4, 1)', fill: 'both' }).onfinish = () => s.remove();
    }
  }
  const W = () => innerWidth, H = () => innerHeight, rnd = (a, b) => a + Math.random() * (b - a);
  const WHIMSY = [
    () => fly(['❄', '✨', '❄'], 18, () => [rnd(0, W()), -40], () => [rnd(0, W()), H() + 40], 2200),                       // a flurry
    () => fly(['🥕', '❄', '⛄'], 10, () => [rnd(0, W()), -50], () => [rnd(0, W()), H() + 60], 2000),                      // Olaf's nose tumbles by
    () => fly(['💙', '🤍', '💜'], 12, () => [rnd(W() * .2, W() * .8), H() + 30], () => [rnd(0, W()), rnd(-60, H() * .3)], 2200), // hearts float up
    () => fly(['❄', '·', '❄', '💨'], 22, () => [-60, rnd(0, H())], () => [W() + 80, rnd(0, H())], 1300),                  // a wind gust
    () => fly(['⭐', '✨', '🌟'], 14, () => [W() / 2, H() / 2], () => [rnd(-50, W() + 50), rnd(-50, H() + 50)], 1300),    // a star pop
    () => fly(['✨'], 16, (k) => [k * W() / 16, H() * .5 + Math.sin(k) * 80], (k) => [k * W() / 16 + 60, H() * .5 + Math.sin(k + 2) * 120 - 140], 1100), // a sparkle wave
  ];
  let lastWhim = -1;
  function whimsy() {
    if (reduce) return;
    let k; do { k = Math.floor(Math.random() * WHIMSY.length); } while (k === lastWhim && WHIMSY.length > 1);
    lastWhim = k; WHIMSY[k]();
  }

  async function speak(text) {
    if (!('speechSynthesis' in window) || !TT.soundOn) return sleep(Math.min(6000, 1200 + text.length * 60));
    TT.music.duck(true);
    await TT.narrator.say(text);  // the voice chosen on the shelf (🗣️)
    TT.music.duck(false);
  }
  function playEntry(p) {
    if (!p.entry) return Promise.resolve();
    TT.music.duck(true);
    player.src = p.box === 'talk' ? `/audio/talk/${p.entry.audio}` : `/audio/story/${storyId}/${p.entry.audio}`;
    return new Promise((res) => {
      const done = () => { TT.music.duck(false); res(); };
      player.onended = done; player.onerror = done; player.play().catch(done);
    });
  }
  function hush() {
    player.pause(); player.onended = null;
    if ('speechSynthesis' in window) speechSynthesis.cancel();
    TT.music.duck(false);
    clearInterval(sb.flip);
  }
  function startFlip(p) {
    clearInterval(sb.flip);
    if (p.frames.length < 2) return;
    let k = 0;
    sb.flip = setInterval(() => {
      k = (k + 1) % p.frames.length;
      const img = document.querySelector('#book .b-photo img');
      if (img) img.src = large(p.frames[k]);
    }, DWELL_MS);
  }

  async function playPage(i, token) {
    const p = sb.pages[i], alive = () => token === sb.token;
    startFlip(p);
    const started = performance.now();
    if (p.bridge) { await speak(p.bridge); if (!alive()) return; await sleep(300); }
    const words = document.querySelector('#book .b-words');
    if (words) words.classList.add('reading');
    await playEntry(p); if (!alive()) return;
    if (words) words.classList.remove('reading');
    const minimum = Math.max(p.entry || p.bridge ? 0 : 3500, p.frames.length > 1 ? p.frames.length * DWELL_MS : 0);
    const left = minimum - (performance.now() - started);
    if (left > 0) await sleep(left);
    await sleep(900);
    if (!alive()) return;
    if (p.kind === 'end') { chime('done'); burst($('book'), '❄', 24, 320); return; }
    if (sb.auto) go(i + 1);
  }

  async function go(to) {
    if (to < 0 || to >= sb.pages.length || to === sb.i || sb.busy) return;
    const token = ++sb.token;
    hush();
    sb.busy = true;
    if (sb.i < 0) { paint(to); } else { await turn(to > sb.i ? 1 : -1, sb.i, to); }
    sb.busy = false;
    sb.i = to;
    $('s-prev').disabled = to === 0;
    $('s-next').disabled = to === sb.pages.length - 1;
    if (token === sb.token) playPage(to, token);
  }

  async function storyTime() {
    const token = ++sb.token;
    clearInterval(flipTimer);
    sb.pages = bookPages(); sb.i = -1; sb.auto = true; sb.busy = false;
    $('s-pause').classList.remove('paused'); $('s-pause').textContent = '⏸';
    $('story-layer').hidden = false;
    if (!sbSnow) { TT.snow($('s-snow'), '--frost', .8); sbSnow = true; }
    layout();
    $('book').style.visibility = 'hidden';
    $('b-cover').hidden = false; $('b-cover').classList.remove('open');
    let n = 0;  // letters freeze in one by one, but words stay whole
    $('c-title').innerHTML = story.title.split(' ').map((word) =>
      `<span class="word">${[...word].map((ch) => `<span class="ch" style="animation-delay:${n++ * 70}ms">${esc(ch)}</span>`).join('')}</span>`).join(' ');
    TT.music.start();
    await sleep(reduce ? 300 : 1300 + story.title.length * 70); if (token !== sb.token) return;
    if (own.cover) { await playEntry({ box: 'story', entry: own.cover }); if (token !== sb.token) return; await sleep(400); }
    paint(0); $('book').style.visibility = 'visible';
    TT.swish(1000);
    $('b-cover').classList.add('open');           // the cover swings open, straight into page 1
    await sleep(reduce ? 0 : 1300); if (token !== sb.token) return;
    $('b-cover').hidden = true;
    go(0);
  }
  let sbSnow = false;

  function closeStory() {
    sb.token++; hush(); TT.music.stop();
    document.querySelectorAll('.b-leaf, .whim').forEach((el) => el.remove());
    $('story-layer').hidden = true;
    if (location.hash === '#play') { location.href = '/'; return; }
    show(at, false);
  }
  function nav(d) {
    if ($('story-layer').hidden || !$('b-cover').hidden) return;
    go(sb.i + d);
  }
  $('s-next').onclick = () => nav(1);
  $('s-prev').onclick = () => nav(-1);
  $('s-pause').onclick = () => {
    sb.auto = !sb.auto;
    $('s-pause').classList.toggle('paused', !sb.auto);
    $('s-pause').textContent = sb.auto ? '⏸' : '▶';
    $('s-pause').setAttribute('aria-label', sb.auto ? 'Pause turning pages' : 'Keep turning pages');
    if (sb.auto && sb.i >= 0 && !sb.busy) playPage(sb.i, ++sb.token);  // pick up where we are
  };
  // tap the left page to go back, the right page to go forward
  $('book').addEventListener('click', (e) => {
    const r = $('book').getBoundingClientRect();
    nav(e.clientX < r.left + r.width / 2 ? -1 : 1);
  });
  addEventListener('resize', () => { if (!$('story-layer').hidden && sb.i >= 0 && !sb.busy) { layout(); paint(sb.i); } });
  $('story').onclick = () => { $('start').hidden = true; chime('tell'); storyTime(); };
  $('start-btn').onclick = () => { $('start').hidden = true; chime('tell'); storyTime(); };
  $('s-close').onclick = closeStory;
  document.addEventListener('keydown', (e) => {
    if ($('story-layer').hidden) return;
    if (e.key === 'Escape') closeStory();
    if (e.key === 'ArrowRight') { e.preventDefault(); nav(1); }
    if (e.key === 'ArrowLeft') { e.preventDefault(); nav(-1); }
    if (e.code === 'Space') { e.preventDefault(); $('s-pause').click(); }
  });

  TT.snow($('snow'), '--frost', .35);
  $('mic').classList.add('idle');
  load();
})();
