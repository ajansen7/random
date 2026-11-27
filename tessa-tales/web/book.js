/* Tessa Tales: make-the-book page. Three stickers do everything:
   ❤️ love it → in the book   ⭐ the best! → in the book as a big page   👎 nope → Marshmallow's pile */
(() => {
  const importId = location.pathname.split('/').filter(Boolean)[1];
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const thumb = (h) => `/thumb/${h}.jpg`;
  const large = (h) => `/large/${h}.jpg`;
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const DWELL_MS = 1800;   // a flip-book lingers this long on each picture
  const HOVER_MS = 1100;   // and a little quicker when you hover a stack
  const REACTIONS = [
    { r: 'heart', icon: '❤️', label: 'Love it: put it in the book' },
    { r: 'star', icon: '⭐', label: 'The best: a big page in the book' },
    { r: 'nope', icon: '👎', label: "Nope: send it to Marshmallow's pile" },
  ];

  let imp, book, stacks = {}, stackOfPhoto = {}, nope = new Set();

  // ---------- loading & saving ----------
  async function load() {
    const [a, b] = await Promise.all([fetch(`/api/import/${importId}`), fetch(`/api/book/${importId}`)]);
    if (!a.ok || !b.ok) { $('moments').innerHTML = '<p class="m-name">⛄❓</p>'; return; }
    imp = await a.json(); book = await b.json();
    for (const st of imp.sieve.stacks) { stacks[st.id] = st; for (const h of st.members) stackOfPhoto[h] = st; }
    book.nope = book.nope || [];
    nope = new Set(book.nope);
    // flip-book pages show only the different-looking pictures
    for (const p of book.pages) {
      const st = stackOfPhoto[p.photo];
      if (p.type === 'flipbook' && st && st.highlights && p.frames.length > st.highlights.length) p.frames = st.highlights.slice();
    }
    renderAll();
  }

  let saveTimer = null;
  function save() {
    $('saved').textContent = '…';
    book.nope = [...nope];
    clearTimeout(saveTimer);
    saveTimer = setTimeout(async () => {
      try {
        const r = await fetch(`/api/book/${importId}`, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(book) });
        if (!r.ok) throw new Error(r.status);
        $('saved').textContent = '✓';
        setTimeout(() => { if ($('saved').textContent === '✓') $('saved').textContent = ''; }, 1500);
      } catch (e) {
        $('saved').textContent = '⚠️';
        $('saved').title = "Couldn't save. Is the app still running?";
      }
    }, 400);
  }

  // ---------- sounds (made on the fly, no files) ----------
  let audio = null;
  let soundOn = true;
  try { soundOn = localStorage.getItem('tt-sound') !== 'off'; } catch (e) {}
  function chime(kind) {
    if (!soundOn) return;
    try {
      audio = audio || new (window.AudioContext || window.webkitAudioContext)();
      const notes = { heart: [660, 880], star: [784, 988, 1175, 1568], nope: [262, 196], off: [523], tell: [523, 659, 784, 1047] }[kind];
      const wave = kind === 'nope' ? 'triangle' : 'sine';
      notes.forEach((f, i) => {
        const o = audio.createOscillator(), g = audio.createGain(), t = audio.currentTime + i * 0.09;
        o.type = wave; o.frequency.value = f;
        g.gain.setValueAtTime(0.0001, t); g.gain.exponentialRampToValueAtTime(0.18, t + 0.02); g.gain.exponentialRampToValueAtTime(0.0001, t + 0.35);
        o.connect(g).connect(audio.destination); o.start(t); o.stop(t + 0.4);
      });
    } catch (e) {}
  }
  function syncSound() { $('sound').textContent = soundOn ? '🔊' : '🔇'; }
  $('sound').onclick = () => { soundOn = !soundOn; try { localStorage.setItem('tt-sound', soundOn ? 'on' : 'off'); } catch (e) {} syncSound(); if (soundOn) chime('heart'); };
  syncSound();

  function burst(el, icon, n = 9) {
    if (reduce || !el) return;
    const r = el.getBoundingClientRect();
    for (let i = 0; i < n; i++) {
      const s = document.createElement('span');
      s.className = 'burst'; s.textContent = icon;
      const a = (Math.PI * 2 * i) / n + Math.random() * .5;
      s.style.left = `${r.left + r.width / 2 - 14}px`; s.style.top = `${r.top + r.height / 2 - 14}px`;
      s.style.setProperty('--dx', `${Math.cos(a) * (60 + Math.random() * 50)}px`);
      s.style.setProperty('--dy', `${Math.sin(a) * (60 + Math.random() * 50) - 40}px`);
      s.style.setProperty('--rot', `${(Math.random() - .5) * 90}deg`);
      document.body.appendChild(s);
      setTimeout(() => s.remove(), 950);
    }
  }

  // ---------- the book model ----------
  const pageFor = (st) => book.pages.find((p) => st.members.includes(p.photo));
  const isNope = (st) => st.members.every((h) => nope.has(h));
  function reactionOf(st) {
    const p = pageFor(st);
    if (p) return p.star ? 'star' : 'heart';
    if (isNope(st)) return 'nope';
    return null;
  }
  function newPage(st, photo, flip) {
    const frames = st.highlights && st.highlights.length ? st.highlights : st.members;
    return {
      id: `p-${st.id}-${Date.now().toString(36)}`,
      type: flip ? 'flipbook' : 'photo',
      photo: photo || st.cover,
      frames: flip ? frames.slice() : [photo || st.cover],
      from_stack: st.id,
      star: false,
    };
  }
  function insertInOrder(page) {
    const t = (p) => (stackOfPhoto[p.photo] ? stackOfPhoto[p.photo].start : '');
    const at = book.pages.findIndex((p) => t(p) > t(page));
    if (at < 0) book.pages.push(page); else book.pages.splice(at, 0, page);
  }

  // r: 'heart' | 'star' | 'nope'. photo: which picture of the stack (from the viewer)
  function react(st, r, btn, photo) {
    const cur = reactionOf(st);
    const page = pageFor(st);
    const samePhoto = !photo || !page || page.photo === photo || page.type === 'flipbook';
    if (r === cur && samePhoto) {                       // tap the same sticker again: undo it
      if (page) book.pages = book.pages.filter((p) => p !== page);
      if (r === 'nope') st.members.forEach((h) => nope.delete(h));
      chime('off');
      return changed();
    }
    if (r === 'nope') {
      if (page) book.pages = book.pages.filter((p) => p !== page);
      st.members.forEach((h) => nope.add(h));
      chime('nope');
      const card = btn && btn.closest('.card');
      if (card && !card.closest('#pile')) {
        card.classList.add('going');
        setTimeout(changed, reduce ? 0 : 350);
      } else changed();
      showUndo(st);
      return;
    }
    st.members.forEach((h) => nope.delete(h));
    if (page) {
      page.star = r === 'star';
      if (photo && page.type === 'photo') { page.photo = photo; page.frames = [photo]; }
    } else {
      const p = newPage(st, photo, st.flipbook && !photo);
      p.star = r === 'star';
      insertInOrder(p);
    }
    chime(r);
    burst(btn, r === 'star' ? '⭐' : '❤️');
    changed();
  }

  let undoTimer = null, undoStack = null;
  function showUndo(st) {
    undoStack = st;
    $('undo').classList.add('show');
    clearTimeout(undoTimer);
    undoTimer = setTimeout(() => $('undo').classList.remove('show'), 4000);
  }
  $('undo-btn').onclick = () => {
    if (undoStack) { undoStack.members.forEach((h) => nope.delete(h)); undoStack = null; chime('off'); changed(); }
    $('undo').classList.remove('show');
  };

  // ---------- rendering ----------
  function momentIcon(iso) {
    const h = new Date(iso).getHours();
    return h < 5 || h >= 20 ? '🌙' : h < 11 ? '☀️' : h < 17 ? '🌤️' : '🌅';
  }
  function reactHTML(st, big) {
    const cur = reactionOf(st);
    return `<div class="react${big ? ' big' : ''}">${REACTIONS.map((x) =>
      `<button data-react="${x.r}" data-r="${x.r}" data-stack="${st.id}" class="${cur === x.r ? 'on' : ''}" title="${x.label}" aria-label="${x.label}" aria-pressed="${cur === x.r}">${x.icon}</button>`).join('')}</div>`;
  }
  function cardHTML(st) {
    const cur = reactionOf(st);
    const shown = pageFor(st) ? pageFor(st).photo : st.cover;
    const behind = st.members.filter((h) => h !== shown).slice(0, 2)
      .map((h, i) => `<img class="behind b${i}" src="${thumb(h)}" alt="" loading="lazy">`).join('');
    const count = st.members.length > 1 ? `<span class="count">${st.flipbook ? '🎞️ ' : '×'}${st.flipbook ? st.highlights.length : st.members.length}</span>` : '';
    return `<div class="card ${cur || ''}" data-card="${st.id}">
      <button class="pola" data-open="${st.id}" aria-label="Look closer">${behind}<img class="cover" src="${thumb(shown)}" alt="" loading="lazy">${st.gold ? '<span class="flake" title="Elsa\'s snowflake: she meant this one" aria-hidden="true">❄</span>' : ''}${count}<span class="sticker" aria-hidden="true">${cur === 'star' ? '⭐' : '❤️'}</span></button>
      ${reactHTML(st)}
    </div>`;
  }

  function renderMoments() {
    $('moments').innerHTML = imp.sieve.moments.map((m) => {
      const mine = m.stacks.map((id) => stacks[id]).filter((s) => !isNope(s));
      if (!mine.length) return '';
      const name = (book.moment_names || {})[m.id] || m.name;
      return `<section class="moment" data-moment="${m.id}">
        <div class="m-head"><span class="m-icon" aria-hidden="true">${momentIcon(m.start)}</span><span class="m-name editable" contenteditable="true" spellcheck="false" data-moment-name="${m.id}" title="Tap to rename">${esc(name)}</span></div>
        <div class="cards">${mine.map(cardHTML).join('')}</div>
      </section>`;
    }).join('');
    wireFlips($('moments'));
  }

  function pileStacks() {
    return imp.sieve.stacks.filter((s) => (s.pile === 'snowdrift' && !pageFor(s)) || isNope(s));
  }
  function renderPileCount() {
    $('pile-n').textContent = pileStacks().reduce((n, s) => n + s.members.length, 0);
  }

  function renderStrip() {
    const ol = $('pages');
    if (!book.pages.length) { ol.innerHTML = '<li class="empty" aria-label="Empty: tap a heart">❤️ ⭐</li>'; return; }
    ol.innerHTML = book.pages.map((p, i) => `<li class="page${p.star ? ' star' : ''}" draggable="true" data-page="${p.id}">
      <span class="num">${i + 1}</span>
      <img src="${thumb(p.photo)}" alt="page ${i + 1}" data-open-page="${p.id}">
      <span class="badge" aria-hidden="true">${p.star ? '⭐' : ''}${p.type === 'flipbook' ? '🎞️' : ''}</span>
      <button class="x" data-remove="${p.id}" aria-label="Take page ${i + 1} out">✕</button>
    </li>`).join('');
  }

  function renderAll() {
    $('title').textContent = book.title;
    document.title = `${book.title} · Tessa Tales`;
    renderMoments(); renderStrip(); renderPileCount();
  }
  function changed() {
    renderMoments(); renderStrip(); renderPileCount();
    if ($('pile').open) renderPile();
    if ($('viewer').open && vSt) showFrame(false);
    save();
  }

  // ---------- clicks ----------
  document.addEventListener('click', (e) => {
    const rb = e.target.closest('[data-react]');
    if (rb) {
      const st = stacks[rb.dataset.stack];
      const fromViewer = rb.closest('#viewer');
      return react(st, rb.dataset.react, rb, fromViewer ? st.members[vIdx] : undefined);
    }
    const open = e.target.closest('[data-open]');
    if (open) return openViewer(stacks[open.dataset.open]);
    const rm = e.target.closest('[data-remove]');
    if (rm) { book.pages = book.pages.filter((p) => p.id !== rm.dataset.remove); chime('off'); return changed(); }
    const op = e.target.closest('[data-open-page]');
    if (op) {
      const p = book.pages.find((x) => x.id === op.dataset.openPage);
      const st = p && stackOfPhoto[p.photo];
      if (st) openViewer(st, st.members.indexOf(p.photo));
    }
  });

  // ---------- renaming (grown-up stuff) ----------
  function finishEdit(el, fallback, apply) {
    const v = el.textContent.trim().slice(0, 80);
    if (!v) { el.textContent = fallback; return; }
    if (v !== fallback) { apply(v); save(); }
  }
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && e.target.isContentEditable) { e.preventDefault(); e.target.blur(); }
  });
  $('title').addEventListener('blur', () => finishEdit($('title'), book.title, (v) => { book.title = v; document.title = `${v} · Tessa Tales`; }));
  document.addEventListener('focusout', (e) => {
    const el = e.target.closest && e.target.closest('[data-moment-name]');
    if (!el) return;
    const id = el.dataset.momentName, m = imp.sieve.moments.find((x) => x.id === id);
    book.moment_names = book.moment_names || {};
    finishEdit(el, book.moment_names[id] || m.name, (v) => { book.moment_names[id] = v; });
  });

  // ---------- drag to reorder the strip ----------
  let dragId = null;
  $('pages').addEventListener('dragstart', (e) => {
    const li = e.target.closest('.page'); if (!li) return;
    dragId = li.dataset.page; li.classList.add('dragging');
    e.dataTransfer.effectAllowed = 'move'; e.dataTransfer.setData('text/plain', dragId);
  });
  $('pages').addEventListener('dragend', () => { dragId = null; document.querySelectorAll('.page').forEach((l) => l.classList.remove('dragging', 'drop-before')); });
  $('pages').addEventListener('dragover', (e) => {
    if (!dragId) return; e.preventDefault();
    document.querySelectorAll('.page').forEach((l) => l.classList.remove('drop-before'));
    const li = e.target.closest('.page'); if (li && li.dataset.page !== dragId) li.classList.add('drop-before');
  });
  $('pages').addEventListener('drop', (e) => {
    if (!dragId) return; e.preventDefault();
    const li = e.target.closest('.page');
    const [moved] = book.pages.splice(book.pages.findIndex((p) => p.id === dragId), 1);
    let to = li ? book.pages.findIndex((p) => p.id === li.dataset.page) : book.pages.length;
    if (to < 0) to = book.pages.length;
    book.pages.splice(to, 0, moved);
    dragId = null; changed();
  });

  // ---------- the big viewer ----------
  let vSt = null, vIdx = 0, playTimer = null, layer = 'a';
  function openViewer(st, idx) {
    stopPlay();
    vSt = st;
    const page = pageFor(st);
    vIdx = Math.max(0, idx ?? st.members.indexOf(page ? page.photo : st.cover));
    const hl = new Set(st.highlights || []);
    $('v-film').innerHTML = st.members.length > 1
      ? st.members.map((h, i) => `<li><button data-frame="${i}" class="${hl.has(h) ? 'hl' : ''}" aria-label="Picture ${i + 1}"><img src="${thumb(h)}" alt="" loading="lazy"></button></li>`).join('') : '';
    $('v-prev').hidden = $('v-next').hidden = st.members.length < 2;
    $('v-play').hidden = (st.highlights || []).length < 2;
    $('v-flip').hidden = !((st.highlights || []).length >= 3);
    if (!$('viewer').open) $('viewer').showModal();
    showFrame(true);
  }
  function setImage(src, fade) {
    const cur = $(`v-img-${layer}`), next = $(`v-img-${layer === 'a' ? 'b' : 'a'}`);
    if (cur.getAttribute('src') === src) return;
    if (!fade || reduce) { cur.src = src; return; }
    next.onload = () => { next.classList.add('on'); cur.classList.remove('on'); layer = layer === 'a' ? 'b' : 'a'; };
    next.src = src;
  }
  function showFrame(fade = true) {
    const st = vSt, h = st.members[vIdx];
    setImage(large(h), fade);
    [vIdx + 1, vIdx + 2].forEach((i) => { if (st.members[i]) { const p = new Image(); p.src = large(st.members[i]); } });
    $('v-count').textContent = st.members.length > 1 ? `${vIdx + 1} / ${st.members.length}` : '';
    document.querySelectorAll('#v-film button').forEach((b) => b.classList.toggle('on', +b.dataset.frame === vIdx));
    const cur = document.querySelector(`#v-film button[data-frame="${vIdx}"]`);
    if (cur) cur.scrollIntoView({ block: 'nearest', inline: 'center' });
    $('v-react').innerHTML = reactHTML(st, true).replace(/^<div class="react big">|<\/div>$/g, '');
    const page = pageFor(st);
    // in the viewer, the heart/star lights up only for the picture that's on the page
    if (page && page.type === 'photo' && page.photo !== h) $('v-react').querySelectorAll('.on').forEach((b) => b.classList.remove('on'));
    $('v-flip').classList.toggle('on', !!page && page.type === 'flipbook');
  }
  function step(d) { if (!vSt) return; vIdx = (vIdx + d + vSt.members.length) % vSt.members.length; showFrame(true); }
  function stopPlay() { clearInterval(playTimer); playTimer = null; $('v-play').textContent = '▶'; }
  $('v-prev').onclick = () => { stopPlay(); step(-1); };
  $('v-next').onclick = () => { stopPlay(); step(1); };
  $('v-play').onclick = () => {
    if (playTimer) return stopPlay();
    const frames = vSt.highlights.map((h) => vSt.members.indexOf(h));
    let k = Math.max(0, frames.findIndex((i) => i >= vIdx));
    $('v-play').textContent = '■';
    vIdx = frames[k]; showFrame(true);
    playTimer = setInterval(() => { k = (k + 1) % frames.length; vIdx = frames[k]; showFrame(true); }, DWELL_MS);
  };
  $('v-flip').onclick = () => {
    const st = vSt, page = pageFor(st);
    if (page && page.type === 'flipbook') { page.type = 'photo'; page.frames = [page.photo]; chime('off'); }
    else if (page) { page.type = 'flipbook'; page.frames = st.highlights.slice(); chime('heart'); burst($('v-flip'), '🎞️', 6); }
    else { const p = newPage(st, st.members[vIdx], true); p.photo = st.members[vIdx]; insertInOrder(p); chime('heart'); burst($('v-flip'), '❤️'); }
    changed();
  };
  $('v-film').addEventListener('click', (e) => { const b = e.target.closest('[data-frame]'); if (b) { stopPlay(); vIdx = +b.dataset.frame; showFrame(true); } });
  $('v-close').onclick = () => $('viewer').close();
  $('viewer').addEventListener('close', () => { stopPlay(); vSt = null; });
  $('viewer').addEventListener('click', (e) => { if (e.target === $('viewer')) $('viewer').close(); });
  document.addEventListener('keydown', (e) => {
    if (!$('viewer').open || e.target.isContentEditable) return;
    if (e.key === 'ArrowRight') { stopPlay(); step(1); }
    if (e.key === 'ArrowLeft') { stopPlay(); step(-1); }
  });

  // ---------- Marshmallow's pile ----------
  function renderPile() { $('pile-grid').innerHTML = pileStacks().map(cardHTML).join(''); wireFlips($('pile-grid')); }
  $('open-pile').onclick = () => { renderPile(); $('pile').showModal(); };
  $('pile-close').onclick = () => $('pile').close();
  $('pile').addEventListener('click', (e) => { if (e.target === $('pile')) $('pile').close(); });

  // ---------- stacks gently flip through their different pictures on hover ----------
  function wireFlips(root) {
    if (reduce) return;
    root.querySelectorAll('.pola').forEach((p) => {
      const st = stacks[p.dataset.open];
      const frames = (st && st.highlights) || [];
      if (frames.length < 2) return;
      const img = p.querySelector('.cover'), rest = img.src;
      let t = null, i = 0;
      p.addEventListener('mouseenter', () => {
        frames.forEach((f) => { const im = new Image(); im.src = thumb(f); });
        t = setInterval(() => { img.src = thumb(frames[++i % frames.length]); }, HOVER_MS);
      });
      p.addEventListener('mouseleave', () => { clearInterval(t); img.src = rest; i = 0; });
    });
  }

  $('tell').onclick = () => {
    if (!book.pages.length) { $('pages').classList.add('wobble'); setTimeout(() => $('pages').classList.remove('wobble'), 450); return chime('nope'); }
    chime('tell'); burst($('tell'), '❄', 12);
    setTimeout(() => { location.href = `/tell/${importId}`; }, reduce ? 0 : 500);
  };

  // ---------- falling snow ----------
  (() => {
    const c = $('snow'), ctx = c.getContext('2d');
    let flakes = [];
    const size = () => {
      c.width = innerWidth; c.height = innerHeight;
      flakes = Array.from({ length: Math.round(innerWidth / 20) }, () => ({ x: Math.random() * innerWidth, y: Math.random() * innerHeight, r: 1.4 + Math.random() * 3, s: .25 + Math.random() * .55, w: Math.random() * 6.3 }));
    };
    const draw = (move) => {
      ctx.clearRect(0, 0, c.width, c.height);
      ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--elsa'); ctx.globalAlpha = .28;
      for (const f of flakes) {
        if (move) { f.y += f.s; f.w += .01; f.x += Math.sin(f.w) * .35; if (f.y > c.height + 6) { f.y = -6; f.x = Math.random() * c.width; } }
        ctx.beginPath(); ctx.arc(f.x, f.y, f.r, 0, 6.283); ctx.fill();
      }
    };
    size(); addEventListener('resize', size);
    if (reduce) draw(false); else (function loop() { draw(true); requestAnimationFrame(loop); })();
  })();

  load();
})();
