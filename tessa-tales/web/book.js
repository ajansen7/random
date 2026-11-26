/* Tessa Tales: make-the-book page. Plain JS, no build step. */
(() => {
  const importId = location.pathname.split('/').filter(Boolean)[1];
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const thumb = (h) => `/thumb/${h}.jpg`;
  const large = (h) => `/large/${h}.jpg`;
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;

  let imp, book, stacks = {}, stackOfPhoto = {};

  // ---------- loading & saving ----------
  async function load() {
    const [a, b] = await Promise.all([fetch(`/api/import/${importId}`), fetch(`/api/book/${importId}`)]);
    if (!a.ok || !b.ok) { $('olaf-line').textContent = "Olaf couldn't find these pictures. Try importing the card again."; return; }
    imp = await a.json(); book = await b.json();
    for (const st of imp.sieve.stacks) { stacks[st.id] = st; for (const h of st.members) stackOfPhoto[h] = st; }
    renderAll();
  }

  let saveTimer = null;
  function save() {
    $('saved').textContent = 'saving…';
    clearTimeout(saveTimer);
    saveTimer = setTimeout(async () => {
      try {
        const r = await fetch(`/api/book/${importId}`, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(book) });
        if (!r.ok) throw new Error((await r.json()).error || r.status);
        $('saved').textContent = 'saved ✓';
        setTimeout(() => { if ($('saved').textContent === 'saved ✓') $('saved').textContent = ''; }, 1800);
      } catch (e) {
        $('saved').textContent = "couldn't save, is the app still running?";
      }
    }, 500);
  }

  // ---------- helpers ----------
  const fmtTime = (iso) => new Date(iso).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' }).toLowerCase().replace(' ', ' ');
  const pageFor = (st) => book.pages.find((p) => st.members.includes(p.photo));
  function newPage(st, photo, flip) {
    return {
      id: `p-${st.id}-${Date.now().toString(36)}`,
      type: flip ? 'flipbook' : 'photo',
      photo: photo || st.cover,
      frames: flip ? st.members.slice() : [photo || st.cover],
      from_stack: st.id,
    };
  }
  function insertInOrder(page) {
    // keep the book in time order when adding; the grown-up can drag to reorder after
    const t = (p) => (stackOfPhoto[p.photo] ? stackOfPhoto[p.photo].start : '');
    const at = book.pages.findIndex((p) => t(p) > t(page));
    if (at < 0) book.pages.push(page); else book.pages.splice(at, 0, page);
  }
  function whyText(st) {
    const sc = st.scores;
    if (st.pile === 'snowdrift') return sc.blank || `looks like ${sc.label}`;
    if (st.gold) return '❄ Elsa\'s snowflake: she meant this one';
    if (st.flipbook) return `${st.members.length} in a row: flip-book!`;
    return sc.label_kind === 'junk' ? `looks like ${sc.label}…?` : `Olaf sees ${sc.label}`;
  }
  let toastTimer;
  function toast(msg) {
    const t = $('toast'); t.textContent = msg; t.classList.add('show');
    clearTimeout(toastTimer); toastTimer = setTimeout(() => t.classList.remove('show'), 2600);
  }

  // ---------- rendering ----------
  function renderAll() {
    const sv = imp.sieve;
    $('title').textContent = book.title;
    document.title = `${book.title} · Tessa Tales`;
    $('olaf-line').innerHTML = `Olaf looked at <b>${imp.total}</b> pictures and found <b>${sv.counts.treasure}</b> warm hugs in <b>${sv.moments.length}</b> moments. ` +
      `He buried <b>${sv.photo_counts.snowdrift}</b> in Marshmallow's pile.` +
      (sv.clock_reliable ? '' : ' (The camera clock looks wrong, so the moments are guesses.)');
    renderMoments();
    renderStrip();
  }

  function cardHTML(st) {
    const behind = st.members.filter((h) => h !== st.cover).slice(0, 2)
      .map((h, i) => `<img class="behind b${i}" src="${thumb(h)}" alt="" loading="lazy">`).join('');
    const count = st.members.length > 1
      ? `<span class="count">×${st.members.length}${st.flipbook ? ' <span class="fl">flip!</span>' : ''}</span>` : '';
    return `<div class="card" data-stack="${st.id}">
      <button class="pola" data-open="${st.id}" aria-label="Look closer">${behind}<img class="cover" src="${thumb(st.cover)}" alt="" loading="lazy">${st.gold ? '<span class="flake" aria-hidden="true">❄</span>' : ''}${count}<span class="ribbon">in the book</span></button>
      <button class="add" data-add="${st.id}" aria-label="Put in the book">+</button>
      <div class="why">${esc(whyText(st))}</div>
    </div>`;
  }

  function renderMoments() {
    const html = imp.sieve.moments.map((m) => {
      const mine = m.stacks.map((id) => stacks[id]);
      const hugs = mine.filter((s) => s.pile === 'treasure');
      const maybes = mine.filter((s) => s.pile === 'maybe');
      const name = (book.moment_names || {})[m.id] || m.name;
      const when = fmtTime(m.start) === fmtTime(m.end) ? fmtTime(m.start) : `${fmtTime(m.start)} – ${fmtTime(m.end)}`;
      return `<section class="moment" data-moment="${m.id}">
        <div class="m-head"><span class="m-name editable" contenteditable="true" spellcheck="false" data-moment-name="${m.id}">${esc(name)}</span><span class="m-when">${when}</span></div>
        <div class="cards">${hugs.map(cardHTML).join('')}</div>
        ${maybes.length ? `<details class="maybes"><summary>Olaf isn't sure about ${maybes.length} more ›</summary><div class="cards small">${maybes.map(cardHTML).join('')}</div></details>` : ''}
      </section>`;
    }).join('');
    $('moments').innerHTML = html;
    markCards();
    wireFlips($('moments'));
  }

  function markCards() {
    document.querySelectorAll('.card').forEach((c) => {
      const st = stacks[c.dataset.stack];
      const on = !!pageFor(st);
      c.classList.toggle('in', on);
      const add = c.querySelector('.add');
      add.textContent = on ? '✓' : '+';
      add.setAttribute('aria-label', on ? 'Take out of the book' : 'Put in the book');
    });
  }

  function renderStrip() {
    $('strip-title').textContent = book.title;
    $('strip-count').textContent = `${book.pages.length} page${book.pages.length === 1 ? '' : 's'}`;
    const ol = $('pages');
    if (!book.pages.length) { ol.innerHTML = '<li class="empty">Tap + on a picture to start the book!</li>'; return; }
    ol.innerHTML = book.pages.map((p, i) => `<li class="page" draggable="true" data-page="${p.id}">
      <span class="num">${i + 1}</span>
      <img src="${thumb(p.photo)}" alt="page ${i + 1}" data-open-page="${p.id}">
      ${p.type === 'flipbook' ? `<span class="kind">flip ×${p.frames.length}</span>` : ''}
      <button class="x" data-remove="${p.id}" aria-label="Take page ${i + 1} out">✕</button>
    </li>`).join('');
  }

  function changed() { markCards(); renderStrip(); save(); }

  // ---------- adding & removing ----------
  function toggle(st, btn) {
    const existing = pageFor(st);
    if (existing) {
      book.pages = book.pages.filter((p) => p !== existing);
      toast('Out of the book. You can always put it back!');
    } else {
      insertInOrder(newPage(st, null, st.flipbook));
      toast(st.flipbook ? 'Flip-book page added! ❄' : 'Into the book! ❄');
    }
    if (btn && !reduce) { btn.classList.remove('pop'); void btn.offsetWidth; btn.classList.add('pop'); }
    changed();
  }

  document.addEventListener('click', (e) => {
    const add = e.target.closest('[data-add]');
    if (add) return toggle(stacks[add.dataset.add], add);
    const open = e.target.closest('[data-open]');
    if (open) return openViewer(stacks[open.dataset.open]);
    const rm = e.target.closest('[data-remove]');
    if (rm) { book.pages = book.pages.filter((p) => p.id !== rm.dataset.remove); return changed(); }
    const op = e.target.closest('[data-open-page]');
    if (op) {
      const p = book.pages.find((x) => x.id === op.dataset.openPage);
      const st = p && stackOfPhoto[p.photo];
      if (st) openViewer(st, st.members.indexOf(p.photo));
    }
  });

  // ---------- editing names ----------
  function wireEditable(el, onSave) {
    el.addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); el.blur(); } });
    el.addEventListener('blur', () => {
      const v = el.textContent.trim().slice(0, 80);
      if (!v) { el.textContent = onSave(null); return; }
      onSave(v);
    });
  }
  wireEditable($('title'), (v) => {
    if (v === null) return book.title;
    book.title = v; document.title = `${v} · Tessa Tales`; renderStrip(); save(); return v;
  });
  document.addEventListener('focusout', (e) => {
    const el = e.target.closest && e.target.closest('[data-moment-name]');
    if (!el) return;
    const id = el.dataset.momentName;
    const v = el.textContent.trim().slice(0, 80);
    const m = imp.sieve.moments.find((x) => x.id === id);
    if (!v) { el.textContent = (book.moment_names || {})[id] || m.name; return; }
    book.moment_names = book.moment_names || {};
    if (v !== ((book.moment_names[id]) || m.name)) { book.moment_names[id] = v; save(); }
  });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && e.target.matches && e.target.matches('[data-moment-name]')) { e.preventDefault(); e.target.blur(); }
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
    const from = book.pages.findIndex((p) => p.id === dragId);
    const [moved] = book.pages.splice(from, 1);
    let to = li ? book.pages.findIndex((p) => p.id === li.dataset.page) : book.pages.length;
    if (to < 0) to = book.pages.length;
    book.pages.splice(to, 0, moved);
    dragId = null; changed();
  });

  // ---------- the big viewer ----------
  let vSt = null, vIdx = 0, vWhere = '', playTimer = null;
  function openViewer(st, idx) {
    vSt = st; vIdx = Math.max(0, idx ?? st.members.indexOf(st.cover));
    const m = imp.sieve.moments.find((x) => x.stacks.includes(st.id));
    vWhere = st.pile === 'snowdrift' ? "From Marshmallow's pile" : ((book.moment_names || {})[m?.id] || m?.name || '');
    $('v-why').textContent = whyText(st);
    $('v-film').innerHTML = st.members.length > 1
      ? st.members.map((h, i) => `<li><button data-frame="${i}" aria-label="Picture ${i + 1}"><img src="${thumb(h)}" alt="" loading="lazy"></button></li>`).join('') : '';
    $('v-prev').hidden = $('v-next').hidden = $('v-play').hidden = st.members.length < 2;
    if (!$('viewer').open) $('viewer').showModal();
    showFrame();
  }
  function showFrame() {
    const st = vSt, h = st.members[vIdx];
    $('v-img').src = large(h);
    // warm the next couple of big pictures so arrows feel instant
    [vIdx + 1, vIdx + 2].forEach((i) => { if (st.members[i]) { const p = new Image(); p.src = large(st.members[i]); } });
    document.querySelectorAll('#v-film button').forEach((b) => b.classList.toggle('on', +b.dataset.frame === vIdx));
    const cur = document.querySelector(`#v-film button[data-frame="${vIdx}"]`);
    if (cur) cur.scrollIntoView({ block: 'nearest', inline: 'center' });
    const page = pageFor(st);
    const many = st.members.length > 1;
    $('v-where').textContent = st.members.length > 1 ? `${vWhere}  ·  ${vIdx + 1} of ${st.members.length}` : vWhere;
    $('v-add').textContent = page ? (page.photo === h && page.type === 'photo' ? 'In the book ✓  (tap to take out)' : 'Use this picture for its page') : 'Put this picture in the book';
    $('v-flip').hidden = !many;
    $('v-flip').textContent = page && page.type === 'flipbook' ? 'Flip-book is in the book ✓' : `Put in the whole flip-book (${st.members.length})`;
  }
  function step(d) { if (!vSt) return; vIdx = (vIdx + d + vSt.members.length) % vSt.members.length; showFrame(); }
  function stopPlay() { clearInterval(playTimer); playTimer = null; $('v-play').textContent = '▶ flip it'; }
  $('v-prev').onclick = () => { stopPlay(); step(-1); };
  $('v-next').onclick = () => { stopPlay(); step(1); };
  $('v-play').onclick = () => {
    if (playTimer) return stopPlay();
    $('v-play').textContent = '■ stop';
    playTimer = setInterval(() => { vIdx = (vIdx + 1) % vSt.members.length; $('v-img').src = thumb(vSt.members[vIdx]); }, 125);
  };
  $('v-film').addEventListener('click', (e) => { const b = e.target.closest('[data-frame]'); if (b) { stopPlay(); vIdx = +b.dataset.frame; showFrame(); } });
  $('v-add').onclick = () => {
    const st = vSt, h = st.members[vIdx], page = pageFor(st);
    if (page && page.photo === h && page.type === 'photo') { book.pages = book.pages.filter((p) => p !== page); toast('Out of the book.'); }
    else if (page) { page.photo = h; page.type = 'photo'; page.frames = [h]; toast('Swapped in this picture ❄'); }
    else { insertInOrder(newPage(st, h, false)); toast('Into the book! ❄'); }
    changed(); showFrame();
  };
  $('v-flip').onclick = () => {
    const st = vSt, h = st.members[vIdx], page = pageFor(st);
    if (page && page.type === 'flipbook') return toast('Already a flip-book page!');
    if (page) { page.type = 'flipbook'; page.frames = st.members.slice(); page.photo = h; }
    else insertInOrder(newPage(st, h, true));
    toast('Flip-book page added! ❄'); changed(); showFrame();
  };
  $('v-close').onclick = () => $('viewer').close();
  $('viewer').addEventListener('close', () => { stopPlay(); vSt = null; });
  $('viewer').addEventListener('click', (e) => { if (e.target === $('viewer')) $('viewer').close(); });
  document.addEventListener('keydown', (e) => {
    if (!$('viewer').open || e.target.isContentEditable) return;
    if (e.key === 'ArrowRight') { stopPlay(); step(1); }
    if (e.key === 'ArrowLeft') { stopPlay(); step(-1); }
  });

  // ---------- Marshmallow's pile ----------
  $('open-pile').onclick = () => {
    const drift = imp.sieve.stacks.filter((s) => s.pile === 'snowdrift');
    $('pile-grid').innerHTML = drift.map(cardHTML).join('');
    markCards(); wireFlips($('pile-grid'));
    $('pile').showModal();
  };
  $('pile-close').onclick = () => $('pile').close();
  $('pile').addEventListener('click', (e) => { if (e.target === $('pile')) $('pile').close(); });

  // ---------- flip-books flip when you hover ----------
  function wireFlips(root) {
    if (reduce) return;
    root.querySelectorAll('.pola').forEach((p) => {
      const st = stacks[p.dataset.open];
      if (!st || st.members.length < 2) return;
      const img = p.querySelector('.cover');
      const step = Math.max(1, Math.floor(st.members.length / 48));
      const frames = st.members.filter((_, i) => i % step === 0).map(thumb);
      let t = null, i = 0;
      p.addEventListener('mouseenter', () => { frames.forEach((f) => { const im = new Image(); im.src = f; }); t = setInterval(() => { img.src = frames[i++ % frames.length]; }, 125); });
      p.addEventListener('mouseleave', () => { clearInterval(t); img.src = thumb(st.cover); i = 0; });
    });
  }

  $('tell').onclick = () => toast(book.pages.length ? "Tessa's turn is the very next thing we build! ❄" : 'Put a few pictures in the book first!');

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
