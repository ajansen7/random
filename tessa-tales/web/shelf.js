/* The ice palace shelf: tap a snow globe to watch its story. Grown-ups: ✏️ to work on it.
   Plug in the camera card and Olaf offers to bring in the new pictures. */
(() => {
  const $ = (id) => document.getElementById(id);
  const { chime, burst, thumb, esc, sleep } = TT;

  async function loadShelf() {
    const list = await (await fetch('/api/shelf')).json();
    $('empty').hidden = list.length > 0;
    $('shelf').innerHTML = list.map((s) => {
      const told = s.pages ? `${'❄'.repeat(Math.min(s.told, 12))}<span class="dim">${'❄'.repeat(Math.max(0, Math.min(s.pages, 12) - Math.min(s.told, 12)))}</span>` : '';
      return `<div class="globe" data-story="${s.id}">
        ${s.new ? '<span class="new" title="New!" aria-hidden="true">✨</span>' : ''}
        <a class="round ghost edit" href="/story/${s.id}" title="Work on this story (grown-ups)" aria-label="Work on ${esc(s.title)}">✏️</a>
        <button class="glass" data-play="${s.id}" aria-label="Watch ${esc(s.title)}">${s.cover ? `<img src="${thumb(s.cover)}" alt="">` : ''}<span class="flakes"></span></button>
        <div class="base" aria-hidden="true"></div>
        <div class="g-title">${esc(s.title)}</div>
        <div class="g-told" title="${s.told} of ${s.pages} pages told">${told}</div>
      </div>`;
    }).join('');
  }

  // tap a globe: it shakes, the snow swirls, then story time
  $('shelf').addEventListener('click', async (e) => {
    const g = e.target.closest('[data-play]');
    if (!g) return;
    g.classList.add('shake'); chime('tell'); burst(g, '❄', 12, 140);
    await sleep(TT.reduce ? 0 : 650);
    location.href = `/story/${g.dataset.play}#play`;
  });

  // ---------- the camera card ----------
  let card = null, polling = null;
  async function checkCard() {
    let s;
    try { s = await (await fetch('/api/card')).json(); } catch (e) { return; }
    const imp = s.import || {};
    const busy = ['copying', 'sorting', 'stories'].includes(imp.phase);
    card = s.card;
    $('card-bar').hidden = !card && !busy;
    $('card-go').hidden = busy || !s.new;
    $('card-n').textContent = s.new || '';
    $('eject').hidden = busy || !card;
    $('progress').hidden = !busy;
    if (busy) {
      const pct = imp.phase === 'copying' && imp.total ? (imp.done / imp.total) * 70 : imp.phase === 'sorting' ? 85 : 97;
      $('bar').style.width = `${pct}%`;
      $('phase').textContent = imp.phase === 'copying' ? '📷' : imp.phase === 'sorting' ? '🧹' : '📚';
    }
    if ((imp.phase === 'done' || imp.phase === 'settled') && !checkCard.shownDone) {
      checkCard.shownDone = true;
      await loadShelf();
      if ((imp.new_stories || []).length) { chime('done'); burst($('shelf'), '✨', 16, 260); }
    }
    if (imp.phase === 'error') $('phase').textContent = '⚠️';
    const bad = imp.damaged || [];
    $('damaged').hidden = !bad.length;
    if (bad.length) $('damaged').textContent = `⚠️ ${bad.length} picture${bad.length === 1 ? ' is' : 's are'} damaged on the card, so Olaf skipped ${bad.length === 1 ? 'it' : 'them'}. Copies already on the Mac are fine. Time for a fresh card soon?`;
  }
  $('card-go').onclick = async () => {
    if (!card) return;
    chime('heart'); burst($('card-go'), '📷', 8, 90);
    checkCard.shownDone = false;
    await fetch('/api/import', { method: 'POST', body: JSON.stringify({ card }) });
    checkCard();
  };
  $('eject').onclick = async () => {
    if (!card) return;
    const r = await fetch('/api/eject', { method: 'POST', body: JSON.stringify({ card }) });
    if (r.ok) { chime('off'); $('card-bar').hidden = true; }
    else $('eject').title = (await r.json()).error || "Couldn't eject";
  };

  // ---------- 🗣️ who tells the story ----------
  const SAMPLE = 'Once upon a time, in a land of snow and sparkles, there was a girl named Tessa.';
  const ACCENTS = { 'en-US': 'American', 'en-GB': 'British', 'en-IE': 'Irish', 'en-AU': 'Australian', 'en-ZA': 'South African', 'en-IN': 'Indian', 'en-NZ': 'New Zealand', 'en-SC': 'Scottish' };
  function renderVoices() {
    const chosen = TT.narrator.chosen();
    const vs = TT.narrator.voices().sort((a, b) => (a.lang === 'en-US' ? 0 : 1) - (b.lang === 'en-US' ? 0 : 1) || a.name.localeCompare(b.name));
    $('voice-list').innerHTML = vs.map((v) => `<li>
      <button class="round ghost v-play" data-hear="${esc(v.name)}" aria-label="Hear ${esc(v.name)}">▶</button>
      <button class="v-name ${chosen && chosen.name === v.name ? 'on' : ''}" data-pick="${esc(v.name)}">${esc(v.name.replace(/ \(English \(.*\)\)/, ''))}<small>${ACCENTS[v.lang.replace('_', '-')] || v.lang}${/premium|enhanced/i.test(v.name) ? ' · ✨ natural' : ''}</small></button>
    </li>`).join('') || '<li>No voices found.</li>';
  }
  $('open-voice').onclick = () => { renderVoices(); $('voice').showModal(); };
  $('voice-close').onclick = () => { speechSynthesis.cancel(); $('voice').close(); };
  $('voice').addEventListener('click', (e) => {
    const vs = TT.narrator.voices();
    const hear = e.target.closest('[data-hear]');
    if (hear) TT.narrator.say(SAMPLE, vs.find((v) => v.name === hear.dataset.hear));
    const pick = e.target.closest('[data-pick]');
    if (pick) { TT.narrator.choose(pick.dataset.pick); renderVoices(); chime('heart'); TT.narrator.say('Hello, Tessa!', vs.find((v) => v.name === pick.dataset.pick)); }
  });
  if ('speechSynthesis' in window) speechSynthesis.onvoiceschanged = () => { if ($('voice').open) renderVoices(); };

  TT.snow($('snow'), '--frost', .35);
  loadShelf();
  checkCard();
  polling = setInterval(checkCard, 2500);
})();
