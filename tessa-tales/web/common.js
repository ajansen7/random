/* Shared bits for Tessa's turn and story time: chimes, bursts, falling snow. */
window.TT = (() => {
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
  let audio = null;
  let soundOn = true;
  try { soundOn = localStorage.getItem('tt-sound') !== 'off'; } catch (e) {}

  const NOTES = {
    heart: [660, 880], star: [784, 988, 1175, 1568], nope: [262, 196], off: [523],
    tell: [523, 659, 784, 1047], start: [880], done: [784, 988, 1175, 1568, 2093], page: [587, 784],
  };
  function chime(kind) {
    if (!soundOn) return;
    try {
      audio = audio || new (window.AudioContext || window.webkitAudioContext)();
      const notes = NOTES[kind] || NOTES.off;
      notes.forEach((f, i) => {
        const o = audio.createOscillator(), g = audio.createGain(), t = audio.currentTime + i * 0.09;
        o.type = kind === 'nope' ? 'triangle' : 'sine'; o.frequency.value = f;
        g.gain.setValueAtTime(0.0001, t); g.gain.exponentialRampToValueAtTime(0.16, t + 0.02); g.gain.exponentialRampToValueAtTime(0.0001, t + 0.35);
        o.connect(g).connect(audio.destination); o.start(t); o.stop(t + 0.4);
      });
    } catch (e) {}
  }
  function setSound(on) { soundOn = on; try { localStorage.setItem('tt-sound', on ? 'on' : 'off'); } catch (e) {} }

  function burst(el, icon, n = 10, spread = 110) {
    if (reduce || !el) return;
    const r = el.getBoundingClientRect();
    for (let i = 0; i < n; i++) {
      const s = document.createElement('span');
      s.className = 'burst'; s.textContent = icon;
      const a = (Math.PI * 2 * i) / n + Math.random() * .5;
      const d = spread * (.55 + Math.random() * .45);
      s.style.left = `${r.left + r.width / 2 - 14}px`; s.style.top = `${r.top + r.height / 2 - 14}px`;
      s.style.setProperty('--dx', `${Math.cos(a) * d}px`);
      s.style.setProperty('--dy', `${Math.sin(a) * d - 40}px`);
      s.style.setProperty('--rot', `${(Math.random() - .5) * 120}deg`);
      document.body.appendChild(s);
      setTimeout(() => s.remove(), 950);
    }
  }

  function snow(canvas, colorVar = '--elsa', alpha = .28) {
    const ctx = canvas.getContext('2d');
    let flakes = [];
    const size = () => {
      canvas.width = innerWidth; canvas.height = innerHeight;
      flakes = Array.from({ length: Math.round(innerWidth / 18) }, () => ({ x: Math.random() * innerWidth, y: Math.random() * innerHeight, r: 1.4 + Math.random() * 3, s: .25 + Math.random() * .55, w: Math.random() * 6.3 }));
    };
    const draw = (move) => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue(colorVar); ctx.globalAlpha = alpha;
      for (const f of flakes) {
        if (move) { f.y += f.s; f.w += .01; f.x += Math.sin(f.w) * .35; if (f.y > canvas.height + 6) { f.y = -6; f.x = Math.random() * canvas.width; } }
        ctx.beginPath(); ctx.arc(f.x, f.y, f.r, 0, 6.283); ctx.fill();
      }
    };
    size(); addEventListener('resize', size);
    if (reduce) draw(false); else (function loop() { draw(true); requestAnimationFrame(loop); })();
  }

  // ---------- a music box: an original little waltz, synthesized, no files ----------
  // 16 bars of 3/4: [note, beats]. Quiet under everything; ducks when someone speaks.
  const WALTZ = [
    ['E5', 1], ['G5', 1], ['C6', 1], ['B5', 2], ['G5', 1], ['A5', 1], ['F5', 1], ['D5', 1], ['G5', 3],
    ['C5', 1], ['E5', 1], ['G5', 1], ['A5', 2], ['E5', 1], ['F5', 1], ['D5', 1], ['B4', 1], ['C5', 3],
    ['E5', 1], ['F5', 1], ['G5', 1], ['C6', 2], ['A5', 1], ['G5', 1], ['E5', 1], ['C5', 1], ['D5', 3],
    ['E5', 1], ['G5', 1], ['F5', 1], ['E5', 1], ['D5', 1], ['B4', 1], ['C5', 3], [null, 3],
  ];
  const ROOTS = ['C3', 'E3', 'F3', 'G2', 'C3', 'A2', 'F3', 'C3', 'C3', 'F3', 'C3', 'G2', 'C3', 'G2', 'C3', 'C3'];
  const BEAT = 0.42;
  const hz = (n) => {
    const i = { C: 0, D: 2, E: 4, F: 5, G: 7, A: 9, B: 11 }[n[0]];
    return 440 * Math.pow(2, (i + 12 * (+n.slice(1) + 1) - 69) / 12);
  };
  const music = (() => {
    let master = null, timer = null, next = 0, step = 0, beatInBar = 0, bar = 0;
    function bell(f, t, vol, decay) {
      [[1, vol], [2, vol * 0.25], [3.01, vol * 0.08]].forEach(([mult, v]) => {
        const o = audio.createOscillator(), g = audio.createGain();
        o.type = 'sine'; o.frequency.value = f * mult;
        g.gain.setValueAtTime(0.0001, t); g.gain.exponentialRampToValueAtTime(v, t + 0.01); g.gain.exponentialRampToValueAtTime(0.0001, t + decay / mult);
        o.connect(g).connect(master); o.start(t); o.stop(t + decay + 0.1);
      });
    }
    function tick() {
      while (next < audio.currentTime + 0.25) {
        if (beatInBar === 0) bell(hz(ROOTS[bar % ROOTS.length]), next, 0.35, 2.2);
        const [note, beats] = WALTZ[step];
        if (note) bell(hz(note), next, 0.5, 1.4);
        next += beats * BEAT;
        beatInBar += beats;
        if (beatInBar >= 3) { beatInBar = 0; bar++; }
        step = (step + 1) % WALTZ.length;
      }
    }
    return {
      start() {
        if (!soundOn || timer) return;
        try {
          audio = audio || new (window.AudioContext || window.webkitAudioContext)();
          master = audio.createGain(); master.gain.value = 0.0001; master.connect(audio.destination);
          master.gain.setTargetAtTime(0.06, audio.currentTime, 0.8);
          next = audio.currentTime + 0.1; step = 0; beatInBar = 0; bar = 0;
          timer = setInterval(tick, 40);
        } catch (e) {}
      },
      duck(on) { if (master) master.gain.setTargetAtTime(on ? 0.015 : 0.06, audio.currentTime, 0.25); },
      stop() {
        if (!timer) return;
        clearInterval(timer); timer = null;
        const m = master; master = null;
        m.gain.setTargetAtTime(0.0001, audio.currentTime, 0.4);
        setTimeout(() => m.disconnect(), 2500);
      },
    };
  })();

  return {
    reduce, chime, burst, snow, setSound, music, get soundOn() { return soundOn; },
    thumb: (h) => `/thumb/${h}.jpg`,
    large: (h) => `/large/${h}.jpg`,
    esc: (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c])),
    sleep: (ms) => new Promise((r) => setTimeout(r, ms)),
  };
})();
