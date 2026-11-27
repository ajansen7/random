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

  return {
    reduce, chime, burst, snow, setSound, get soundOn() { return soundOn; },
    thumb: (h) => `/thumb/${h}.jpg`,
    large: (h) => `/large/${h}.jpg`,
    esc: (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c])),
    sleep: (ms) => new Promise((r) => setTimeout(r, ms)),
  };
})();
