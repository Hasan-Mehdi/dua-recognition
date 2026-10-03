// Kids mode: the same app dressed for children. The recognizer, the tracker and
// the page's flow are unchanged; this adds what a child sees instead of the
// manuscript: Noor the lantern (who listens, and answers the voice through
// meter() in app.js), a garland with one lantern per line, a sticker for each
// du'a finished from the microphone, and a celebration at the end. The look is
// kids.css, scoped to body.kids. A grown-up leaves by pressing and holding the
// lock on the home screen.

const NAVY = "#1F2A5A";
const COLORS = ["#FFC93C", "#7CC7FF", "#5ED3A3", "#B39DFF", "#FF7BAC", "#FF9F43"];
const FONT = "https://fonts.googleapis.com/css2?family=Baloo+Bhaijaan+2:wght@400..800&display=swap";

// The shelf in the sticker book, in order; stickers earned for any other du'a follow.
const SHELF = ["dua-hujjat", "dua-hadithkisa", "dua-tawassul", "ziyarat-ashura", "dua-aahad", "dua-kumayl", "dua-friday",
  "sahifa-24", "duasorg-sahifa-fatimiya-dua-for-sleep", "dua-noor", "dua-eid-takbirat", "sahifa-06"];
const NAMES = {
  "dua-hujjat": "Dua Hujjah", "dua-hadithkisa": "Hadith al-Kisa", "sahifa-24": "For my parents", "sahifa-06": "Morning & evening",
  "duasorg-sahifa-fatimiya-dua-for-sleep": "Before sleep", "dua-eid-takbirat": "Eid Takbir",
};
const ICON_OF = {
  "dua-hujjat": "lantern", "dua-hadithkisa": "cloak", "dua-tawassul": "link", "ziyarat-ashura": "drop", "dua-aahad": "flag",
  "dua-kumayl": "moon", "sahifa-24": "heart", "duasorg-sahifa-fatimiya-dua-for-sleep": "sleep", "dua-noor": "sun",
  "dua-eid-takbirat": "star", "sahifa-06": "sun", "dua-iftitah": "moon", "dua-nudbah": "lantern",
};
const COLOR_OF = { "dua-hujjat": 0, "dua-hadithkisa": 2, "dua-tawassul": 3, "ziyarat-ashura": 4, "dua-aahad": 5, "dua-kumayl": 3,
  "dua-friday": 2, "sahifa-24": 4, "duasorg-sahifa-fatimiya-dua-for-sleep": 1, "dua-noor": 0, "dua-eid-takbirat": 5, "sahifa-06": 1 };
const DAY = /^(dua|ziyarat)-(sunday|monday|tuesday|wednesday|thursday|friday|saturday)$/;

const ICONS = {
  lantern: '<circle cx="12" cy="3.6" r="1.4"/><path d="M7.8 8.4Q12 4.2 16.2 8.4Z" fill="#fff"/><rect x="8.4" y="8.4" width="7.2" height="9" rx="2.6" fill="#fff"/><path d="M9.8 17.4h4.4L12 20.8z"/>',
  calendar: '<rect x="4" y="5" width="16" height="15" rx="3.5"/><path d="M4 10h16M8.5 3v4M15.5 3v4"/><circle cx="12" cy="15" r="1.6" fill="currentColor"/>',
  cloak: '<path d="M12 4c-4 0-5 3-6 7l-2 9h16l-2-9c-1-4-2-7-6-7z"/><path d="M12 4v16"/>',
  link: '<path d="M10 14a4 4 0 0 0 5.7 0l3-3a4 4 0 0 0-5.7-5.7l-1 1"/><path d="M14 10a4 4 0 0 0-5.7 0l-3 3a4 4 0 0 0 5.7 5.7l1-1"/>',
  drop: '<path d="M12 3s6 7 6 11a6 6 0 0 1-12 0c0-4 6-11 6-11z"/>',
  flag: '<path d="M6 21V4"/><path d="M6 4.5h11l-2.5 4 2.5 4H6"/>',
  moon: '<path d="M19 14.5A7.5 7.5 0 1 1 9.5 5a6 6 0 0 0 9.5 9.5z"/><path d="M17 3.5v3M15.5 5h3"/>',
  dome: '<path d="M5 20v-5a7 7 0 0 1 14 0v5"/><path d="M3 20h18M12 8V4.5M10.5 6h3"/>',
  arch: '<path d="M5 21V11a7 7 0 0 1 14 0v10"/><path d="M9.5 21v-5a2.5 2.5 0 0 1 5 0v5M3 21h18"/>',
  heart: '<path d="M12 20s-7.5-4.6-7.5-10.2A4.3 4.3 0 0 1 12 7.4a4.3 4.3 0 0 1 7.5 2.4C19.5 15.4 12 20 12 20z"/>',
  sleep: '<path d="M16 15.5A6.5 6.5 0 1 1 8.5 8a5 5 0 0 0 7.5 7.5z"/><path d="M14 4h4l-4 4h4"/>',
  sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2.5v2M12 19.5v2M2.5 12h2M19.5 12h2M5.3 5.3l1.4 1.4M17.3 17.3l1.4 1.4M5.3 18.7l1.4-1.4M17.3 6.7l1.4-1.4"/>',
  star: '<path d="M12 2.8l2.7 5.6 6.1.8-4.5 4.2 1.1 6.1L12 16.6l-5.4 2.9 1.1-6.1-4.5-4.2 6.1-.8z"/>',
  replay: '<path d="M4.5 12a7.5 7.5 0 1 0 2.2-5.3"/><path d="M4.5 3.5v4h4"/>',
  home: '<path d="M4 11.5l8-7 8 7"/><path d="M6.5 9.5V20h11V9.5"/>',
  lock: '<rect x="5" y="11" width="14" height="10" rx="3"/><path d="M8 11V8a4 4 0 0 1 8 0v3"/>',
  book: '<path d="M5 5.5A2.5 2.5 0 0 1 7.5 3H19v15H7.5A2.5 2.5 0 0 0 5 20.5z"/><path d="M5 20.5A2.5 2.5 0 0 0 7.5 23H19"/>',
};
const ROSETTE = "M0-46 7.2-36.3 17.6-42.5 20.6-30.8 32.5-32.5 30.8-20.6 42.5-17.6 36.3-7.2 46 0 36.3 7.2 42.5 17.6 30.8 20.6 " +
  "32.5 32.5 20.6 30.8 17.6 42.5 7.2 36.3 0 46-7.2 36.3-17.6 42.5-20.6 30.8-32.5 32.5-30.8 20.6-42.5 17.6-36.3 7.2-46 " +
  "0-36.3-7.2-42.5-17.6-30.8-20.6-32.5-32.5-20.6-30.8-17.6-42.5-7.2-36.3Z";

const icon = (name, stroke = NAVY) =>
  `<svg viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="${stroke}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="color:${stroke}">${ICONS[name]}</svg>`;

// Noor. `face`: smile (home), talk (listening), happy (the celebration).
function lantern(face) {
  const faces = {
    smile: '<ellipse cx="51" cy="86" rx="4.5" ry="6" fill="#1F2A5A"/><ellipse cx="69" cy="86" rx="4.5" ry="6" fill="#1F2A5A"/>' +
      '<circle cx="52.6" cy="83.8" r="1.7" fill="#fff"/><circle cx="70.6" cy="83.8" r="1.7" fill="#fff"/>' +
      '<path d="M53 98Q60 105 67 98" fill="none" stroke="#1F2A5A" stroke-width="3.5" stroke-linecap="round"/>',
    talk: '<ellipse cx="51" cy="84" rx="5" ry="7" fill="#1F2A5A"/><ellipse cx="69" cy="84" rx="5" ry="7" fill="#1F2A5A"/>' +
      '<circle cx="52.8" cy="81.5" r="1.9" fill="#fff"/><circle cx="70.8" cy="81.5" r="1.9" fill="#fff"/>' +
      '<ellipse cx="60" cy="101" rx="5" ry="4.5" fill="#1F2A5A"/>',
    happy: '<path d="M43 88Q50 79 57 88M63 88Q70 79 77 88" fill="none" stroke="#1F2A5A" stroke-width="4" stroke-linecap="round"/>' +
      '<path d="M50 96H70Q68 110 60 110 52 110 50 96Z" fill="#1F2A5A" stroke="#1F2A5A" stroke-width="2" stroke-linejoin="round"/>' +
      '<path d="M54.5 105.5Q60 101.5 65.5 105.5 63.5 109 60 109 56.5 109 54.5 105.5Z" fill="#FF7BAC"/>',
  };
  return `<svg class="lantern" viewBox="0 0 120 170" aria-hidden="true">
<circle cx="60" cy="18" r="6.5" fill="none" stroke="#1F2A5A" stroke-width="4"/>
<path d="M28 44Q60 14 92 44Z" fill="#FFC93C" stroke="#1F2A5A" stroke-width="4" stroke-linejoin="round"/>
<rect x="24" y="42" width="72" height="13" rx="6.5" fill="#FF9F43" stroke="#1F2A5A" stroke-width="4"/>
<rect x="31" y="55" width="58" height="70" rx="20" fill="#FFF4C4" stroke="#1F2A5A" stroke-width="4"/>
<path d="M45 58V122M75 58V122" stroke="#F4D06A" stroke-width="3"/>
<ellipse cx="41.5" cy="97" rx="5" ry="3.5" fill="#FF7BAC" opacity=".75"/><ellipse cx="78.5" cy="97" rx="5" ry="3.5" fill="#FF7BAC" opacity=".75"/>
${faces[face]}
<rect x="24" y="123" width="72" height="13" rx="6.5" fill="#FF9F43" stroke="#1F2A5A" stroke-width="4"/>
<path d="M42 136H78L60 156Z" fill="#FFC93C" stroke="#1F2A5A" stroke-width="4" stroke-linejoin="round"/></svg>`;
}

// One lantern of the garland; kids.css colours it by state (lit, now, or not yet).
const MINI = '<svg viewBox="0 0 24 34" aria-hidden="true"><circle cx="12" cy="3" r="2" fill="none"/><path class="cap" d="M5.5 10Q12 3.5 18.5 10Z"/>' +
  '<rect class="band" x="4" y="9.5" width="16" height="4" rx="2"/><rect class="glass" x="6" y="13.5" width="12" height="12" rx="4"/>' +
  '<rect class="band" x="4" y="25" width="16" height="4" rx="2"/><path class="cap" d="M9 29H15L12 32.5Z"/></svg>';

// The seal of the start button, made squishy: two rounded squares with one outline around both.
const SEAL = (() => {
  const sq = 'x="-38" y="-38" width="76" height="76" rx="17"';
  const pair = (attrs) => `<rect ${sq} ${attrs}/><rect ${sq} transform="rotate(45)" ${attrs}/>`;
  return `<svg class="kid-seal" viewBox="-60 -60 120 120" aria-hidden="true"><g class="kid-seal-star">
<g transform="translate(0 7)">${pair(`fill="${NAVY}" stroke="${NAVY}" stroke-width="8"`)}</g>
${pair(`fill="${NAVY}" stroke="${NAVY}" stroke-width="8"`)}${pair('fill="#FF7BAC"')}</g>
<circle r="33" fill="#fff" stroke="${NAVY}" stroke-width="4"/>
<ellipse cx="-30" cy="-34" rx="9" ry="4.5" fill="#fff" opacity=".6" transform="rotate(-40 -30 -34)"/>
<g transform="translate(-19 -19) scale(1.6)"><rect x="8.5" y="2.5" width="7" height="12" rx="3.5" fill="#FF7BAC" stroke="${NAVY}" stroke-width="2"/>
<path d="M5 11a7 7 0 0 0 14 0M12 18v3.2M8.6 21.4h6.8" fill="none" stroke="${NAVY}" stroke-width="2" stroke-linecap="round"/></g></svg>`;
})();

const CLOUD = '<svg viewBox="0 0 120 60" aria-hidden="true"><path d="M22 52C8 52 4 36 16 31 14 17 32 10 42 19 48 5 72 4 78 20 92 14 106 24 100 36 112 38 112 52 100 52Z" fill="#fff"/></svg>';
const SPARK = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 2c.8 5.4 4.6 9.2 10 10-5.4.8-9.2 4.6-10 10-.8-5.4-4.6-9.2-10-10 5.4-.8 9.2-4.6 10-10z"/></svg>';

const el = (html) => {
  const t = document.createElement("template");
  t.innerHTML = html.trim();
  return t.content.firstElementChild;
};
const store = (key, value) => {
  try {
    if (value === undefined) return localStorage.getItem(key);
    localStorage.setItem(key, value);
  } catch {} // private windows: kids mode still works, it just isn't remembered
  return null;
};
const night = (h = new Date().getHours()) => h >= 18 || h < 5;

export const kids = {
  on: false,
  hooks: {},
  duas: {},
  built: false,

  // hooks: { pick(id), openPicker(), again(id), home() } from app.js.
  init(duas, hooks) {
    this.duas = duas;
    this.hooks = hooks;
    document.getElementById("kids-on").onclick = () => this.set(true);
    const asked = new URLSearchParams(location.search).get("kids");
    this.set(asked != null ? asked !== "0" : store("kids") === "1", false);
  },

  set(on, remember = true) {
    if (on) this.build();
    this.on = on;
    document.body.classList.toggle("kids", on);
    document.body.classList.toggle("kids-night", on && night());
    if (remember) store("kids", on ? "1" : "0");
  },

  name(id) {
    const m = id.match(DAY);
    if (m) return `${m[2][0].toUpperCase()}${m[2].slice(1)} ${m[1] === "dua" ? "Du'a" : "Ziyarat"}`;
    return NAMES[id] || this.duas[id]?.name_en || id;
  },
  iconOf(id) {
    return ICON_OF[id] || (DAY.test(id) ? (id.startsWith("dua") ? "calendar" : "arch") : id.startsWith("ziyarat") ? "arch" : /ramadan/.test(id) ? "moon" : "star");
  },
  colorOf(id) {
    if (id in COLOR_OF) return COLORS[COLOR_OF[id]];
    let h = 0;
    for (const c of id) h = (h * 31 + c.charCodeAt(0)) >>> 0;
    return COLORS[h % COLORS.length];
  },

  // A du'a's button (today's du'as, the "Is it…" guesses): an icon and a friendly
  // name, which only kids.css shows.
  chip(button, id) {
    const friendly = this.name(id);
    const text = [...button.childNodes].find((n) => n.nodeType === Node.TEXT_NODE);
    if (text && friendly !== this.duas[id]?.name_en) {
      const grown = Object.assign(document.createElement("span"), { className: "grown" });
      text.replaceWith(grown);
      grown.append(text);
      grown.after(Object.assign(document.createElement("span"), { className: "kid-name", textContent: friendly }));
    }
    button.prepend(el(`<span class="kid-ico" style="--c:${this.colorOf(id)}">${icon(this.iconOf(id))}</span>`));
  },

  // Everything kids mode adds to the page, made once.
  build() {
    if (this.built) return;
    this.built = true;
    document.head.append(Object.assign(document.createElement("link"), { rel: "stylesheet", href: FONT }));
    const $ = (id) => document.getElementById(id);
    const sky = el('<div class="kids-sky" aria-hidden="true"></div>');
    for (const [x, y, w] of [[-7, 20, 130], [78, 12, 150], [72, 38, 92]]) sky.append(Object.assign(el(`<i class="cloud">${CLOUD}</i>`), { style: `left:${x}%;top:${y}%;width:${w}px` }));
    for (const [k, [x, y, s]] of [[20, 5, 14], [42, 9, 10], [74, 18, 16], [88, 30, 11], [9, 36, 12], [6, 24, 9], [64, 25, 9], [92, 45, 13], [16, 50, 10], [78, 56, 9]].entries()) {
      sky.append(Object.assign(el(`<i class="star">${SPARK}</i>`), { style: `left:${x}%;top:${y}%;width:${s}px;animation-delay:-${(k * 0.7) % 2.6}s` }));
    }
    sky.append(el('<i class="moon"><svg viewBox="0 0 56 56"><path d="M36 5A22 22 0 1 0 52 38 17 17 0 1 1 36 5Z"/></svg></i>'));
    document.body.prepend(sky, el('<div class="kids-town" aria-hidden="true"></div>'));

    $("home").prepend(el(`<div class="kids-hang" aria-hidden="true"><i class="string"></i><i class="glow"></i>${lantern("smile")}</div>`));
    const lock = el(`<button class="kids-exit" aria-label="Grown-ups: press and hold to leave kids mode" title="Grown-ups: press and hold">${icon("lock")}<i></i></button>`);
    const book = el(`<button class="kids-book" aria-label="My du'as">${icon("book")}<b></b></button>`);
    book.onclick = () => this.hooks.openPicker();
    $("home").append(lock, book);
    this.holdToLeave(lock); // after it's in the page: it places its hint beside it
    this.book = book;
    $("start").append(el(SEAL));

    const noor = el(`<div class="noor" aria-hidden="true"><i class="glow"></i><i class="ring"></i><i class="ring"></i><i class="ring"></i>${lantern("talk")}</div>`);
    document.querySelector(".voice").append(noor);
    const mini = el(`<div class="noor-mini" aria-hidden="true"><i class="glow"></i><i class="ring"></i><i class="ring"></i>${lantern("smile")}</div>`);
    document.body.append(mini);
    this.voiceEls = { listen: noor.querySelector(".lantern"), listenGlow: noor.querySelector(".glow"),
      follow: mini.querySelector(".lantern"), followGlow: mini.querySelector(".glow") };

    this.garland = el('<div class="garland" aria-hidden="true"><svg viewBox="0 0 100 40" preserveAspectRatio="none"><path d="M-10 -6L0 4Q50 44 100 4L110 -6"/></svg></div>');
    document.querySelector(".top").append(this.garland);

    this.shelf = el('<div class="stickers"><div class="stickers-head"><b>My du\'as</b><span class="stickers-count"></span></div><div class="stickers-grid"></div></div>');
    $("search").before(this.shelf);

    this.party = el(`<div class="celebrate" hidden>
<div class="confetti" aria-hidden="true"></div>
<div class="bubble"><span lang="ar" dir="rtl">ما شاء الله!</span><span>MashaAllah!</span></div>
<div class="hero" aria-hidden="true"><i class="glow"></i><i class="spark"></i><i class="spark"></i><i class="spark"></i>${lantern("happy")}</div>
<div class="badge"><svg viewBox="-50 -50 100 100" aria-hidden="true"><path d="${ROSETTE}"/><circle r="28"/></svg><span class="badge-ico"></span><b class="badge-count"></b></div>
<p class="badge-name"></p>
<div class="party-buttons"><button class="again">${icon("replay")}Again</button><button class="go-home">${icon("home")}Home</button></div>
</div>`);
    const confetti = this.party.querySelector(".confetti");
    for (let k = 0; k < 16; k++) {
      const shape = ["dot", "bit", "moon", "dot"][k % 4];
      const piece = el(shape === "moon" ? '<i class="moon"><svg viewBox="0 0 20 20"><path d="M13 2A8 8 0 1 0 17 15 6.5 6.5 0 1 1 13 2Z"/></svg></i>' : `<i class="${shape}"></i>`);
      piece.style.cssText = `left:${(k * 6.3 + 3) % 97}%;--c:${COLORS[k % COLORS.length]};animation-duration:${4 + (k % 5) * 0.5}s;animation-delay:-${(k * 0.83) % 5}s`;
      confetti.append(piece);
    }
    this.party.querySelector(".again").onclick = () => {
      this.party.hidden = true;
      this.hooks.again(this.partyFor);
    };
    this.party.querySelector(".go-home").onclick = () => {
      this.party.hidden = true;
      this.hooks.home();
    };
    document.body.append(this.party);
    this.updateBook();
  },

  // Leaving kids mode takes a grown-up's patience: press and hold, not a tap.
  holdToLeave(button) {
    let timer;
    const stop = () => {
      clearTimeout(timer);
      button.classList.remove("pressing");
    };
    button.addEventListener("pointerdown", (e) => {
      e.preventDefault();
      button.classList.add("pressing");
      timer = setTimeout(() => {
        stop();
        this.set(false);
      }, 1200);
    });
    for (const ev of ["pointerup", "pointerleave", "pointercancel"]) button.addEventListener(ev, stop);
    // A tap says how: a grown-up who taps it is looking for the way back.
    const hint = el('<span class="kids-exit-hint" role="status">Hold to go back to the grown-up app</span>');
    button.after(hint);
    let hintTimer;
    button.addEventListener("click", () => {
      button.classList.remove("nudge");
      void button.offsetWidth;
      button.classList.add("nudge");
      hint.classList.add("show");
      clearTimeout(hintTimer);
      hintTimer = setTimeout(() => hint.classList.remove("show"), 2600);
    });
    button.addEventListener("contextmenu", (e) => e.preventDefault());
  },

  // Meter hook: Noor sways and glows with the voice (listening), and so does the
  // small Noor in the corner while following. voice(null) puts them back.
  voice(listening, level = 0, size = 0, now = 0, still = false) {
    const els = this.voiceEls;
    if (!els) return;
    if (listening == null) {
      for (const k of ["listen", "listenGlow", "follow", "followGlow"]) els[k].style.transform = els[k].style.opacity = "";
      return;
    }
    const [body, glow] = listening ? [els.listen, els.listenGlow] : [els.follow, els.followGlow];
    glow.style.opacity = (0.45 + 0.55 * Math.min(1, level * 1.15)).toFixed(3);
    if (still) return;
    const sway = Math.sin((now / 1000) * (2 * Math.PI / 2.4)) * (2 + 5 * level);
    body.style.transform = `rotate(${sway.toFixed(2)}deg) scale(${(1 + 0.1 * size).toFixed(4)})`;
  },

  // The garland: one lantern per line, for du'as short enough to hang them all.
  hang(dua) {
    if (!this.built) return;
    const n = dua.segments.length;
    const hang = n <= 15;
    document.body.classList.toggle("has-garland", hang);
    this.garland.querySelectorAll(".g").forEach((g) => g.remove());
    this.lanterns = [];
    if (!hang) return;
    for (let i = 0; i < n; i++) {
      const x = n === 1 ? 50 : 96 - (i * 92) / (n - 1); // line 1 on the right, as the text reads
      const y = 4 + 20 * (1 - ((x - 50) / 50) ** 2);
      const g = el(`<i class="g">${MINI}</i>`);
      g.style.cssText = `left:${x}%;top:${(y - 3).toFixed(1)}px`;
      this.garland.append(g);
      this.lanterns.push(g);
    }
  },

  progress(idx) {
    this.lanterns?.forEach((g, i) => {
      g.classList.toggle("lit", i < idx);
      g.classList.toggle("now", i === idx);
    });
  },

  // The last word is said: the last lantern lights and, a moment later (unless
  // the tracker takes it back), the celebration; from the microphone, a sticker.
  finished(on, id, fromMic) {
    if (!this.on || !this.built) return;
    if (this.lanterns?.length) {
      const last = this.lanterns.at(-1);
      last.classList.toggle("lit", on);
      last.classList.toggle("now", !on && last.classList.contains("now"));
    }
    clearTimeout(this.partyTimer);
    if (!on || this.celebrated) return;
    this.partyTimer = setTimeout(() => {
      this.celebrated = true;
      const stickers = this.earned();
      if (fromMic) {
        stickers[id] = (stickers[id] || 0) + 1;
        store("kids-stickers", JSON.stringify(stickers));
      }
      this.partyFor = id;
      const n = stickers[id] || 0;
      this.party.querySelector(".badge").style.setProperty("--c", this.colorOf(id));
      this.party.querySelector(".badge-ico").innerHTML = icon(this.iconOf(id));
      this.party.querySelector(".badge-count").textContent = n ? `×${n}` : "";
      this.party.querySelector(".badge-name").textContent = this.name(id);
      this.party.hidden = false;
      this.updateBook();
    }, 1400);
  },

  reset() {
    clearTimeout(this.partyTimer);
    this.celebrated = false;
    if (this.party) this.party.hidden = true;
    this.voice(null);
    document.body.classList.toggle("kids-night", this.on && night());
  },

  earned() {
    try {
      const s = JSON.parse(store("kids-stickers") || "{}");
      return s && typeof s === "object" ? s : {};
    } catch {
      return {};
    }
  },

  updateBook() {
    const got = Object.keys(this.earned()).filter((id) => this.duas[id]).length;
    if (this.book) this.book.querySelector("b").textContent = got || "";
  },

  // The sticker book, at the top of the du'a picker: the shelf, then any other
  // du'a with a sticker. Earned ones in colour with their count; the rest waiting.
  fillShelf() {
    if (!this.built) return;
    const got = this.earned();
    const ids = [...SHELF, ...Object.keys(got).filter((id) => !SHELF.includes(id))].filter((id) => this.duas[id]);
    const grid = this.shelf.querySelector(".stickers-grid");
    grid.replaceChildren(...ids.map((id, k) => {
      const n = got[id] || 0;
      const b = el(`<button class="sticker${n ? " got" : ""}" style="--c:${this.colorOf(id)};animation-delay:-${(k * 0.7) % 4}s">
<span class="rosette"><svg viewBox="-50 -50 100 100" aria-hidden="true"><path d="${ROSETTE}"/><circle r="27"/></svg>${icon(this.iconOf(id))}${n ? `<b>×${n}</b>` : ""}</span>
<span class="sticker-name"></span></button>`);
      b.querySelector(".sticker-name").textContent = this.name(id);
      b.onclick = () => this.hooks.pick(id);
      return b;
    }));
    const earned = ids.filter((id) => got[id]).length;
    this.shelf.querySelector(".stickers-count").textContent = `${earned} / ${ids.length}`;
  },
};
