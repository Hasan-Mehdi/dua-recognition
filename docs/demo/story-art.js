// The drawings in story.html: a mosque on a Thursday night, the people inside, a boy, his father
// and the mosque's cat. Every function returns SVG markup for one moment, so story.html draws each
// frame afresh from the film's time and any frame can be drawn on its own.
//
// Coordinates are a 1920x1080 shot. People and the cat are drawn about the point on the floor
// between their knees, in their own units (an adult's head is 62 across the middle), and scaled.

export const C = {
  ink: "#2a1b16", // outlines: a warm near-black
  gold: "#d9b46a", goldHi: "#f0d596", goldDeep: "#9c7434",
  lamp: "#ffd88a", lampHot: "#fff3cf",
  cream: "#f4ecdc", creamShade: "#d8ccb6",
  black: "#2b2a33", blackHi: "#3b3a46",
  charcoal: "#3d3f4c", navy: "#28324f", grey: "#6f6f7c", olive: "#56604a", brown: "#6b4b3a",
  teal: "#3f9a8c", tealDeep: "#2c776c",
  skin: ["#e0a77c", "#c98d62", "#b77a52", "#a46a45", "#8c5a3c", "#d29a70"],
  hair: "#2b201c",
  blush: "#ff8f84",
  mouth: "#7b2b2c", tongue: "#e2786d",
  wall: "#1b2843", wallHi: "#24345a", wallLow: "#141e33",
  carpet: "#6e1f2e", carpetHi: "#86283a", carpetLow: "#561725",
  cat: "#f0a65c", catStripe: "#d07f3c", catCream: "#fbe4c4", catNose: "#e98a86",
};

export const r2 = (x) => Math.round(x * 100) / 100;
// Scales and the camera: a zoom rounded to 0.01 holds for frames and then jumps (10 px at the
// frame's edge), so these keep five places.
export const r5 = (x) => Math.round(x * 1e5) / 1e5;
export const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
export const ease = (k) => (k <= 0 ? 0 : k >= 1 ? 1 : k * k * (3 - 2 * k));
export const easeIO = (k) => (k <= 0 ? 0 : k >= 1 ? 1 : k < 0.5 ? 4 * k ** 3 : 1 - (-2 * k + 2) ** 3 / 2);
export function seeded(seed) {
  let s = seed >>> 0;
  return () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 4294967296);
}
// A blink every few seconds, at times fixed by a seed: 0 open, 1 shut (0.14 s down and up).
export function blinkAt(t, seed) {
  const rnd = seeded(seed);
  let at = 0.6 + rnd() * 2;
  while (at < t + 1) {
    const d = t - at;
    if (d >= 0 && d < 0.16) return Math.sin((d / 0.16) * Math.PI);
    at += 2.6 + rnd() * 2.8;
  }
  return 0;
}

// -- the shared paints: gradients, masks and patterns (one <defs>, in story.html) -------------
export const DEFS = `
<radialGradient id="g-glow"><stop offset="0" stop-color="#ffd38a" stop-opacity=".62"/><stop offset=".35" stop-color="#ffb860" stop-opacity=".22"/><stop offset="1" stop-color="#ff9a40" stop-opacity="0"/></radialGradient>
<radialGradient id="g-glow-soft"><stop offset="0" stop-color="#ffd9a0" stop-opacity=".34"/><stop offset=".5" stop-color="#ffb870" stop-opacity=".1"/><stop offset="1" stop-color="#ff9a40" stop-opacity="0"/></radialGradient>
<radialGradient id="g-screen"><stop offset="0" stop-color="#c9d8ff" stop-opacity=".42"/><stop offset=".5" stop-color="#9fb4ff" stop-opacity=".12"/><stop offset="1" stop-color="#8aa0ff" stop-opacity="0"/></radialGradient>
<radialGradient id="g-glass" cx=".5" cy=".45" r=".6"><stop offset="0" stop-color="#fff6d6"/><stop offset=".55" stop-color="#ffcf70"/><stop offset="1" stop-color="#e48c3a"/></radialGradient>
<radialGradient id="g-glass-off" cx=".5" cy=".45" r=".6"><stop offset="0" stop-color="#8a7350"/><stop offset="1" stop-color="#4a3a2a"/></radialGradient>
<linearGradient id="g-sky" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#050914"/><stop offset=".55" stop-color="#101839"/><stop offset=".85" stop-color="#262a5c"/><stop offset="1" stop-color="#3a3468"/></linearGradient>
<linearGradient id="g-win-sky" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#070c1c"/><stop offset=".7" stop-color="#18204a"/><stop offset="1" stop-color="#2d2d62"/></linearGradient>
<radialGradient id="g-moon"><stop offset="0" stop-color="#fff2c4" stop-opacity=".32"/><stop offset=".25" stop-color="#ffe7a8" stop-opacity=".13"/><stop offset=".55" stop-color="#ffe7a8" stop-opacity=".04"/><stop offset="1" stop-color="#ffe7a8" stop-opacity="0"/></radialGradient>
<linearGradient id="g-dome" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#8e6a2e"/><stop offset=".3" stop-color="#e9c46f"/><stop offset=".45" stop-color="#f6dc93"/><stop offset=".8" stop-color="#b9893c"/><stop offset="1" stop-color="#7d5c26"/></linearGradient>
<linearGradient id="g-wall-ext" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#57506f"/><stop offset="1" stop-color="#3a3553"/></linearGradient>
<linearGradient id="g-minaret" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#433d5b"/><stop offset=".4" stop-color="#625a7c"/><stop offset="1" stop-color="#39344e"/></linearGradient>
<linearGradient id="g-door" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#ffe9b0"/><stop offset=".6" stop-color="#ffc66c"/><stop offset="1" stop-color="#f09a45"/></linearGradient>
<linearGradient id="g-ground" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#1a1f36"/><stop offset="1" stop-color="#0a0d18"/></linearGradient>
<linearGradient id="g-wall-in" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#121c33"/><stop offset=".55" stop-color="#1d2b48"/><stop offset="1" stop-color="#22324f"/></linearGradient>
<linearGradient id="g-floor" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#4f1522"/><stop offset="1" stop-color="#7a2433"/></linearGradient>
<radialGradient id="g-bloom"><stop offset="0" stop-color="#fff4dc"/><stop offset=".4" stop-color="#ffdca0" stop-opacity=".96"/><stop offset=".75" stop-color="#ffc06a" stop-opacity=".55"/><stop offset="1" stop-color="#ffa850" stop-opacity="0"/></radialGradient>
<radialGradient id="g-warm" cx=".55" cy=".1" r=".95"><stop offset="0" stop-color="#ffcf80"/><stop offset=".6" stop-color="#e7a35a" stop-opacity=".5"/><stop offset="1" stop-color="#2a1a3a" stop-opacity="0"/></radialGradient>
<radialGradient id="g-vignette" cx=".5" cy=".5" r=".75"><stop offset=".55" stop-color="#05070c" stop-opacity="0"/><stop offset="1" stop-color="#05070c" stop-opacity=".6"/></radialGradient>
<mask id="m-crescent" maskContentUnits="userSpaceOnUse" x="-200" y="-200" width="400" height="400"><circle r="66" fill="#fff"/><circle cx="28" cy="-16" r="58" fill="#000"/></mask>
<pattern id="p-lattice" width="36" height="36" patternUnits="userSpaceOnUse"><g fill="none" stroke="#d9b46a" stroke-opacity=".55" stroke-width="1.6"><rect x="11" y="11" width="14" height="14"/><rect x="11" y="11" width="14" height="14" transform="rotate(45 18 18)"/><path d="M18 0v6M18 30v6M0 18h6M30 18h6"/></g></pattern>
<pattern id="p-lattice-dark" width="30" height="30" patternUnits="userSpaceOnUse"><g fill="none" stroke="#7a5a2c" stroke-width="2.2"><rect x="9" y="9" width="12" height="12"/><rect x="9" y="9" width="12" height="12" transform="rotate(45 15 15)"/><path d="M15 0v5M15 25v5M0 15h5M25 15h5"/></g></pattern>
<pattern id="p-tile" width="28" height="28" patternUnits="userSpaceOnUse"><rect width="28" height="28" fill="#1f5c6c"/><path d="M14 3 25 14 14 25 3 14Z" fill="none" stroke="#e6cf96" stroke-width="1.6" stroke-opacity=".8"/><circle cx="14" cy="14" r="2.4" fill="#e6cf96"/></pattern>
<pattern id="p-band" width="44" height="22" patternUnits="userSpaceOnUse"><path d="M0 11 11 2 22 11 33 2 44 11M0 11 11 20 22 11 33 20 44 11" fill="none" stroke="#d9b46a" stroke-width="1.8" stroke-opacity=".7"/></pattern>
<filter id="f-soft" x="-20%" y="-20%" width="140%" height="140%"><feGaussianBlur stdDeviation="3"/></filter>
`;

// -- small pieces -----------------------------------------------------------------------------
const sw = (w) => `stroke="${C.ink}" stroke-width="${w}" stroke-linejoin="round" stroke-linecap="round"`;

// A four-pointed sparkle, gold.
export function sparkle(x, y, size, opacity = 1, color = C.goldHi) {
  if (opacity <= 0.01 || size <= 0.1) return "";
  const a = size, b = size * 0.22;
  return `<path transform="translate(${r2(x)} ${r2(y)})" opacity="${r2(opacity)}" fill="${color}"
    d="M0 ${-a}Q${b} ${-b} ${a} 0Q${b} ${b} 0 ${a}Q${-b} ${b} ${-a} 0Q${-b} ${-b} 0 ${-a}Z"/>`;
}

// A thought bubble with a question mark, as a cartoon puts one over a puzzled head. pop 0..1.
export function puzzled(x, y, s, pop, wobble = 0) {
  if (pop <= 0.01) return "";
  const k = pop < 1 ? 1 + 0.18 * Math.sin(pop * Math.PI) : 1; // a small overshoot on the way in
  const sc = s * ease(pop) * k;
  return `<g transform="translate(${r2(x)} ${r2(y)}) rotate(${r2(wobble)}) scale(${r2(sc)})">
    <circle cx="-62" cy="58" r="9" fill="#fffaf0" ${sw(4)}/>
    <circle cx="-40" cy="36" r="14" fill="#fffaf0" ${sw(4)}/>
    <path d="M-30 -8C-46 -40 -16 -70 14 -62C30 -84 70 -76 74 -48C100 -40 98 -2 74 6C70 30 34 36 18 22C0 38 -34 26 -30 -8Z" fill="#fffaf0" ${sw(5)}/>
    <text x="22" y="10" text-anchor="middle" font-family="'Baloo 2', 'Nunito', sans-serif" font-weight="800" font-size="74" fill="${C.navy}">?</text>
  </g>`;
}

// -- people -----------------------------------------------------------------------------------
// o: x, y, s (scale), kid, turn (-1 looking left .. 1 right), sway (degrees), skin, cloth, trim,
// cap ("kufi" | "kufi-black" | "turban" | "none"), capColor, beard ("full" | "short" | null),
// eyes ("open" | "closed" | "happy"), blink 0..1, look [x, y] -1..1, brows (0 calm, 1 worried,
// -0.6 raised), mouth (0 shut .. 1 wide open), smile (0 flat .. 1), hands {l, r} as [x, y] in the
// person's units (null: in the lap), grip (how the hands hold what they hold: "thumb", a thumb
// over a book's page; "fingers", fingers round a phone's back; null: plain hands), held (markup
// over the body and under the hands), split (true: return {back, front} with the arms and hands
// apart, for a phone between the two), lit (a warm light from the side: -1 left .. 1 right, 0
// none), glow (a phone's light on the face, 0..1).
export function person(o) {
  const {
    x = 0, y = 0, s = 1, kid = false, turn = 0, sway = 0, skin = C.skin[1], cloth = C.black,
    trim = null, cap = "kufi", capColor = C.cream, beard = null, eyes = "open", blink = 0,
    look = [0, 0], brows = 0, mouth = 0, smile = 0.5, hands = null, held = "", split = false,
    glow = 0, bob = 0, headTilt = 0, hair = C.hair, pitch = 0, reading = 0, grip = null,
  } = o;
  const P = kid
    ? { R: 74, hy: -205, sx: 44, sy: -128, lap: 112, waist: 54, arm: 30, hand: 14, ow: 4.4, ua: 46, fa: 46 }
    : { R: 62, hy: -262, sx: 64, sy: -172, lap: 150, waist: 76, arm: 38, hand: 15.5, ow: 4.6, ua: 68, fa: 64 };
  const { R, sx, sy, lap, ow } = P;
  const hy = P.hy + bob;
  const hx = turn * R * 0.08;
  const off = turn * R * 0.3;
  const dark = shade(cloth, -0.3);

  // Crossed legs: two knees wide to the sides, the near shin across the front.
  const kh = lap * 0.36; // knee height
  const legs = `<path d="M${-lap} -4C${-lap - 4} ${-kh} ${-lap * 0.62} ${-kh - 18} ${-lap * 0.3} ${-kh - 14}L${lap * 0.3} ${-kh - 14}C${lap * 0.62} ${-kh - 18} ${lap + 4} ${-kh} ${lap} -4C${lap - 4} 12 ${lap * 0.6} 14 0 12C${-lap * 0.6} 14 ${-lap + 4} 12 ${-lap} -4Z" fill="${dark}" ${sw(ow)}/>
    <path d="M${-lap * 0.82} -10C${-lap * 0.4} -${kh * 0.55} ${lap * 0.35} -${kh * 0.5} ${lap * 0.8} -14" fill="none" stroke="${C.ink}" stroke-opacity=".4" stroke-width="${ow * 0.8}" stroke-linecap="round"/>`;
  // The body: shoulders, a little wider at the waist, the kurta's hem over the lap.
  const wx = P.waist, hemY = -kh * 0.42;
  const body = `<path d="M${-sx} ${sy + 4}Q0 ${sy - 12} ${sx} ${sy + 4}C${sx + 16} ${sy + 10} ${wx + 6} ${sy + 50} ${wx + 4} ${-kh - 10}C${wx + 8} ${hemY - 8} ${wx + 4} ${hemY + 4} ${wx - 10} ${hemY + 6}Q0 ${hemY + 18} ${-wx + 10} ${hemY + 6}C${-wx - 4} ${hemY + 4} ${-wx - 8} ${hemY - 8} ${-wx - 4} ${-kh - 10}C${-wx - 6} ${sy + 50} ${-sx - 16} ${sy + 10} ${-sx} ${sy + 4}Z" fill="${cloth}" ${sw(ow)}/>
    <path d="M${wx - 4} ${sy + 40}C${wx + 2} ${sy + 80} ${wx} ${-kh} ${wx - 12} ${hemY + 4}Q${wx * 0.6} ${hemY + 10} ${wx * 0.45} ${hemY + 9}C${wx * 0.6} ${-kh} ${wx * 0.62} ${sy + 80} ${wx - 4} ${sy + 40}Z" fill="#000" opacity=".12"/>
    <ellipse cx="${r2(hx + off * 0.2)}" cy="${sy + 6}" rx="${r2(sx * 0.62)}" ry="${R * 0.2}" fill="#000" opacity=".22"/>
    <path d="M${r2(off * 0.45 - 11)} ${sy - 2}Q${r2(off * 0.45)} ${sy + 15} ${r2(off * 0.45 + 11)} ${sy - 2}" fill="none" ${sw(ow * 0.7)}/>
    ${trim ? `<path d="M${r2(off * 0.45)} ${sy + 12}L${r2(off * 0.38)} ${sy + 64}" stroke="${trim}" stroke-width="${ow * 1.3}" stroke-linecap="round"/>
      ${[0, 1, 2].map((i) => `<circle cx="${r2(off * 0.44 - i * off * 0.03)}" cy="${sy + 22 + i * 17}" r="${ow * 0.75}" fill="${trim}"/>`).join("")}` : ""}`;

  // Arms: soft sleeves from the shoulder to the hand, the elbow found from the two lengths of the
  // arm (two-bone IK), out to the side.
  const rest = { l: [-lap * 0.6, -kh * 0.6], r: [lap * 0.6, -kh * 0.6] };
  const H = { l: hands?.l ?? rest.l, r: hands?.r ?? rest.r };
  const arm = (side) => {
    const sgn = side === "l" ? -1 : 1;
    const S = [sgn * (sx - P.arm * 0.32), sy + P.arm * 0.5];
    const [hxp, hyp] = H[side];
    const dx = hxp - S[0], dy = hyp - S[1];
    const d0 = Math.hypot(dx, dy) || 1;
    const a = P.ua, b = P.fa, dd = Math.min(d0, a + b - 0.5);
    const A = Math.acos(clamp((a * a + dd * dd - b * b) / (2 * a * dd), -1, 1));
    const base = Math.atan2(dy, dx);
    // Of the two elbows, the one out to this side and down, as an arm hangs.
    const e1 = [S[0] + a * Math.cos(base + A), S[1] + a * Math.sin(base + A)];
    const e2 = [S[0] + a * Math.cos(base - A), S[1] + a * Math.sin(base - A)];
    const score = (e) => e[0] * sgn + e[1] * 1.2;
    const E = score(e1) > score(e2) ? e1 : e2;
    const d = `M${r2(S[0])} ${r2(S[1])}Q${r2(E[0])} ${r2(E[1])} ${r2(hxp)} ${r2(hyp)}`;
    return `<path d="${d}" fill="none" stroke="${C.ink}" stroke-width="${P.arm + ow * 2}" stroke-linecap="round"/>
      <path d="${d}" fill="none" stroke="${cloth}" stroke-width="${P.arm}" stroke-linecap="round"/>`;
  };
  const hand = (side) => {
    const [a, b] = H[side];
    if (grip) return gripHand(a, b, P.hand, side === "l" ? 1 : -1, grip, skin, ow);
    return `<circle cx="${r2(a)}" cy="${r2(b)}" r="${P.hand}" fill="${skin}" ${sw(ow * 0.9)}/>`;
  };
  const arms = arm("l") + arm("r");
  const handsSvg = hand("l") + hand("r");

  const headSvg = head({ hx, hy, R, kid, off, turn, skin, cap, capColor, beard, eyes, blink, look, brows, mouth, smile, glow, ow, headTilt, hair, pitch, reading });
  const g = (inner) => `<g transform="translate(${r2(x)} ${r2(y)}) rotate(${r2(sway)}) scale(${r5(s)})">${inner}</g>`;
  // Fingers round a phone held out before him: his forearms are behind it, only his hands before.
  if (split && grip === "fingers") return { back: g(legs + body + headSvg + arms + held), front: g(handsSvg) };
  if (split) return { back: g(legs + body + headSvg + held), front: g(arms + handsSvg) };
  return g(legs + body + headSvg + arms + held + handsSvg);
}

// A hand holding the edge of something, at (x, y), r its size; in (1: the left hand, the thing
// to its right; -1 the right hand). "thumb": the palm at the edge and the thumb over the page;
// "fingers": the palm at the edge and three fingers round the back of a phone, toward us.
function gripHand(x, y, r, in_, grip, skin, ow) {
  const crease = shade(skin, -0.2);
  let s = "";
  if (grip === "fingers") {
    // Four fingers lying across the phone's back, outlined as one shape with the palm, a line
    // between each two.
    const fingers = [-0.78, -0.26, 0.26, 0.74].map((k, i) => ({
      y0: y + k * r * 0.9, x0: x - in_ * r * 0.3, x1: x + in_ * r * [0.95, 1.12, 1.05, 0.8][i],
    }));
    const fw = r * 0.5;
    const palm = (grow) => `<ellipse cx="${r2(x - in_ * r * 0.35)}" cy="${r2(y + r * 0.05)}" rx="${r2(r * 0.85 + grow)}" ry="${r2(r * 1.1 + grow)}" fill="${grow ? C.ink : skin}"/>`;
    for (const f of fingers) s += `<path d="M${r2(f.x0)} ${r2(f.y0)}L${r2(f.x1)} ${r2(f.y0)}" stroke="${C.ink}" stroke-width="${r2(fw + ow * 1.8)}" stroke-linecap="round"/>`;
    s += palm(ow * 0.9);
    for (const f of fingers) s += `<path d="M${r2(f.x0)} ${r2(f.y0)}L${r2(f.x1)} ${r2(f.y0)}" stroke="${skin}" stroke-width="${r2(fw)}" stroke-linecap="round"/>`;
    s += palm(0);
    for (let i = 0; i < 3; i++) {
      const ym = (fingers[i].y0 + fingers[i + 1].y0) / 2, x1 = Math.min(fingers[i].x1 * in_, fingers[i + 1].x1 * in_) * in_;
      s += `<path d="M${r2(x + in_ * r * 0.15)} ${r2(ym)}L${r2(x1)} ${r2(ym)}" stroke="${crease}" stroke-width="${r2(ow * 0.6)}" stroke-linecap="round"/>`;
    }
    return s;
  }
  // "thumb": the palm round the edge, and the thumb lying in over the page.
  const tx = x + in_ * r * 0.95, ty = y - r * 0.55;
  s += `<ellipse cx="${r2(x - in_ * r * 0.2)}" cy="${r2(y + r * 0.1)}" rx="${r2(r * 0.95)}" ry="${r2(r * 1.1)}" fill="${skin}" ${sw(ow * 0.9)}/>
    <path d="M${r2(x + in_ * r * 0.05)} ${r2(y - r * 0.3)}L${r2(tx)} ${r2(ty)}" stroke="${C.ink}" stroke-width="${r2(r * 0.6 + ow * 1.8)}" stroke-linecap="round"/>
    <path d="M${r2(x + in_ * r * 0.05)} ${r2(y - r * 0.3)}L${r2(tx)} ${r2(ty)}" stroke="${skin}" stroke-width="${r2(r * 0.6)}" stroke-linecap="round"/>`;
  return s;
}

// A phone seen from behind, held up to the face: its back, the camera, the screen's light round
// its edges. (0, 0) its centre, w x h, in the holder's units.
export function phoneBack({ x = 0, y = 0, w = 54, h = 112, glow = 1 }) {
  const rx = w * 0.17, ow = w * 0.06;
  const cam = w * 0.4;
  return `<g transform="translate(${r2(x)} ${r2(y)})">
    ${glow > 0.01 ? `<rect x="${r2(-w / 2 - 6)}" y="${r2(-h / 2 - 8)}" width="${r2(w + 12)}" height="${r2(h + 8)}" rx="${r2(rx + 6)}" fill="#bfd0ff" opacity="${r2(0.22 * glow)}" filter="url(#f-soft)"/>` : ""}
    <rect x="${r2(-w / 2)}" y="${r2(-h / 2)}" width="${r2(w)}" height="${r2(h)}" rx="${r2(rx)}" fill="#2a2e3a" ${sw(ow)}/>
    <rect x="${r2(-w / 2 + ow * 1.2)}" y="${r2(-h / 2 + ow * 1.2)}" width="${r2(w - ow * 2.4)}" height="${r2(h - ow * 2.4)}" rx="${r2(rx * 0.8)}" fill="none" stroke="#4a5062" stroke-width="${r2(ow * 0.6)}"/>
    <rect x="${r2(-w / 2 + w * 0.1)}" y="${r2(-h / 2 + w * 0.1)}" width="${r2(cam)}" height="${r2(cam)}" rx="${r2(cam * 0.3)}" fill="#1b1e27" stroke="#4a5062" stroke-width="${r2(ow * 0.5)}"/>
    <circle cx="${r2(-w / 2 + w * 0.1 + cam * 0.3)}" cy="${r2(-h / 2 + w * 0.1 + cam * 0.3)}" r="${r2(cam * 0.17)}" fill="#0b0d12" stroke="#5c6378" stroke-width="${r2(ow * 0.4)}"/>
    <circle cx="${r2(-w / 2 + w * 0.1 + cam * 0.3)}" cy="${r2(-h / 2 + w * 0.1 + cam * 0.72)}" r="${r2(cam * 0.17)}" fill="#0b0d12" stroke="#5c6378" stroke-width="${r2(ow * 0.4)}"/>
    <circle cx="${r2(-w / 2 + w * 0.1 + cam * 0.72)}" cy="${r2(-h / 2 + w * 0.1 + cam * 0.3)}" r="${r2(cam * 0.08)}" fill="#f6e7b0" opacity=".7"/>
    <path d="M${r2(w * 0.3)} ${r2(-h * 0.36)}L${r2(w * 0.3)} ${r2(h * 0.1)}" stroke="#fff" stroke-opacity=".08" stroke-width="${r2(w * 0.08)}" stroke-linecap="round"/>
  </g>`;
}

function head(o) {
  const { hx, hy, R, kid, off, turn, skin, cap, capColor, beard, eyes, blink, look, brows, mouth, smile, glow, ow, headTilt, hair, pitch, reading } = o;
  const skinDark = shade(skin, -0.18);
  const ex = R * (kid ? 0.37 : 0.35) * (1 - Math.abs(turn) * 0.1);
  // A bowed head (pitch > 0) shows its face lower down and more of its cap.
  const pz = (pitch ?? 0) * R * 0.16;
  const ey = hy + R * (kid ? 0.16 : 0.08) + pz;
  const cx = hx + off;
  const lx = look[0] * R * 0.09, ly = look[1] * R * 0.05;
  const ers = R * 0.2;
  const earShift = -turn * R * 0.1;
  let s = "";
  // Ears, then the head over them.
  for (const sgn of [-1, 1]) {
    const ex2 = hx + sgn * R * 0.95 + earShift;
    s += `<circle cx="${r2(ex2)}" cy="${r2(hy + R * 0.12 - pz * 0.4)}" r="${r2(ers)}" fill="${skin}" ${sw(ow)}/>
      <path d="M${r2(ex2 - sgn * ers * 0.1)} ${r2(hy + R * 0.04 - pz * 0.4)}q${r2(sgn * ers * 0.5)} ${r2(ers * 0.2)} 0 ${r2(ers * 0.6)}" fill="none" stroke="${skinDark}" stroke-width="${ow * 0.8}" stroke-linecap="round"/>`;
  }
  s += `<ellipse cx="${r2(hx)}" cy="${r2(hy)}" rx="${R}" ry="${r2(R * 0.95)}" fill="${skin}" ${sw(ow)}/>`;
  // A little hair at the temples under the cap.
  if (cap !== "turban") {
    for (const sgn of [-1, 1]) {
      const tx = hx + sgn * R * 0.86 + earShift * 0.6;
      s += `<path d="M${r2(tx)} ${r2(hy - R * 0.36)}q${r2(sgn * R * 0.1)} ${r2(R * 0.22)} ${r2(-sgn * R * 0.02)} ${r2(R * 0.4)}q${r2(-sgn * R * 0.08)} ${r2(-R * 0.2)} ${r2(-sgn * R * 0.06)} ${r2(-R * 0.42)}Z" fill="${hair}"/>`;
    }
  }
  // The phone's light from below.
  if (glow > 0.01) s += `<ellipse cx="${r2(cx)}" cy="${r2(hy + R * 0.45)}" rx="${R * 0.95}" ry="${R * 0.7}" fill="url(#g-screen)" opacity="${r2(glow)}"/>`;
  // Beard: around the jaw, the mouth showing through.
  if (beard) {
    const b = beard === "full" ? 1 : 0.7;
    s += `<path d="M${r2(hx - R * 0.96)} ${r2(hy + R * 0.02)}C${r2(hx - R * 0.98)} ${r2(hy + R * (0.6 + 0.4 * b))} ${r2(cx - R * 0.5)} ${r2(hy + R * (1.0 + 0.3 * b))} ${r2(cx)} ${r2(hy + R * (1.0 + 0.32 * b))}C${r2(cx + R * 0.5)} ${r2(hy + R * (1.0 + 0.3 * b))} ${r2(hx + R * 0.98)} ${r2(hy + R * (0.6 + 0.4 * b))} ${r2(hx + R * 0.96)} ${r2(hy + R * 0.02)}C${r2(hx + R * 0.8)} ${r2(hy + R * 0.5)} ${r2(cx + R * 0.45)} ${r2(hy + R * 0.42)} ${r2(cx)} ${r2(hy + R * 0.36)}C${r2(cx - R * 0.45)} ${r2(hy + R * 0.42)} ${r2(hx - R * 0.8)} ${r2(hy + R * 0.5)} ${r2(hx - R * 0.96)} ${r2(hy + R * 0.02)}Z" fill="${hair}" ${sw(ow * 0.8)}/>`;
    s += `<ellipse cx="${r2(cx)}" cy="${r2(hy + R * 0.6)}" rx="${r2(R * 0.2)}" ry="${r2(R * 0.13)}" fill="${skin}"/>`;
  }
  // Cheeks.
  for (const sgn of [-1, 1]) {
    s += `<ellipse cx="${r2(cx + sgn * (ex + R * 0.1))}" cy="${r2(ey + R * 0.24)}" rx="${r2(R * 0.13)}" ry="${r2(R * 0.075)}" fill="${C.blush}" opacity="${kid ? 0.5 : 0.3}"/>`;
  }
  // Eyes.
  const ew = R * (kid ? 0.12 : 0.1), eh = R * (kid ? 0.16 : 0.135);
  for (const sgn of [-1, 1]) {
    const x0 = cx + sgn * ex + lx, y0 = ey + ly;
    if (eyes === "closed" || (eyes === "open" && blink > 0.85)) {
      s += `<path d="M${r2(x0 - ew * 1.1)} ${r2(y0)}Q${r2(x0)} ${r2(y0 + R * 0.09)} ${r2(x0 + ew * 1.1)} ${r2(y0)}" fill="none" stroke="${C.ink}" stroke-width="${r2(R * 0.045)}" stroke-linecap="round"/>`;
    } else if (eyes === "happy") {
      s += `<path d="M${r2(x0 - ew * 1.15)} ${r2(y0 + R * 0.03)}Q${r2(x0)} ${r2(y0 - R * 0.11)} ${r2(x0 + ew * 1.15)} ${r2(y0 + R * 0.03)}" fill="none" stroke="${C.ink}" stroke-width="${r2(R * 0.05)}" stroke-linecap="round"/>`;
    } else {
      // Looking down: the eyes sit lower; reading, the lids come down to a flat line over them.
      const down = clamp((look[1] - 0.4) / 0.6, 0, 1);
      if (o.reading > 0.5 && blink < 0.5) {
        const yc = y0 + eh * 0.5;
        s += `<path d="M${r2(x0 - ew * 1.05)} ${r2(yc - eh * 0.15)}Q${r2(x0)} ${r2(yc - eh * 0.32)} ${r2(x0 + ew * 1.05)} ${r2(yc - eh * 0.15)}C${r2(x0 + ew * 1.0)} ${r2(yc + eh * 0.75)} ${r2(x0 - ew * 1.0)} ${r2(yc + eh * 0.75)} ${r2(x0 - ew * 1.05)} ${r2(yc - eh * 0.15)}Z" fill="${C.ink}"/>
          <path d="M${r2(x0 - ew * 1.3)} ${r2(yc - eh * 0.12)}Q${r2(x0)} ${r2(yc - eh * 0.36)} ${r2(x0 + ew * 1.3)} ${r2(yc - eh * 0.12)}" fill="none" stroke="${C.ink}" stroke-width="${r2(R * 0.045)}" stroke-linecap="round"/>
          <circle cx="${r2(x0 + ew * 0.35)}" cy="${r2(yc + eh * 0.12)}" r="${r2(ew * 0.26)}" fill="#fff" opacity=".9"/>`;
        continue;
      }
      const k = (1 - blink) * (1 - 0.14 * down);
      const yc = y0 + eh * 0.42 * down;
      s += `<ellipse cx="${r2(x0)}" cy="${r2(yc)}" rx="${r2(ew)}" ry="${r2(eh * k)}" fill="${C.ink}"/>`;
      if (k > 0.4) {
        s += `<circle cx="${r2(x0 + ew * 0.32)}" cy="${r2(yc - eh * 0.36 * k)}" r="${r2(ew * 0.4)}" fill="#fff"/>
          <circle cx="${r2(x0 - ew * 0.3)}" cy="${r2(yc + eh * 0.42 * k)}" r="${r2(ew * 0.17)}" fill="#fff" opacity=".8"/>`;
      }
    }
  }
  // Brows: worried lifts the inner ends; raised lifts both.
  const by = ey - R * (kid ? 0.3 : 0.27);
  const raise = brows < 0 ? -brows * R * 0.08 : 0, worry = brows > 0 ? brows * R * 0.07 : 0;
  for (const sgn of [-1, 1]) {
    const b0 = cx + sgn * ex, w = R * 0.11;
    const inner = b0 - sgn * w, outer = b0 + sgn * w;
    s += `<path d="M${r2(inner)} ${r2(by - raise - worry)}Q${r2(b0)} ${r2(by - raise - R * 0.05 - worry * 0.4)} ${r2(outer)} ${r2(by - raise + worry * 0.4)}" fill="none" stroke="${C.hair}" stroke-width="${r2(R * (kid ? 0.04 : 0.055))}" stroke-linecap="round"/>`;
  }
  // Nose, mouth.
  s += `<path d="M${r2(cx + off * 0.25 - R * 0.045)} ${r2(ey + R * 0.17)}q${r2(R * 0.045)} ${r2(R * 0.05)} ${r2(R * 0.09)} 0" fill="none" stroke="${skinDark}" stroke-width="${r2(R * 0.035)}" stroke-linecap="round"/>`;
  const mx = cx + off * 0.15, my = ey + R * (kid ? 0.34 : 0.4);
  if (mouth > 0.06) {
    const w = R * (0.13 - 0.03 * mouth), h = R * (0.05 + 0.2 * mouth);
    s += `<path d="M${r2(mx - w)} ${r2(my - R * 0.01)}Q${r2(mx)} ${r2(my - R * 0.04)} ${r2(mx + w)} ${r2(my - R * 0.01)}Q${r2(mx + w * 0.9)} ${r2(my + h)} ${r2(mx)} ${r2(my + h)}Q${r2(mx - w * 0.9)} ${r2(my + h)} ${r2(mx - w)} ${r2(my - R * 0.01)}Z" fill="${C.mouth}" ${sw(R * 0.035)}/>`;
    if (mouth > 0.3) s += `<ellipse cx="${r2(mx)}" cy="${r2(my + h * 0.7)}" rx="${r2(w * 0.55)}" ry="${r2(h * 0.25)}" fill="${C.tongue}"/>`;
  } else {
    const w = R * 0.12;
    s += `<path d="M${r2(mx - w)} ${r2(my)}Q${r2(mx)} ${r2(my + R * 0.13 * smile)} ${r2(mx + w)} ${r2(my)}" fill="none" stroke="${C.ink}" stroke-width="${r2(R * 0.04)}" stroke-linecap="round"/>`;
  }
  // Cap.
  if (cap === "kufi" || cap === "kufi-black") {
    const fill = cap === "kufi" ? capColor : "#24232b";
    const band = cap === "kufi" ? C.gold : "#8c7a5a";
    const b0 = hy - R * 0.32 + pz * 0.7, top = hy - R * 1.12;
    s += `<path d="M${r2(hx - R * 1.0)} ${r2(b0)}C${r2(hx - R * 1.04)} ${r2(top - R * 0.12)} ${r2(hx + R * 1.04)} ${r2(top - R * 0.12)} ${r2(hx + R * 1.0)} ${r2(b0)}Q${r2(hx)} ${r2(b0 + R * 0.16)} ${r2(hx - R * 1.0)} ${r2(b0)}Z" fill="${fill}" ${sw(ow)}/>
      <path d="M${r2(hx - R * 0.98)} ${r2(b0 - R * 0.16)}Q${r2(hx)} ${r2(b0)} ${r2(hx + R * 0.98)} ${r2(b0 - R * 0.16)}" fill="none" stroke="${band}" stroke-width="${r2(R * 0.07)}" stroke-dasharray="${r2(R * 0.05)} ${r2(R * 0.06)}" stroke-linecap="round"/>
      <path d="M${r2(hx - R * 0.62)} ${r2(top + R * 0.36)}Q${r2(hx - R * 0.42)} ${r2(top + R * 0.2)} ${r2(hx - R * 0.12)} ${r2(top + R * 0.16)}" fill="none" stroke="#fff" stroke-opacity="${cap === "kufi" ? 0.7 : 0.15}" stroke-width="${r2(R * 0.06)}" stroke-linecap="round"/>`;
  } else if (cap === "turban") {
    const b0 = hy - R * 0.3, top = hy - R * 1.32;
    s += `<path d="M${r2(hx - R * 1.08)} ${r2(b0)}C${r2(hx - R * 1.2)} ${r2(top)} ${r2(hx + R * 1.2)} ${r2(top)} ${r2(hx + R * 1.08)} ${r2(b0)}Q${r2(hx)} ${r2(b0 + R * 0.2)} ${r2(hx - R * 1.08)} ${r2(b0)}Z" fill="${capColor}" ${sw(ow)}/>
      ${[0.25, 0.5, 0.72].map((k) => `<path d="M${r2(hx - R * (1.0 - k * 0.2))} ${r2(b0 - R * k * 0.9)}Q${r2(hx)} ${r2(b0 - R * k * 0.9 + R * 0.28)} ${r2(hx + R * (1.0 - k * 0.2))} ${r2(b0 - R * k * 0.9 - R * 0.12)}" fill="none" stroke="${C.ink}" stroke-opacity=".28" stroke-width="${r2(R * 0.04)}" stroke-linecap="round"/>`).join("")}`;
  }
  return headTilt ? `<g transform="rotate(${r2(headTilt)} ${r2(hx)} ${r2(hy + R * 0.8)})">${s}</g>` : s;
}

// A colour lighter (k > 0) or darker (k < 0).
export function shade(hex, k) {
  const n = parseInt(hex.slice(1), 16);
  const f = (c) => Math.round(k < 0 ? c * (1 + k) : c + (255 - c) * k);
  const [r, g, b] = [f(n >> 16), f((n >> 8) & 255), f(n & 255)];
  return `#${((1 << 24) | (r << 16) | (g << 8) | b).toString(16).slice(1)}`;
}

// -- walking, seen from behind ---------------------------------------------------------------------
// Standing on (x, y), going away from us. phase: the stride (radians); hold: [x, y] in the shot
// where the near hand holds someone else's (null: both arms swing); side: which hand holds (-1
// left, 1 right).
export function walker({ x, y, s = 1, kid = false, phase = 0, cloth = C.black, skin = C.skin[1], cap = "kufi",
  hold = null, side = 1, hair = C.hair }) {
  const P = kid
    ? { R: 74, hy: -322, sx: 40, sy: -252, hem: -92, leg: 92, lw: 30, arm: 26, ow: 4.6 }
    : { R: 62, hy: -446, sx: 58, sy: -380, hem: -126, leg: 126, lw: 36, arm: 32, ow: 4.8 };
  const { R, sx, sy, hem, leg, lw, ow } = P;
  const bob = -Math.abs(Math.sin(phase)) * 9;
  const dark = shade(cloth, -0.32);
  let g = "";
  // Legs: the one stepping lifts and shortens.
  for (const sgn of [-1, 1]) {
    const lift = Math.max(0, Math.sin(phase) * sgn);
    const lx = sgn * sx * 0.38, top = hem - 20, foot = -lift * leg * 0.22;
    g += `<path d="M${r2(lx)} ${r2(top)}V${r2(foot - 8)}" stroke="${C.ink}" stroke-width="${lw + ow * 2}" stroke-linecap="round"/>
      <path d="M${r2(lx)} ${r2(top)}V${r2(foot - 8)}" stroke="${dark}" stroke-width="${lw}" stroke-linecap="round"/>
      <ellipse cx="${r2(lx)}" cy="${r2(foot - 4)}" rx="${r2(lw * 0.62)}" ry="${r2(lw * 0.36)}" fill="#1c1a22" ${sw(ow * 0.8)}/>`;
  }
  // Body: the kurta from behind.
  g += `<g transform="translate(0 ${r2(bob)})">`;
  const armPath = (sgn) => {
    const S = [sgn * (sx - P.arm * 0.4), sy + P.arm * 0.6];
    let H;
    if (hold && sgn === side) H = [(hold[0] - x) / s, (hold[1] - y) / s - bob];
    else {
      const a = Math.sin(phase) * 0.35 * -sgn;
      H = [S[0] + sgn * 18 + Math.sin(a) * 120, S[1] + Math.cos(a) * 150];
    }
    const E = [(S[0] + H[0]) / 2 + sgn * 22, (S[1] + H[1]) / 2];
    const d = `M${r2(S[0])} ${r2(S[1])}Q${r2(E[0])} ${r2(E[1])} ${r2(H[0])} ${r2(H[1])}`;
    return `<path d="${d}" fill="none" stroke="${C.ink}" stroke-width="${P.arm + ow * 2}" stroke-linecap="round"/>
      <path d="${d}" fill="none" stroke="${cloth}" stroke-width="${P.arm}" stroke-linecap="round"/>
      <circle cx="${r2(H[0])}" cy="${r2(H[1])}" r="${r2(P.arm * 0.48)}" fill="${skin}" ${sw(ow * 0.8)}/>`;
  };
  g += `<path d="M${-sx} ${sy}Q0 ${sy - 12} ${sx} ${sy}C${sx + 14} ${sy + 8} ${sx + 18} ${sy + 60} ${sx + 26} ${hem}Q0 ${hem + 14} ${-sx - 26} ${hem}C${-sx - 18} ${sy + 60} ${-sx - 14} ${sy + 8} ${-sx} ${sy}Z" fill="${cloth}" ${sw(ow)}/>
    <path d="M0 ${sy + 10}V${hem + 6}" stroke="#000" stroke-opacity=".18" stroke-width="${ow}"/>`;
  g += armPath(-1) + armPath(1);
  // The head from behind: the neck, ears, the back of the head (hair, close-cropped), the cap.
  g += `<rect x="${r2(-R * 0.3)}" y="${r2(P.hy + R * 0.6)}" width="${r2(R * 0.6)}" height="${r2(R * 0.5)}" rx="${r2(R * 0.12)}" fill="${skin}" ${sw(ow * 0.8)}/>`;
  for (const sgn of [-1, 1]) g += `<circle cx="${r2(sgn * R * 0.95)}" cy="${r2(P.hy + R * 0.12)}" r="${r2(R * 0.2)}" fill="${skin}" ${sw(ow)}/>`;
  g += `<ellipse cx="0" cy="${P.hy}" rx="${R}" ry="${r2(R * 0.95)}" fill="${shade(hair, 0.16)}" ${sw(ow)}/>
    <path d="M${r2(-R * 0.55)} ${r2(P.hy + R * 0.2)}q${r2(R * 0.1)} ${r2(R * 0.3)} ${r2(R * 0.3)} ${r2(R * 0.5)}M${r2(R * 0.55)} ${r2(P.hy + R * 0.2)}q${r2(-R * 0.1)} ${r2(R * 0.3)} ${r2(-R * 0.3)} ${r2(R * 0.5)}" fill="none" stroke="${shade(hair, 0.3)}" stroke-width="${r2(ow * 0.8)}" stroke-linecap="round"/>`;
  const b0 = P.hy - R * 0.3, top = P.hy - R * 1.12;
  const fill = cap === "kufi" ? C.cream : "#24232b";
  g += `<path d="M${r2(-R * 1.0)} ${r2(b0)}C${r2(-R * 1.04)} ${r2(top - R * 0.12)} ${r2(R * 1.04)} ${r2(top - R * 0.12)} ${r2(R * 1.0)} ${r2(b0)}Q0 ${r2(b0 + R * 0.16)} ${r2(-R * 1.0)} ${r2(b0)}Z" fill="${fill}" ${sw(ow)}/>
    <path d="M${r2(-R * 0.98)} ${r2(b0 - R * 0.16)}Q0 ${r2(b0)} ${r2(R * 0.98)} ${r2(b0 - R * 0.16)}" fill="none" stroke="${C.gold}" stroke-width="${r2(R * 0.07)}" stroke-dasharray="${r2(R * 0.05)} ${r2(R * 0.06)}" stroke-linecap="round"/>`;
  g += `</g>`;
  return `<g transform="translate(${r2(x)} ${r2(y)}) scale(${r5(s)})">${g}</g>`;
}

// -- the cat ------------------------------------------------------------------------------------
// Curled up facing left. awake 0..1 lifts the head and opens the eyes; happy closes them in arcs;
// breathe 0..1 swells the body; tail -1..1 flicks its tip.
export function cat({ x, y, s = 1, awake = 0, happy = 0, breathe = 0, tail = 0, look = 0 }) {
  const b = 1 + 0.03 * breathe;
  const hy = -78 - 40 * awake, hxx = -112 + 6 * awake;
  const ow = 4.4;
  const tip = tail * 18;
  let g = "";
  g += `<ellipse cx="0" cy="-2" rx="150" ry="16" fill="#000" opacity=".25"/>`;
  g += `<g transform="scale(1 ${r2(b)})" style="transform-origin: 0 0">
    <path d="M-120 -10C-140 -60 -100 -120 -10 -124C80 -128 140 -80 136 -36C134 -8 110 0 60 0L-90 0C-110 0 -118 -4 -120 -10Z" fill="${C.cat}" ${sw(ow)}/>
    <path d="M-20 -121C-12 -100 -14 -88 -24 -76M24 -122C32 -100 30 -86 20 -74M66 -112C70 -94 66 -82 56 -72M100 -92C100 -78 94 -68 84 -62" fill="none" stroke="${C.catStripe}" stroke-width="10" stroke-linecap="round"/>
  </g>`;
  // Tail around the front.
  g += `<path d="M126 -24C140 6 90 14 20 12C-30 10 -64 8 ${r2(-80 + tip * 0.3)} ${r2(-4 - Math.abs(tip) * 0.4)}" fill="none" stroke="${C.ink}" stroke-width="${30 + ow * 2}" stroke-linecap="round"/>
    <path d="M126 -24C140 6 90 14 20 12C-30 10 -64 8 ${r2(-80 + tip * 0.3)} ${r2(-4 - Math.abs(tip) * 0.4)}" fill="none" stroke="${C.cat}" stroke-width="30" stroke-linecap="round"/>
    <path d="M70 1L66 22M30 2L28 22" stroke="${C.catStripe}" stroke-width="9" stroke-linecap="round"/>`;
  // Front paws.
  g += `<ellipse cx="-78" cy="-10" rx="24" ry="14" fill="${C.catCream}" ${sw(ow)}/><ellipse cx="-40" cy="-8" rx="22" ry="13" fill="${C.catCream}" ${sw(ow)}/>`;
  // Head.
  const R = 52;
  const fx = hxx - 8 * look;
  g += `<g transform="translate(${r2(hxx)} ${r2(hy)})">
    <path d="M-44 -30L-50 -78L-12 -50Z" fill="${C.cat}" ${sw(ow)}/><path d="M-40 -42L-44 -66L-22 -50Z" fill="${C.catNose}" opacity=".7"/>
    <path d="M44 -30L50 -78L12 -50Z" fill="${C.cat}" ${sw(ow)}/><path d="M40 -42L44 -66L22 -50Z" fill="${C.catNose}" opacity=".7"/>
    <ellipse cx="0" cy="0" rx="${R + 6}" ry="${R - 2}" fill="${C.cat}" ${sw(ow)}/>
    <path d="M-14 -46C-12 -36 -12 -30 -14 -24M0 -50V-26M14 -46C12 -36 12 -30 14 -24" fill="none" stroke="${C.catStripe}" stroke-width="6" stroke-linecap="round"/>
    <ellipse cx="${r2(fx - hxx)}" cy="18" rx="30" ry="20" fill="${C.catCream}"/>`;
  const eo = awake > 0.5 && happy < 0.5;
  for (const sgn of [-1, 1]) {
    const ex = fx - hxx + sgn * 22, ey = 2;
    if (eo) {
      g += `<ellipse cx="${r2(ex)}" cy="${ey}" rx="8" ry="${r2(10 * clamp((awake - 0.5) * 2, 0.2, 1))}" fill="${C.ink}"/><circle cx="${r2(ex + 3)}" cy="${ey - 4}" r="3.2" fill="#fff"/>`;
    } else if (happy >= 0.5) {
      g += `<path d="M${r2(ex - 10)} ${ey + 3}Q${r2(ex)} ${ey - 9} ${r2(ex + 10)} ${ey + 3}" fill="none" stroke="${C.ink}" stroke-width="4" stroke-linecap="round"/>`;
    } else {
      g += `<path d="M${r2(ex - 10)} ${ey}Q${r2(ex)} ${ey + 8} ${r2(ex + 10)} ${ey}" fill="none" stroke="${C.ink}" stroke-width="4" stroke-linecap="round"/>`;
    }
  }
  const nx = fx - hxx;
  g += `<path d="M${r2(nx - 5)} 14L${r2(nx + 5)} 14L${r2(nx)} 19Z" fill="${C.catNose}" stroke="${C.catNose}" stroke-width="2" stroke-linejoin="round"/>
    <path d="M${r2(nx - 9)} 22q4.5 5 9 0q4.5 5 9 0" fill="none" stroke="${C.ink}" stroke-width="3" stroke-linecap="round"/>
    <path d="M${r2(nx - 24)} 16l-26 -4M${r2(nx - 24)} 22l-26 4M${r2(nx + 24)} 16l26 -4M${r2(nx + 24)} 22l26 4" stroke="${C.ink}" stroke-opacity=".5" stroke-width="2" stroke-linecap="round"/>
    <ellipse cx="${r2(nx - 30)}" cy="12" rx="9" ry="5" fill="${C.blush}" opacity=".4"/><ellipse cx="${r2(nx + 30)}" cy="12" rx="9" ry="5" fill="${C.blush}" opacity=".4"/>
  </g>`;
  return `<g transform="translate(${r2(x)} ${r2(y)}) scale(${r2(s)})">${g}</g>`;
}

// -- lanterns -----------------------------------------------------------------------------------
// Hung from (x, y) on a chain; lit 0..1; swing in degrees about the hook.
export function lantern({ x, y, s = 1, chain = 60, lit = 1, swing = 0, halo = 1 }) {
  const c = chain;
  const glass = lit > 0.02 ? "url(#g-glass)" : "url(#g-glass-off)";
  let g = `<path d="M0 0V${c}" stroke="${C.goldDeep}" stroke-width="3" stroke-dasharray="5 3"/>`;
  if (halo > 0 && lit > 0.02) g += `<circle cx="0" cy="${c + 72}" r="${r2(150 * halo)}" fill="url(#g-glow)" opacity="${r2(lit)}" style="mix-blend-mode:screen"/>`;
  g += `<circle cx="0" cy="${c + 4}" r="6" fill="none" stroke="${C.gold}" stroke-width="3"/>
    <path d="M-24 ${c + 32}Q-22 ${c + 12} 0 ${c + 10}Q22 ${c + 12} 24 ${c + 32}Z" fill="${C.gold}" stroke="${C.goldDeep}" stroke-width="2.5" stroke-linejoin="round"/>
    <path d="M-22 ${c + 32}H22L32 ${c + 72}L17 ${c + 112}H-17L-32 ${c + 72}Z" fill="${glass}" stroke="${C.goldDeep}" stroke-width="3" stroke-linejoin="round"/>
    <g opacity="${r2(0.35 + 0.65 * lit)}"><path d="M-22 ${c + 32}H22L32 ${c + 72}L17 ${c + 112}H-17L-32 ${c + 72}Z" fill="url(#g-glass)" opacity="${r2(lit)}"/></g>
    <path d="M0 ${c + 32}V${c + 112}M-11 ${c + 32}L-16 ${c + 72}L-8 ${c + 112}M11 ${c + 32}L16 ${c + 72}L8 ${c + 112}M-32 ${c + 72}H32" fill="none" stroke="${C.goldDeep}" stroke-width="2.5"/>
    <path d="M-18 ${c + 112}H18L10 ${c + 124}H-10Z" fill="${C.gold}" stroke="${C.goldDeep}" stroke-width="2.5" stroke-linejoin="round"/>
    <path d="M0 ${c + 124}V${c + 138}" stroke="${C.gold}" stroke-width="3"/><circle cx="0" cy="${c + 141}" r="4" fill="${C.gold}"/>`;
  return `<g transform="translate(${r2(x)} ${r2(y)}) rotate(${r2(swing)}) scale(${r2(s)})">${g}</g>`;
}

// -- the book -------------------------------------------------------------------------------------
// A du'a book at (x, y), w across its two open pages. sq: how much of its height we see (1: the
// pages face us, as on the reciter's stand; about 0.5: open in a lap, seen from in front and a
// little above). flip 0..1: a page turning right to left. shut 0..1: the right half, cover and
// all, closing over the left; at 1 the book lies closed, cover up, where its left half was.
const PAGE = "#fbf4e3", PAGE_BACK = "#f1e6cc", COVER = "#2c6a52";
export function book({ x = 0, y = 0, w = 200, flip = -1, rot = 0, sq = 1, shut = 0 }) {
  const h = w * 0.62 * sq, half = w / 2, k = w / 200;
  const dip = 8 * k * sq; // the pages curve down into the spine
  const th = 9 * k * (1.4 - sq); // the pages' thickness, at the near edge
  const lw = r2(w * 0.016);
  // Out of the page's plane is up the screen, by how far the book lies back.
  const up = sq < 1 ? Math.sqrt(1 - sq * sq) * 0.9 : 0.25 * sq * 0.62;
  const line = (x0, y0, len) => `<path d="M${r2(x0)} ${r2(y0)}h${r2(-len)}" stroke="#6a5a48" stroke-width="${r2(w * 0.012 * Math.max(sq, 0.7))}" stroke-linecap="round" stroke-dasharray="${r2(w * 0.05)} ${r2(w * 0.012)} ${r2(w * 0.02)} ${r2(w * 0.012)}"/>`;
  const lines = (side) => {
    let s = "";
    for (let i = 0; i < 6; i++) {
      const yy = -h * 0.36 + i * h * 0.13;
      s += side < 0 ? line(-w * 0.06, yy, half * 0.78 - (i === 5 ? half * 0.3 : 0)) : line(half * 0.88, yy, half * 0.76 - (i === 5 ? half * 0.3 : 0));
    }
    return s;
  };
  // A leaf turned about the spine (a = 0: lying right, PI: lying left): its free edge at px,
  // lifted toward the reader.
  const leaf = (a, fill, extra = "") => {
    const px = half * Math.cos(a), lift = Math.sin(a) * half * up;
    const bow = Math.sin(a) * half * 0.12 * Math.sign(Math.cos(a) || 1);
    return `<path d="M0 ${r2(-h / 2 + dip)}Q${r2(px / 2 + bow)} ${r2(-h / 2 - dip - lift)} ${r2(px)} ${r2(-h / 2 - lift)}V${r2(h / 2 - lift)}Q${r2(px / 2 + bow)} ${r2(h / 2 - dip - lift)} 0 ${r2(h / 2 + dip)}Z" fill="${fill}" ${sw(lw)}/>${extra}`;
  };
  const coverHalf = (side) => `<path d="M0 ${r2(-h / 2 + 4 * k)}H${r2(side * (half + 6 * k))}V${r2(h / 2 + th + 6 * k)}Q${r2(side * half * 0.5)} ${r2(h / 2 + th + 2 * k)} 0 ${r2(h / 2 + th + 4 * k)}Z" fill="${COVER}" ${sw(w * 0.02)}/>`;
  const block = (side) => `<path d="M0 ${r2(h / 2)}H${r2(side * half)}V${r2(h / 2 + th)}Q${r2(side * half * 0.5)} ${r2(h / 2 + th - dip * 0.5)} 0 ${r2(h / 2 + th + dip * 0.4)}Z" fill="#eadcbc" ${sw(lw)}/>`;
  const page = (side) => `<path d="M${r2(side * half)} ${r2(-h / 2)}Q${r2(side * half / 2)} ${r2(-h / 2 - dip)} 0 ${r2(-h / 2 + dip)}V${r2(h / 2 + dip)}Q${r2(side * half / 2)} ${r2(h / 2 - dip)} ${r2(side * half)} ${r2(h / 2)}Z" fill="${PAGE}" ${sw(lw)}/>${lines(side)}`;
  let g = coverHalf(-1) + (shut > 0 ? "" : coverHalf(1)) + block(-1) + (shut > 0 ? "" : block(1)) + page(-1);
  if (shut <= 0) g += page(1);
  g += `<path d="M0 ${r2(-h / 2 + dip)}V${r2(h / 2 + dip)}" stroke="#c9b897" stroke-width="${lw}"/>`;
  if (shut <= 0 && flip >= 0 && flip <= 1) {
    // The turning page shows its face until it stands up, then its back.
    const a = flip * Math.PI;
    g += leaf(a, Math.cos(a) > 0 ? PAGE : PAGE_BACK);
  }
  if (shut > 0) {
    // The right half closing: its pages until it stands up, then the cover's outside.
    const a = Math.min(1, shut) * Math.PI;
    if (Math.cos(a) > 0) g += leaf(a, PAGE);
    else {
      g += leaf(a, COVER);
      if (shut >= 0.98) {
        // The ornament on the closed cover.
        const cx = -half / 2, cy = 0;
        g += `<path d="M${r2(cx)} ${r2(cy - h * 0.22)}L${r2(cx + half * 0.18)} ${r2(cy)}L${r2(cx)} ${r2(cy + h * 0.22)}L${r2(cx - half * 0.18)} ${r2(cy)}Z" fill="none" stroke="${C.gold}" stroke-width="${r2(w * 0.014)}"/>
          <circle cx="${r2(cx)}" cy="${r2(cy)}" r="${r2(half * 0.04)}" fill="${C.gold}"/>`;
      }
    }
  }
  return `<g transform="translate(${r2(x)} ${r2(y)}) rotate(${r2(rot)})">${g}</g>`;
}

// -- outside: the mosque on a Thursday night ------------------------------------------------------
const STARS = (() => {
  const rnd = seeded(7);
  return Array.from({ length: 170 }, () => ({ x: rnd() * 1920, y: rnd() * 640, r: 0.7 + rnd() * 1.6, p: rnd() * 6.28, v: 0.6 + rnd() * 1.6 }));
})();
const BIG_STARS = [[220, 120, 14], [470, 300, 9], [760, 90, 11], [1180, 150, 10], [1320, 330, 8], [1760, 420, 9], [90, 420, 8], [1010, 60, 7]];

export function sky(t, { stars = 1 } = {}) {
  let s = `<rect width="1920" height="1080" fill="url(#g-sky)"/>`;
  for (const st of STARS) {
    const o = (0.45 + 0.55 * (0.5 + 0.5 * Math.sin(t * st.v + st.p))) * stars;
    s += `<circle cx="${r2(st.x)}" cy="${r2(st.y)}" r="${st.r}" fill="#fff6dc" opacity="${r2(o)}"/>`;
  }
  BIG_STARS.forEach(([x, y, z], i) => {
    const k = 0.75 + 0.25 * Math.sin(t * 1.3 + i * 1.7);
    s += sparkle(x, y, z * k, 0.9 * stars);
  });
  return s;
}

export function moon(x, y, s = 1) {
  return `<g transform="translate(${x} ${y}) scale(${s})"><circle r="260" fill="url(#g-moon)"/>
    <g transform="rotate(-18)"><circle r="66" fill="#fbe7b0" mask="url(#m-crescent)"/></g></g>`;
}

function palm(x, y, s, flip = 1) {
  const leaf = (a, l) => `<path transform="rotate(${a})" d="M0 0C${l * 0.3} -26 ${l * 0.7} -30 ${l} -6C${l * 0.7} -16 ${l * 0.3} -12 0 0Z" fill="#0b1124"/>`;
  return `<g transform="translate(${x} ${y}) scale(${s * flip} ${s})">
    <path d="M-10 0C-8 -120 10 -260 40 -360L56 -356C30 -260 16 -120 14 0Z" fill="#0b1124"/>
    <g transform="translate(48 -360)">${[-160, -130, -95, -60, -25, 10, 35].map((a, i) => leaf(a, 150 + (i % 2) * 30)).join("")}</g></g>`;
}

export function exterior(t, { door = 1 } = {}) {
  let s = sky(t);
  s += moon(1470, 250, 0.9);
  // The town beyond: low roofs and domes, a few lit windows.
  s += `<path d="M0 820V760h70v-30h60v40h50v-60h40l20-26 20 26h20v70h90v-36q40-50 80 0v36h60V680h50v150h40V780h60v40h40v-50h70v60H0Z" fill="#121933"/>
    <path d="M1920 820V770h-60v-50h-50v60h-80v-30q-45-55-90 0v30h-60v-80h-40v110h-50v-50h-70v70h1000Z" fill="#121933"/>`;
  const lit = [[95, 790], [150, 745], [205, 720], [370, 790], [420, 760], [510, 700], [1830, 740], [1745, 750], [1650, 790], [1515, 760], [1430, 800]];
  for (const [x, y] of lit) s += `<rect x="${x}" y="${y}" width="12" height="16" rx="2" fill="#ffcc77" opacity=".75"/>`;
  s += `<rect y="860" width="1920" height="220" fill="url(#g-ground)"/>`;
  s += palm(250, 880, 1.15) + palm(1700, 880, 1.05, -1) + palm(120, 900, 0.8, -1);
  // The path to the door, lit from it.
  s += `<path d="M845 905L700 1080H1220L1075 905Z" fill="#2a2a44"/>
    <path d="M845 905L700 1080H1220L1075 905Z" fill="url(#g-glow-soft)" opacity=".9"/>
    ${[930, 960, 995, 1040].map((y, i) => `<path d="M${r2(845 - (y - 905) * 1.0)} ${y}H${r2(1075 + (y - 905) * 1.0)}" stroke="#1d1a30" stroke-width="${2 + i}"/>`).join("")}`;
  // Low hedges and flowers either side of the path.
  for (const [hx, hy, k] of [[520, 975, 1], [380, 1010, 1.2], [1400, 975, 1], [1540, 1010, 1.2]]) {
    s += `<g transform="translate(${hx} ${hy}) scale(${k})"><path d="M-110 30q-10-50 40-50q20-30 60-10q40-20 60 10q50 0 40 50Z" fill="#14233a"/>
      ${[-70, -30, 10, 50].map((fx, i) => `<circle cx="${fx}" cy="${-8 - (i % 2) * 14}" r="6" fill="${["#e98a86", "#f0d596", "#e98a86", "#f6efe0"][i]}" opacity=".85"/>`).join("")}</g>`;
  }
  // The mosque: a gold dome, two minarets, its door open and lit.
  s += `<ellipse cx="960" cy="905" rx="300" ry="50" fill="url(#g-glow)" opacity="${r2(0.8 * door)}" style="mix-blend-mode:screen"/>`;
  for (const mx of [560, 1360]) {
    s += `<path d="M${mx - 26} 880V420h52v460Z" fill="url(#g-minaret)" stroke="#1d1a2e" stroke-width="3"/>
      <rect x="${mx - 40}" y="404" width="80" height="18" rx="4" fill="#6c6489" stroke="#1d1a2e" stroke-width="3"/>
      <path d="M${mx - 36} 404v-20h72v20" fill="none" stroke="#1d1a2e" stroke-width="3"/>
      ${[-24, -12, 0, 12, 24].map((d) => `<path d="M${mx + d} 404v-20" stroke="#1d1a2e" stroke-width="2.5"/>`).join("")}
      <path d="M${mx - 34} 384h68" stroke="#1d1a2e" stroke-width="4"/>
      <path d="M${mx - 20} 384V320h40v64Z" fill="url(#g-minaret)" stroke="#1d1a2e" stroke-width="3"/>
      <path d="M${mx - 26} 320C${mx - 36} 296 ${mx - 12} 276 ${mx} 254C${mx + 12} 276 ${mx + 36} 296 ${mx + 26} 320Z" fill="url(#g-dome)" stroke="#5d4519" stroke-width="3"/>
      <path d="M${mx} 254V232" stroke="${C.gold}" stroke-width="3"/><circle cx="${mx}" cy="230" r="4" fill="${C.goldHi}"/>
      <path d="M${mx - 8} 214a9 9 0 1 0 16 0a7 7 0 1 1 -16 0Z" fill="${C.goldHi}" transform="rotate(180 ${mx} 216)"/>
      <path d="M${mx - 26} 560h52M${mx - 26} 700h52" stroke="#2a2640" stroke-width="5"/>
      <path d="M${mx - 6} 470v-26a6 6 0 0 1 12 0v26Z" fill="#ffce7a" opacity=".85"/>
      <path d="M${mx - 6} 620v-26a6 6 0 0 1 12 0v26Z" fill="#ffce7a" opacity=".75"/>`;
  }
  // Hall, tile band and parapet.
  s += `<rect x="640" y="600" width="640" height="280" fill="url(#g-wall-ext)" stroke="#1d1a2e" stroke-width="3"/>
    <rect x="640" y="600" width="640" height="34" fill="url(#p-tile)" stroke="#1d1a2e" stroke-width="3"/>
    ${Array.from({ length: 20 }, (_, i) => `<path d="M${646 + i * 32} 600v-10a10 10 0 0 1 20 0v10Z" fill="#57506f" stroke="#1d1a2e" stroke-width="2.5"/>`).join("")}`;
  // Drum and dome.
  s += `<rect x="820" y="500" width="280" height="100" fill="url(#g-wall-ext)" stroke="#1d1a2e" stroke-width="3"/>
    <rect x="820" y="500" width="280" height="16" fill="url(#p-tile)" stroke="#1d1a2e" stroke-width="2.5"/>
    ${[870, 925, 980, 1035].map((x) => `<path d="M${x} 586v-40a15 15 0 0 1 30 0v40Z" fill="#ffd27e" opacity=".9" stroke="#1d1a2e" stroke-width="2.5"/>`).join("")}
    <path d="M810 500C760 440 790 362 872 322C922 298 952 262 960 214C968 262 998 298 1048 322C1130 362 1160 440 1110 500Z" fill="url(#g-dome)" stroke="#5d4519" stroke-width="3.5"/>
    <path d="M868 480C846 420 880 360 930 330" fill="none" stroke="#fff3cf" stroke-opacity=".55" stroke-width="7" stroke-linecap="round"/>
    <path d="M900 500C880 420 910 330 960 214M1020 500C1040 420 1010 330 960 214" fill="none" stroke="#7d5c26" stroke-opacity=".5" stroke-width="2.5"/>
    <path d="M960 214V176" stroke="${C.gold}" stroke-width="4"/><circle cx="960" cy="196" r="6" fill="${C.goldHi}"/>
    <g transform="translate(960 156) rotate(180)"><path d="M-12 0a13 13 0 1 0 24 0a10 10 0 1 1 -24 0Z" fill="${C.goldHi}"/></g>`;
  // Windows and the door.
  for (const wx of [700, 790, 1080, 1170]) {
    s += `<path d="M${wx} 800v-70a25 25 0 0 1 50 0v70Z" fill="#ffcf7c" stroke="#1d1a2e" stroke-width="3"/>
      <path d="M${wx} 800v-70a25 25 0 0 1 50 0v70Z" fill="url(#p-lattice-dark)" opacity=".7"/>`;
  }
  s += `<path d="M884 880V752C884 712 920 690 960 670C1000 690 1036 712 1036 752V880Z" fill="#2a3b63" stroke="#1d1a2e" stroke-width="3"/>
    <path d="M884 880V752C884 712 920 690 960 670C1000 690 1036 712 1036 752V880Z" fill="url(#p-tile)" opacity=".9"/>
    <path d="M904 880V758C904 728 930 710 960 694C990 710 1016 728 1016 758V880Z" fill="url(#g-door)" stroke="#1d1a2e" stroke-width="3" opacity="${r2(0.4 + 0.6 * door)}"/>
    <rect x="850" y="880" width="220" height="14" fill="#6d6688" stroke="#1d1a2e" stroke-width="2.5"/>
    <rect x="830" y="894" width="260" height="14" fill="#5a5377" stroke="#1d1a2e" stroke-width="2.5"/>`;
  s += lantern({ x: 870, y: 690, s: 0.42, chain: 20, lit: door, swing: Math.sin(t * 1.1) * 2, halo: 0.6 });
  s += lantern({ x: 1050, y: 690, s: 0.42, chain: 20, lit: door, swing: Math.sin(t * 1.1 + 1) * 2, halo: 0.6 });
  // Shoes by the steps: big pairs, and one small red pair.
  const shoe = (x, y, c, k = 1) => `<g transform="translate(${x} ${y}) scale(${k})"><path d="M-14 0q0 -9 10 -10q12 -1 18 6q2 4 -2 4Z" fill="${c}" stroke="#1d1a2e" stroke-width="2"/></g>`;
  s += shoe(808, 918, "#5a3b2a") + shoe(830, 920, "#5a3b2a") + shoe(1110, 918, "#2d2a3a") + shoe(1132, 920, "#2d2a3a") +
    shoe(1158, 924, "#c84a3a", 0.7) + shoe(1172, 926, "#c84a3a", 0.7);
  // Bushes.
  s += `<path d="M600 880q30-46 70-20q30-30 60 0q20-14 30 20Z M1190 880q20-34 50-14q30-30 60 0q30-20 50 14Z" fill="#0f1a2c"/>`;
  // Fireflies.
  const rnd = seeded(31);
  for (let i = 0; i < 14; i++) {
    const bx = 300 + rnd() * 1320, by = 760 + rnd() * 150, p = rnd() * 6.28;
    const fx = bx + Math.sin(t * 0.4 + p) * 26, fy = by + Math.cos(t * 0.55 + p * 1.3) * 14;
    const o = 0.5 + 0.5 * Math.sin(t * 2.1 + p * 3);
    s += `<circle cx="${r2(fx)}" cy="${r2(fy)}" r="9" fill="url(#g-glow)" opacity="${r2(o)}"/><circle cx="${r2(fx)}" cy="${r2(fy)}" r="2.2" fill="#fff0b0" opacity="${r2(0.4 + 0.6 * o)}"/>`;
  }
  return s;
}

// -- inside: the hall -------------------------------------------------------------------------------
// The room without its people: back wall with windows onto the night, the mihrab, the carpet's
// rows, and a string of lanterns. lit: per lantern 0..1.
export const GARLAND = 11;
export function hall(t, { lit = [], moonIn = true } = {}) {
  let s = `<rect width="1920" height="1080" fill="url(#g-wall-in)"/>`;
  // Windows onto the sky, with a lattice.
  [[560, 1], [930, 0], [1300, 0], [1670, 0]].forEach(([wx, m], i) => {
    const w = 170, top = 170, bot = 500;
    const arch = `M${wx - w / 2} ${bot}V${top + 70}C${wx - w / 2} ${top + 20} ${wx - 20} ${top} ${wx} ${top - 22}C${wx + 20} ${top} ${wx + w / 2} ${top + 20} ${wx + w / 2} ${top + 70}V${bot}Z`;
    s += `<path d="${arch}" fill="url(#g-win-sky)"/>`;
    const rnd = seeded(50 + i);
    for (let k = 0; k < 16; k++) {
      const sx = wx - w / 2 + 10 + rnd() * (w - 20), sy = top + 10 + rnd() * (bot - top - 30), p = rnd() * 6;
      s += `<circle cx="${r2(sx)}" cy="${r2(sy)}" r="${r2(1 + rnd() * 1.3)}" fill="#fff6dc" opacity="${r2(0.4 + 0.5 * (0.5 + 0.5 * Math.sin(t * 1.4 + p)))}"/>`;
    }
    if (m && moonIn) s += `<g transform="translate(${wx + 18} ${top + 110}) scale(.6)"><circle r="160" fill="url(#g-moon)"/><g transform="rotate(-18)"><circle r="66" fill="#fbe7b0" mask="url(#m-crescent)"/></g></g>`;
    s += `<path d="${arch}" fill="url(#p-lattice)" opacity=".75"/>
      <path d="${arch}" fill="none" stroke="${C.gold}" stroke-width="7"/>
      <path d="${arch}" fill="none" stroke="${C.goldDeep}" stroke-width="2" transform="translate(0 0)"/>
      <rect x="${wx - w / 2 - 16}" y="${bot}" width="${w + 32}" height="16" rx="3" fill="${C.goldDeep}"/>`;
  });
  // Mihrab, where the reciter sits.
  const mx = 230;
  s += `<path d="M${mx - 150} 712V330C${mx - 150} 230 ${mx - 60} 180 ${mx} 130C${mx + 60} 180 ${mx + 150} 230 ${mx + 150} 330V712Z" fill="#1f5c6c"/>
    <path d="M${mx - 150} 712V330C${mx - 150} 230 ${mx - 60} 180 ${mx} 130C${mx + 60} 180 ${mx + 150} 230 ${mx + 150} 330V712Z" fill="url(#p-tile)" opacity=".85"/>
    <path d="M${mx - 104} 712V350C${mx - 104} 270 ${mx - 40} 236 ${mx} 200C${mx + 40} 236 ${mx + 104} 270 ${mx + 104} 350V712Z" fill="#101a2e" stroke="${C.gold}" stroke-width="6"/>
    <path d="M${mx - 150} 712V330C${mx - 150} 230 ${mx - 60} 180 ${mx} 130C${mx + 60} 180 ${mx + 150} 230 ${mx + 150} 330V712" fill="none" stroke="${C.gold}" stroke-width="6"/>`;
  s += lantern({ x: mx, y: 200, s: 0.55, chain: 60, lit: 1, swing: Math.sin(t * 0.9) * 1.5, halo: 1.2 });
  // Band along the wall, and the lower wall.
  s += `<rect x="0" y="540" width="1920" height="26" fill="${C.wallHi}"/><rect x="0" y="542" width="1920" height="22" fill="url(#p-band)"/>
    <rect x="0" y="566" width="1920" height="146" fill="${C.wallLow}"/>
    ${Array.from({ length: 14 }, (_, i) => `<path d="M${400 + i * 120} 700V616a40 40 0 0 1 80 0V700" fill="none" stroke="${C.gold}" stroke-opacity=".25" stroke-width="3"/>`).join("")}`;
  // The carpet: flat on the floor, its rows (one for each line of people) marked by a border, a
  // small medallion now and then between them. (Raised, lit prayer arches read as rows of seats.)
  s += `<rect x="0" y="712" width="1920" height="368" fill="url(#g-floor)"/>`;
  const rows = [712, 742, 780, 830, 896, 980, 1090];
  for (let i = 0; i < rows.length - 1; i++) {
    const y0 = rows[i], y1 = rows[i + 1], h = y1 - y0, w = h * 3.2;
    s += `<path d="M0 ${r2(y0 + h * 0.06)}H1920" stroke="${C.carpetLow}" stroke-width="${r2(h * 0.14)}"/>
      <path d="M0 ${r2(y0 + h * 0.06)}H1920" stroke="${C.gold}" stroke-opacity=".28" stroke-width="${r2(0.8 + h * 0.022)}" stroke-dasharray="${r2(h * 0.1)} ${r2(h * 0.08)}"/>`;
    for (let x = (i % 2) * w * 0.5; x < 1920 + w; x += w) {
      const cy = y0 + h * 0.56, rx = h * 0.42, ry = h * 0.2;
      s += `<path d="M${r2(x - rx)} ${r2(cy)}L${r2(x)} ${r2(cy - ry)}L${r2(x + rx)} ${r2(cy)}L${r2(x)} ${r2(cy + ry)}Z" fill="${C.carpetHi}" fill-opacity=".45" stroke="${C.gold}" stroke-opacity=".22" stroke-width="${r2(0.8 + h * 0.016)}"/>`;
    }
  }
  // The lanterns, across the hall.
  for (let i = 0; i < GARLAND; i++) {
    const x = 470 + i * 140, chain = 30 + 26 * Math.sin(i * 1.9) ** 2;
    s += lantern({ x, y: -10, s: 0.62, chain: chain / 0.62, lit: lit[i] ?? 1, swing: Math.sin(t * 0.8 + i) * 1.6, halo: 1 });
  }
  return s;
}

// The low platform the reciter sits on, before the mihrab, under a white sheet: x0..x1 across,
// its top at y (the back edge) to y + 22 (the front), its front face below.
export function dais(x0, x1, y) {
  const d = 22, face = 30;
  let s = `<ellipse cx="${(x0 + x1) / 2}" cy="${y + d + face + 4}" rx="${r2((x1 - x0) * 0.56)}" ry="12" fill="#000" opacity=".25"/>
    <path d="M${x0 - 10} ${y + d}L${x0 + 6} ${y}H${x1 - 6}L${x1 + 10} ${y + d}V${y + d + face}H${x0 - 10}Z" fill="#4a2e22" ${sw(3)}/>`;
  // The sheet over its top, falling a little over the front in soft folds.
  let hem = "";
  for (let x = x0 - 10, i = 0; x < x1 + 10; x += 30, i++) hem += `Q${r2(x + 15)} ${y + d + 16 + (i % 2) * 3} ${r2(Math.min(x1 + 10, x + 30))} ${y + d + 10}`;
  s += `<path d="M${x0 - 10} ${y + d + 10}V${y + d}L${x0 + 6} ${y}H${x1 - 6}L${x1 + 10} ${y + d}V${y + d + 10}${hem}Z" fill="${C.cream}" ${sw(3)}/>
    <path d="M${x0 + 4} ${y + d - 2}H${x1 - 4}" stroke="${C.creamShade}" stroke-width="3" stroke-linecap="round"/>`;
  return s;
}

// The reciter's book stand (rahl), an open book on it, at (x, y), s.
export function rahl(x, y, s) {
  return `<g transform="translate(${x} ${y}) scale(${s})">
    <path d="M-60 0L40 -70M60 0L-40 -70" stroke="#6b4320" stroke-width="16" stroke-linecap="round"/>
    <path d="M-60 0L40 -70M60 0L-40 -70" stroke="#8a5a2c" stroke-width="10" stroke-linecap="round"/>
    ${book({ x: 0, y: -84, w: 120, rot: 0 })}</g>`;
}

// A microphone on a short stand, reaching up to (x, y).
export function micStand(x, y, s, base) {
  return `<path d="M${x} ${base}L${x} ${y + 30 * s}" stroke="#1c1c22" stroke-width="${6 * s}"/>
    <ellipse cx="${x}" cy="${base}" rx="${30 * s}" ry="${8 * s}" fill="#1c1c22"/>
    <g transform="translate(${x} ${y}) rotate(-35) scale(${s})"><rect x="-9" y="-6" width="18" height="44" rx="6" fill="#26262e" stroke="${C.ink}" stroke-width="3"/>
    <circle cx="0" cy="-10" r="15" fill="#9aa0ad" stroke="${C.ink}" stroke-width="3"/><path d="M-10 -14h20M-12 -8h24M-10 -2h20" stroke="#5d6270" stroke-width="2"/></g>`;
}
