#!/usr/bin/env python
"""The scenario bench's noise cells, plus a phone in a real hall: measured rooms, a PA, a crowd.

bench.py's hall and far cells reverberate the reader's own voice with a synthetic response (a
noise tail with the direct sound on top), the same family the models were trained on. These
cells use rooms measured with a microphone (OpenSLR 28's real responses, data/testbed/rirs/real;
none used in training), and a masjid evening: the reciter through a PA, 1-3 loudspeakers heard
from one seat, people talking and a fan in the same hall.

    rhall        a measured large room (Aula Carolina T20 ~4.5 s, RWCP E2B ~2 s, a lecture hall,
                 the REVERB challenge's large room, RWCP E1C), the voice alone
    masjid       PA chain -> 1-3 loudspeakers in one of those rooms -> people talking (real
                 speech clips, 12 dB under the voice, in the same room) + fan noise at 22 dB
    <cell>_wpe   the same audio after WPE dereverberation (halls.wpe), as a front end: offline,
                 the whole recording at once (the best a front end can do)
    <cell>_owpe  ...causal WPE (halls.wpe_online), frame by frame as a phone could run it

Items go to data/testbed/items_noise/<cell>/, apart from the reading grid, so `bench.py score`
without cell names never sees them. Everything else is bench.py's: name cells to its commands.

    python scripts/noise_bench.py build                       # the cells above
    python scripts/noise_bench.py front owpe hall rhall ...   # copies of cells through a front end
    python scripts/noise_bench.py variant masjid masjid_room pa,crowd,hum   # a cell without some effects
    python scripts/noise_bench.py asr rhall masjid hall_wpe --split dev [--model ... --ctc-model ...]
    python scripts/noise_bench.py score rhall masjid hall hall_wpe --name x --display stream --split dev
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench  # noqa: E402
import halls  # noqa: E402

ITEMS = bench.BENCH / "items_noise"
RIRS = bench.BENCH / "rirs" / "real"
# Measured rooms with a reverberation time of a hall (T20 in rirs/real_index.json); the AIR
# "phone" and RWCP type4 responses are left out (their decay runs into the noise floor).
REAL_HALLS = ("air_type1_air_binaural_aula_carolina", "RWCP_type1_rir_cirline_e2b", "RWCP_type2_rir_cirline_e2b",
              "air_type1_air_binaural_lecture", "RVB2014_type1_rir_largeroom2", "RWCP_type1_rir_circle_e1c")
CELLS = ("rhall", "masjid")
FRONT = ("wpe",)
STD = {"reverb", "noise", "babble", "bgrecite", "gain", "clip", "codec"}


def _rooms() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for f in sorted(RIRS.glob("*.wav")):
        for room in REAL_HALLS:
            if f.name.startswith(room):
                out.setdefault(room, []).append(f.name)
    return out


_RIR_CACHE: dict = {}


def load_rir(name: str) -> np.ndarray:
    """A measured response, first channel, its direct sound moved to t=0 (labels stay aligned)."""
    if name not in _RIR_CACHE:
        import soundfile as sf

        x, sr = sf.read(RIRS / name, always_2d=True)
        assert sr == bench.SR, (name, sr)
        h = x[:, 0].astype(np.float64)
        k = int(np.argmax(np.abs(h)))
        h = h[max(0, k - 16):]
        e = np.cumsum(h[::-1] ** 2)[::-1]
        end = int(np.argmax(10 * np.log10(e / e[0] + 1e-20) < -60)) or h.size  # cut at -60 dB
        _RIR_CACHE[name] = (h[:end] / np.abs(h).max()).astype(np.float32)
    return _RIR_CACHE[name]


def _mix_speakers(spec: list) -> np.ndarray:
    """[[rir file, delay s, gain], ...] -> one response: each loudspeaker's path to the seat."""
    hs = [(load_rir(f), int(d * bench.SR), g) for f, d, g in spec]
    n = max(h.size + d for h, d, _ in hs)
    out = np.zeros(n, np.float32)
    for h, d, g in hs:
        out[d : d + h.size] += g * h
    return out


def render(it: dict) -> np.ndarray:
    """bench.render plus this file's effects. Standard effects come first and are bench.py's own
    (so hall_wpe hears exactly hall's audio before WPE); then pa / room / crowd / hum / wpe."""
    fx = it["fx"]
    k = 0
    while k < len(fx) and fx[k][0] in STD:
        k += 1
    if any(e[0] in STD for e in fx[k:]):
        raise ValueError(f"{it['id']}: bench effects must come before this file's")
    y = _orig_render({**it, "fx": fx[:k]})
    rng = np.random.default_rng(it.get("seed", 0) + 1)
    for e in fx[k:]:
        if e[0] == "pa":
            y = halls.pa(y, rng)
        elif e[0] == "room":  # ["room", [[file, delay, gain], ...]]
            y = halls.convolve(y, _mix_speakers(e[1]))
        elif e[0] == "crowd":  # ["crowd", snr, [clips], rir file]
            parts = []
            for c in e[2]:
                x = np.load(bench.ROOT / c).astype(np.float32)
                x = x / 32768.0 if np.abs(x).max() > 2.0 else x
                x = x / (np.sqrt(np.mean(x ** 2)) + 1e-9)
                parts.append(np.roll(np.tile(x, int(np.ceil(len(y) / max(1, len(x))))), int(rng.integers(0, len(x))))[: len(y)])
            nz = halls.convolve(np.sum(parts, axis=0).astype(np.float32), load_rir(e[3]))
            y = halls.add_at_snr(y, nz, e[1])
        elif e[0] == "hum":  # ["hum", snr]
            y = halls.add_at_snr(y, bench.colored_noise(len(y), rng, "brown"), e[1])
        elif e[0] == "wpe":
            y = halls.wpe(y, taps=40, delay=2, n_fft=512, hop=256)
        elif e[0] == "owpe":
            y = halls.wpe_online(y, taps=20, delay=2, alpha=0.9995, n_fft=512, hop=256)
        else:
            raise ValueError(f"unknown effect {e[0]}")
    pk = np.abs(y).max()
    return (y / pk * 0.9 if pk > 1.0 else y).astype(np.float32)


def load_items(scenarios=None, split: str = "all", practice: bool = False) -> list[dict]:
    """bench.load_items, with this file's cells taken from items_noise/ when named."""
    names = list(scenarios or [])
    std = [s for s in names if (bench.BENCH / "items" / s).is_dir()]
    out = _orig_load_items(std, split, practice) if std or not names else []
    for s in names:
        if s in std:
            continue
        for f in sorted((ITEMS / s).glob("*.json")):
            it = json.loads(f.read_text(encoding="utf-8"))
            if split == "all" or bench.split_of(it["voice"]) == split:
                out.append(it)
    return out


def cmd_build(args) -> None:
    import evaluate as ev

    ix = ev.CorpusIndex(ev.load_all())
    rooms = _rooms()
    print({r: len(v) for r, v in rooms.items()})
    srcs = [s for s in (bench.fresh_source(s, ix) for s in bench.load_sources()) if s is not None]
    known = [s for s in srcs if s["dua"]]
    talk = bench._talk_clips()
    hv_by_voice: dict = {}
    for s in known:
        if s["lane"] == "harvest":
            hv_by_voice.setdefault(s["voice"], []).append(s)
    for ci, name in enumerate(CELLS):
        pool = [s for s in known if s["lane"] == "studio"]
        voices = sorted(hv_by_voice)
        rng = random.Random(100 + ci)
        rng.shuffle(voices)
        for v in voices[: args.harvest_per_cell]:
            runs = sorted(hv_by_voice[v], key=lambda s: s["sid"])
            pool.append(runs[(100 + ci) % len(runs)])
        out = ITEMS / name
        out.mkdir(parents=True, exist_ok=True)
        n = 0
        for s in pool:
            rng = random.Random(f"{name}|{s['sid']}")
            p = bench.sc_cond(s, ix, rng)
            if p is None or p.t < 20.0:
                continue
            room = rng.choice(sorted(rooms))
            files = rooms[room]
            if name == "rhall":
                fx = [["room", [[rng.choice(files), 0.0, 1.0]]]]
            else:
                k = rng.choice([1, 2, 3])
                spk = [[f, 0.0 if j == 0 else round(rng.uniform(0.005, 0.06), 4), 1.0 if j == 0 else round(rng.uniform(0.3, 0.9), 2)]
                       for j, f in enumerate(rng.sample(files, min(k, len(files))))]
                fx = [["pa"], ["room", spk], ["crowd", 12.0, [c["clip"] for c in rng.sample(talk, 6)], rng.choice(files)],
                      ["hum", 22.0]]
            it = p.item(name, fx, 1.0, extra={"seed": rng.randrange(1 << 30), "room": room})
            for nm, ff in ((name, fx), (f"{name}_wpe", fx + [["wpe"]])):
                d = ITEMS / nm
                d.mkdir(parents=True, exist_ok=True)
                jt = dict(it, id=it["id"].replace(f"{name}-", f"{nm}-", 1), scenario=nm, fx=ff)
                (d / f"{jt['id']}.json").write_text(json.dumps(jt, ensure_ascii=False), encoding="utf-8")
            n += 1
        print(f"{name:8s} {n} items (+ {name}_wpe)", flush=True)
    for name in args.front_of:  # bench.py cells again, through the front end
        d = ITEMS / f"{name}_wpe"
        d.mkdir(parents=True, exist_ok=True)
        n = 0
        for f in sorted((bench.BENCH / "items" / name).glob("*.json")):
            it = json.loads(f.read_text(encoding="utf-8"))
            jt = dict(it, id=it["id"].replace(f"{name}-", f"{name}_wpe-", 1), scenario=f"{name}_wpe",
                      fx=it["fx"] + [["wpe"]])
            (d / f"{jt['id']}.json").write_text(json.dumps(jt, ensure_ascii=False), encoding="utf-8")
            n += 1
        print(f"{name}_wpe {n} items", flush=True)


_orig_load_items = bench.load_items
_orig_render = bench.render


def cmd_front(front: str, cells: list[str]) -> None:
    """<cell>_<front>: a cell's items (bench.py's or this file's) with the front end applied last."""
    for name in cells:
        src = bench.BENCH / "items" / name
        src = src if src.is_dir() else ITEMS / name
        d = ITEMS / f"{name}_{front}"
        d.mkdir(parents=True, exist_ok=True)
        n = 0
        for f in sorted(src.glob("*.json")):
            it = json.loads(f.read_text(encoding="utf-8"))
            jt = dict(it, id=it["id"].replace(f"{name}-", f"{name}_{front}-", 1), scenario=f"{name}_{front}",
                      fx=it["fx"] + [[front]])
            (d / f"{jt['id']}.json").write_text(json.dumps(jt, ensure_ascii=False), encoding="utf-8")
            n += 1
        print(f"{name}_{front} {n} items", flush=True)


def cmd_variant(src: str, new: str, drop: list[str]) -> None:
    """<new>: the items of cell <src> without some of its effects (same rooms, seeds and clips)."""
    d = ITEMS / new
    d.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted((ITEMS / src).glob("*.json")):
        it = json.loads(f.read_text(encoding="utf-8"))
        jt = dict(it, id=it["id"].replace(f"{src}-", f"{new}-", 1), scenario=new,
                  fx=[e for e in it["fx"] if e[0] not in drop])
        (d / f"{jt['id']}.json").write_text(json.dumps(jt, ensure_ascii=False), encoding="utf-8")
        n += 1
    print(f"{new} {n} items ({src} without {', '.join(drop)})", flush=True)


def main() -> None:
    if len(sys.argv) > 4 and sys.argv[1] == "variant":  # variant SRC NEW effect,effect
        cmd_variant(sys.argv[2], sys.argv[3], sys.argv[4].split(","))
        return
    if len(sys.argv) > 2 and sys.argv[1] == "front":
        cmd_front(sys.argv[2], sys.argv[3:])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        import argparse

        ap = argparse.ArgumentParser()
        ap.add_argument("cmd")
        ap.add_argument("--harvest-per-cell", type=int, default=30)
        ap.add_argument("--front-of", nargs="*", default=["hall", "far", "flow"],
                        help="bench.py cells to copy through WPE")
        cmd_build(ap.parse_args())
        return
    # Everything else is bench.py's command, with these cells known to it.
    extra = [d.name for d in sorted(ITEMS.glob("*")) if d.is_dir()]
    for name in extra:
        bench.SCENARIOS.setdefault(name, (None, [], (), 1.0))
    bench.load_items = load_items
    bench.render = render
    bench.main()


if __name__ == "__main__":
    main()
