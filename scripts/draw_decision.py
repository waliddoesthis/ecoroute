"""Draw one saved routing decision as the README's worked example (SVG).

    python scripts/draw_decision.py --decisions reports/decisions_v14.json \\
        --prompt "4k+3" --out docs/img/decision-example.svg

Reads the output of scripts/decision_example.py, so every number in the figure comes from
a real route() call. The figure follows the graph: prompt -> skill -> difficulty ->
model, with the edge weights the graph used, then each cleared model's calibrated
P(correct) against the profile's threshold, its cost, and the choice.
"""

from __future__ import annotations

import argparse
import json
from html import escape
from pathlib import Path

W, H = 1200, 770
FONT = '-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif'
STYLE = f"""
.bg{{fill:#fcfcfb}}.t1{{fill:#0b0b0b}}.t2{{fill:#52514e}}.t3{{fill:#8a8984}}
.card{{fill:#ffffff;stroke:#e4e3df}}.node{{fill:#f3f2ee;stroke:#d6d5d0}}
.edge{{stroke:#cfcec8;fill:none}}.hot{{stroke:#2a78d6;fill:none}}.hotfill{{fill:#2a78d6}}
.hotnode{{fill:#e8f1fc;stroke:#2a78d6}}.track{{fill:#eeede9}}.bar{{fill:#9fc3ee}}
.barhot{{fill:#2a78d6}}.tau{{stroke:#d64545}}.taut{{fill:#d64545}}.ok{{fill:#1f8a4c}}
.chip{{fill:#f3f2ee;stroke:#d6d5d0}}.pick{{fill:#e8f1fc;stroke:#2a78d6}}.grid{{stroke:#ecebe7}}
text{{font-family:{FONT}}}
@media (prefers-color-scheme:dark){{
.bg{{fill:#1a1a19}}.t1{{fill:#ffffff}}.t2{{fill:#c3c2b7}}.t3{{fill:#8f8e86}}
.card{{fill:#222220;stroke:#3a3a37}}.node{{fill:#2a2a28;stroke:#45453f}}
.edge{{stroke:#4a4a45}}.hot{{stroke:#5a9df0}}.hotfill{{fill:#5a9df0}}
.hotnode{{fill:#1d2f47;stroke:#5a9df0}}.track{{fill:#2e2e2b}}.bar{{fill:#35567d}}
.barhot{{fill:#5a9df0}}.tau{{stroke:#f07a7a}}.taut{{fill:#f07a7a}}.ok{{fill:#4cc27f}}
.chip{{fill:#2a2a28;stroke:#45453f}}.pick{{fill:#1d2f47;stroke:#5a9df0}}.grid{{stroke:#2c2c29}}}}
"""


def t(x, y, s, cls="t1", size=13, weight=400, anchor="start", extra=""):
    return (
        f'<text class="{cls}" x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
        f'font-weight="{weight}" text-anchor="{anchor}"{extra}>{escape(str(s))}</text>'
    )


def chip(x, y, label, value, cls="chip"):
    w = 9 + 6.6 * (len(label) + len(value)) + 14
    return (
        f'<rect class="{cls}" x="{x}" y="{y}" width="{w:.0f}" height="26" rx="13"/>'
        + t(x + 12, y + 17.5, label, "t2", 12)
        + t(x + 12 + 6.6 * len(label) + 4, y + 17.5, value, "t1", 12, 600)
    ), w


def curve(x1, y1, x2, y2, cls, width, extra=""):
    mx = (x1 + x2) / 2
    return (
        f'<path class="{cls}" d="M{x1:.1f},{y1:.1f} C{mx:.1f},{y1:.1f} {mx:.1f},{y2:.1f} '
        f'{x2:.1f},{y2:.1f}" stroke-width="{width:.2f}"{extra}/>'
    )


def pretty_skill(s: str) -> str:
    return s.replace("_", " ").replace("-", " ")


def money(usd: float) -> str:
    return f"${usd * 1000:.2f}" if usd >= 0.00001 else "$0.00"


def why(rec: dict) -> str:
    """The decisive step in one line (the router's own wording is longer)."""
    pick = next(c for c in rec["candidates"] if c["name"] == rec["model"])
    if not pick["qualifies"]:
        return rec["steps"][-1][:150]
    allowed = [c for c in rec["candidates"] if c["allowed"]]
    top = max(allowed, key=lambda c: c["cost_usd"])
    saved = 1 - pick["cost_usd"] / top["cost_usd"]
    return (
        f"{rec['model']} is the cheapest model whose predicted P(correct), "
        f"{pick['p_success']:.2f}, reaches the {rec['profile']} threshold {rec['tau']:.2f}. "
        f"It costs {saved:.0%} less than {top['name']}."
    )


def draw(rec: dict, others: list[dict]) -> str:
    g = rec["graph"]
    chosen = rec["model"]
    anchor = rec["anchors"].get(chosen, {}).get("model")
    cands = sorted(rec["candidates"], key=lambda c: c["cost_usd"])
    tau = rec["tau"]
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" role="img" aria-labelledby="t d">',
        '<title id="t">How EcoRoute routed one request</title>',
        f'<desc id="d">Prompt: {escape(rec["prompt"])} Profile {rec["profile"]}, level '
        f"{rec['level']}. Chosen model {chosen}. {escape(rec['steps'][-1])}</desc>",
        f"<style>{STYLE}</style>",
        f'<rect class="bg" width="{W}" height="{H}" rx="10"/>',
        t(32, 44, "How EcoRoute routes one request", "t1", 22, 650),
        t(
            32,
            68,
            "Real output of route() with router v14. Edge labels are the weights the "
            "decision graph used for this prompt; blue marks the paths that decided it.",
            "t2",
            13.5,
        ),  # fmt: skip
    ]

    # Prompt card and the decision summary chips.
    out.append(f'<rect class="card" x="32" y="86" width="{W - 64}" height="92" rx="10"/>')
    out.append(t(52, 112, "PROMPT", "t3", 11, 600, extra=' letter-spacing="1"'))
    out.append(t(52, 138, f"“{rec['prompt']}”", "t1", 17, 500))
    x = 52
    cleared = sum(c["allowed"] for c in rec["candidates"])
    diff = rec["difficulty"]
    for label, value, cls in [
        ("Profile", f"{rec['profile']} (threshold {tau:.2f})", "chip"),
        ("Sensitivity", f"{rec['level']}, {cleared} of {len(cands)} models cleared", "chip"),
        ("Difficulty", f"{diff:.2f} ({rec['difficulty_label']})", "chip"),
        ("Routed to", chosen, "pick"),
    ]:
        s, w = chip(x, 146, label, value, cls)
        out.append(s)
        x += w + 10

    # Column positions of the graph.
    top, bottom = 230, 600
    px, sx, lx, mx = 70, 250, 460, 625
    bx, bw, cx = mx + 175, 170, 1018  # P(correct) bars, cost column
    out += [
        t(sx + 70, 214, "SKILL, P(skill | prompt)", "t3", 11, 600, "middle"),
        t(lx + 50, 214, "DIFFICULTY, P(level | prompt)", "t3", 11, 600, "middle"),
        t(mx + 4, 214, "CLEARED MODEL", "t3", 11, 600, "start"),
        t(bx, 214, "P(CORRECT), CALIBRATED", "t3", 11, 600, "start"),
        t(cx, 214, "$ / 1K REQ.", "t3", 11, 600, "start"),
    ]

    # Strongest paths to the chosen model's anchor: these get highlighted.
    paths = g["edges"].get(anchor, [])[:2] if anchor else []
    hot_skills = {p[0] for p in paths}
    hot_levels = {p[1] for p in paths}

    skills = [(name, w) for name, w in g["skills"][:3] if w >= 0.01]
    rest = 1 - sum(w for _, w in skills)
    nodes_s = skills + ([("other skills", rest)] if rest > 0.005 else [])
    gap = min(110, (bottom - top - 60) / max(1, len(nodes_s) - 1))
    mid = (len(nodes_s) - 1) / 2
    ys = [(top + bottom) / 2 - 20 + (i - mid) * gap for i in range(len(nodes_s))]
    levels = list(g["levels"].items())
    yl = [top + 70 + i * (bottom - top - 140) / 2 for i in range(len(levels))]
    py = (top + bottom) / 2 - 20
    rows = len(cands)
    row_h = (bottom - top + 30) / rows
    ym = {c["name"]: top + 14 + i * row_h for i, c in enumerate(cands)}

    # Edges first so nodes sit on top.
    for (name, w), y in zip(nodes_s, ys):
        hot = name in hot_skills
        out.append(curve(px + 46, py, sx, y, "hot" if hot else "edge", 1.5 + 14 * w))
        out.append(t(sx - 8, y - 7, f"{w:.2f}", "t2", 11.5, 600, "end"))
    for (name, _), y in zip(nodes_s, ys):
        for (lv, pl), y2 in zip(levels, yl):
            hot = (
                name in hot_skills
                and lv in hot_levels
                and any(p[0] == name and p[1] == lv for p in paths)
            )
            out.append(curve(sx + 140, y, lx, y2, "hot" if hot else "edge",
                             2.5 if hot else 0.8))  # fmt: skip
    for (lv, _), y in zip(levels, yl):
        hot = lv in hot_levels
        out.append(curve(lx + 100, y, mx - 8, ym[chosen], "hot" if hot else "edge",
                         2.5 if hot else 0.8, "" if hot else ' stroke-dasharray="3 4"'))  # fmt: skip

    # Similar-prompt edge, straight from the prompt to the chosen model.
    nb = g.get("neighbours", {}).get(anchor) if anchor else None
    if g.get("beta", 0) > 0 and nb:
        solved, answered, _ = nb
        yb = bottom + 92
        out.append(
            f'<path class="hot" d="M{px},{py + 22} L{px},{yb} L{mx - 30},{yb} '
            f'C{mx - 15},{yb} {mx - 22},{ym[chosen]} {mx - 8},{ym[chosen]}" '
            f'stroke-width="1.6" stroke-dasharray="5 4"/>'
        )
        out.append(t(px + 12, yb - 8, f"Similar past prompts: {anchor} solved {solved} of the "
                     f"{answered} nearest ones (edge weight {g['beta']:.2f})", "t2", 12))  # fmt: skip

    # Nodes.
    out.append(
        f'<rect class="hotnode" x="{px - 46}" y="{py - 22}" width="92" height="44" rx="22"/>'
    )
    out.append(t(px, py + 5, "prompt", "t1", 13, 600, "middle"))
    for (name, w), y in zip(nodes_s, ys):
        cls = "hotnode" if name in hot_skills else "node"
        out.append(f'<rect class="{cls}" x="{sx}" y="{y - 17}" width="140" height="34" rx="8"/>')
        out.append(t(sx + 70, y + 4.5, pretty_skill(name)[:20], "t1", 12.5, 500, "middle"))
    for (lv, pl), y in zip(levels, yl):
        cls = "hotnode" if lv in hot_levels else "node"
        out.append(f'<rect class="{cls}" x="{lx}" y="{y - 17}" width="100" height="34" rx="8"/>')
        out.append(t(lx + 50, y + 4.5, f"{lv} {pl:.2f}", "t1", 12.5, 500, "middle"))
    if paths:
        via = " and ".join(f"{a:.0%} of {pretty_skill(s)}/{lv}" for s, lv, a, _ in paths)
        out.append(t(px + 12, bottom + 36, f"Strongest paths: {chosen} is scored through the "
                     f"benchmark model {anchor},", "t2", 12))  # fmt: skip
        out.append(t(px + 12, bottom + 54, f"which solved {via} training prompts.", "t2", 12))

    # Model rows: P(correct) bar against the threshold, cost, status.
    for c in cands:
        y = ym[c["name"]]
        is_pick = c["name"] == chosen
        if is_pick:
            out.append(f'<rect class="pick" x="{mx - 8}" y="{y - 15}" width="{W - mx - 24}" '
                       f'height="30" rx="7"/>')  # fmt: skip
        name_cls = "t1" if c["allowed"] else "t3"
        out.append(t(mx + 4, y + 4.5, c["name"], name_cls, 13, 650 if is_pick else 500))
        out.append(f'<rect class="track" x="{bx}" y="{y - 6}" width="{bw}" height="12" rx="6"/>')
        p = c["p_success"]
        if c["allowed"] and p is not None:
            cls = "barhot" if is_pick else "bar"
            out.append(f'<rect class="{cls}" x="{bx}" y="{y - 6}" width="{bw * p:.1f}" '
                       f'height="12" rx="6"/>')  # fmt: skip
            out.append(t(bx + bw + 8, y + 4.5, f"{p:.2f}", "t1" if is_pick else "t2", 12, 600))
        else:
            out.append(t(bx + 8, y + 4, "not cleared for this data", "t3", 11))
        out.append(t(cx, y + 4.5, money(c["cost_usd"]), "t1" if is_pick else "t2", 12.5))
        status = (
            "chosen" if is_pick else "passes" if c["qualifies"] else
            "below threshold" if c["allowed"] else "excluded"
        )  # fmt: skip
        out.append(t(W - 40, y + 4.5, status, "ok" if is_pick else "t3", 12,
                     650 if is_pick else 400, "end"))  # fmt: skip
    # Threshold line across the bars.
    tx = bx + bw * tau
    y0, y1 = ym[cands[0]["name"]] - 16, ym[cands[-1]["name"]] + 16
    out.append(f'<line class="tau" x1="{tx:.1f}" x2="{tx:.1f}" y1="{y0}" y2="{y1}" '
               f'stroke-width="1.5" stroke-dasharray="4 3"/>')  # fmt: skip
    out.append(t(tx, y1 + 14, f"threshold {tau:.2f}", "taut", 11, 600, "middle"))

    # Why: the decisive step, and the same prompt under the other profiles.
    out.append(f'<rect class="card" x="32" y="{H - 62}" width="{W - 64}" height="44" rx="10"/>')
    out.append(t(52, H - 34, "Why:", "t1", 13.5, 650))
    out.append(t(92, H - 34, why(rec), "t2", 13.5))
    if others:
        alt = ",  ".join(f"{o['profile']} → {o['model']}" for o in others)
        out.append(t(W - 52, 112, f"Same prompt under the other profiles: {alt}", "t2", 12,
                     400, "end"))  # fmt: skip
    out.append("</svg>")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decisions", type=Path, default=Path("reports/decisions.json"))
    parser.add_argument("--prompt", required=True, help="part of the example prompt to draw")
    parser.add_argument("--profile", default="balanced")
    parser.add_argument("--out", type=Path, default=Path("docs/img/decision-example.svg"))
    args = parser.parse_args()
    recs = [r for r in json.loads(args.decisions.read_text()) if args.prompt in r["prompt"]]
    rec = next(r for r in recs if r["profile"] == args.profile)
    others = [r for r in recs if r["profile"] != args.profile]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(draw(rec, others))
    print(f"saved {args.out}: {rec['prompt']} -> {rec['model']} ({rec['profile']})")


if __name__ == "__main__":
    main()
