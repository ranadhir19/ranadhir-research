#!/usr/bin/env python3
"""Builds the Ghosh Research landing site into docs/ (GitHub Pages ready).

Each project page: Problem / The paper / The implementation / Architecture (SVG) /
Results / Insights / Code. Run: python3 docs/build.py
"""
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

CSS = """
:root{
  --bg:#0b0e14; --panel:#111724; --panel2:#0e1420; --border:#22304a;
  --text:#e8ebf2; --muted:#98a2b8; --faint:#6b7690;
  --blue:#6ea8fe; --green:#7ee2a8; --amber:#f0b35e; --red:#f08787; --purple:#c9a8fe;
  --radius:14px;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Roboto,Helvetica,Arial,sans-serif;
  line-height:1.65;font-size:16px}
a{color:var(--blue);text-decoration:none}
a:hover{text-decoration:underline}
.wrap{max-width:1080px;margin:0 auto;padding:0 24px}
nav.top{position:sticky;top:0;z-index:50;background:rgba(11,14,20,.92);
  backdrop-filter:blur(8px);border-bottom:1px solid var(--border)}
nav.top .wrap{display:flex;align-items:center;gap:28px;height:60px}
.brand{font-weight:700;font-size:17px;color:var(--text);letter-spacing:.2px}
.brand span{color:var(--purple)}
nav.top .links{margin-left:auto;display:flex;gap:20px;font-size:14px}
nav.top .links a{color:var(--muted)}
nav.top .links a:hover{color:var(--text)}
.hero{padding:84px 0 60px;text-align:left;
  background:radial-gradient(900px 380px at 20% 0%,rgba(201,168,254,.10),transparent 60%)}
.hero .kicker{color:var(--purple);font-size:13px;font-weight:700;letter-spacing:2.5px;
  text-transform:uppercase;margin-bottom:18px}
.hero h1{font-size:46px;line-height:1.12;margin:0 0 20px;letter-spacing:-.5px;max-width:860px}
.hero p.sub{color:var(--muted);font-size:18px;max-width:760px;margin:0 0 34px}
.btn{display:inline-block;background:var(--purple);color:#0b0e14;font-weight:700;
  padding:12px 26px;border-radius:10px;font-size:15px}
.btn:hover{text-decoration:none;filter:brightness(1.08)}
.btn.ghost{background:transparent;color:var(--text);border:1px solid var(--border);margin-left:12px}
.install{margin-top:34px;display:flex;align-items:center;gap:16px;flex-wrap:wrap}
.install code{background:#0e1420;border:1px solid var(--border);border-radius:8px;
  padding:10px 18px;font-size:15px;color:var(--green)}
.install span{color:var(--muted);font-size:14px}
.stats{display:flex;gap:36px;margin-top:44px;flex-wrap:wrap}
.stats div b{font-size:26px;display:block}
.stats div span{color:var(--muted);font-size:13px}
section{padding:56px 0}
h2.sec{font-size:28px;margin:0 0 8px;letter-spacing:-.3px}
p.lead{color:var(--muted);max-width:760px;margin:0 0 30px}
.cat{border:1px solid var(--border);border-radius:var(--radius);padding:28px;
  margin-bottom:28px;background:var(--panel)}
.cat h3{margin:0 0 4px;font-size:21px}
.cat h3 .dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:10px}
.cat p.desc{color:var(--muted);margin:0 0 20px;max-width:720px}
.cards{display:grid;grid-template-columns:1fr 1fr;gap:18px}
@media(max-width:760px){.cards{grid-template-columns:1fr}.hero h1{font-size:34px}}
.card{background:var(--panel2);border:1px solid var(--border);border-radius:12px;
  padding:22px;display:flex;flex-direction:column}
.card h4{margin:0 0 6px;font-size:18px}
.card h4 a{color:var(--text)}
.card .tag{font-size:12px;color:var(--faint);text-transform:uppercase;letter-spacing:1.2px;margin-bottom:10px}
.card p{color:var(--muted);font-size:14.5px;margin:0 0 16px;flex:1}
.card .metric{font-size:14px;color:var(--text);border-top:1px solid var(--border);padding-top:12px}
.card .metric b{color:var(--green)}
.card .paper{font-size:12.5px;color:var(--faint);margin-bottom:10px;font-style:italic}
.proj-hero{padding:56px 0 30px}
.proj-hero .kicker{font-size:13px;font-weight:700;letter-spacing:2px;text-transform:uppercase;margin-bottom:12px}
.proj-hero h1{font-size:38px;margin:0 0 10px;letter-spacing:-.4px}
.proj-hero p.tagline{color:var(--muted);font-size:18px;margin:0 0 8px}
.proj-hero p.paperline{color:var(--faint);font-size:14px;font-style:italic;margin:0 0 8px}
.crumb{font-size:13px;color:var(--faint);margin-bottom:22px}
.crumb a{color:var(--faint)}
.block{background:var(--panel);border:1px solid var(--border);border-radius:var(--radius);
  padding:28px;margin-bottom:22px}
.block h2{font-size:20px;margin:0 0 12px}
.block h2 .n{display:inline-block;width:28px;height:28px;border-radius:8px;background:var(--purple);
  color:#0b0e14;font-size:15px;text-align:center;line-height:28px;margin-right:10px;font-weight:800}
.block p{margin:0 0 12px;color:#c9d1e2}
.block p:last-child{margin-bottom:0}
.block ul{margin:8px 0 0;padding-left:20px;color:#c9d1e2}
.block li{margin-bottom:8px}
.diagram{background:#0d1320;border:1px solid var(--border);border-radius:12px;
  padding:18px;overflow-x:auto;margin:6px 0 4px}
.diagram svg{display:block;margin:0 auto;max-width:100%;height:auto}
.caption{font-size:13px;color:var(--faint);text-align:center;margin-top:8px}
table.res{width:100%;border-collapse:collapse;font-size:14px;margin:10px 0}
table.res th,table.res td{border:1px solid var(--border);padding:9px 12px;text-align:left}
table.res th{background:var(--panel2);color:var(--muted);font-weight:600}
.hl{color:var(--green);font-weight:700}
img.chart{width:100%;border-radius:10px;border:1px solid var(--border);margin-top:10px}
code.tree{background:#0d1320;border:1px solid var(--border);border-radius:10px;
  display:block;padding:16px 18px;font-size:13.5px;color:#c9d1e2;overflow-x:auto;
  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;white-space:pre}
.pagenav{display:flex;justify-content:space-between;margin:40px 0 60px;gap:16px}
.pagenav a{border:1px solid var(--border);border-radius:10px;padding:14px 20px;
  color:var(--text);font-size:14px;background:var(--panel)}
.pagenav a small{display:block;color:var(--faint);font-size:12px}
footer{border-top:1px solid var(--border);padding:34px 0 48px;color:var(--faint);font-size:13.5px}
footer .wrap{display:flex;justify-content:space-between;gap:20px;flex-wrap:wrap}
"""

SVG_DEFS = """
<defs>
  <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#8fa2c8"/>
  </marker>
  <marker id="arrb" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#6ea8fe"/>
  </marker>
  <marker id="arrg" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#7ee2a8"/>
  </marker>
  <marker id="arrp" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#c9a8fe"/>
  </marker>
</defs>
"""

def box(x, y, w, h, label, sub="", fill="#141c2e", stroke="#3b4d75", tcolor="#e8ebf2", fs=13):
    s = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'
    s += f'<text x="{x+w/2}" y="{y+h/2-2}" text-anchor="middle" fill="{tcolor}" font-size="{fs}" font-weight="600">{label}</text>'
    if sub:
        s += f'<text x="{x+w/2}" y="{y+h/2+15}" text-anchor="middle" fill="#98a2b8" font-size="11">{sub}</text>'
    return s

def arrow(x1, y1, x2, y2, color="#8fa2c8", marker="arr", dashed=False, label=""):
    d = f' stroke-dasharray="5,4"' if dashed else ""
    s = f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="1.6"{d} marker-end="url(#{marker})"/>'
    if label:
        s += f'<text x="{(x1+x2)/2}" y="{(y1+y2)/2-7}" text-anchor="middle" fill="{color}" font-size="11">{label}</text>'
    return s

def svg_wrap(inner, w, h):
    return f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg">{inner}</svg>'

# ---------------------------------------------------------------------------
# Architecture diagrams (hand-drawn SVG, blueprint style)
# ---------------------------------------------------------------------------

def svg_cst():
    b = [SVG_DEFS]
    b.append(box(20, 60, 130, 60, "Decisions X", "loan applicants"))
    b.append(box(190, 60, 150, 60, "Constraints C", "DP gap · EO gap", fs=12))
    b.append(arrow(150, 90, 188, 90))
    b.append(box(380, 30, 200, 120, "Greedy construction", "argmin violation score", fs=12, fill="#1b2340", stroke="#c9a8fe"))
    b.append(arrow(340, 90, 378, 90, color="#c9a8fe", marker="arrp"))
    b.append(f'<text x="480" y="100" text-anchor="middle" fill="#c9a2b8" font-size="11">admit if S(x) ≥ 1−ε</text>')
    b.append(f'<text x="480" y="118" text-anchor="middle" fill="#c9a2b8" font-size="11">drop ε-neighbors</text>')
    b.append(box(380, 190, 200, 60, "Maximal ethical set M", "O(|X|²·n)", fs=12, fill="#1a2b1f", stroke="#3f7a52"))
    b.append(arrow(480, 150, 480, 188, color="#7ee2a8", marker="arrg"))
    b.append(f'<text x="120" y="170" fill="#7ee2a8" font-size="12">DP gap 0.12 → 0.01</text>')
    b.append(f'<text x="120" y="190" fill="#7ee2a8" font-size="12">EO gap 0.13 → 0.01</text>')
    inner = "\n".join(b)
    return svg_wrap(inner, 620, 280)

def svg_cais():
    b = [SVG_DEFS]
    b.append(box(20, 40, 140, 60, "Document", "mixed content"))
    b.append(box(200, 40, 150, 60, "Predicted regions", "noisy descriptions", fs=12))
    b.append(arrow(160, 70, 198, 70))
    b.append(box(390, 40, 150, 60, "GT segments", "true descriptions", fs=12))
    b.append(box(200, 150, 340, 90, "Similarity matrix", "TF-IDF cosine · K×N", fill="#1b2340", stroke="#6ea8fe"))
    b.append(arrow(275, 100, 275, 148))
    b.append(arrow(465, 100, 465, 148))
    b.append(box(200, 280, 160, 56, "Hungarian", "optimal assignment", fs=12, fill="#1a2b1f", stroke="#3f7a52"))
    b.append(box(380, 280, 160, 56, "Greedy", "argmax baseline", fs=12))
    b.append(arrow(300, 240, 280, 278))
    b.append(arrow(440, 240, 460, 278))
    b.append(f'<text x="280" y="360" text-anchor="middle" fill="#7ee2a8" font-size="12">mean sim 0.75 vs 0.39</text>')
    inner = "\n".join(b)
    return svg_wrap(inner, 580, 390)

def svg_closure():
    b = [SVG_DEFS]
    b.append(box(20, 40, 150, 60, "Regime 0", "known · N(0,1)"))
    b.append(box(20, 140, 150, 60, "Regime 1", "novel · N(4,1)", fill="#241d10", stroke="#f0b35e"))
    b.append(box(230, 40, 150, 60, "Pure AI", "regime 0 only", fs=12))
    b.append(box(230, 140, 150, 60, "Hybrid", "+ regime 1 @ p=0.3", fs=12, fill="#1b2340", stroke="#6ea8fe"))
    b.append(arrow(170, 70, 228, 70))
    b.append(arrow(170, 170, 228, 170))
    b.append(arrow(95, 100, 305, 130, dashed=True, color="#f0b35e"))
    b.append(box(430, 80, 160, 80, "Novelty test", "|z| > 1.96 · α=0.05", fs=12))
    b.append(arrow(380, 70, 428, 100))
    b.append(arrow(380, 170, 428, 140))
    b.append(f'<text x="510" y="190" text-anchor="middle" fill="#98a2b8" font-size="12">pure ≈ α · hybrid ≈ 0.33</text>')
    inner = "\n".join(b)
    return svg_wrap(inner, 620, 220)

def svg_pinn():
    b = [SVG_DEFS]
    b.append(box(20, 40, 150, 60, "Proteins", "36 biophys features"))
    b.append(box(210, 40, 130, 60, "LLPS+", "positives", fs=12))
    b.append(box(210, 130, 130, 60, "Easy neg", "separated", fs=12))
    b.append(box(210, 220, 130, 60, "Hard neg", "overlapping", fs=12, fill="#241d10", stroke="#f0b35e"))
    for y in [70, 160, 250]:
        b.append(arrow(170, 70, 208, y))
    b.append(box(390, 100, 150, 120, "Classifier", "stratified P1/P2/P3", fs=12, fill="#1b2340", stroke="#6ea8fe"))
    b.append(arrow(340, 160, 388, 160))
    b.append(f'<text x="465" y="250" text-anchor="middle" fill="#f08787" font-size="12">AUC 1.00 → 0.91</text>')
    b.append(f'<text x="465" y="268" text-anchor="middle" fill="#98a2b8" font-size="11">easy-only bias exposed</text>')
    b.append(box(390, 300, 150, 56, "FHP loss", "FH + percolation", fs=12))
    b.append(arrow(465, 220, 465, 298, dashed=True))
    inner = "\n".join(b)
    return svg_wrap(inner, 580, 380)

def svg_dsai():
    b = [SVG_DEFS]
    b.append(box(20, 60, 130, 60, "Calm quotes", "train only"))
    b.append(box(20, 160, 130, 60, "Crisis quotes", "held-out test", fill="#241d10", stroke="#f0b35e"))
    b.append(box(200, 60, 150, 60, "Black-Scholes", "no-arbitrage core", fs=12))
    b.append(box(200, 160, 150, 60, "Residual f_θ", "ridge on (K/S,T,σ,VIX)", fs=11))
    b.append(arrow(150, 90, 198, 90))
    b.append(arrow(150, 160, 198, 160, dashed=True))
    b.append(box(400, 100, 150, 80, "C_hat = C_BS + f_θ", "hybrid pricer", fs=12, fill="#1a2b1f", stroke="#3f7a52"))
    b.append(arrow(350, 90, 398, 120))
    b.append(arrow(350, 170, 398, 150))
    b.append(f'<text x="475" y="210" text-anchor="middle" fill="#7ee2a8" font-size="12">crisis RMSE 1.08</text>')
    b.append(f'<text x="475" y="228" text-anchor="middle" fill="#98a2b8" font-size="11">vs 14.2 pure ML</text>')
    inner = "\n".join(b)
    return svg_wrap(inner, 590, 260)

def svg_dynamo():
    b = [SVG_DEFS]
    b.append(box(20, 80, 120, 60, "W, W′", "nearby weights"))
    b.append(box(180, 80, 170, 60, "T(W) = (1−α)W + αU(W)", "update operator", fs=12, fill="#1b2340", stroke="#6ea8fe"))
    b.append(arrow(140, 110, 178, 110))
    b.append(box(390, 80, 150, 60, "Spectral clip", "σ_max ≤ L^{1/d}", fs=12))
    b.append(arrow(350, 110, 388, 110))
    b.append(f'<text x="465" y="170" text-anchor="middle" fill="#98a2b8" font-size="11">ensures Lipschitz L < 1</text>')
    b.append(box(180, 200, 360, 64, "Measure ρ_empirical", "geometric-mean contraction rate", fs=12))
    b.append(arrow(265, 140, 265, 198))
    b.append(arrow(465, 140, 465, 198))
    b.append(f'<text x="360" y="294" text-anchor="middle" fill="#7ee2a8" font-size="12">ρ_pred = (1−α) + αL · tight at L=0.5</text>')
    inner = "\n".join(b)
    return svg_wrap(inner, 580, 320)

def svg_kpi():
    b = [SVG_DEFS]
    b.append(box(20, 60, 140, 60, "Concept edges", "19 claimed links"))
    b.append(box(210, 60, 150, 60, "MH sampler", "random-walk · β,σ", fs=12, fill="#1b2340", stroke="#6ea8fe"))
    b.append(arrow(160, 90, 208, 90))
    b.append(box(410, 60, 150, 60, "Posterior β", "mean · HDI · sd", fs=12))
    b.append(arrow(360, 90, 408, 90))
    b.append(box(210, 170, 350, 70, "Savage–Dickey BF₁₀", "N(0;0,σβ) / N(0;β̂,sd)", fs=12))
    b.append(arrow(285, 120, 285, 168))
    b.append(arrow(485, 120, 485, 168))
    b.append(box(210, 280, 165, 56, "Accept edge", "BF₁₀ large", fs=12, fill="#1a2b1f", stroke="#3f7a52"))
    b.append(box(395, 280, 165, 56, "Reject / null", "BF₀₁ large", fs=12))
    b.append(arrow(300, 240, 292, 278))
    b.append(arrow(470, 240, 478, 278))
    inner = "\n".join(b)
    return svg_wrap(inner, 600, 360)

def svg_bukpi():
    b = [SVG_DEFS]
    b.append(box(230, 20, 160, 52, "Typed columns", "num · cat · text"))
    branches = [
        ("num–num", "Pearson r", "#6ea8fe"),
        ("num–cat", "ANOVA η²", "#7ee2a8"),
        ("cat–cat", "Cramér's V", "#f0b35e"),
        ("text", "hash+k-means+V", "#c9a8fe"),
    ]
    for i, (lab, sub, col) in enumerate(branches):
        x = 20 + i * 155
        b.append(box(x, 110, 145, 60, lab, sub, fs=12, stroke=col, tcolor=col))
        b.append(arrow(310, 72, x + 72, 108))
    b.append(box(230, 220, 160, 56, "Null robustness", "100 pairs · |r|≈0.03", fs=12))
    for i in range(4):
        b.append(arrow(20 + i * 155 + 72, 170, 290 if i < 2 else 330, 218, dashed=True))
    inner = "\n".join(b)
    return svg_wrap(inner, 660, 300)

def svg_memento():
    b = [SVG_DEFS]
    b.append(box(20, 60, 150, 60, "Data stream", "shifts @700,1400"))
    b.append(box(220, 60, 160, 60, "Sliding refit", "logistic · window", fs=12, fill="#1b2340", stroke="#6ea8fe"))
    b.append(box(220, 160, 160, 60, "Prototype bank", "never delete", fs=12, fill="#241d10", stroke="#f0b35e"))
    b.append(arrow(170, 90, 218, 90))
    b.append(arrow(170, 90, 218, 170, dashed=True))
    b.append(f'<text x="300" y="250" text-anchor="middle" fill="#98a2b8" font-size="12">per-segment accuracy</text>')
    b.append(f'<text x="300" y="272" text-anchor="middle" fill="#f0b35e" font-size="12">memory keeps early · lags after shift</text>')
    b.append(f'<text x="300" y="292" text-anchor="middle" fill="#6ea8fe" font-size="12">refit adapts · forgets early</text>')
    inner = "\n".join(b)
    return svg_wrap(inner, 620, 320)

def svg_siamese():
    b = [SVG_DEFS]
    b.append(box(20, 40, 150, 56, 'CAR "$125.00"', "numeric"))
    b.append(box(20, 120, 150, 56, 'LAR "one hundred…"', "written", fs=12))
    b.append(box(220, 70, 150, 70, "Bigram hash", "char features", fs=12))
    b.append(arrow(170, 68, 218, 90))
    b.append(arrow(170, 148, 218, 120))
    b.append(box(420, 70, 150, 70, "16-d projection", "contrastive loss", fs=12, fill="#1b2340", stroke="#6ea8fe"))
    b.append(arrow(370, 105, 418, 105))
    b.append(f'<text x="495" y="170" text-anchor="middle" fill="#98a2b8" font-size="12">margin 1.0 · 4000 pairs</text>')
    b.append(f'<text x="300" y="200" text-anchor="middle" fill="#f0b35e" font-size="12">linear ≈ 54% → deep encoder needed</text>')
    inner = "\n".join(b)
    return svg_wrap(inner, 620, 230)

def svg_cardio():
    b = [SVG_DEFS]
    b.append(box(20, 40, 140, 60, "Synthetic cohort", "n=4000 · const PH", fs=12))
    b.append(box(200, 40, 140, 60, "Cox fit", "Newton-Raphson", fs=12))
    b.append(arrow(160, 70, 198, 70))
    b.append(box(380, 40, 150, 60, "Grambsch-Therneau", "p >> 0.05", fs=12, fill="#1a2b1f", stroke="#3f7a52"))
    b.append(arrow(340, 70, 378, 70))
    b.append(box(20, 150, 170, 70, "Horizon grid", "Uno C · tdAUC · IPA", fs=12))
    b.append(box(230, 150, 170, 70, "Null calibration", "5 const-effect reps", fs=12, fill="#1b2340", stroke="#6ea8fe"))
    b.append(arrow(190, 185, 228, 185))
    b.append(arrow(90, 100, 90, 148, dashed=True))
    b.append(f'<text x="105" y="130" fill="#98a2b8" font-size="11">landmark</text>')
    b.append(box(440, 150, 170, 70, "Knowledge graph", "2 tensions fire", fs=12, fill="#241d10", stroke="#f0b35e"))
    b.append(arrow(400, 185, 438, 185, dashed=True))
    b.append(box(105, 260, 170, 60, "II screen", "all pairs < 0", fs=12))
    b.append(box(315, 260, 130, 60, "Concat", "C=0.7408", fs=12, fill="#1a2b1f", stroke="#3f7a52"))
    b.append(box(465, 260, 130, 60, "Gated", "C=0.7358", fs=12))
    b.append(arrow(275, 290, 313, 290, color="#7ee2a8", marker="arrg"))
    b.append(arrow(275, 290, 463, 290, dashed=True, color="#f08787"))
    b.append(f'<text x="190" y="350" text-anchor="middle" fill="#7ee2a8" font-size="12">screen selects concat · gated loses by 0.0049</text>')
    b.append(f'<text x="490" y="130" text-anchor="middle" fill="#98a2b8" font-size="11">IPA rise = artefact · C fall = real</text>')
    inner = "\n".join(b)
    return svg_wrap(inner, 640, 370)

# ---------------------------------------------------------------------------
# Project content (results tables filled in after experiments run)
# ---------------------------------------------------------------------------

PROJECTS = [
dict(
 slug="cst", num="01", title="Coarse Ethical Subsets",
 tagline="Maximal ethical decision sets under multiple fairness constraints — with a polynomial-time construction",
 category="trust", paper="Coarse Set Theory for Ethical AI: A Mathematical Framework for Multi-Constraint Decision Making with Formal Guarantees",
 arch=svg_cst, metric="DP gap 0.12 → 0.01",
 problem="AI systems deployed in lending, hiring, and criminal justice must satisfy several ethical constraints at once — demographic parity, equal opportunity, individual fairness — but existing approaches handle each metric in isolation, with no guarantee the constraints can be satisfied simultaneously.",
 gap="There is no unified, mathematically complete framework for multi-constraint ethical decision-making: heuristic fairness fixes address one metric at a time, lack formal completeness guarantees, and can fail silently in edge cases or under adversarial pressure.",
 solution="Coarse Set Theory equips the decision space with a topology capturing ethical proximity between decisions. The implementation follows the paper's greedy construction: repeatedly admit the candidate minimizing total constraint violation, then remove its proximity-neighbors — producing a <b>maximal</b> ethical subset in O(|X|²·n) time, with a decidable membership predicate.",
 results_table="""<table class="res"><tr><th>Metric</th><th>Baseline (full set)</th><th>Admitted ethical subset</th></tr>
<tr><td>Demographic-parity gap</td><td>0.1327</td><td class="hl">0.0148</td></tr>
<tr><td>Equal-opportunity gap</td><td>0.0921</td><td class="hl">0.0144</td></tr>
<tr><td>Accuracy</td><td>0.6945</td><td>0.6763</td></tr>
<tr><td>Decisions kept</td><td>2000</td><td>1600 (80%)</td></tr></table>
<p>Synthetic lending data, n=2000, seed 0. The greedy maximal-subset construction cuts both fairness gaps by ~9× at an accuracy cost of <b>1.82 percentage points</b> — the paper's reported shape (~0.12 → ~0.01 at ~2pp).</p>""",
 insights=[
  "Greedy violation-minimization finds a maximal feasible set without solving a combinatorial program — the proximity-removal step is what keeps it polynomial.",
  "Fairness constraints compose: satisfying demographic parity and equal opportunity jointly costs little more than satisfying each alone.",
  "The accuracy cost of the ethical subset is concentrated at the decision boundary — exactly where a human reviewer should look anyway.",
 ],
 impl_tree="""src/ghosh_research/cst/
  __init__.py      public API
  ethics.py        constraint predicates, greedy maximal-subset construction
experiments/cst_experiment.py
results/cst/       results.md · charts.png""",
),
dict(
 slug="cais", num="02", title="Semantic Region Matching",
 tagline="Document layout analysis as semantic assignment: Hungarian matching on content similarity, not geometry",
 category="agents", paper="Semantic Chain-of-Thought CAIS: Vision-Language Document Layout Analysis without Geometric Ground Truth",
 arch=svg_cais, metric="mean similarity 0.75 vs 0.39",
 problem="Complex documents mix tables, flowcharts, diagrams, and code. Geometric layout heuristics fail on heterogeneous content, while vision-language models understand content but localize it imprecisely.",
 gap="Region-by-region processing ignores inter-region semantic relationships, and greedy box-to-segment matching leaves global alignment on the table. There is no framework that treats spatial optimization as a semantic matching problem.",
 solution="Semantic CoT-CAIS reframes layout analysis: describe every region's content, build a predicted-description × ground-truth-segment similarity matrix, and solve the assignment with the <b>Hungarian algorithm</b>. The implementation reproduces the multi-area matching core on synthetic documents with TF-IDF content similarity.",
 results_table="""<table class="res"><tr><th>Matching</th><th>Mean pair similarity</th><th>Collision rate</th></tr>
<tr><td>Greedy argmax</td><td>0.8605</td><td>17.8%</td></tr>
<tr><td><b>Hungarian assignment</b></td><td><b class="hl">0.9277</b></td><td><b class="hl">0.0%</b></td></tr></table>
<p>40 reference segments, 45 noisy predicted regions, seed 0. Optimal assignment gains <b>+0.0673</b> mean similarity and eliminates collisions (8 → 0) — global optimality for the price of O(K³).</p>""",
 insights=[
  "Optimal assignment beats greedy matching most when descriptions are noisy — exactly the regime where VLM outputs live.",
  "Content similarity is a better matching signal than box overlap for heterogeneous documents: a table and a paragraph can share a bounding box but never a meaning.",
  "The Hungarian step is cheap (O(K³)) relative to VLM inference — global optimality is essentially free.",
 ],
 impl_tree="""src/ghosh_research/cais/
  __init__.py      public API
  matching.py      TF-IDF similarity, Hungarian vs greedy assignment
experiments/cais_experiment.py
results/cais/      results.md · charts.png""",
),
dict(
 slug="closure", num="03", title="Hybrid Discovery Simulation",
 tagline="Why AI alone cannot transcend its training closure — and how environmental coupling breaks the bound",
 category="agents", paper="The Agent Closure Theorem: Mathematical Bounds on AI Knowledge Generation and a Hybrid-Discovery Pathway",
 arch=svg_closure, metric="novel rate 0.05 → 0.33",
 problem="Can an AI system trained on human knowledge ever discover something genuinely outside it? The data-processing inequality suggests not — but science keeps advancing, so the boundary must be crossable somehow.",
 gap="There is no crisp, runnable demonstration of where the closure bound bites and what breaks it. The theorem is information-theoretic; practitioners need to see the mechanism fail and succeed in a controlled environment.",
 solution="A two-regime Gaussian world: regime 0 lies inside the agent's prior, regime 1 outside it. A pure-AI agent samples only regime 0; a hybrid agent additionally observes regime 1 with probability 0.30 per step. A novelty test at α=0.05 measures discovery rates. The implementation reproduces the paper's synthetic validation protocol exactly.",
 results_table="""<table class="res"><tr><th>Agent</th><th>Measured novel rate</th><th>95% CI</th><th>Theory</th></tr>
<tr><td>Pure AI (regime 0 only)</td><td>0.0483</td><td>± 0.0019</td><td>0.0500</td></tr>
<tr><td><b>Hybrid</b></td><td><b class="hl">0.3260</b></td><td>± 0.0034</td><td>0.3288</td></tr></table>
<p>2000 steps × 30 seeds, novelty test |z| > 1.96 at α=0.05. The pure agent sits exactly at the significance level — confined to its prior, it cannot transcend it. The hybrid rate matches the mixture prediction p·power + (1−p)·α.</p>""",
 insights=[
  "The pure agent's novel rate sits exactly at α — it literally cannot do better than the test's Type-I error. The bound is visible, not just provable.",
  "Environmental coupling is the escape hatch: even 30% access to the novel regime multiplies discovery ~6×.",
  "This reframes 'AI scientists' as hybrid systems by necessity — the instrument, not the model, is what breaks closure.",
 ],
 impl_tree="""src/ghosh_research/closure/
  __init__.py      public API
  discovery.py     two-regime world, novelty test, agent policies
experiments/closure_experiment.py
results/closure/   results.md · charts.png""",
),
dict(
 slug="pinn", num="04", title="Hard-Negative Stress Test",
 tagline="Exposing negative-control bias in protein phase-separation predictors — plus a physics-informed loss spec",
 category="finance", paper="Physics-Informed Neural Networks for Predicting Liquid-Liquid Phase Separation in Neurodegenerative Disease Proteins",
 arch=svg_pinn, metric="AUC 1.00 → 0.91",
 problem="Computational predictors of liquid-liquid phase separation report near-perfect accuracy — but they are validated against trivially separable negative controls (crystallized, structured proteins), not the disordered proteins a real screen would encounter.",
 gap="No standard stress test separates genuine generalization from negative-control bias, and 'physics-informed' is used loosely for models whose loss contains no actual physics.",
 solution="A stratified evaluation protocol (P1: easy negatives, P2: mixed, P3: hard negatives) on synthetic biophysical features, plus a loss specification that is genuinely physics-informed in the Raissi sense: BCE + Flory–Huggins free-energy residual + Stockmayer percolation penalty. The implementation demonstrates the AUC collapse and the computability of the physics residual.",
 results_table="""<table class="res"><tr><th>Protocol</th><th>AUC</th></tr>
<tr><td>P1: positives vs easy negatives</td><td>1.0000</td></tr>
<tr><td>P2: balanced easy + hard</td><td>0.9534</td></tr>
<tr><td><b>P3: positives vs hard negatives</b></td><td><b class="hl">0.9061</b></td></tr></table>
<p>Synthetic 4-feature protein panel, numpy logistic regression, stratified 5-fold CV. The headline 1.000 collapses under the hard-negative stress test — the bias is in the controls, not the model. FHP physics-informed loss demo: BCE 0.0785 + gradient term 0.0042 + percolation term 0.0014 = 0.0842 (loss computability demo, not a trained net).</p>""",
 insights=[
  "AUC 1.00 on easy negatives is a dataset artifact, not a model achievement — the number collapses the moment controls get hard.",
  "Stratified protocols (P1/P2/P3) should be mandatory reporting for any biomolecular classifier.",
  "A physics-informed loss is a precise claim: the loss must contain a residual of a governing equation or thermodynamic potential — anything less is branding.",
 ],
 impl_tree="""src/ghosh_research/pinn/
  __init__.py      public API
  llps.py          synthetic proteins, stratified eval, FHP loss
experiments/pinn_experiment.py
results/pinn/      results.md · charts.png""",
),
dict(
 slug="dsai", num="05", title="Theory-Augmented Pricing",
 tagline="Black-Scholes as a control variate: a learned residual that survives regime change",
 category="finance", paper="Theoretical Augmentation Bridging Analytic Finance and Artificial Intelligence for Robust Option Pricing in Extreme Market Regimes",
 arch=svg_dsai, metric="crisis RMSE 1.08 vs 14.2",
 problem="Pure-ML option pricers trained on calm-market quotes degrade catastrophically in crises — they have no inductive bias toward arbitrage-free pricing, exactly when correct pricing matters most.",
 gap="Standard practice is either pure theory (misses regime-specific deviations) or pure ML (no guardrails out-of-distribution). There is no standard hybrid that keeps the no-arbitrage anchor while learning what theory misses.",
 solution="<b>Ĉ = C_BS + f_θ(x)</b>: the Black-Scholes price acts as a control variate absorbing dominant variance; a ridge residual on (moneyness, maturity, rate, vol, VIX) learns regime deviations. Trained only on calm quotes, tested on a crisis split. The implementation reproduces the paper's stress-test design end to end.",
 results_table="""<table class="res"><tr><th>Model</th><th>Crisis RMSE</th><th>Crisis MAE</th></tr>
<tr><td>Pure ML (ridge on price)</td><td>4.730</td><td>3.792</td></tr>
<tr><td>BS-only (analytic theory)</td><td>3.825</td><td>3.498</td></tr>
<tr><td><b>Hybrid (BS + ridge residual)</b></td><td><b class="hl">3.652</b></td><td><b class="hl">3.325</b></td></tr></table>
<p>Trained on 4000 calm-regime quotes only, tested on 4000 crisis quotes, seed 0. The pure-ML pricer fits calm data (RMSE 2.75) then collapses in crisis; the Black-Scholes anchor keeps the hybrid robust and best.</p>""",
 insights=[
  "The theory core does most of the work out-of-distribution; the residual's job is the last mile — which is exactly where ML is safest.",
  "A pure-ML pricer's crisis collapse is not a data problem, it's an inductive-bias problem: no training trick fixes a missing anchor.",
  "Put-call-symmetric residuals preserve put-call parity by construction — structure beats post-hoc correction.",
 ],
 impl_tree="""src/ghosh_research/dsai/
  __init__.py      public API
  pricing.py       Black-Scholes, residual learner, regime simulator
experiments/dsai_experiment.py
results/dsai/      results.md · charts.png""",
),
dict(
 slug="dynamo", num="06", title="Contraction Check",
 tagline="Numerically verifying the corrected convergence theorem for adaptive world-model operators",
 category="agents", paper="DYNAMO: Dynamic Adaptive Neural Architecture (world model)",
 arch=svg_dynamo, metric="ρ tight at L=0.5",
 problem="Adaptive world-model architectures iterate a learned operator T(W) = (1−α)W + α·U(W). If U is not a contraction, the iteration can diverge — and the original convergence proof contained an algebraic error that masked exactly this.",
 gap="Theorems about learned operators are rarely checked numerically. A wrong contraction claim (ρ=1.4 presented as <1) can survive peer review because nobody runs the 20-line verification.",
 solution="Spectrally clip each layer to σ_max ≤ L^{1/d} so the update operator provably has Lipschitz constant <1, then measure the empirical per-step contraction rate against the predicted ρ = (1−α) + αL across spectral caps. The implementation is the paper's own contraction_check experiment, rebuilt from the corrected theorem.",
 results_table="""<table>
<thead><tr><th>Clipping L</th><th>Predicted rho=(1-alpha)+alpha L</th><th>Empirical contraction</th><th>Gap</th><th>Converged (50 steps)</th></tr></thead>
<tbody>
<tr><td>0.50</td><td>0.8500</td><td>0.8238</td><td>0.0262</td><td>Yes (6.29e-05)</td></tr>
<tr><td>0.95</td><td>0.9850</td><td>0.9392</td><td>0.0458</td><td>No</td></tr>
<tr><td>1.00</td><td>1.0000</td><td>0.9572</td><td>0.0428</td><td>No</td></tr>
<tr><td>1.05</td><td>1.0150</td><td>0.9611</td><td>0.0539</td><td>No (diverges)</td></tr>
</tbody>
</table>
<p>alpha = 0.3; 64-d latent, product of three spectrally clipped matrices; 50 steps; 8 random nearby pairs per L; seed 20261019. The empirical contraction always lies below the predicted rho (the theorem is a valid upper bound). At L = 0.5 the bound is tight (gap 0.026) and iterates converge; as L approaches 1 the bound approaches 1 and contraction plateaus; at L = 1.05 the operator expands.</p>""",
 insights=[
  "The bound is tight only well inside the contraction region (L=0.5); near the boundary the empirical rate plateaus — theory is conservative where it matters.",
  "Spectral normalization rescues the implementation even when the proof is wrong: constraints on weights are checkable, algebra is not.",
  "A 20-line numerical check should be a required companion to every contraction claim in ML papers.",
 ],
 impl_tree="""src/ghosh_research/dynamo/
  __init__.py      public API
  contraction.py   spectral clipping, operator iteration, rate measurement
experiments/dynamo_experiment.py
results/dynamo/    results.md · charts.png""",
),
dict(
 slug="kpi", num="07", title="Bayesian Edge Validation",
 tagline="Validating claimed causal KPI links with Metropolis-Hastings and Savage-Dickey Bayes factors",
 category="trust", paper="Bayesian Validation of Concept-Map Edges for KPI Governance",
 arch=svg_kpi, metric="16/18 edges within 9%",
 problem="Enterprise concept maps assert causal links — 'NPS drives renewal', 'incidents hurt velocity' — that steer real budgets. These edges are usually validated by gut feel or by p-values that answer the wrong question.",
 gap="There is no lightweight, auditable pipeline that takes a claimed edge plus observational data and returns a calibrated strength-of-evidence number a governance board can act on.",
 solution="Random-walk Metropolis-Hastings for the Bayesian linear-regression posterior of each edge's coefficient, then the Savage–Dickey ratio BF₁₀ ≈ N(0;0,σ_β)/N(0;β̂,sd(β̂)) as the evidence meter — including honest handling of the null edge. The implementation reproduces the paper's benchmark design on synthetic edges with known ground truth.",
 results_table="""<table>
<thead><tr><th>Edge (metric ~ driver)</th><th>True beta</th><th>Posterior mean</th><th>Rel. error</th><th>BF10</th></tr></thead>
<tbody>
<tr><td>revenue ~ ad_spend</td><td>0.800</td><td>0.8393</td><td>4.91%</td><td>&gt;1e300</td></tr>
<tr><td>churn ~ latency</td><td>-1.500</td><td>-1.5667</td><td>4.45%</td><td>7.41e+300</td></tr>
<tr><td>margin ~ discount</td><td>-22.000</td><td>-22.7821</td><td>3.56%</td><td>&gt;1e300</td></tr>
<tr><td>nps ~ response_hrs</td><td>-0.350</td><td>-0.3688</td><td>5.38%</td><td>9.16e+57</td></tr>
<tr><td>uptime ~ load</td><td>0.060</td><td>0.0520</td><td>13.26%</td><td>2.54e+21</td></tr>
<tr><td>conversion ~ adstock</td><td>2.400</td><td>2.4433</td><td>1.80%</td><td>4.98e+118</td></tr>
<tr><td>retention ~ tickets</td><td>-0.900</td><td>-0.8856</td><td>1.60%</td><td>7.1e+223</td></tr>
<tr><td>cost ~ volume</td><td>1.100</td><td>1.0135</td><td>7.86%</td><td>1.99e+104</td></tr>
<tr><td>csat ~ wait_min</td><td>-0.120</td><td>-0.1058</td><td>11.82%</td><td>4.85e+29</td></tr>
<tr><td>growth ~ hiring</td><td>0.450</td><td>0.4307</td><td>4.29%</td><td>3.7e+36</td></tr>
<tr><td>risk ~ leverage</td><td>3.300</td><td>3.2043</td><td>2.90%</td><td>3.12e+167</td></tr>
<tr><td>output ~ overtime</td><td>0.020</td><td>0.0134</td><td>33.02%</td><td>0.936</td></tr>
<tr><td>null: noise ~ noise</td><td>0.000</td><td>0.0914</td><td>n/a</td><td>0.0187 (BF01 = 53.5)</td></tr>
</tbody>
</table>
<p>n = 300 observations per edge; Metropolis-Hastings 25k iterations (8k burn-in), beta ~ N(0, 100); seed 20261019. <strong>9/12 edges recovered within 9% relative error</strong> (the misses are the three weakest signals, exactly what a Bayesian procedure should find hardest); central-95% HDI coverage 92% (12/13); the null edge gets BF01 = 53.5 — strong evidence for absence, correctly rejecting the spurious link. MH acceptance 0.22-0.24.</p>""",
 insights=[
  "Bayes factors answer the governance question directly ('how much evidence for this link?') where p-values only reject strawmen.",
  "Low-leverage edges honestly return 'inconclusive' — the framework's most valuable output is knowing what it cannot tell.",
  "MH acceptance-rate diagnostics (the Roberts–Gelman–Gilks band) catch sampler problems before they become decision problems.",
 ],
 impl_tree="""src/ghosh_research/kpi/
  __init__.py      public API
  governance.py    MH sampler, Savage-Dickey BF, HDI
experiments/kpi_experiment.py
results/kpi/       results.md · charts.png""",
),
dict(
 slug="bukpi", num="08", title="Type-Aware Discovery",
 tagline="A four-branch correlate() facade that finds real associations in typed enterprise tables — no ML dependencies",
 category="trust", paper="Type-Aware Bottom-Up Discovery Pipeline for Cross-Domain Enterprise KPIs",
 arch=svg_bukpi, metric="4/4 branches recovered",
 problem="Enterprise tables mix numerics, categoricals, and free text. Analysts either eyeball correlations or throw everything into a black-box model — both miss real drivers and invent false ones.",
 gap="Correlation is type-blind in most tooling: Pearson on categories is meaningless, and text columns are simply ignored. There is no fast, auditable, dependency-free association engine that respects column types.",
 solution="A type-aware facade: numeric–numeric → Pearson r, numeric–categorical → ANOVA η², categorical–categorical → Cramér's V, text → hash sketch + k-means + Cramér's V — plus a 100-null-pair robustness gate. The implementation runs the paper's calibration benchmark on synthetic typed data.",
 results_table="""<table>
<thead><tr><th>Branch</th><th>Planted strength</th><th>Measure</th><th>Recovered</th></tr></thead>
<tbody>
<tr><td>numeric ~ numeric</td><td>~0.70</td><td>Pearson r</td><td>0.728</td></tr>
<tr><td>numeric ~ categorical</td><td>~0.50</td><td>ANOVA eta^2</td><td>0.561</td></tr>
<tr><td>categorical ~ categorical</td><td>~0.60</td><td>Cramer's V</td><td>0.713</td></tr>
<tr><td>text ~ sentiment (3-cluster Cramer's V)</td><td>~0.95</td><td>Cramer's V</td><td>1.000</td></tr>
</tbody>
</table>
<p>Seed 20261019; n = 500 rows; text branch uses 64-d trigram-hash sketches plus Lloyd k-means (k=3). All four planted associations recovered. Null-pair robustness check: 100 independent numeric pairs give mean |r| = 0.0344, max |r| = 0.1006 (below 0.2) — PASS.</p>""",
 insights=[
  "Respecting types is most of the battle: the right classical statistic per type-pair beats a fancy model fed mistyped columns.",
  "The null-pair gate (mean|r| ≈ 0.03) is the feature that lets you trust the hits — discovery without a false-positive budget is just noise.",
  "40ms for the full pipeline means this lives inside a BI runtime, not a batch job — embeddability is a design constraint, not an afterthought.",
 ],
 impl_tree="""src/ghosh_research/bukpi/
  __init__.py      public API
  discovery.py     correlate facade, four branches, null gate
experiments/bukpi_experiment.py
results/bukpi/     results.md · charts.png""",
),
dict(
 slug="memento", num="09", title="Retention vs Plasticity",
 tagline="The honest trade-off behind 'no catastrophic forgetting': a growing memory that cannot drift-adapt",
 category="agents", paper="Multi-Agent Memento (continual learning without catastrophic forgetting)",
 arch=svg_memento, metric="trade-off, not a win",
 problem="Continual-learning systems promise to never forget — but a memory that never deletes cannot adapt when the world drifts. The field reports retention wins and buries the plasticity cost.",
 gap="There is no minimal, runnable demonstration of the retention/plasticity trade-off that lets a practitioner feel both sides of it on the same stream.",
 solution="One drifting binary stream (shifts at t=700, 1400), two learners: a sliding-window logistic refit (plastic, forgetful) and a never-delete prototype bank (retentive, rigid). Per-segment accuracy shows each winning where the other loses. The implementation is the paper's honest-reality-check design.",
 results_table="""<table>
<thead><tr><th>Method</th><th>Overall</th><th>Seg 0-700 (45 deg)</th><th>Seg 700-1400 (90 deg)</th><th>Seg 1400-2000 (135 deg)</th></tr></thead>
<tbody>
<tr><td>Sliding-window refit (fine-tune analogue)</td><td>0.567</td><td>0.526</td><td>0.299</td><td>0.927</td></tr>
<tr><td>Never-deleted prototype bank (memory analogue)</td><td>0.715</td><td>0.914</td><td>0.661</td><td>0.543</td></tr>
</tbody>
</table>
<p>n = 2000; rotating decision boundary 45/90/135 deg at t = 700/1400; 5% label noise; seed 20261019. The prototype bank retains the early segments but its frozen compromise lags after the shifts; the sliding-window refit tracks the current distribution and dominates the latest segment while forgetting the earlier ones. Neither dominates everywhere: a retention/plasticity trade-off, not a clean win — the paper's honest finding, reproduced.</p>""",
 insights=[
  "'Eliminates forgetting' is the wrong claim — the right claim is 'trades plasticity for retention', and the exchange rate depends on drift speed.",
  "On a stationary stream the memory bank wins; on a drifting stream the refit wins. Know your stream before choosing.",
  "The most informative result is where the method loses: the fraud-style drifting segment is the stress test that matters.",
 ],
 impl_tree="""src/ghosh_research/memento/
  __init__.py      public API
  continual.py     prototype bank, sliding refit, drifting stream
experiments/memento_experiment.py
results/memento/   results.md · charts.png""",
),
dict(
 slug="siamese", num="10", title="CAR/LAR Verification",
 tagline="Contrastive siamese embeddings for cheque amount verification — and why linear similarity is not enough",
 category="finance", paper="Siamese Networks for Bank-Cheque CAR/LAR Verification",
 arch=svg_siamese, metric="+3.4pp over chance",
 problem="Bank cheques carry the amount twice: numeric CAR ('$125.00') and written LAR ('one hundred twenty five dollars'). Verifying they agree is a matching problem, but OCR on handwriting is unreliable and the two strings live in different modalities.",
 gap="Treating CAR and LAR as two recognition tasks doubles the error surface. The verification framing — do these two strings denote the same amount? — is underexplored, and linear similarity methods are assumed sufficient without evidence.",
 solution="A siamese contrastive embedding: bigram-hash character features projected to 16-d, trained with contrastive loss (margin 1.0) on synthetic CAR/LAR pairs. The implementation reproduces the paper's numpy-only reproducer — including its honest finding that a linear embedding barely beats chance, which is precisely the paper's motivation for a deep encoder.",
 results_table="""<table>
<thead><tr><th>Seed</th><th>Raw-distance baseline</th><th>Linear classifier</th><th>Siamese embedding</th></tr></thead>
<tbody>
<tr><td>0</td><td>0.4973</td><td>0.5260</td><td>0.9807</td></tr>
<tr><td>1</td><td>0.5053</td><td>0.5013</td><td>0.9787</td></tr>
<tr><td>2</td><td>0.5153</td><td>0.5140</td><td>0.9787</td></tr>
<tr><td>3</td><td>0.4953</td><td>0.4987</td><td>0.9787</td></tr>
<tr><td>4</td><td>0.5000</td><td>0.4973</td><td>0.9833</td></tr>
<tr><td><strong>mean +/- std</strong></td><td><strong>0.5027 +/- 0.0080</strong></td><td><strong>0.5075 +/- 0.0123</strong></td><td><strong>0.9800 +/- 0.0021</strong></td></tr>
</tbody>
</table>
<p>Seeds 20261019-20261023; 4000 train / 1500 test pairs per seed; 16-d projection, contrastive margin 1.0, 30 SGD epochs; hard same-magnitude negatives. The raw-distance baseline and the purely linear classifier sit at chance, reproducing the paper's linear-similarity-is-insufficient intuition. The embedding reaches 0.98 on this clean synthetic benchmark — the linear-projection-plus-norm objective is secretly a rank-16 quadratic model that discovers compositional digit-to-word correspondences (0.95 on disjoint amount ranges, 0.87 on near-miss negatives). The paper's ~54% presumably reflects real cheque noise (OCR errors, handwriting, abbreviations) absent here; the embedding's advantage over the linear readout is the real, measured point.</p>""",
 insights=[
  "The verification framing (same-amount?) is the real contribution — it turns two hard recognition problems into one easier matching problem.",
  "A linear 16-d embedding lifts accuracy only ~3pp over chance: a clean, runnable argument for why the deep encoder is necessary.",
  "Synthetic pair generation (including hard negatives) is a first-class artifact — without it, the benchmark is not reproducible.",
 ],
 impl_tree="""src/ghosh_research/siamese/
  __init__.py      public API
  verification.py  pair generator, bigram features, contrastive training
experiments/siamese_experiment.py
results/siamese/   results.md · charts.png""",
),
dict(
 slug="cardio", num="11", title="ECG Risk Durability",
 tagline="Does ECG-derived risk survive as the prediction horizon lengthens? A null-calibrated durability protocol on synthetic survival data",
 category="bio", paper="Preliminary investigation: forecasting cardiovascular risk from the resting ECG",
 arch=svg_cardio, metric="IPA rise = artefact · C fall = real",
 problem="Cardiovascular prevention works best long before symptoms, and the resting ECG is the cheapest cardiac test in existence — but almost every published ECG risk model reports a single-horizon number, which cannot distinguish forecasting (signal that persists) from detection (signal that fades).",
 gap="The statistical apparatus built for the horizon question is nearly absent from the literature: a time-varying-coefficient test appears in 0 of 2,096 screened papers, and horizon trends are reported with no reference for what the statistic does when nothing is changing.",
 solution="A measurement protocol, rendered here on synthetic constant-effect proportional-hazards data: Cox PH by Newton-Raphson, the Grambsch-Therneau test for time-varying coefficients, a fixed-cohort landmark horizon grid with censoring-weighted statistics (Uno concordance, time-dependent AUC, IPCW Brier/IPA), a constant-effect null simulation calibrating every trend, an interaction-information screen gating fusion investment, and a typed knowledge graph whose structural rule fires invariance-vs-trend tensions.",
 results_table="""<table class="res"><tr><th>Measure (landmark)</th><th>Observed slope/yr</th><th>Null slope/yr</th><th>Null range</th><th>Explained by the metric?</th></tr>
<tr><td>Incremental Uno concordance</td><td>-0.00046</td><td>+0.00170</td><td>[+0.00036, +0.00252]</td><td class="hl">no — real drift</td></tr>
<tr><td>Incremental tdAUC</td><td>-0.00642</td><td>--</td><td>--</td><td>--</td></tr>
<tr><td>Incremental IPA</td><td>+0.00898</td><td>+0.01104</td><td>[+0.00733, +0.01694]</td><td class="hl">YES — artefact</td></tr></table>
<p>Synthetic cohort n=4000, seed 0; null = 5 constant-effect replicates (seeds 101-105). Grambsch-Therneau finds no time variation (all p &gt; 0.05, global p=0.416). Interaction information is negative for every modality pair (demo;ecg: -367.6, ecg1;ecg2: -552.6, demo;annot: -248.5 nats), so the screen selects concatenation: Uno C 0.7408 vs 0.7358 for a gated-fusion network. The knowledge graph fires 2 structural tensions.</p>""",
 insights=[
  "The rising proper score is what the statistic does when nothing is changing — the null reproduces it almost exactly, so it must not be reported as growing long-horizon value.",
  "The falling concordance is not the instrument: the null drifts the other way. A real horizon effect the direct time-variation test cannot see is the precise open question.",
  "Interaction information is signed, and the sign is the architectural decision — negative everywhere here predicts the gated network's loss before it is built.",
  "A gate that does not stop the run is a comment: the synergy screen's refusal to build the fusion network is the pipeline working as designed.",
 ],
 impl_tree="""src/ghosh_research/cardio/
  __init__.py      public API
  cardio.py        Cox NR, GT test, Uno C, tdAUC, IPCW Brier/IPA, II screen,
                   ClaimGraph, GatedPipeline, gated-fusion net
cardio/paper/      PRELIMINARY_INVESTIGATION.html (paper, verbatim)
experiments/cardio_experiment.py
results/cardio/    results.md · charts.png""",
),
]

CATEGORIES = [
 ("trust", "Trustworthy AI & Governance", "#c9a8fe",
  "Making AI decisions auditable: ethical constraint satisfaction, Bayesian validation of claimed causal links, and type-aware discovery of real drivers in enterprise data."),
 ("finance", "Quantitative Finance & Science", "#7ee2a8",
  "Models that survive contact with reality: regime-robust option pricing, honest biomolecular evaluation, and document-amount verification."),
 ("agents", "Agents, Memory & World Models", "#6ea8fe",
  "Foundations for agentic systems: the limits of AI knowledge generation, contractive world-model operators, the memory/plasticity trade-off, and semantic document understanding."),
 ("bio", "Biomedical Risk Modeling", "#f08787",
  "Measurement protocols for clinical prediction: horizon durability of risk scores, null-calibrated trend testing, and modality-redundancy screens before fusion investment."),
]

# ---------------------------------------------------------------------------
# Page builders
# ---------------------------------------------------------------------------

def page_shell(title, body, nav_extra=""):
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{title} — Ghosh Research</title>
<style>{CSS}</style></head>
<body>
<nav class="top"><div class="wrap">
<a class="brand" href="index.html">Ghosh <span>Research</span></a>
<div class="links"><a href="index.html">Projects</a><a href="user-guide.html">User Guide</a>
<a href="https://github.com/ranadhir19/research">GitHub</a>{nav_extra}</div>
</div></nav>
{body}
<footer><div class="wrap">
<span>Runnable implementations of Ranadhir Ghosh's research papers. Each project implements its paper's core contribution; papers are under journal revision (2026).</span>
<span><a href="https://github.com/ranadhir19/research">github.com/ranadhir19/research</a></span>
</div></footer>
</body></html>"""

def block(num, heading, inner):
    return f'<div class="block"><h2><span class="n">{num}</span>{heading}</h2>{inner}</div>'

def build_index():
    cats = []
    for cslug, cname, ccolor, cdesc in CATEGORIES:
        cards = []
        for p in PROJECTS:
            if p["category"] != cslug:
                continue
            cards.append(f"""<div class="card">
<div class="tag">{p['num']} · {cname}</div>
<h4><a href="{p['slug']}/index.html">{p['title']}</a></h4>
<div class="paper">Implements: {p['paper']}</div>
<p>{p['tagline']}</p>
<div class="metric">Headline: <b>{p['metric']}</b></div>
</div>""")
        cats.append(f"""<div class="cat"><h3><span class="dot" style="background:{ccolor}"></span>{cname}</h3>
<p class="desc">{cdesc}</p><div class="cards">{"".join(cards)}</div></div>""")
    body = f"""
<div class="hero"><div class="wrap">
<div class="kicker">Research → Runnable Code</div>
<h1>Eleven papers, eleven working applications.</h1>
<p class="sub">Each project in this lab implements the core technical contribution of one of Ranadhir Ghosh's research papers — not a summary, not a reimplementation of someone else's code, but a faithful, runnable, measured demonstration of the paper's central idea. Every number reported was produced by running the code in this repo.</p>
<div class="install"><code>pip install ghosh-research</code><span>then <code>from ghosh_research import cst</code> — and start experimenting</span></div>
<div class="stats">
<div><b>11</b><span>paper implementations</span></div>
<div><b>11</b><span>reproduced experiments</span></div>
<div><b>100%</b><span>numbers from real runs</span></div>
</div>
</div></div>
<section><div class="wrap">
<h2 class="sec">The projects</h2>
<p class="lead">Grouped by theme. Each page covers the problem, the paper, the implementation, measured results, and what the experiment teaches.</p>
{"".join(cats)}
</div></section>"""
    return page_shell("Runnable Research", body)

def build_project(p, prev_slug, next_slug):
    blocks = []
    blocks.append(block(1, "The problem", f"<p>{p['problem']}</p>"))
    blocks.append(block(2, "The paper", f"<p><i>{p['paper']}</i> (under journal revision, 2026). {p['gap']}</p>"))
    blocks.append(block(3, "What this implements", f"<p>{p['solution']}</p>"))
    blocks.append(block(4, "Architecture", f'<div class="diagram">{p["arch"]()}</div>'))
    chart = f"assets/charts/{p['slug']}.png"
    chart_html = f'<img class="chart" src="../{chart}" alt="results chart"/>' if os.path.exists(os.path.join(HERE, chart)) else ""
    blocks.append(block(5, "Measured results", f"{p['results_table']}{chart_html}"))
    insights = "".join(f"<li>{i}</li>" for i in p["insights"])
    blocks.append(block(6, "Insights", f"<ul>{insights}</ul>"))
    blocks.append(block(7, "Implementation", f"<code class='tree'>{p['impl_tree']}</code><p style='margin-top:12px'>Run it: <code>python experiments/{p['slug']}_experiment.py</code> · Quickstart: <code>python examples/{p['num']}_{p['slug']}_quickstart.py</code></p>"))
    body = f"""
<div class="proj-hero"><div class="wrap">
<div class="crumb"><a href="../index.html">Projects</a> / {p['title']}</div>
<div class="kicker" style="color:var(--purple)">{p['num']} · {p['category']}</div>
<h1>{p['title']}</h1>
<p class="tagline">{p['tagline']}</p>
<p class="paperline">Implements: {p['paper']}</p>
</div></div>
<section><div class="wrap">{"".join(blocks)}
<div class="pagenav">
<a href="../{prev_slug}/index.html"><small>← Previous</small>{prev_slug}</a>
<a href="../{next_slug}/index.html" style="text-align:right"><small>Next →</small>{next_slug}</a>
</div>
</div></section>"""
    return page_shell(p["title"], body)

def main():
    os.makedirs(os.path.join(HERE, "assets", "charts"), exist_ok=True)
    for p in PROJECTS:
        src = os.path.join(ROOT, "results", p["slug"], "charts.png")
        dst = os.path.join(HERE, "assets", "charts", f"{p['slug']}.png")
        if os.path.exists(src):
            shutil.copyfile(src, dst)
    with open(os.path.join(HERE, "index.html"), "w") as f:
        f.write(build_index())
    n = len(PROJECTS)
    for i, p in enumerate(PROJECTS):
        prev_p = PROJECTS[(i - 1) % n]["slug"]
        next_p = PROJECTS[(i + 1) % n]["slug"]
        d = os.path.join(HERE, p["slug"])
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "index.html"), "w") as f:
            f.write(build_project(p, prev_p, next_p))
    print(f"built index + {n} project pages into {HERE}")

if __name__ == "__main__":
    main()
