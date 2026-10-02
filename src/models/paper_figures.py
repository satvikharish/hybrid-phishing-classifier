"""Builds the four three-panel paper figures (teaser, architecture, result).

Result panels use the CSVs written by significance_test.py and
latency_analysis.py. SHAP values are the mean |SHAP| values from
shap_analysis.py on the 300-URL hybrid sample.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Patch

PROCESSED = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
OUT = os.path.join(os.path.dirname(__file__), "..", "..", "reports", "figures")

LEX, HOST, GREY, INK = "#3A6EA5", "#E08E45", "#8A8F98", "#222222"
LEX_FILL, HOST_FILL, NEUTRAL_FILL = "#DCE8F5", "#FBE6D2", "#F1F2F4"

SHAP = [
    ("path_length", 1.583, "lex"), ("num_subdomains", 0.304, "lex"), ("has_https", 0.232, "lex"),
    ("num_slashes", 0.220, "lex"), ("has_txt_record", 0.111, "host"), ("url_length", 0.039, "lex"),
    ("domain_age_days", 0.030, "host"), ("has_valid_ssl", 0.028, "host"),
    ("has_whois_record", 0.025, "host"), ("has_mx_record", 0.021, "host"),
]


def box(ax, x, y, w, h, text, fc=NEUTRAL_FILL, ec=INK, fs=8.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.005,rounding_size=0.02",
                                fc=fc, ec=ec, lw=1.0))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color=INK)


def arrow(ax, x1, y1, x2, y2, style="-|>"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, lw=1.2, color="#444444", shrinkA=0, shrinkB=0))


def panel(ax, label):
    ax.text(0.0, 1.04, label, transform=ax.transAxes, fontsize=10.5, fontweight="bold", va="bottom")


def blank(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def new_fig(left=0.09):
    fig = plt.figure(figsize=(7.0, 9.2))
    gs = fig.add_gridspec(3, 1, height_ratios=[0.8, 1.0, 1.35], hspace=0.28,
                          left=left, right=0.97, top=0.96, bottom=0.06)
    return fig, [fig.add_subplot(gs[i]) for i in range(3)]


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name), dpi=170)
    plt.close(fig)
    print("wrote", name)


def fig1(sig):
    fig, (a, b, c) = new_fig()
    blank(a); panel(a, "(a) Teaser")
    box(a, 0.03, 0.2, 0.36, 0.6, "URL text only\n\nAUROC 0.9954", LEX_FILL, LEX, fs=10)
    box(a, 0.61, 0.2, 0.36, 0.6, "URL text + server checks\n\nAUROC 0.9981", HOST_FILL, HOST, fs=10, bold=False)
    arrow(a, 0.40, 0.5, 0.60, 0.5)
    a.text(0.5, 0.62, "+0.0027", ha="center", fontsize=11, fontweight="bold")
    a.text(0.5, 0.06, "Small gain, but it holds up across 100 paired folds (p = 0.002)", ha="center", fontsize=8.5, style="italic")

    blank(b); panel(b, "(b) Architecture")
    box(b, 0.01, 0.38, 0.17, 0.26, "126,620 URLs\nPhishTank,\nOpenPhish,\nMajestic", fs=7.5)
    box(b, 0.25, 0.62, 0.2, 0.24, "Lexical features\n16, read from\nthe URL text", LEX_FILL, LEX, fs=7.5)
    box(b, 0.25, 0.22, 0.2, 0.24, "Host telemetry\n12, DNS / WHOIS /\nSSL / HTTP (n=300)", HOST_FILL, HOST, fs=7.5)
    box(b, 0.52, 0.38, 0.14, 0.26, "Feature\nmatrix", fs=8)
    box(b, 0.73, 0.6, 0.13, 0.24, "XGBoost\nweight 2", fs=7.5)
    box(b, 0.73, 0.24, 0.13, 0.24, "Random\nForest\nweight 1", fs=7.5)
    box(b, 0.9, 0.38, 0.09, 0.26, "Soft\nvote", fs=7.5)
    arrow(b, 0.18, 0.55, 0.25, 0.72); arrow(b, 0.18, 0.47, 0.25, 0.34)
    arrow(b, 0.45, 0.72, 0.52, 0.56); arrow(b, 0.45, 0.34, 0.52, 0.46)
    arrow(b, 0.66, 0.56, 0.73, 0.70); arrow(b, 0.66, 0.46, 0.73, 0.38)
    arrow(b, 0.86, 0.72, 0.9, 0.56); arrow(b, 0.86, 0.36, 0.9, 0.46)
    box(b, 0.01, 0.02, 0.98, 0.1, "Baseline for comparison: lexical features only, Random Forest (same folds)", NEUTRAL_FILL, GREY, fs=7.5)

    d = (sig["hybrid_auroc"] - sig["lexical_auroc"]).to_numpy()
    better, same, worse = int((d > 1e-12).sum()), int((np.abs(d) <= 1e-12).sum()), int((d < -1e-12).sum())
    rng = np.random.default_rng(42)
    boots = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(10000)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    panel(c, "(c) Result")
    c.hist(d, bins=np.linspace(d.min() - 0.001, d.max() + 0.001, 30), color=HOST, edgecolor="white")
    c.axvspan(lo, hi, color=LEX, alpha=0.18, label=f"95% CI of mean [{lo:.4f}, {hi:.4f}]")
    c.axvline(0, color=INK, lw=1)
    c.axvline(d.mean(), color=LEX, lw=2, label=f"mean difference {d.mean():+.4f}")
    c.set_xlabel("Per-fold AUROC difference (hybrid minus lexical-only)")
    c.set_ylabel("Number of folds")
    c.legend(loc="upper right", fontsize=8, frameon=False)
    c.text(0.98, 0.5, f"{better} folds hybrid better\n{same} folds equal\n{worse} folds hybrid worse",
           transform=c.transAxes, ha="right", fontsize=9)
    c.spines[["top", "right"]].set_visible(False)
    save(fig, "fig1_hybrid_auroc.png")


def fig2(sig):
    fig, (a, b, c) = new_fig()
    blank(a); panel(a, "(a) Teaser")
    box(a, 0.03, 0.2, 0.38, 0.6, "URL text only\n\nPrecision 1.000\nFalse alarms 0%", LEX_FILL, LEX, fs=9.5)
    box(a, 0.59, 0.2, 0.38, 0.6, "URL text + server checks\n\nPrecision 1.000\nFalse alarms 0%", HOST_FILL, HOST, fs=9.5)
    a.text(0.5, 0.5, "=", ha="center", va="center", fontsize=26, fontweight="bold")
    a.text(0.5, 0.06, "Already at the ceiling, so there was nothing left to improve", ha="center", fontsize=8.5, style="italic")

    blank(b); panel(b, "(b) Architecture")
    steps = ["10 random\nseeds", "Stratified\n10-fold split", "Train both models\non the same\n9 folds", "Score the\nheld-out fold", "Precision, FPR,\nAUROC per fold"]
    xs = np.linspace(0.01, 0.80, 5)
    for i, (x, s) in enumerate(zip(xs, steps)):
        box(b, x, 0.4, 0.17, 0.32, s, HOST_FILL if i == 2 else NEUTRAL_FILL, HOST if i == 2 else INK, fs=7.5)
        if i < 4:
            arrow(b, x + 0.17, 0.56, xs[i + 1], 0.56)
    box(b, 0.18, 0.05, 0.64, 0.17, "100 paired folds → t-test, Wilcoxon, bootstrap CI", NEUTRAL_FILL, GREY, fs=7.5)
    arrow(b, 0.89, 0.4, 0.62, 0.22)

    panel(c, "(c) Result")
    rng = np.random.default_rng(1)
    groups = [("Precision\nlexical-only", sig["lexical_precision"]), ("Precision\nhybrid", sig["hybrid_precision"]),
              ("False positive rate\nlexical-only", sig["lexical_fpr"]), ("False positive rate\nhybrid", sig["hybrid_fpr"])]
    cols = [LEX, HOST, LEX, HOST]
    for i, ((lab, vals), col) in enumerate(zip(groups, cols)):
        c.scatter(i + rng.uniform(-0.18, 0.18, len(vals)), vals, s=14, color=col, alpha=0.7)
    c.set_xticks(range(4)); c.set_xticklabels([g[0] for g in groups], fontsize=8)
    c.set_ylim(-0.05, 1.08); c.set_ylabel("Value in each of the 100 folds")
    ident = bool((sig["lexical_precision"] == sig["hybrid_precision"]).all() and (sig["lexical_fpr"] == sig["hybrid_fpr"]).all())
    c.text(0.5, 0.5, "Identical in all 100 folds" if ident else "Differences found", transform=c.transAxes,
           ha="center", fontsize=10, fontweight="bold")
    c.spines[["top", "right"]].set_visible(False)
    save(fig, "fig2_precision_ceiling.png")


def fig3():
    fig, (a, b, c) = new_fig(left=0.25)
    blank(a); panel(a, "(a) Teaser")
    for i, (_, _, src) in enumerate(SHAP):
        x = 0.03 + i * 0.097
        box(a, x, 0.4, 0.085, 0.28, str(i + 1), LEX_FILL if src == "lex" else HOST_FILL, LEX if src == "lex" else HOST, fs=10)
    a.text(0.5, 0.82, "Top 10 features by influence", ha="center", fontsize=9)
    a.text(0.5, 0.14, "4 of the top 10 come from server checks (orange)", ha="center", fontsize=9, style="italic")

    blank(b); panel(b, "(b) Architecture")
    box(b, 0.01, 0.35, 0.2, 0.3, "Tuned XGBoost\n(trained on the\n300-URL sample)", fs=8)
    box(b, 0.31, 0.35, 0.2, 0.3, "SHAP\nTreeExplainer", fs=8)
    box(b, 0.61, 0.35, 0.2, 0.3, "Mean |SHAP|\nper feature", fs=8)
    box(b, 0.88, 0.35, 0.11, 0.3, "Rank,\ntag by\nsource", fs=7.5)
    arrow(b, 0.21, 0.5, 0.31, 0.5); arrow(b, 0.51, 0.5, 0.61, 0.5); arrow(b, 0.81, 0.5, 0.88, 0.5)
    b.text(0.5, 0.12, "Explains the XGBoost part only, not the full soft vote", ha="center", fontsize=8, style="italic")

    panel(c, "(c) Result")
    names = [s[0] for s in SHAP][::-1]
    vals = [s[1] for s in SHAP][::-1]
    cols = [LEX if s[2] == "lex" else HOST for s in SHAP][::-1]
    c.barh(names, vals, color=cols)
    for y, v in enumerate(vals):
        c.text(v + 0.02, y, f"{v:.3f}", va="center", fontsize=8)
    c.set_xlabel("Mean |SHAP value| (average influence on a prediction)")
    c.legend(handles=[Patch(color=LEX, label="URL text"), Patch(color=HOST, label="Server check")], frameon=False, loc="lower right")
    c.spines[["top", "right"]].set_visible(False)
    save(fig, "fig3_shap_features.png")


def fig4(lat):
    fig, (a, b, c) = new_fig(left=0.13)
    blank(a); panel(a, "(a) Teaser")
    lm, hm = lat["lexical_ms"].mean(), lat["host_telemetry_ms"].mean()
    a.barh([0.68], [lm / hm], height=0.2, color=LEX)
    a.barh([0.3], [1.0], height=0.2, color=HOST)
    a.text(lm / hm + 0.02, 0.68, f"URL text: {lm:.1f} ms", va="center", fontsize=9.5)
    a.text(0.5, 0.3, f"Server checks: {hm:,.0f} ms", va="center", ha="center", fontsize=9.5, color="white", fontweight="bold")
    a.text(0.5, 0.06, f"Roughly {round(hm / lm, -1):,.0f} times slower per URL", ha="center", fontsize=9, style="italic")
    a.set_xlim(0, 1.0); a.set_ylim(0, 1); a.axis("off")

    blank(b); panel(b, "(b) Architecture")
    box(b, 0.01, 0.4, 0.13, 0.22, "Hostname", fs=8)
    for y, t in [(0.78, "DNS: A, MX, TXT"), (0.55, "WHOIS: domain age"), (0.32, "TLS: certificate"), (0.09, "HTTP: headers")]:
        box(b, 0.24, y, 0.22, 0.15, t, HOST_FILL, HOST, fs=7.5)
        arrow(b, 0.14, 0.51, 0.24, y + 0.075)
        arrow(b, 0.46, y + 0.075, 0.55, 0.51)
    box(b, 0.55, 0.38, 0.16, 0.26, "Run together\n(up to 50 at\nonce)", fs=7.5)
    box(b, 0.77, 0.6, 0.21, 0.18, "Disk cache\n(7 days)", NEUTRAL_FILL, GREY, fs=7.5)
    box(b, 0.77, 0.22, 0.21, 0.18, "Feature row\n(12 features)", fs=7.5)
    arrow(b, 0.71, 0.56, 0.77, 0.66); arrow(b, 0.71, 0.46, 0.77, 0.32)

    panel(c, "(c) Result")
    rng = np.random.default_rng(3)
    c.scatter(rng.uniform(-0.15, 0.15, len(lat)), lat["lexical_ms"], s=28, color=LEX, alpha=0.8)
    c.scatter(1 + rng.uniform(-0.15, 0.15, len(lat)), lat["host_telemetry_ms"], s=28, color=HOST, alpha=0.8)
    c.scatter([0, 1], [lm, hm], marker="_", s=900, color=INK, linewidths=2)
    c.set_yscale("log"); c.set_xlim(-0.6, 1.6)
    c.set_xticks([0, 1]); c.set_xticklabels(["URL text", "Server checks"])
    c.set_ylabel("Time per URL in milliseconds (log scale)")
    c.text(0.1, lm * 1.5, f"mean {lm:.1f} ms", fontsize=8.5)
    c.text(1.17, hm * 0.8, f"mean {hm:,.0f} ms", fontsize=8.5)
    c.set_xlabel(f"{len(lat)} live URLs, each point is one URL", fontsize=8.5)
    c.spines[["top", "right"]].set_visible(False)
    save(fig, "fig4_latency_cost.png")


if __name__ == "__main__":
    sig = pd.read_csv(os.path.join(PROCESSED, "significance_test_results.csv"))
    lat = pd.read_csv(os.path.join(PROCESSED, "latency_results.csv"))
    fig1(sig)
    fig2(sig)
    fig3()
    fig4(lat)
