"""Five three-panel paper figures (teaser, architecture, result), one per claim.

Reads the CSV/JSON results written by fair_eval.py, temporal_validation.py and
latency_analysis.py. Nothing here is typed in by hand except a few counts that
are printed in the matching report JSON files.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Patch
from sklearn.metrics import roc_curve

P = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
OUT = os.path.join(os.path.dirname(__file__), "..", "..", "reports", "figures")

LEX, HOST, HYB, GREY, INK = "#3A6EA5", "#E08E45", "#3F8F6B", "#8A8F98", "#222222"
LEX_F, HOST_F, HYB_F, NEUT_F = "#DCE8F5", "#FBE6D2", "#DDEFE6", "#F1F2F4"


def box(ax, x, y, w, h, text, fc=NEUT_F, ec=INK, fs=8.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.005,rounding_size=0.02", fc=fc, ec=ec, lw=1.0))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color=INK)


def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", lw=1.2, color="#444444", shrinkA=0, shrinkB=0))


def panel(ax, label):
    ax.text(0.0, 1.04, label, transform=ax.transAxes, fontsize=10.5, fontweight="bold", va="bottom")


def blank(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def new_fig(left=0.1):
    fig = plt.figure(figsize=(7.0, 9.2))
    gs = fig.add_gridspec(3, 1, height_ratios=[0.8, 1.0, 1.35], hspace=0.3, left=left, right=0.97, top=0.96, bottom=0.07)
    return fig, [fig.add_subplot(gs[i]) for i in range(3)]


def widen(ax, left):
    pos = ax.get_position()
    ax.set_position([left, pos.y0, 0.97 - left, pos.height])


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name), dpi=170)
    plt.close(fig)
    print("wrote", name)


def rd(name):
    return pd.read_csv(os.path.join(P, name))


def fig1(infl, rep):
    fig, (a, b, c) = new_fig()
    o = infl[infl.data.str.startswith("original")].iloc[1]
    f = infl[infl.data.str.startswith("fair")].iloc[1]
    blank(a); panel(a, "(a) Teaser")
    box(a, 0.02, 0.2, 0.4, 0.6, f"Original comparison\nbenign = bare domain\n\nAUROC {o.auroc:.3f}", LEX_F, LEX, fs=9.5)
    box(a, 0.58, 0.2, 0.4, 0.6, f"Fair comparison\nsame form for both classes\n\nAUROC {f.auroc:.3f}", HOST_F, HOST, fs=9.5)
    arrow(a, 0.43, 0.5, 0.57, 0.5)
    a.text(0.5, 0.06, "Most of the original score came from the shape of the URLs", ha="center", fontsize=8.5, style="italic")

    blank(b); panel(b, "(b) Architecture")
    ph = rep["distinct_phishing_hostnames"]; dropped = rep["phishing_hostnames_on_popular_legit_domains_dropped"]
    row1 = [("PhishTank,\nOpenPhish,\nMajestic", NEUT_F, INK), ("Keep the hostname\nonly, for both\nclasses", HYB_F, HYB),
            (f"Drop duplicates\n({ph:,} phishing\nhostnames left)", NEUT_F, INK), (f"Drop phishing on popular\nlegit domains\n({dropped:,}, {dropped / ph:.0%})", NEUT_F, INK)]
    xs = [0.01, 0.26, 0.51, 0.76]
    for x, (t, fc, ec) in zip(xs, row1):
        box(b, x, 0.58, 0.23, 0.32, t, fc, ec, fs=7.2)
    for i in range(3):
        arrow(b, xs[i] + 0.23, 0.74, xs[i + 1], 0.74)
    box(b, 0.01, 0.1, 0.3, 0.32, "Keep at most 3 hosts\nper registered domain", NEUT_F, INK, fs=7.5)
    box(b, 0.37, 0.1, 0.26, 0.32, f"{rep['final_phishing'] + rep['final_benign']:,} hosts\n({rep['final_phishing']:,} phishing,\n{rep['final_benign']:,} benign)", NEUT_F, INK, fs=7.5)
    box(b, 0.69, 0.1, 0.3, 0.32, "Cross-validation keeps\neach domain on one side", HYB_F, HYB, fs=7.5)
    arrow(b, 0.88, 0.58, 0.16, 0.42); arrow(b, 0.31, 0.26, 0.37, 0.26); arrow(b, 0.63, 0.26, 0.69, 0.26)

    panel(c, "(c) Result")
    labels = ["Original comparison", "Fair comparison"]
    r = [infl[infl.data.str.startswith(k)] for k in ("original", "fair")]
    w = 0.34
    for j, (cv, col) in enumerate([("random rows", GREY), ("domain-grouped", LEX)]):
        vals = [x[x.cv == cv].auroc.iloc[0] for x in r]
        bars = c.bar(np.arange(2) + (j - 0.5) * w, vals, w, color=col, label=f"Cross-validation: {cv}")
        for bar, v in zip(bars, vals):
            c.text(bar.get_x() + bar.get_width() / 2, v + 0.004, f"{v:.3f}", ha="center", fontsize=8.5)
    c.set_xticks(range(2)); c.set_xticklabels([f"Original comparison\n(accuracy {o.accuracy:.1%})", f"Fair comparison\n(accuracy {f.accuracy:.1%})"]); c.set_ylim(0.8, 1.04)
    c.set_ylabel("AUROC, URL text only (XGBoost)")
    c.legend(loc="upper right", fontsize=8, frameon=False)
    c.spines[["top", "right"]].set_visible(False)
    save(fig, "fig1_dataset_fairness.png")


def fig2(folds, oof):
    s = folds.groupby("model")[["auroc", "recall", "fpr", "fpr_at_99tpr"]].mean()
    fig, (a, b, c) = new_fig()
    blank(a); panel(a, "(a) Teaser")
    box(a, 0.02, 0.2, 0.4, 0.6, f"URL text only\n\nAUROC {s.loc['xgb_lexical','auroc']:.3f}\nfinds {s.loc['xgb_lexical','recall']:.0%} of phishing", LEX_F, LEX, fs=9.5)
    box(a, 0.58, 0.2, 0.4, 0.6, f"URL text + server checks\n\nAUROC {s.loc['xgb_hybrid','auroc']:.3f}\nfinds {s.loc['xgb_hybrid','recall']:.0%} of phishing", HYB_F, HYB, fs=9.5)
    arrow(a, 0.43, 0.5, 0.57, 0.5)
    a.text(0.5, 0.06, f"False alarm rate stays about the same ({s.loc['xgb_lexical','fpr']:.1%} vs {s.loc['xgb_hybrid','fpr']:.1%})", ha="center", fontsize=8.5, style="italic")

    blank(b); panel(b, "(b) Architecture")
    box(b, 0.01, 0.38, 0.14, 0.26, "Hostname", fs=8)
    box(b, 0.22, 0.64, 0.24, 0.26, "URL text features\n12 (length, dots, digits,\nentropy, subdomains...)", LEX_F, LEX, fs=7)
    box(b, 0.22, 0.14, 0.24, 0.26, "Server checks\n12 (DNS, SSL, WHOIS,\nHTTP headers)", HOST_F, HOST, fs=7)
    box(b, 0.54, 0.38, 0.14, 0.26, "Feature\ntable", fs=8)
    box(b, 0.76, 0.38, 0.2, 0.26, "XGBoost\n(tuned on a\nseparate 20%)", fs=7.5)
    arrow(b, 0.15, 0.54, 0.22, 0.74); arrow(b, 0.15, 0.48, 0.22, 0.28)
    arrow(b, 0.46, 0.74, 0.54, 0.58); arrow(b, 0.46, 0.28, 0.54, 0.44); arrow(b, 0.68, 0.51, 0.76, 0.51)
    b.text(0.5, 0.02, "Baseline: the same model on URL text features only. 10 repeats x 5 folds, split by domain.", ha="center", fontsize=7.5, style="italic")

    panel(c, "(c) Result")
    y = oof.label.to_numpy()
    for col, lab, colr in [("p_lexical", "URL text only", LEX), ("p_host_only", "Server checks only", HOST), ("p_hybrid", "Hybrid", HYB)]:
        fpr, tpr, _ = roc_curve(y, oof[col])
        auc = s.loc[{"p_lexical": "xgb_lexical", "p_host_only": "xgb_host_only", "p_hybrid": "xgb_hybrid"}[col], "auroc"]
        c.plot(fpr, tpr, color=colr, lw=2, label=f"{lab} (AUROC {auc:.3f})")
    c.plot([0, 1], [0, 1], color=GREY, lw=1, ls=":")
    c.axhline(0.99, color=INK, lw=0.8, ls="--")
    c.text(0.3, 0.3, f"At 99% recall, false alarm rate: {s.loc['xgb_lexical','fpr_at_99tpr']:.2f} (text only) vs {s.loc['xgb_hybrid','fpr_at_99tpr']:.2f} (hybrid)", fontsize=8, va="top", wrap=True)
    c.set_xlabel("False positive rate (real sites flagged)"); c.set_ylabel("True positive rate (phishing found)")
    c.set_xlim(0, 1); c.set_ylim(0, 1.02); c.legend(loc="lower right", fontsize=8.5, frameon=False)
    c.spines[["top", "right"]].set_visible(False)
    save(fig, "fig2_hybrid_detection.png")


def fig3(folds, fast):
    stat = lambda df, m: (df[df.model == m].auroc.mean(), df[df.model == m].auroc.std())
    rows = [("URL text only", *stat(folds, "xgb_lexical"), LEX), ("+ HTTP headers", *stat(folds, "xgb_lexical+http"), HOST),
            ("+ WHOIS", *stat(folds, "xgb_lexical+whois"), HOST), ("+ DNS", *stat(folds, "xgb_lexical+dns"), HOST),
            ("+ SSL", *stat(folds, "xgb_lexical+ssl"), HOST), ("+ DNS + SSL", *stat(fast, "xgb_lexical+dns+ssl"), HYB),
            ("All checks, XGBoost", *stat(folds, "xgb_hybrid"), HYB), ("All checks, ensemble", *stat(folds, "ensemble_hybrid"), HYB),
            ("All checks, Random Forest", *stat(folds, "rf_hybrid"), HYB)]
    base = rows[0][1]
    fig, (a, b, c) = new_fig()
    blank(a); panel(a, "(a) Teaser")
    for i, (nm, k) in enumerate([("DNS", 3), ("SSL", 4), ("WHOIS", 2), ("HTTP", 1)]):
        gain = rows[k][1] - base
        strong = gain > 0.05
        box(a, 0.03 + i * 0.245, 0.25, 0.22, 0.5, f"{nm}\n{gain:+.3f}", HYB_F if strong else (HOST_F if gain > 0.01 else NEUT_F),
            HYB if strong else (HOST if gain > 0.01 else GREY), fs=10)
    a.text(0.5, 0.88, f"AUROC gained when one check is added to URL text ({base:.3f})", ha="center", fontsize=9)
    a.text(0.5, 0.06, "SSL and DNS carry most of the gain; HTTP headers add nothing", ha="center", fontsize=8.5, style="italic")

    blank(b); panel(b, "(b) Architecture")
    box(b, 0.01, 0.55, 0.3, 0.32, "URL text features\n+ ONE group of checks", LEX_F, LEX, fs=7.5)
    box(b, 0.38, 0.55, 0.28, 0.32, "Same XGBoost,\nsame domain-split\nfolds", fs=7.5)
    box(b, 0.73, 0.55, 0.25, 0.32, "Compare AUROC\nacross 50 folds", fs=7.5)
    arrow(b, 0.31, 0.71, 0.38, 0.71); arrow(b, 0.66, 0.71, 0.73, 0.71)
    box(b, 0.01, 0.1, 0.3, 0.3, "All checks together", HYB_F, HYB, fs=7.5)
    box(b, 0.38, 0.1, 0.6, 0.3, "XGBoost alone vs Random Forest alone\nvs weighted 2:1 ensemble of both", fs=7.5)
    arrow(b, 0.31, 0.25, 0.38, 0.25)

    widen(c, 0.33)
    panel(c, "(c) Result")
    names = [r[0] for r in rows][::-1]; means = [r[1] for r in rows][::-1]; sds = [r[2] for r in rows][::-1]; cols = [r[3] for r in rows][::-1]
    c.barh(names, means, xerr=sds, color=cols, error_kw=dict(lw=1, capsize=2))
    for yy, v in enumerate(means):
        c.text(v + 0.012, yy, f"{v:.3f}", va="center", fontsize=8)
    c.set_xlim(0.8, 1.04); c.set_xlabel("AUROC (mean of 50 domain-split folds; bars = std. dev.)", fontsize=8.5)
    c.legend(handles=[Patch(color=LEX, label="URL text only"), Patch(color=HOST, label="One check group added"), Patch(color=HYB, label="Several checks")],
             loc="upper right", fontsize=8, frameon=False)
    c.spines[["top", "right"]].set_visible(False)
    save(fig, "fig3_which_checks.png")


def fig4(lat, fast, casc):
    L, U, C = lat["lexical_ms"], lat["host_uncached_ms"], lat["host_cached_ms"]
    F = fast["dns_plus_ssl_uncached_ms"]
    fig, (a, b, c) = new_fig()
    widen(a, 0.3)
    panel(a, "(a) Teaser")
    names = ["URL text model", "DNS + SSL checks", "All server checks", "All checks, cached"]
    med = [L["median"], F["median"], U["median"], C["median"]]
    p90 = [L["p90"], F["p90"], U["p90"], C["p90"]]
    yy = np.arange(4)[::-1]
    a.barh(yy + 0.18, med, 0.34, color=LEX, label="median")
    a.barh(yy - 0.18, p90, 0.34, color=HOST, label="slowest 10% (90th percentile)")
    a.set_xscale("log"); a.set_yticks(yy); a.set_yticklabels(names, fontsize=8.5)
    for y_, m_, p_ in zip(yy, med, p90):
        a.text(m_ * 1.3, y_ + 0.18, f"{m_:,.2f} ms" if m_ < 10 else f"{m_:,.0f} ms", va="center", fontsize=7.5)
        a.text(p_ * 1.3, y_ - 0.18, f"{p_:,.2f} ms" if p_ < 10 else f"{p_:,.0f} ms", va="center", fontsize=7.5)
    a.set_xlim(0.005, 60000); a.set_xlabel("Time per host (milliseconds, log scale)", fontsize=8.5)
    a.legend(loc="lower right", fontsize=7.5, frameon=False); a.spines[["top", "right"]].set_visible(False)

    blank(b); panel(b, "(b) Architecture")
    box(b, 0.01, 0.4, 0.14, 0.24, "Hostname", fs=8)
    box(b, 0.22, 0.4, 0.2, 0.24, "URL text model\n(instant)", LEX_F, LEX, fs=7.5)
    box(b, 0.5, 0.4, 0.2, 0.24, "Score within\n0.5 ± δ ?", fs=8, bold=True)
    box(b, 0.78, 0.7, 0.2, 0.22, "Yes: run DNS + SSL\nthen hybrid model", HYB_F, HYB, fs=7)
    box(b, 0.78, 0.1, 0.2, 0.22, "No: keep the\ntext-model verdict", LEX_F, LEX, fs=7)
    arrow(b, 0.15, 0.52, 0.22, 0.52); arrow(b, 0.42, 0.52, 0.5, 0.52)
    arrow(b, 0.7, 0.58, 0.78, 0.78); arrow(b, 0.7, 0.46, 0.78, 0.22)
    b.text(0.5, 0.02, "δ sets how often the slow checks run", ha="center", fontsize=8, style="italic")

    panel(c, "(c) Result")
    c.plot(casc.share_needing_server_checks * 100, casc.auroc, "-o", color=HYB, lw=2, ms=5)
    for _, r in casc.iterrows():
        if r.delta in (0.0, 0.2, 0.3, 0.5):
            c.annotate(f"δ={r.delta:g}", (r.share_needing_server_checks * 100, r.auroc), textcoords="offset points", xytext=(-8, 10) if r.delta == 0.5 else (6, -12), fontsize=8)
    sel = casc[casc.delta == 0.3].iloc[0]
    c.annotate(f"δ=0.3: {sel.share_needing_server_checks:.0%} of hosts need checks,\nAUROC {sel.auroc:.3f} (full hybrid {casc.auroc.iloc[-1]:.3f})",
               (sel.share_needing_server_checks * 100, sel.auroc), xytext=(45, 0.89), fontsize=8, arrowprops=dict(arrowstyle="->", color=GREY))
    c.set_xlim(-5, 108); c.set_xlabel("Share of hosts that need the slow server checks (%)"); c.set_ylabel("AUROC of the cascade")
    c.text(0.02, 0.97, f"Average time = share x {U['mean']:,.0f} ms (all checks, measured mean)", transform=c.transAxes, fontsize=8, va="top")
    c.spines[["top", "right"]].set_visible(False)
    save(fig, "fig4_latency_cascade.png")


def fig5(tres, tl, trep, base):
    t = tres.set_index("model")
    fig, (a, b, c) = new_fig()
    blank(a); panel(a, "(a) Teaser")
    box(a, 0.02, 0.2, 0.4, 0.6, f"URL text only\n\nAUROC {t.loc['xgb_lexical','auroc']:.3f}", LEX_F, LEX, fs=10)
    box(a, 0.58, 0.2, 0.4, 0.6, f"URL text + server checks\n\nAUROC {t.loc['xgb_hybrid','auroc']:.3f}", HYB_F, HYB, fs=10)
    arrow(a, 0.43, 0.5, 0.57, 0.5)
    a.text(0.5, 0.06, "On phishing domains the model had never seen", ha="center", fontsize=8.5, style="italic")

    blank(b); panel(b, "(b) Architecture")
    box(b, 0.01, 0.62, 0.3, 0.3, "Train: September 16\nsnapshot", LEX_F, LEX, fs=7.5)
    box(b, 0.01, 0.12, 0.3, 0.3, f"Fresh feeds, Oct 22:\n{trep['new_phishing_hosts_in_feed']:,} phishing hosts", NEUT_F, INK, fs=7.5)
    box(b, 0.4, 0.12, 0.28, 0.3, f"Keep unseen domains\nonly: {trep['final_phishing']:,} phishing\n+ {trep['final_benign']:,} benign\n(Majestic ranks 50k-60k)", HYB_F, HYB, fs=7)
    box(b, 0.78, 0.37, 0.2, 0.3, "Score the model\ntrained on\nSeptember", fs=7.5)
    arrow(b, 0.31, 0.27, 0.4, 0.27); arrow(b, 0.68, 0.34, 0.78, 0.46); arrow(b, 0.31, 0.77, 0.78, 0.6)

    panel(c, "(c) Result")
    models = [("xgb_lexical", "URL text only", LEX), ("xgb_host_only", "Server checks only", HOST), ("xgb_hybrid", "Hybrid", HYB)]
    w = 0.34
    for j, (key, lab) in enumerate([("in", "Same period (domain-split CV)"), ("new", "Newer data (unseen domains)")]):
        vals = [base[m] if key == "in" else t.loc[m, "auroc"] for m, _, _ in models]
        bars = c.bar(np.arange(3) + (j - 0.5) * w, vals, w, color=[m[2] for m in models], alpha=1.0 if key == "in" else 0.55,
                     edgecolor=INK, hatch=None if key == "in" else "//", label=lab)
        for bar, v in zip(bars, vals):
            c.text(bar.get_x() + bar.get_width() / 2, v + 0.004, f"{v:.3f}", ha="center", fontsize=8)
    c.set_xticks(range(3)); c.set_xticklabels([m[1] for m in models]); c.set_ylim(0.75, 1.1); c.set_ylabel("AUROC")
    c.legend(handles=[Patch(fc="white", ec=INK, label="Same period (domain-split CV)"), Patch(fc="white", ec=INK, hatch="//", label="Newer data (unseen domains)")],
             loc="upper right", fontsize=8, frameon=False)
    d = tl[0]
    c.text(0.02, 0.97, f"Hybrid minus text-only on newer data: {d['mean_diff']:+.3f} AUROC\n(95% CI {d['ci_low']:+.3f} to {d['ci_high']:+.3f}, resampling whole domains)",
           transform=c.transAxes, fontsize=8, va="top")
    c.spines[["top", "right"]].set_visible(False)
    save(fig, "fig5_newer_data.png")


if __name__ == "__main__":
    folds = pd.concat([rd("fair_hybrid_folds.csv")])
    fast = rd("fair_fast_subset_folds.csv")
    base = folds.groupby("model").auroc.mean().to_dict()
    fig1(rd("fair_lexical_inflation.csv"), json.load(open(os.path.join(P, "fair_dataset_report.json"))))
    fig2(folds, rd("fair_oof_predictions.csv"))
    fig3(folds, fast)
    fig4(json.load(open(os.path.join(P, "latency_extended.json"))), json.load(open(os.path.join(P, "latency_fast_path.json"))), rd("fair_cascade.csv"))
    fig5(rd("temporal_hybrid_results.csv"), json.load(open(os.path.join(P, "temporal_hybrid_bootstrap.json"))),
         json.load(open(os.path.join(P, "temporal_report.json"))), base)
