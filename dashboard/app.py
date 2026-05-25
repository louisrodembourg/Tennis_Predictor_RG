"""
Dashboard Streamlit — Roland Garros 2026
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

DATA_DIR = str(Path(__file__).parent.parent / "data" / "raw")
RG2026_PATH = str(Path(__file__).parent.parent / "data" / "rg2026" / "results.jsonl")

ROUND_LABELS = {
    -2: "Qualifs — 1er tour",
    -1: "Qualifs — 2e tour",
    0:  "Qualifs — 3e tour (dernier)",
    1:  "1er tour (R128)",
    2:  "2e tour (R64)",
    3:  "3e tour (R32)",
    4:  "8e de finale",
    5:  "Quarts de finale",
    6:  "Demi-finale",
    7:  "Finale",
}

PLAYER_LIST_DEFAULT = [
    "Carlos Alcaraz", "Jannik Sinner", "Alexander Zverev", "Casper Ruud",
    "Stefanos Tsitsipas", "Hubert Hurkacz", "Andrey Rublev", "Taylor Fritz",
    "Tommy Paul", "Ben Shelton", "Holger Rune", "Grigor Dimitrov",
    "Felix Auger-Aliassime", "Francisco Cerundolo", "Lorenzo Musetti",
    "Sebastian Baez", "Flavio Cobolli", "Arthur Fils", "Gael Monfils",
    "Stan Wawrinka", "Rafael Nadal", "Novak Djokovic",
]


# ------------------------------------------------------------------
# Initialisation du prédicteur (mis en cache)
# ------------------------------------------------------------------

@st.cache_resource(show_spinner="Chargement des données et entraînement du modèle...")
def get_predictor():
    from predictor import RolandGarrosPredictor
    return RolandGarrosPredictor(
        historical_data_path=DATA_DIR,
        rg2026_path=RG2026_PATH,
    )


# ------------------------------------------------------------------
# Composants UI réutilisables
# ------------------------------------------------------------------

def proba_bar(proba_a: float, player_a: str, player_b: str):
    color_a = "#e8473f" if proba_a >= 0.5 else "#aaaaaa"
    color_b = "#aaaaaa" if proba_a >= 0.5 else "#e8473f"
    fig = go.Figure(go.Bar(
        x=[proba_a * 100, (1 - proba_a) * 100],
        y=[player_a, player_b],
        orientation="h",
        marker_color=[color_a, color_b],
        text=[f"{proba_a*100:.1f}%", f"{(1-proba_a)*100:.1f}%"],
        textposition="inside",
    ))
    fig.update_layout(
        height=120, margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(range=[0, 100], showticklabels=False),
        showlegend=False,
    )
    return fig


def confidence_gauge(score: float, label: str):
    color = "#e8473f" if score >= 40 else ("#f5a623" if score >= 20 else "#aaaaaa")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": "%", "font": {"size": 28}},
        title={"text": f"Confiance — {label}", "font": {"size": 14}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 20], "color": "#f0f0f0"},
                {"range": [20, 40], "color": "#ffe0b2"},
                {"range": [40, 100], "color": "#ffcdd2"},
            ],
            "threshold": {"line": {"color": "#333", "width": 3}, "thickness": 0.75, "value": score},
        },
    ))
    fig.update_layout(height=220, margin=dict(l=20, r=20, t=40, b=10))
    return fig


def radar_chart(features_a: dict, features_b: dict, player_a: str, player_b: str):
    cats = ["Clay Elo", "WElo", "Clay WR 12m", "RG Win Rate", "1st Serve Won", "BP Saved"]
    vals_a = [
        features_a.get("clay_elo", 1500) / 2000,
        features_a.get("welo", 1500) / 2000,
        features_a.get("win_rate_clay_12m", 0.5),
        features_a.get("rg_win_rate", 0.5),
        features_a.get("first_serve_won_pct", 0.7),
        features_a.get("bp_saved_pct", 0.6),
    ]
    vals_b = [
        features_b.get("clay_elo", 1500) / 2000,
        features_b.get("welo", 1500) / 2000,
        features_b.get("win_rate_clay_12m", 0.5),
        features_b.get("rg_win_rate", 0.5),
        features_b.get("first_serve_won_pct", 0.7),
        features_b.get("bp_saved_pct", 0.6),
    ]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=vals_a + [vals_a[0]], theta=cats + [cats[0]],
                                   fill="toself", name=player_a, line_color="#e8473f"))
    fig.add_trace(go.Scatterpolar(r=vals_b + [vals_b[0]], theta=cats + [cats[0]],
                                   fill="toself", name=player_b, line_color="#3f7de8"))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                       height=350, margin=dict(l=20, r=20, t=30, b=20))
    return fig


def features_table(top_features: list, player_a: str, player_b: str):
    """Affiche les features décisives avec direction et importance."""
    if not top_features:
        return
    rows = []
    for f in top_features:
        favors = f["favors"]
        if favors == player_a:
            badge = f"✅ {player_a}"
        elif favors == player_b:
            badge = f"✅ {player_b}"
        else:
            badge = "—"
        rows.append({
            "Feature": f["feature"],
            "Valeur (A−B)": f["value"],
            "Faveur": badge,
            "Importance modèle": f["importance"],
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, width='stretch', hide_index=True)


# ------------------------------------------------------------------
# Pages
# ------------------------------------------------------------------

def page_results(predictor):
    st.title("Saisie des Résultats RG 2026")

    st.info(
        "Saisissez chaque résultat après le match — qualifications comprises. "
        "Cliquez sur **Entraîner le modèle** une fois tous les résultats saisis."
    )

    from data_loader import get_player_list
    all_players_hist = set(get_player_list(predictor.df) or PLAYER_LIST_DEFAULT)

    # Ajoute les joueurs du tirage (inclut les qualifiés non présents en base)
    try:
        from rg2026_bracket import load_draw, DRAW_PATH
        if DRAW_PATH.exists():
            draw_players = [r["player_name"] for r in load_draw() if r.get("player_name")]
            all_players_hist.update(draw_players)
    except Exception:
        pass

    all_players = sorted(all_players_hist)

    # Formulaire de saisie
    with st.form("result_form"):
        col1, col2 = st.columns(2)
        with col1:
            winner = st.selectbox("Vainqueur", all_players, key="res_winner")
        with col2:
            loser_opts = [p for p in all_players if p != winner]
            loser = st.selectbox("Perdant", loser_opts, key="res_loser")

        col3, col4 = st.columns(2)
        with col3:
            score = st.text_input("Score (ex: 6-4 7-5 6-3)", "6-4 6-3")
        with col4:
            round_num = st.select_slider(
                "Tour",
                options=list(ROUND_LABELS.keys()),
                format_func=lambda x: ROUND_LABELS[x],
                value=1,
                key="res_round",
            )

        match_date = st.date_input(
            "Date du match",
            value=pd.Timestamp("2026-05-25").date(),
            min_value=pd.Timestamp("2026-05-18").date(),
            max_value=pd.Timestamp("2026-06-08").date(),
        )

        submitted = st.form_submit_button("Enregistrer le résultat", type="primary")

    if submitted:
        predictor.add_result(winner, loser, score, round_num, match_date=str(match_date))
        st.success(f"Résultat enregistré : **{winner}** bat {loser} ({score}) — {ROUND_LABELS[round_num]}")
        st.rerun()

    # Bouton d'entraînement manuel
    st.divider()
    col_train, col_info = st.columns([1, 3])
    with col_train:
        if st.button("Entraîner le modèle", type="primary", disabled=not predictor._rg2026_results):
            with st.spinner("Entraînement en cours…"):
                predictor.retrain()
            st.success("Modèle réentraîné sur les résultats actuels.")
            st.cache_resource.clear()
            st.rerun()
    with col_info:
        n = len(predictor._rg2026_results)
        if n == 0:
            st.caption("Aucun résultat saisi — entraînement non disponible.")
        else:
            st.caption(f"{n} résultat(s) disponible(s) pour l'entraînement.")

    # Résultats déjà saisis
    st.subheader("Résultats enregistrés")
    if predictor._rg2026_results:
        header = st.columns([2, 2, 2, 2, 1])
        for col, label in zip(header, ["Tour", "Vainqueur", "Perdant", "Score", ""]):
            col.markdown(f"**{label}**")
        st.divider()

        for i, r in enumerate(predictor._rg2026_results):
            col_tour, col_w, col_l, col_s, col_del = st.columns([2, 2, 2, 2, 1])
            col_tour.write(ROUND_LABELS.get(r.get("round_number"), r.get("round_number", "")))
            col_w.write(r.get("winner", ""))
            col_l.write(r.get("loser", ""))
            col_s.write(r.get("score", ""))
            if col_del.button("Supprimer", key=f"del_result_{i}"):
                predictor.delete_result(i)
                st.cache_resource.clear()
                st.rerun()

        # Stats de fatigue par joueur
        st.subheader("État de fatigue intra-tournoi")
        fatigue_rows = []
        for player, state in predictor._intra_rg.items():
            fatigue_rows.append({
                "Joueur": player,
                "Matchs joués": state.get("matches_played", 0),
                "Sets joués": state.get("sets_played", 0),
            })
        if fatigue_rows:
            fat_df = pd.DataFrame(fatigue_rows).sort_values("Sets joués", ascending=False)
            st.dataframe(fat_df, width='stretch', hide_index=True)
    else:
        st.write("Aucun résultat encore saisi.")


def _compare_optuna_vs_default(results_default: list, results_opt: list):
    """Tableau comparatif Brier Base — params défaut vs params Optuna."""
    st.subheader("Comparaison : défaut vs paramètres optimisés")
    rows = []
    opt_by_year = {r.year: r for r in results_opt}
    for r in results_default:
        r_opt = opt_by_year.get(r.year)
        rows.append({
            "Année": r.year,
            "Brier défaut": f"{r.brier:.4f}",
            "Brier Optuna": f"{r_opt.brier:.4f}" if r_opt else "—",
            "Δ Brier": f"{r_opt.brier - r.brier:+.4f}" if r_opt else "—",
            "Acc défaut": f"{r.accuracy:.1%}",
            "Acc Optuna": f"{r_opt.accuracy:.1%}" if r_opt else "—",
        })
    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

    # Moyennes
    def _mean(rs, attr):
        vals = [getattr(r, attr) for r in rs]
        return np.mean(vals)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Brier moyen — défaut", f"{_mean(results_default, 'brier'):.4f}")
    with col2:
        st.metric("Brier moyen — Optuna", f"{_mean(results_opt, 'brier'):.4f}")
    with col3:
        delta = _mean(results_opt, "brier") - _mean(results_default, "brier")
        st.metric("Δ Brier", f"{delta:+.4f}", delta_color="inverse")


def page_backtest(predictor):
    st.title("Backtesting & Performance Historique")

    results_path = Path(__file__).parent.parent / "results" / "backtest_history.txt"

    col_run1, col_run2 = st.columns([3, 1])
    with col_run1:
        run_label = st.text_input(
            "Étiquette du run (optionnel)",
            placeholder="ex: v2_39feat_optuna, clay_cal, temporal_0.1…",
            help="Identifie ce run dans l'historique et le comparateur.",
        )
    with col_run2:
        st.markdown("<br>", unsafe_allow_html=True)
        launch_bt = st.button("Lancer le backtesting (RG 2017-2025)", type="primary")

    if launch_bt:
        with st.spinner("Backtesting en cours (peut prendre quelques minutes)..."):
            from backtesting import run_full_backtest
            from features import FEATURE_COLS
            from backtest_logger import save_backtest_results

            bt = run_full_backtest(str(Path(predictor.data_path)))
            results = bt["backtest_results"]
            baseline_df = bt["baseline_comparison"]

            config = {
                "features": FEATURE_COLS,
                "n_features": len(FEATURE_COLS),
                "model": "XGBoost (n_estimators=481, max_depth=6, lr=0.135) + CalibratedCV isotonic cv=4",
                "données": "ATP 2000-2025 clay",
                "test": "RG 2017-2025 expanding window",
            }
            saved_path = save_backtest_results(results, baseline_df, config, label=run_label.strip())

        st.success(f"Résultats sauvegardés dans `{saved_path.relative_to(Path(__file__).parent.parent)}`")

        # Persister les résultats dans session_state pour survivre aux re-renders
        st.session_state["bt_results"] = results
        st.session_state["bt_baseline"] = baseline_df
        st.session_state["bt_all_feat"] = bt["all_features"]
        st.session_state["bt_rg_feat"]  = bt["rg_features"]
        if "predictions_df" in bt:
            st.session_state["predictions_df"] = bt["predictions_df"]

    # Afficher les résultats s'ils existent en session (persistent après le clic)
    if "bt_results" in st.session_state and st.session_state["bt_results"]:
        _display_backtest(st.session_state["bt_results"], st.session_state["bt_baseline"])

    # ----------------------------------------------------------------
    # Comparaison des runs historiques
    # ----------------------------------------------------------------
    st.markdown("---")
    _display_runs_comparison()

    # ----------------------------------------------------------------
    # Section Optuna — optimisation des hyperparamètres
    # ----------------------------------------------------------------
    st.markdown("---")
    st.subheader("Optimisation des hyperparamètres (Optuna)")

    has_feats = "bt_all_feat" in st.session_state and "bt_rg_feat" in st.session_state
    if not has_feats:
        st.info("Lancez d'abord le backtesting ci-dessus pour charger les features nécessaires à Optuna.")
    else:
        from hyperopt import (
            CV_YEARS, BEST_PARAMS_PATH, load_best_params,
            run_optuna, save_best_params, get_xgb_params,
        )

        existing = load_best_params()
        if existing:
            st.success(
                f"Meilleurs params trouvés : Brier = **{existing['_best_brier']:.4f}** "
                f"({existing['_n_trials']} trials) — `{BEST_PARAMS_PATH.name}`"
            )

        col_opt1, col_opt2 = st.columns([1, 3])
        with col_opt1:
            n_trials = st.number_input("Nombre de trials", min_value=10, max_value=500, value=50, step=10)
        with col_opt2:
            st.caption(
                f"CV sur {len(CV_YEARS)} années ({CV_YEARS[0]}–{CV_YEARS[-1]}) — "
                f"2023–2025 gardés comme holdout. "
                f"Durée estimée : {n_trials * 6 * 3 // 60 + 1}–{n_trials * 6 * 8 // 60 + 1} min."
            )

        if st.button("Lancer Optuna", type="primary"):
            progress_placeholder = st.empty()
            progress_bar = st.progress(0)
            log_lines: list[str] = []

            def _on_trial(n_done: int, best_val: float, _best_p: dict) -> None:
                pct = int(n_done / n_trials * 100)
                progress_bar.progress(pct)
                log_lines.append(f"Trial {n_done:3d} | Best Brier : {best_val:.4f}")
                progress_placeholder.code("\n".join(log_lines[-8:]), language="text")

            with st.spinner("Optuna en cours..."):
                study = run_optuna(
                    st.session_state["bt_all_feat"],
                    st.session_state["bt_rg_feat"],
                    n_trials=int(n_trials),
                    on_trial_end=_on_trial,
                )
            best = save_best_params(study)
            st.session_state["optuna_best"] = best
            progress_bar.progress(100)
            st.success(f"Optimisation terminée — Best Brier : **{best['_best_brier']:.4f}**")
            st.rerun()

        # Affichage des meilleurs paramètres
        display_best = st.session_state.get("optuna_best") or existing
        if display_best:
            st.markdown("**Meilleurs paramètres trouvés :**")
            xgb_p = get_xgb_params(display_best)
            cal_method = display_best.get("calibration_method", "isotonic")
            cal_cv = int(display_best.get("calibration_cv", 3))

            col_a, col_b = st.columns(2)
            with col_a:
                params_df = pd.DataFrame(
                    [{"Paramètre": k, "Valeur": v} for k, v in xgb_p.items()]
                )
                st.dataframe(params_df, width='stretch', hide_index=True)
            with col_b:
                st.metric("calibration_method", cal_method)
                st.metric("calibration_cv", cal_cv)
                st.metric("CV Brier (2017-2022)", f"{display_best['_best_brier']:.4f}")

            with st.expander("Snippet à copier dans model.py"):
                snippet_lines = ["XGB_PARAMS = {"]
                for k, v in xgb_p.items():
                    val_repr = f'"{v}"' if isinstance(v, str) else repr(v)
                    snippet_lines.append(f'    "{k}": {val_repr},')
                snippet_lines.append("}")
                st.code("\n".join(snippet_lines), language="python")

            # Valider avec un backtest complet sur les mêmes données
            has_default_results = (
                "bt_results" in st.session_state and st.session_state["bt_results"]
            )
            if has_default_results and st.button("Valider avec le backtest (params optimisés vs défaut)"):
                from model import expanding_window_backtest
                with st.spinner("Backtest avec params optimisés..."):
                    results_opt = expanding_window_backtest(
                        st.session_state["bt_all_feat"],
                        st.session_state["bt_rg_feat"],
                        xgb_params=xgb_p,
                        calibration_method=cal_method,
                        calibration_cv=cal_cv,
                    )
                st.session_state["bt_results_opt"] = results_opt

            if (
                has_default_results
                and "bt_results_opt" in st.session_state
                and st.session_state["bt_results_opt"]
            ):
                _compare_optuna_vs_default(
                    st.session_state["bt_results"],
                    st.session_state["bt_results_opt"],
                )

    # ----------------------------------------------------------------
    # Blend Optuna — Étape 2
    st.markdown("---")
    with st.expander("⚙️ Optimisation Blend (Étape 2 — après XGB Optuna)"):
        from hyperopt import (
            run_blend_optuna, save_best_blend_params, load_best_blend_params, get_blend_params,
            get_xgb_params as _get_xgb_params, load_best_params as _load_best_params,
        )
        existing_xgb = _load_best_params()
        existing_blend = load_best_blend_params()

        if existing_xgb is None:
            st.warning("⚠️ Lancez d'abord l'optimisation XGB (Étape 1) avant d'optimiser le blend.")
        else:
            st.caption(
                "Optimise BLEND_ALPHA, decay_lo/hi, alpha_floor_late_round et min_rg_for_blend. "
                "Les modèles XGBoost sont pré-entraînés une seule fois, chaque trial est rapide."
            )
            n_trials_blend = st.number_input("Nombre de trials Blend", min_value=10, max_value=200, value=50, step=10)
            if st.button("Lancer Optuna Blend", type="secondary"):
                if "bt_all_feat" not in st.session_state or "bt_rg_feat" not in st.session_state:
                    st.error("Lancez le backtesting d'abord pour avoir les features en mémoire.")
                else:
                    xgb_p_blend = _get_xgb_params(existing_xgb)
                    cal_m = existing_xgb.get("calibration_method", "isotonic")
                    cal_cv_v = int(existing_xgb.get("calibration_cv", 4))
                    prog_b = st.progress(0)
                    log_b: list[str] = []
                    log_ph = st.empty()

                    def _on_blend(n_done: int, best_val: float, _p: dict) -> None:
                        prog_b.progress(int(n_done / n_trials_blend * 100))
                        log_b.append(f"Trial {n_done:3d} | Best Brier blend : {best_val:.4f}")
                        log_ph.code("\n".join(log_b[-6:]), language="text")

                    with st.spinner("Blend Optuna en cours..."):
                        study_blend = run_blend_optuna(
                            st.session_state["bt_all_feat"],
                            st.session_state["bt_rg_feat"],
                            xgb_params=xgb_p_blend,
                            calibration_method=cal_m,
                            calibration_cv=cal_cv_v,
                            n_trials=int(n_trials_blend),
                            on_trial_end=_on_blend,
                        )
                    best_blend = save_best_blend_params(study_blend)
                    st.session_state["blend_best"] = best_blend
                    prog_b.progress(100)
                    st.success(f"Terminé — Best Brier blend : **{best_blend['_best_brier']:.4f}**")
                    st.rerun()

            display_blend = st.session_state.get("blend_best") or existing_blend
            if display_blend:
                bp = get_blend_params(display_blend)
                st.markdown("**Meilleurs paramètres blend :**")
                st.dataframe(
                    pd.DataFrame([{"Paramètre": k, "Valeur": round(v, 4) if isinstance(v, float) else v}
                                  for k, v in bp.items()]),
                    width='stretch', hide_index=True,
                )
                with st.expander("Snippet à copier dans model.py"):
                    lines = []
                    for k, v in bp.items():
                        lines.append(f"{k.upper()} = {repr(v)}")
                    st.code("\n".join(lines), language="python")

    # ----------------------------------------------------------------
    # Affichage de l'historique sauvegardé
    if results_path.exists():
        st.subheader("Historique des runs")
        with open(results_path, encoding="utf-8") as f:
            content = f.read()
        st.download_button(
            "Télécharger backtest_history.txt",
            data=content,
            file_name="backtest_history.txt",
            mime="text/plain",
        )
        with st.expander("Voir le fichier complet"):
            st.code(content, language="text")


def _display_runs_comparison():
    """Section comparaison de tous les runs historiques (chargés depuis JSON)."""
    from backtest_logger import load_all_runs
    runs = load_all_runs()
    if not runs:
        st.info("Aucun run enregistré. Lancez un backtest pour démarrer l'historique.")
        return

    st.subheader("Comparaison des runs")

    # Table résumé : une ligne par run
    summary_rows = []
    for r in runs:
        s = r.get("summary", {})
        summary_rows.append({
            "Run": r.get("label", r.get("timestamp", "?")),
            "Timestamp": r.get("timestamp", "")[:16].replace("T", " "),
            "N feat.": r.get("n_features", "?"),
            "Acc Base": s.get("avg_acc_base"),
            "Brier Base": s.get("avg_brier_base"),
            "Acc Blend": s.get("avg_acc_blend"),
            "Brier Blend": s.get("avg_brier_blend"),
        })

    summary_df = pd.DataFrame(summary_rows)

    # Sélectionner le run de référence
    run_labels = [r["Run"] for r in summary_rows]
    ref_idx = st.selectbox(
        "Run de référence (pour le calcul des deltas)",
        options=list(range(len(run_labels))),
        format_func=lambda i: run_labels[i],
        index=0,
    )
    ref = summary_rows[ref_idx]

    # Afficher la table avec deltas colorés
    def _fmt_delta(val, ref_val, lower_is_better: bool = False) -> str:
        if val is None or ref_val is None:
            return "—"
        delta = val - ref_val
        if abs(delta) < 1e-5:
            return f"{val:.4f}"
        arrow = ("▼" if lower_is_better else "▲") if delta > 0 else ("▲" if lower_is_better else "▼")
        sign = "+" if delta > 0 else ""
        return f"{val:.4f} ({sign}{delta:+.4f} {arrow})"

    display_rows = []
    for row in summary_rows:
        is_ref = (row["Run"] == ref["Run"] and row["Timestamp"] == ref["Timestamp"])
        display_rows.append({
            "Run": ("⭐ " if is_ref else "") + row["Run"],
            "Timestamp": row["Timestamp"],
            "N feat.": row["N feat."],
            "Acc Base": _fmt_delta(row["Acc Base"], ref["Acc Base"], lower_is_better=False),
            "Brier Base": _fmt_delta(row["Brier Base"], ref["Brier Base"], lower_is_better=True),
            "Acc Blend": _fmt_delta(row["Acc Blend"], ref["Acc Blend"], lower_is_better=False),
            "Brier Blend": _fmt_delta(row["Brier Blend"], ref["Brier Blend"], lower_is_better=True),
        })

    st.dataframe(pd.DataFrame(display_rows), width='stretch', hide_index=True)

    # Graphique évolution Brier Blend dans le temps
    if len(runs) >= 2:
        plot_rows = [
            {"Run": r.get("label", r.get("timestamp", ""))[:20], "Brier Blend": s.get("avg_brier_blend"),
             "Acc Blend": s.get("avg_acc_blend")}
            for r, s in [(r, r.get("summary", {})) for r in runs]
            if s.get("avg_brier_blend") is not None
        ]
        if plot_rows:
            fig = px.line(
                pd.DataFrame(plot_rows), x="Run", y="Brier Blend", markers=True,
                title="Évolution du Brier Blend (↓ meilleur)",
                color_discrete_sequence=["#e8473f"],
            )
            fig.update_layout(xaxis_tickangle=-30)
            st.plotly_chart(fig, width='stretch')

    # Détail par année pour deux runs sélectionnés
    if len(runs) >= 2:
        with st.expander("Détail par année — comparer deux runs"):
            col_a, col_b = st.columns(2)
            with col_a:
                idx_a = st.selectbox("Run A", list(range(len(run_labels))),
                                     format_func=lambda i: run_labels[i], key="cmp_a",
                                     index=len(run_labels) - 1)
            with col_b:
                idx_b = st.selectbox("Run B (référence)", list(range(len(run_labels))),
                                     format_func=lambda i: run_labels[i], key="cmp_b",
                                     index=0)
            run_a = runs[idx_a]
            run_b = runs[idx_b]
            years_a = {r["year"]: r for r in run_a.get("per_year", [])}
            years_b = {r["year"]: r for r in run_b.get("per_year", [])}
            common_years = sorted(set(years_a) & set(years_b))
            if common_years:
                cmp_rows = []
                for y in common_years:
                    ra, rb = years_a[y], years_b[y]
                    d_acc   = round(ra["acc_blend"]   - rb["acc_blend"],   4)
                    d_brier = round(ra["brier_blend"] - rb["brier_blend"], 4)
                    cmp_rows.append({
                        "Année": y,
                        f"Acc A ({run_a.get('label','A')[:12]})": f"{ra['acc_blend']:.3f}",
                        f"Acc B ({run_b.get('label','B')[:12]})": f"{rb['acc_blend']:.3f}",
                        "ΔAcc (A-B)":   f"{d_acc:+.3f}",
                        f"Brier A": f"{ra['brier_blend']:.4f}",
                        f"Brier B": f"{rb['brier_blend']:.4f}",
                        "ΔBrier (A-B)": f"{d_brier:+.4f}",
                    })
                st.dataframe(pd.DataFrame(cmp_rows), width='stretch', hide_index=True)


def _display_backtest(results: list, baseline_df: pd.DataFrame):
    """Affiche les graphes et tableaux de résultats de backtest."""
    st.subheader("Accuracy par édition (Base vs Blend)")
    rows = []
    for r in results:
        rows.append({
            "Année": r.year,
            "Acc Base": f"{r.accuracy:.1%}",
            "Brier Base": f"{r.brier:.4f}",
            "Acc Blend": f"{r.accuracy_blend:.1%}",
            "Brier Blend": f"{r.brier_blend:.4f}",
            "Log-Loss Blend": f"{r.log_loss_blend:.4f}",
            "N matchs": r.n_matches,
        })
    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

    # Courbe base vs blend
    line_data = []
    for r in results:
        line_data.append({"Année": r.year, "Accuracy": r.accuracy, "Modèle": "Base"})
        line_data.append({"Année": r.year, "Accuracy": r.accuracy_blend, "Modèle": "Blend"})
    fig = px.line(
        pd.DataFrame(line_data), x="Année", y="Accuracy", color="Modèle", markers=True,
        title="Accuracy Base vs Blend par édition Roland Garros",
        color_discrete_map={"Base": "#aaaaaa", "Blend": "#e8473f"},
    )
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, width='stretch')

    # Décomposition par tour
    st.subheader("Accuracy par phase du tournoi (moyenne 2017-2025)")
    all_by_round = []
    for r in results:
        df_r = r.by_round.copy()
        df_r["year"] = r.year
        all_by_round.append(df_r)

    if all_by_round:
        combined = pd.concat(all_by_round, ignore_index=True)
        pivot = combined.groupby("round_name")[["acc_base", "acc_blend", "n"]].agg(
            {"acc_base": "mean", "acc_blend": "mean", "n": "sum"}
        ).reset_index()
        round_order = ["R128", "R64", "R32", "R16", "QF", "SF", "F", "Q1", "Q2", "Q3"]
        pivot["_order"] = pivot["round_name"].map({r: i for i, r in enumerate(round_order)})
        pivot = pivot.sort_values("_order").drop(columns="_order")
        pivot.columns = ["Tour", "Acc Base", "Acc Blend", "N matchs"]

        st.dataframe(
            pivot.assign(**{
                "Acc Base": pivot["Acc Base"].map("{:.1%}".format),
                "Acc Blend": pivot["Acc Blend"].map("{:.1%}".format),
            }),
            width='stretch',
            hide_index=True,
        )

        # Graphique par tour
        round_melt = pivot.rename(columns={"Tour": "Tour"}).melt(
            id_vars=["Tour", "N matchs"], value_vars=["Acc Base", "Acc Blend"],
            var_name="Modèle", value_name="Accuracy",
        )
        fig_round = px.bar(
            round_melt, x="Tour", y="Accuracy", color="Modèle", barmode="group",
            title="Accuracy par phase du tournoi — Base vs Blend",
            color_discrete_map={"Acc Base": "#aaaaaa", "Acc Blend": "#e8473f"},
            category_orders={"Tour": ["R128", "R64", "R32", "R16", "QF", "SF", "F"]},
        )
        fig_round.update_yaxes(tickformat=".0%", range=[0.5, 1.0])
        st.plotly_chart(fig_round, width='stretch')

    if not baseline_df.empty:
        st.subheader("Comparaison avec les baselines")
        display_cols = {
            "year": "Année",
            "baseline_rank_acc": "Ranking seul",
            "baseline_clay_elo_acc": "Clay Elo",
            "baseline_welo_adj_acc": "WElo Adj.",
            "xgboost_acc": "XGBoost",
        }
        avail = {k: v for k, v in display_cols.items() if k in baseline_df.columns}
        df_disp = baseline_df[list(avail.keys())].rename(columns=avail).copy()
        for col in df_disp.columns:
            if col != "Année":
                df_disp[col] = df_disp[col].map(
                    lambda x: f"{x:.1%}" if pd.notna(x) else "N/A"
                )
        st.dataframe(df_disp, width='stretch', hide_index=True)

        melt_cols = ["year"] + [c for c in ["baseline_rank_acc", "baseline_clay_elo_acc",
                                               "baseline_welo_adj_acc", "xgboost_acc"]
                                 if c in baseline_df.columns]
        melted = baseline_df[melt_cols].melt(id_vars="year", var_name="Modèle", value_name="Accuracy")
        melted["Modèle"] = melted["Modèle"].map({
            "baseline_rank_acc": "Ranking",
            "baseline_clay_elo_acc": "Clay Elo",
            "baseline_welo_adj_acc": "WElo Adj.",
            "xgboost_acc": "XGBoost",
        })
        fig2 = px.bar(melted, x="year", y="Accuracy", color="Modèle", barmode="group",
                      title="Accuracy par modèle et par édition")
        fig2.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig2, width='stretch')


# ------------------------------------------------------------------
# Page Tableau RG 2026
# ------------------------------------------------------------------

def page_bracket(predictor):
    st.title("Tableau Roland Garros 2026")

    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from rg2026_bracket import (
        DRAW_PATH, load_draw, simulate_bracket, get_player_path,
        bracket_summary_df, generate_draw_template,
    )

    # ── Mise à jour des données ──────────────────────────────────────
    with st.expander("Mettre à jour les données ATP", expanded=False):
        col_upd1, col_upd2 = st.columns([2, 1])
        with col_upd1:
            st.caption(
                "Télécharge les dernières données ATP depuis le dépôt Jeff Sackmann "
                "(utile pour récupérer les résultats de Rome 2026 et la saison clay complète)."
            )
        with col_upd2:
            if st.button("Télécharger données 2025-2026", type="secondary"):
                import subprocess, sys as _sys
                script = Path(__file__).parent.parent / "src" / "update_data.py"
                with st.spinner("Téléchargement en cours…"):
                    result = subprocess.run(
                        [_sys.executable, str(script), "--years", "2025", "2026", "--clear-cache"],
                        capture_output=True, text=True, cwd=str(Path(__file__).parent.parent),
                    )
                if result.returncode == 0:
                    st.success("Données mises à jour. Rechargez la page pour utiliser le nouveau cache.")
                    st.code(result.stdout[-1000:], language="text")
                else:
                    st.error(f"Erreur : {result.stderr[-500:]}")

    # ── Saisie / chargement du draw ──────────────────────────────────
    st.subheader("Saisie du tableau")

    draw = load_draw(DRAW_PATH)

    # --- Saisie texte directe (méthode principale) ---
    st.markdown(
        "**Colle les 128 joueurs dans l'ordre du tirage** (un nom par ligne = position dans le tableau).\n"
        "Format accepté : `Jannik Sinner`, `[1] Jannik Sinner` ou `Jannik Sinner [1]`"
    )

    # Pré-remplir avec le draw existant si disponible
    existing_text = ""
    if draw:
        existing_text = "\n".join(
            f"[{p['seed']}] {p['player_name']}" if p.get("seed")
            else p.get("player_name", "")
            for p in sorted(draw, key=lambda x: int(x.get("draw_pos") or 0))
        )

    draw_text = st.text_area(
        "128 joueurs en ordre du tirage",
        value=existing_text,
        height=400,
        placeholder="[1] Jannik Sinner\nArthur Rinderknech\nTomas Machac\n[WC] Joueur wildcard\n…",
        help="Colle directement depuis une source (ATP, Tennis Abstract, etc.). Ligne 1 = position 1 du tirage.",
    )

    col_save, col_csv, col_reset = st.columns([1, 1, 1])

    with col_save:
        if st.button("Résoudre et enregistrer", type="primary"):
            import re as _re, csv as _csv, pandas as _pd3
            from rg2026_bracket import resolve_draw_names

            lines = [l.strip() for l in draw_text.strip().split("\n") if l.strip()]
            if len(lines) < 2:
                st.error("Besoin d'au moins 2 joueurs.")
            else:
                # Load historical data for name resolution
                raw_dir = Path(__file__).parent.parent / "data" / "raw"
                df_hist = _pd3.concat(
                    [_pd3.read_csv(f, usecols=["winner_name", "loser_name"], low_memory=False)
                     for f in sorted(raw_dir.glob("atp_matches_2[0-9][0-9][0-9].csv"))],
                    ignore_index=True,
                )
                resolved = resolve_draw_names(lines, df_hist)
                st.session_state["resolution_preview"] = resolved
                st.session_state["resolution_lines"] = lines

    with col_csv:
        uploaded = st.file_uploader("Ou importer un draw.csv", type="csv", label_visibility="collapsed")
        if uploaded is not None:
            DRAW_PATH.parent.mkdir(parents=True, exist_ok=True)
            DRAW_PATH.write_bytes(uploaded.read())
            st.session_state.pop("bracket_result", None)
            st.success("Draw CSV importé.")
            st.rerun()

    with col_reset:
        if st.button("Regénérer depuis rankings 2026"):
            import pandas as _pd2
            raw_dir = Path(__file__).parent.parent / "data" / "raw"
            df26 = _pd2.read_csv(raw_dir / "atp_matches_2026.csv")
            ranks_dict: dict = {}
            for _, row in df26.iterrows():
                for prefix in ['winner', 'loser']:
                    name = str(row[f'{prefix}_name'])
                    rank = row[f'{prefix}_rank']
                    ioc  = row.get(f'{prefix}_ioc', '')
                    if _pd2.notna(rank) and name not in ranks_dict:
                        ranks_dict[name] = {'atp_rank': int(rank), 'ioc': str(ioc) if _pd2.notna(ioc) else ''}
            ranked = sorted(
                [{'player_name': n, **info} for n, info in ranks_dict.items()],
                key=lambda x: x['atp_rank'],
            )
            generate_draw_template(ranked)
            st.session_state.pop("bracket_result", None)
            st.success("Draw regénéré depuis rankings.")
            st.rerun()

    # ── Tableau de résolution des noms ───────────────────────────────
    if "resolution_preview" in st.session_state:
        import re as _re2, csv as _csv2
        resolved_list = st.session_state["resolution_preview"]

        n_exact   = sum(1 for r in resolved_list if r["confidence"] == "exact")
        n_fuzzy   = sum(1 for r in resolved_list if r["confidence"] == "fuzzy")
        n_unres   = sum(1 for r in resolved_list if r["confidence"] == "unresolved")

        st.markdown(f"**Résolution des noms** — {n_exact} exacts · {n_fuzzy} fuzzy · {n_unres} non trouvés")

        # Show fuzzy + unresolved for manual correction
        needs_review = [r for r in resolved_list if r["confidence"] != "exact"]
        if needs_review:
            st.caption("Vérifie / corrige les noms ci-dessous avant de confirmer :")
            corrected: dict[int, str] = {}
            cols4 = st.columns(2)
            for idx, r in enumerate(needs_review):
                col = cols4[idx % 2]
                badge = "🟡" if r["confidence"] == "fuzzy" else "🔴"
                corrected[resolved_list.index(r)] = col.text_input(
                    f"{badge} pos {resolved_list.index(r)+1}: {r['original']}",
                    value=r["resolved"],
                    key=f"corr_{idx}",
                )
        else:
            st.success("Tous les noms ont été résolus exactement.")
            corrected = {}

        if st.button("Confirmer et sauvegarder le tirage", type="primary"):
            # Apply corrections
            for global_idx, new_name in corrected.items():
                resolved_list[global_idx]["resolved"] = new_name.strip()

            new_draw = []
            for i, r in enumerate(resolved_list):
                seed_str = r.get("seed", "")
                seed = seed_str if seed_str.isdigit() else ""
                new_draw.append({
                    "draw_pos": i + 1,
                    "player_name": r["resolved"],
                    "seed": seed,
                    "atp_rank": "",
                    "nationality": "",
                    "status": "seeded" if seed else "ranked",
                })

            DRAW_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(DRAW_PATH, "w", newline="", encoding="utf-8") as f:
                writer = _csv2.DictWriter(
                    f, fieldnames=["draw_pos", "player_name", "seed", "atp_rank", "nationality", "status"]
                )
                writer.writeheader()
                writer.writerows(new_draw)

            del st.session_state["resolution_preview"]
            del st.session_state["resolution_lines"]
            st.session_state.pop("bracket_result", None)
            st.session_state.pop("bracket_draw", None)
            st.success(f"Tirage sauvegardé ({len(new_draw)} joueurs).")
            st.rerun()

        if st.button("Annuler"):
            del st.session_state["resolution_preview"]
            del st.session_state["resolution_lines"]
            st.rerun()

        st.stop()

    draw = load_draw(DRAW_PATH)
    if not draw:
        st.info("Colle les 128 joueurs ci-dessus et clique 'Résoudre et enregistrer'.")
        return

    # Résumé du draw chargé
    n_known = sum(1 for p in draw if "TBD" not in p.get("player_name","") and "Qualifier" not in p.get("player_name",""))
    n_seeded = sum(1 for p in draw if p.get("seed"))
    st.caption(f"Tirage chargé : {len(draw)} positions — {n_known} joueurs identifiés — {n_seeded} têtes de série")

    # ── Simulation du tableau ────────────────────────────────────────
    st.subheader("Simulation du tableau")

    confirmed = []
    if hasattr(predictor, '_rg2026_results'):
        confirmed = predictor._rg2026_results

    if st.button("Simuler le tableau complet", type="primary"):
        with st.spinner("Simulation en cours…"):
            result = simulate_bracket(draw, predictor, confirmed_results=confirmed)
        st.session_state["bracket_result"] = result
        st.session_state["bracket_draw"] = draw

    if "bracket_result" not in st.session_state:
        st.info("Cliquez sur 'Simuler le tableau complet' pour lancer les prédictions.")
        return

    result = st.session_state["bracket_result"]
    draw_snap = st.session_state.get("bracket_draw", draw)

    if "error" in result:
        st.error(result["error"])
        return

    # Champion prédit
    champion = result.get("champion", "?")
    st.success(f"Champion prédit : **{champion}**")

    # ── Vue par tour ──────────────────────────────────────────────────
    round_tabs = st.tabs(["R128", "R64", "R32", "R16", "QF", "SF", "Finale"])
    round_numbers = [1, 2, 3, 4, 5, 6, 7]

    for tab, rnd in zip(round_tabs, round_numbers):
        with tab:
            matches = result["rounds"].get(rnd, [])
            rows = []
            for m in matches:
                fav = m["a"] if m["proba_a"] >= 0.5 else m["b"]
                fav_p = max(m["proba_a"], m["proba_b"])
                rows.append({
                    "A": m["a"],
                    "% A": f"{m['proba_a']:.1%}",
                    "B": m["b"],
                    "% B": f"{m['proba_b']:.1%}",
                    "Favori prédit": fav,
                    "Confiance": f"{fav_p:.1%}",
                    "✓": "✓" if m["confirmed"] else "",
                })
            if rows:
                import pandas as _pd
                st.dataframe(_pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Parcours d'un joueur ──────────────────────────────────────────
    st.subheader("Parcours prédit d'un joueur")
    all_players = sorted(set(
        p["player_name"] for p in draw_snap
        if "TBD" not in p.get("player_name", "") and "Qualifier" not in p.get("player_name", "")
    ))

    selected_player = st.selectbox("Choisir un joueur", [""] + all_players)
    if selected_player:
        path = get_player_path(selected_player, result, draw_snap)
        if path:
            path_rows = []
            for step in path:
                path_rows.append({
                    "Tour": step["round_name"],
                    "Adversaire": step["opponent"],
                    "Prob victoire": f"{step['player_proba']:.1%}",
                    "Issue prédite": "Victoire" if step["player_wins"] else "Défaite",
                    "Confirmé": "✓" if step["confirmed"] else "",
                })
            import pandas as _pd
            st.dataframe(_pd.DataFrame(path_rows), use_container_width=True, hide_index=True)

            # Prob de gagner le titre = produit des probas victoire sur chaque tour
            survival = 1.0
            for step in path:
                if step["player_wins"]:
                    survival *= step["player_proba"]
                else:
                    break
            rounds_survived = sum(1 for s in path if s["player_wins"])
            st.metric("Probabilité de titre (modèle)", f"{survival:.1%}",
                      help="Produit des probabilités de victoire à chaque tour")
            st.caption(f"Parcours prédit : {rounds_survived} victoires consécutives")
        else:
            st.warning(f"{selected_player} non trouvé dans le tableau simulé.")

    # ── Tableau complet (téléchargeable) ──────────────────────────────
    with st.expander("Tableau complet (tous les matchs)"):
        full_df = bracket_summary_df(result)
        st.dataframe(full_df, width='stretch', hide_index=True)
        csv_bytes = full_df.to_csv(index=False).encode()
        st.download_button(
            "Télécharger le tableau (CSV)",
            data=csv_bytes,
            file_name="rg2026_bracket_predictions.csv",
            mime="text/csv",
        )


# ------------------------------------------------------------------
# Page Simulation Paris
# ------------------------------------------------------------------

def page_betting(_predictor):
    st.title("Simulation Paris — Roland Garros 2017-2025")

    try:
        from betting import download_rg_odds, load_rg_odds, run_betting_backtest, compute_metrics
    except ImportError as e:
        st.error(f"Module betting non disponible : {e}")
        return

    if "bt_results" not in st.session_state or not st.session_state["bt_results"]:
        st.warning("⚠️ Lancez d'abord le backtesting (onglet **Backtesting**) pour générer les prédictions.")
        return

    if "predictions_df" not in st.session_state:
        st.warning("⚠️ Relancez le backtesting : les prédictions match par match ne sont pas encore en mémoire.")
        return

    preds_df = st.session_state["predictions_df"]

    # --- Chargement des cotes ---
    st.subheader("1. Cotes historiques")
    odds_path = Path(__file__).parent.parent / "data" / "odds"
    has_odds = any(odds_path.glob("rg_odds_*.csv")) if odds_path.exists() else False

    col_dl, col_info = st.columns([1, 2])
    with col_dl:
        if st.button("📥 Télécharger cotes (tennis-data.co.uk)"):
            with st.spinner("Téléchargement en cours..."):
                df_odds = download_rg_odds(years=list(range(2017, 2026)))
            if df_odds.empty:
                st.error("Aucune cote téléchargée. Vérifiez votre connexion.")
            else:
                st.success(f"{len(df_odds)} matchs téléchargés.")
                st.rerun()
    with col_info:
        if has_odds:
            st.success("Cotes locales disponibles.")
        else:
            st.info("Cotes non encore téléchargées.")

    if not has_odds:
        return

    odds_df = load_rg_odds(years=list(range(2017, 2026)))
    st.caption(f"{len(odds_df)} entrées de cotes chargées ({odds_df['year'].nunique()} éditions)")

    # --- Paramètres simulation ---
    st.subheader("2. Paramètres")
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        bankroll_init = st.number_input("Bankroll initiale (€)", min_value=100, max_value=100000,
                                         value=1000, step=100)
        min_ev = st.slider("EV minimum pour parier", min_value=0.0, max_value=0.20,
                            value=0.03, step=0.01, format="%.2f")
    with col_p2:
        strategies_sel = st.multiselect(
            "Stratégies à comparer",
            ["full_kelly", "half_kelly", "capped_kelly", "fixed_ev_tier", "kelly_by_round"],
            default=["half_kelly", "capped_kelly", "fixed_ev_tier"],
        )

    if st.button("▶️ Lancer la simulation", type="primary"):
        with st.spinner("Simulation en cours..."):
            results = run_betting_backtest(
                preds_df, odds_df,
                strategies=strategies_sel,
                initial_bankroll=bankroll_init,
                min_ev_threshold=min_ev,
            )
        st.session_state["betting_results"] = results
        st.rerun()

    if "betting_results" not in st.session_state or not st.session_state["betting_results"]:
        return

    results = st.session_state["betting_results"]

    # --- Bankroll evolution chart ---
    st.subheader("3. Évolution du bankroll (2017–2025)")
    all_histories = []
    for strat, res in results.items():
        h = res["history"].copy()
        h["Stratégie"] = strat
        h["Match #"] = range(len(h))
        all_histories.append(h)

    if all_histories:
        hist_all = pd.concat(all_histories, ignore_index=True)
        fig_bk = px.line(
            hist_all, x="Match #", y="bankroll_after", color="Stratégie",
            title="Évolution du bankroll par stratégie",
        )
        fig_bk.add_hline(y=bankroll_init, line_dash="dash", line_color="gray",
                          annotation_text="Bankroll initiale")
        st.plotly_chart(fig_bk, width='stretch')

    # --- Tableau comparatif ---
    st.subheader("4. Comparaison des stratégies")
    comp_rows = []
    for strat, res in results.items():
        m = res["metrics"]
        comp_rows.append({
            "Stratégie": strat,
            "ROI": f"{m.get('roi', float('nan')):.1%}",
            "Bankroll finale": f"{m.get('final_bankroll', float('nan')):.0f} €",
            "Max Drawdown": f"{m.get('max_drawdown', float('nan')):.1%}",
            "Sharpe": f"{m.get('sharpe', float('nan')):.2f}",
            "Années rentables": m.get("profitable_years", "?"),
            "Paris": m.get("n_bets", 0),
            "Gagnés": m.get("n_wins", 0),
        })
    st.dataframe(pd.DataFrame(comp_rows), width='stretch', hide_index=True)

    # --- ROI par tour ---
    st.subheader("5. ROI par tour")
    roi_rows = []
    from model import ROUND_NAMES as _RN
    for strat, res in results.items():
        for rn, roi_v in res["metrics"].get("roi_by_round", {}).items():
            roi_rows.append({"Tour": _RN.get(int(rn), str(rn)), "Stratégie": strat, "ROI": roi_v})
    if roi_rows:
        roi_df = pd.DataFrame(roi_rows)
        fig_roi = px.bar(
            roi_df, x="Tour", y="ROI", color="Stratégie", barmode="group",
            title="ROI par tour du tournoi",
            category_orders={"Tour": ["R128", "R64", "R32", "R16", "QF", "SF", "F"]},
        )
        fig_roi.update_yaxes(tickformat=".0%")
        fig_roi.add_hline(y=0, line_color="black", line_width=1)
        st.plotly_chart(fig_roi, width='stretch')


# ------------------------------------------------------------------
# Page Calculateur de Paris
# ------------------------------------------------------------------

def _bet_metrics(p_a: float, p_b: float, odds_a: float, odds_b: float, bankroll: float, confidence: float):
    """Calcule EV, edge, Kelly pour les deux joueurs d'un match."""
    overround = 1 / odds_a + 1 / odds_b
    implied_a = (1 / odds_a) / overround
    implied_b = (1 / odds_b) / overround

    ev_a = odds_a * p_a - 1
    ev_b = odds_b * p_b - 1

    def kelly(p, o):
        b = o - 1
        return max((b * p - (1 - p)) / b, 0.0)

    stake_a = round(0.5 * kelly(p_a, odds_a) * bankroll, 2)
    stake_b = round(0.5 * kelly(p_b, odds_b) * bankroll, 2)

    def stars(ev, conf):
        if ev <= 0 or conf < 20:
            return ""
        if ev > 0.10 and conf > 40:
            return "★★★"
        if ev > 0.05:
            return "★★"
        return "★"

    return {
        "implied_a": implied_a, "implied_b": implied_b,
        "margin": (overround - 1) * 100,
        "ev_a": ev_a, "ev_b": ev_b,
        "edge_a": p_a - implied_a, "edge_b": p_b - implied_b,
        "stake_a": stake_a, "stake_b": stake_b,
        "stars_a": stars(ev_a, confidence), "stars_b": stars(ev_b, confidence),
    }


def page_bet_calculator(predictor):
    st.title("Calculateur de Paris")

    tab_bulk, tab_single = st.tabs(["Meilleurs paris du tour", "Analyse match unique"])

    # ── TAB 1 : Meilleurs paris du tour ─────────────────────────────
    with tab_bulk:
        st.caption("Saisis les cotes pour tous les matchs d'un tour — le classement des meilleurs paris s'affiche automatiquement.")

        # Charger les matchs du bracket
        bracket = st.session_state.get("bracket_result")
        if bracket is None:
            try:
                from rg2026_bracket import DRAW_PATH, load_draw, simulate_bracket
                draw = load_draw(DRAW_PATH)
                if draw:
                    from rg2026_bracket import RESULTS_PATH
                    import json as _json
                    confirmed: dict = {}
                    if RESULTS_PATH.exists():
                        with open(RESULTS_PATH) as _f:
                            for _line in _f:
                                _line = _line.strip()
                                if _line:
                                    _r = _json.loads(_line)
                                    confirmed[(_r["winner"], _r["loser"])] = _r
                    bracket = simulate_bracket(draw, predictor, confirmed_results=confirmed)
            except Exception:
                bracket = None

        if bracket is None:
            st.warning("Générez d'abord le tableau dans **Tableau RG 2026**.")
        else:
            from rg2026_bracket import ROUND_NAMES
            available_rounds = sorted(bracket["rounds"].keys())

            # Détecte le premier tour avec des matchs non confirmés
            default_round = available_rounds[0]
            for r in available_rounds:
                if any(not m.get("confirmed") for m in bracket["rounds"].get(r, [])):
                    default_round = r
                    break

            round_idx = st.select_slider(
                "Tour à analyser",
                options=available_rounds,
                value=default_round,
                format_func=lambda x: ROUND_NAMES.get(x, str(x)),
                key="bulk_round",
            )

            matches = bracket["rounds"].get(round_idx, [])
            unconfirmed = [m for m in matches if not m.get("confirmed")]

            if not unconfirmed:
                st.info("Tous les matchs de ce tour ont déjà un résultat confirmé.")
            else:
                bankroll_bulk = st.number_input(
                    "Bankroll (€)", min_value=10.0, value=1000.0,
                    step=50.0, format="%.0f", key="bulk_bank",
                )

                st.markdown(
                    f"**{len(unconfirmed)} matchs à venir** — saisis les cotes bookmaker "
                    f"(laisse à **0** les matchs à ignorer), puis clique sur Calculer."
                )

                # Formulaire de saisie des cotes
                with st.form("bulk_odds_form"):
                    odds_inputs: list[dict] = []
                    for i, m in enumerate(unconfirmed):
                        pa, pb = m["a"], m["b"]
                        p_a = m.get("proba_a", 0.5)
                        p_b = 1.0 - p_a
                        pred_label = pa if p_a >= 0.5 else pb

                        col_match, col_oa, col_ob = st.columns([3, 1, 1])
                        col_match.markdown(
                            f"**{pa}** vs **{pb}**  \n"
                            f"<small>Modèle : {pa} {p_a:.0%} · {pb} {p_b:.0%} · favori : **{pred_label}**</small>",
                            unsafe_allow_html=True,
                        )
                        oa = col_oa.number_input(
                            pa[:18], min_value=0.0, max_value=99.0,
                            value=0.0, step=0.05, format="%.2f", key=f"bulk_oa_{i}",
                        )
                        ob = col_ob.number_input(
                            pb[:18], min_value=0.0, max_value=99.0,
                            value=0.0, step=0.05, format="%.2f", key=f"bulk_ob_{i}",
                        )
                        odds_inputs.append({"pa": pa, "pb": pb, "p_a": p_a, "p_b": p_b,
                                            "oa": oa, "ob": ob, "round": round_idx})

                    submitted = st.form_submit_button("Calculer les meilleurs paris", type="primary")

                if submitted:
                    rows = []
                    for item in odds_inputs:
                        pa, pb = item["pa"], item["pb"]
                        p_a, p_b = item["p_a"], item["p_b"]
                        oa, ob = item["oa"], item["ob"]

                        # Ignorer les matchs sans cotes saisies
                        if oa < 1.01 or ob < 1.01:
                            continue

                        pred_full = predictor.predict_match(pa, pb, item["round"])
                        conf = pred_full["confidence_score"]

                        m = _bet_metrics(p_a, p_b, oa, ob, bankroll_bulk, conf)

                        for player, p, odds, ev, edge, stake, stars in [
                            (pa, p_a, oa, m["ev_a"], m["edge_a"], m["stake_a"], m["stars_a"]),
                            (pb, p_b, ob, m["ev_b"], m["edge_b"], m["stake_b"], m["stars_b"]),
                        ]:
                            rows.append({
                                "Match": f"{pa} / {pb}",
                                "Pari sur": player,
                                "Cote": round(odds, 2),
                                "Prob. modèle": p,
                                "EV": ev,
                                "Edge": edge,
                                "Confiance": conf,
                                "Mise ½K (€)": stake,
                                "Signal": stars,
                            })

                    if not rows:
                        st.warning("Aucune cote saisie — entre au moins une cote (> 1.01) pour analyser.")
                        st.stop()

                    df_bets = pd.DataFrame(rows).sort_values("EV", ascending=False).reset_index(drop=True)

                    # Sépare les bons paris des mauvais
                    good = df_bets[df_bets["Signal"] != ""].copy()
                    bad  = df_bets[df_bets["Signal"] == ""].copy()

                    if not good.empty:
                        st.subheader(f"Paris recommandés ({len(good)})")
                        display_good = good.copy()
                        display_good["Prob. modèle"] = display_good["Prob. modèle"].map("{:.1%}".format)
                        display_good["EV"]    = display_good["EV"].map("{:+.1%}".format)
                        display_good["Edge"]  = display_good["Edge"].map("{:+.1%}".format)
                        display_good["Confiance"] = display_good["Confiance"].map("{:.0f}/100".format)
                        st.dataframe(display_good, hide_index=True, width='stretch')
                    else:
                        st.warning("Aucun pari avec un edge suffisant sur ce tour.")

                    with st.expander("Tous les paris (dont EV négatif)"):
                        display_all = df_bets.copy()
                        display_all["Prob. modèle"] = display_all["Prob. modèle"].map("{:.1%}".format)
                        display_all["EV"]   = display_all["EV"].map("{:+.1%}".format)
                        display_all["Edge"] = display_all["Edge"].map("{:+.1%}".format)
                        display_all["Confiance"] = display_all["Confiance"].map("{:.0f}/100".format)
                        st.dataframe(display_all, hide_index=True, width='stretch')

    # ── TAB 2 : Analyse match unique ─────────────────────────────────
    with tab_single:
        st.caption("Analyse détaillée d'un match précis.")

        from data_loader import get_player_list
        all_players_hist = set(get_player_list(predictor.df) or PLAYER_LIST_DEFAULT)
        try:
            from rg2026_bracket import load_draw, DRAW_PATH
            if DRAW_PATH.exists():
                all_players_hist.update(r["player_name"] for r in load_draw() if r.get("player_name"))
        except Exception:
            pass
        all_players = sorted(all_players_hist)

        col1, col2, col3 = st.columns(3)
        with col1:
            player_a = st.selectbox("Joueur A", all_players, key="bc_pa")
        with col2:
            player_b = st.selectbox("Joueur B", [p for p in all_players if p != player_a], key="bc_pb")
        with col3:
            round_num = st.select_slider(
                "Tour", options=list(ROUND_LABELS.keys()),
                format_func=lambda x: ROUND_LABELS[x], value=1, key="bc_round",
            )

        col_oa, col_ob, col_bank = st.columns(3)
        with col_oa:
            odds_a = st.number_input(f"Cote {player_a}", min_value=1.01, max_value=100.0,
                                      value=1.80, step=0.05, format="%.2f", key="bc_oa")
        with col_ob:
            odds_b = st.number_input(f"Cote {player_b}", min_value=1.01, max_value=100.0,
                                      value=2.00, step=0.05, format="%.2f", key="bc_ob")
        with col_bank:
            bankroll = st.number_input("Bankroll (€)", min_value=10.0, value=1000.0,
                                        step=50.0, format="%.0f", key="bc_bank")

        if st.button("Analyser", type="primary"):
            with st.spinner("Calcul…"):
                pred = predictor.predict_match(player_a, player_b, round_num)

            p_a = pred["proba_a"]
            p_b = pred["proba_b"]
            confidence = pred["confidence_score"]
            m = _bet_metrics(p_a, p_b, odds_a, odds_b, bankroll, confidence)

            col_pred, col_conf = st.columns([3, 1])
            with col_pred:
                st.plotly_chart(proba_bar(p_a, player_a, player_b), width='stretch')
            with col_conf:
                st.plotly_chart(confidence_gauge(confidence, pred["confidence"]), width='stretch')

            st.info(f"Vainqueur prédit : **{pred['winner_predicted']}** — confiance **{confidence:.0f}/100**")

            st.divider()
            comp_data = {
                "": [player_a, player_b],
                "Cote book": [f"{odds_a:.2f}", f"{odds_b:.2f}"],
                "Prob. implicite book": [f"{m['implied_a']:.1%}", f"{m['implied_b']:.1%}"],
                "Prob. modèle": [f"{p_a:.1%}", f"{p_b:.1%}"],
                "Edge": [f"{m['edge_a']:+.1%}", f"{m['edge_b']:+.1%}"],
                "EV": [f"{m['ev_a']:+.1%}", f"{m['ev_b']:+.1%}"],
                "Mise ½ Kelly (€)": [f"{m['stake_a']:.0f}", f"{m['stake_b']:.0f}"],
            }
            st.dataframe(pd.DataFrame(comp_data), hide_index=True, width='stretch')
            st.caption(f"Marge bookmaker : **{m['margin']:.1f}%**")

            st.divider()
            for player, ev, edge, stake, odds_val, stars in [
                (player_a, m["ev_a"], m["edge_a"], m["stake_a"], odds_a, m["stars_a"]),
                (player_b, m["ev_b"], m["edge_b"], m["stake_b"], odds_b, m["stars_b"]),
            ]:
                if ev <= 0:
                    st.error(f"**{player}** — EV {ev:+.1%} : ne pas parier")
                elif ev < 0.03 or confidence < 20:
                    st.warning(f"**{player}** — EV {ev:+.1%} · confiance {confidence:.0f}/100 : prudence")
                else:
                    st.success(f"**{player}** {stars} — EV **{ev:+.1%}** · edge **{edge:+.1%}** · mise **{stake:.0f} €** sur {odds_val:.2f}")

            if pred.get("conformal_calibrated"):
                st.caption(
                    f"Intervalle conformal 80 % pour {player_a} : "
                    f"**{pred['proba_a_low']:.1%}** – **{pred['proba_a_high']:.1%}**"
                )


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Roland Garros 2026 Predictor",
        page_icon="🎾",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.sidebar.title("Roland Garros 2026")
    st.sidebar.markdown("*Prédicteur ML — WElo + XGBoost*")

    page = st.sidebar.radio(
        "Navigation",
        ["Tableau RG 2026", "Saisie résultats", "Calculateur Paris", "Backtesting", "Simulation Paris"],
    )

    raw_dir = Path(__file__).parent.parent / "data" / "raw"
    if not list(raw_dir.glob("atp_matches_*.csv")):
        st.error(
            "Aucune donnée trouvée dans `data/raw/`.\n\n"
            "Lancez d'abord :\n```bash\nbash scripts/download_data.sh\n```"
        )
        st.stop()

    predictor = get_predictor()

    if page == "Tableau RG 2026":
        page_bracket(predictor)
    elif page == "Saisie résultats":
        page_results(predictor)
    elif page == "Calculateur Paris":
        page_bet_calculator(predictor)
    elif page == "Backtesting":
        page_backtest(predictor)
    elif page == "Simulation Paris":
        page_betting(predictor)


if __name__ == "__main__":
    main()
