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
    st.dataframe(df, use_container_width=True, hide_index=True)


# ------------------------------------------------------------------
# Pages
# ------------------------------------------------------------------

def page_overview(predictor):
    st.title("Roland Garros 2026 — Vue d'ensemble")

    st.subheader("Joueurs du tournoi")
    players_input = st.text_area(
        "Un joueur par ligne :",
        value="\n".join(PLAYER_LIST_DEFAULT),
        height=200,
    )
    players = [p.strip() for p in players_input.strip().split("\n") if p.strip()]

    if len(players) < 2:
        st.warning("Entrez au moins 2 joueurs.")
        return

    st.subheader(f"Simulation — {len(players)} joueurs")
    n_sim = st.slider("Simulations Monte Carlo", 1000, 20000, 5000, step=1000)

    if st.button("Simuler le tournoi", type="primary"):
        with st.spinner("Simulation en cours..."):
            sim = predictor.simulate_tournament(players, n_simulations=n_sim)

        st.subheader("Probabilités de victoire finale")
        fig = px.bar(
            sim.head(20), x="win_proba", y="player", orientation="h",
            color="win_proba", color_continuous_scale="Reds",
            labels={"win_proba": "Probabilité", "player": "Joueur"},
        )
        fig.update_layout(height=600, coloraxis_showscale=False,
                           yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(
            sim.assign(win_proba=sim["win_proba"].map("{:.1%}".format)),
            use_container_width=True,
        )

    st.subheader("Profils Elo des joueurs")
    profiles = []
    for p in players:
        prof = predictor.get_player_profile(p)
        profiles.append({
            "Joueur": p,
            "Rang": int(prof.get("rank", 999)) if pd.notna(prof.get("rank")) else None,
            "Clay Elo": f"{prof.get('clay_elo', 1500):.0f}",
            "WElo": f"{prof.get('welo', 1500):.0f}",
            "Adj. Elo": f"{prof.get('adjusted_elo', 1500):.0f}",
        })
    st.dataframe(pd.DataFrame(profiles), use_container_width=True, hide_index=True)


def page_match(predictor):
    st.title("Prédiction Match par Match")

    from data_loader import get_player_list
    all_players = get_player_list(predictor.df) or PLAYER_LIST_DEFAULT

    col1, col2 = st.columns(2)
    with col1:
        player_a = st.selectbox("Joueur A", all_players, index=0)
    with col2:
        player_b = st.selectbox("Joueur B", all_players,
                                 index=min(1, len(all_players) - 1))

    round_num = st.select_slider(
        "Tour",
        options=list(ROUND_LABELS.keys()),
        format_func=lambda x: ROUND_LABELS[x],
        value=4,
    )

    if player_a == player_b:
        st.warning("Sélectionnez deux joueurs différents.")
        return

    if st.button("Prédire", type="primary"):
        with st.spinner("Calcul des features..."):
            pred = predictor.predict_match(player_a, player_b, round_num)

        winner = pred["winner_predicted"]
        score = pred["confidence_score"]
        label = pred["confidence"]

        # Résumé
        st.success(f"**Vainqueur prédit : {winner}** — confiance {label} ({score:.0f}/100)")

        col_bar, col_gauge = st.columns([2, 1])
        with col_bar:
            st.plotly_chart(proba_bar(pred["proba_a"], player_a, player_b),
                            use_container_width=True)
        with col_gauge:
            st.plotly_chart(confidence_gauge(score, label), use_container_width=True)

        # Explication du score de confiance
        with st.expander("Comment interpréter le score de confiance ?"):
            st.markdown(
                f"""
Le score de confiance est calculé à partir de l'écart de la probabilité prédite à 50 % :

> `score = |P(A gagne) − 0.5| × 200`

| Score | Interprétation |
|-------|----------------|
| 0–20 | Match très incertain, les deux joueurs sont au coude à coude |
| 20–40 | Légère avance, le modèle penche sans certitude |
| 40–100 | Avantage marqué, le modèle est confiant |

Ici P({player_a}) = **{pred['proba_a']:.1%}** → score **{score:.0f}/100**.
                """
            )

        # Radar comparatif
        st.subheader("Comparaison des profils")
        prof_a = predictor.get_player_profile(player_a)
        prof_b = predictor.get_player_profile(player_b)
        st.plotly_chart(radar_chart(prof_a, prof_b, player_a, player_b),
                        use_container_width=True)

        # Features décisives
        if pred["top_features"]:
            st.subheader("Features décisives")
            features_table(pred["top_features"], player_a, player_b)


def page_results(predictor):
    st.title("Saisie des Résultats RG 2026")

    st.info(
        "Saisissez chaque résultat après le match — qualifications comprises. "
        "Le modèle se réentraîne automatiquement."
    )

    from data_loader import get_player_list
    all_players = get_player_list(predictor.df) or PLAYER_LIST_DEFAULT

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
        with st.spinner("Mise à jour du modèle..."):
            predictor.add_result(winner, loser, score, round_num,
                                  match_date=str(match_date))
        st.success(f"Résultat enregistré : **{winner}** bat {loser} ({score}) — {ROUND_LABELS[round_num]}")
        st.cache_resource.clear()
        st.rerun()

    # Résultats déjà saisis
    st.subheader("Résultats enregistrés")
    if predictor._rg2026_results:
        df_res = pd.DataFrame(predictor._rg2026_results)
        df_res["Tour"] = df_res["round_number"].map(ROUND_LABELS)
        display_cols = {"Tour": "Tour", "winner": "Vainqueur", "loser": "Perdant", "score": "Score"}
        avail = {k: v for k, v in display_cols.items() if k in df_res.columns}
        st.dataframe(
            df_res[list(avail.keys())].rename(columns=avail),
            use_container_width=True,
            hide_index=True,
        )

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
            st.dataframe(fat_df, use_container_width=True, hide_index=True)
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
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

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

    if st.button("Lancer le backtesting (RG 2017-2025)", type="primary"):
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
                "model": "XGBoost (n_estimators=300, max_depth=4, lr=0.05) + CalibratedCV",
                "données": "ATP 2000-2025 clay",
                "test": "RG 2017-2025 expanding window",
            }
            saved_path = save_backtest_results(results, baseline_df, config)

        st.success(f"Résultats sauvegardés dans `{saved_path.relative_to(Path(__file__).parent.parent)}`")

        # Persister les résultats dans session_state pour survivre aux re-renders
        st.session_state["bt_results"] = results
        st.session_state["bt_baseline"] = baseline_df
        st.session_state["bt_all_feat"] = bt["all_features"]
        st.session_state["bt_rg_feat"]  = bt["rg_features"]

    # Afficher les résultats s'ils existent en session (persistent après le clic)
    if "bt_results" in st.session_state and st.session_state["bt_results"]:
        _display_backtest(st.session_state["bt_results"], st.session_state["bt_baseline"])

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
                st.dataframe(params_df, use_container_width=True, hide_index=True)
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
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

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
    st.plotly_chart(fig, use_container_width=True)

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
            use_container_width=True,
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
        st.plotly_chart(fig_round, use_container_width=True)

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
        st.dataframe(df_disp, use_container_width=True, hide_index=True)

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
        st.plotly_chart(fig2, use_container_width=True)


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
        ["Vue d'ensemble", "Match par match", "Saisie résultats", "Backtesting"],
    )

    raw_dir = Path(__file__).parent.parent / "data" / "raw"
    if not list(raw_dir.glob("atp_matches_*.csv")):
        st.error(
            "Aucune donnée trouvée dans `data/raw/`.\n\n"
            "Lancez d'abord :\n```bash\nbash scripts/download_data.sh\n```"
        )
        st.stop()

    predictor = get_predictor()

    if page == "Vue d'ensemble":
        page_overview(predictor)
    elif page == "Match par match":
        page_match(predictor)
    elif page == "Saisie résultats":
        page_results(predictor)
    elif page == "Backtesting":
        page_backtest(predictor)


if __name__ == "__main__":
    main()
