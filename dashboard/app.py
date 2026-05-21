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
# Helpers UI
# ------------------------------------------------------------------

ROUND_LABELS = {1: "1er tour", 2: "2e tour", 3: "3e tour", 4: "8e de finale",
                5: "Quarts", 6: "Demis", 7: "Finale"}

PLAYER_LIST_DEFAULT = [
    "Carlos Alcaraz", "Jannik Sinner", "Alexander Zverev", "Casper Ruud",
    "Stefanos Tsitsipas", "Hubert Hurkacz", "Andrey Rublev", "Taylor Fritz",
    "Tommy Paul", "Ben Shelton", "Holger Rune", "Grigor Dimitrov",
    "Felix Auger-Aliassime", "Francisco Cerundolo", "Lorenzo Musetti",
    "Sebastian Baez", "Flavio Cobolli", "Arthur Fils", "Gael Monfils",
    "Stan Wawrinka", "Rafael Nadal", "Novak Djokovic",
]


def proba_bar(proba_a: float, player_a: str, player_b: str):
    fig = go.Figure(go.Bar(
        x=[proba_a * 100, (1 - proba_a) * 100],
        y=[player_a, player_b],
        orientation="h",
        marker_color=["#e8473f" if proba_a >= 0.5 else "#aaa", "#aaa" if proba_a >= 0.5 else "#e8473f"],
        text=[f"{proba_a*100:.1f}%", f"{(1-proba_a)*100:.1f}%"],
        textposition="inside",
    ))
    fig.update_layout(
        height=120, margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(range=[0, 100], showticklabels=False),
        yaxis=dict(showticklabels=True),
        showlegend=False,
    )
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


# ------------------------------------------------------------------
# Pages
# ------------------------------------------------------------------

def page_overview(predictor):
    st.title("Roland Garros 2026 — Vue d'ensemble")

    # Saisie du tableau
    st.subheader("Joueurs du tournoi")
    players_input = st.text_area(
        "Un joueur par ligne (128 joueurs pour un tableau complet) :",
        value="\n".join(PLAYER_LIST_DEFAULT),
        height=200,
    )
    players = [p.strip() for p in players_input.strip().split("\n") if p.strip()]

    if len(players) < 2:
        st.warning("Entrez au moins 2 joueurs.")
        return

    st.subheader(f"Simulation — {len(players)} joueurs")
    n_sim = st.slider("Nombre de simulations Monte Carlo", 1000, 20000, 5000, step=1000)

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

    # Profils joueurs
    st.subheader("Profils Elo des joueurs")
    profiles = []
    for p in players:
        prof = predictor.get_player_profile(p)
        profiles.append({
            "Joueur": p,
            "Rang": int(prof.get("rank", 999)) if pd.notna(prof.get("rank")) else "N/A",
            "Clay Elo": f"{prof.get('clay_elo', 1500):.0f}",
            "WElo": f"{prof.get('welo', 1500):.0f}",
            "Adj. Elo": f"{prof.get('adjusted_elo', 1500):.0f}",
        })
    st.dataframe(pd.DataFrame(profiles), use_container_width=True)


def page_match(predictor):
    st.title("Prédiction Match par Match")

    # Récupération de la liste des joueurs
    from data_loader import get_player_list
    all_players = get_player_list(predictor.df)
    if not all_players:
        all_players = PLAYER_LIST_DEFAULT

    col1, col2 = st.columns(2)
    with col1:
        player_a = st.selectbox("Joueur A", all_players, index=0)
    with col2:
        player_b = st.selectbox("Joueur B", all_players, index=min(1, len(all_players) - 1))

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
        st.success(f"Vainqueur prédit : **{winner}** (confiance : {pred['confidence']})")

        st.plotly_chart(proba_bar(pred["proba_a"], player_a, player_b), use_container_width=True)

        # Radar comparatif
        st.subheader("Comparaison des profils")
        prof_a = predictor.get_player_profile(player_a)
        prof_b = predictor.get_player_profile(player_b)
        st.plotly_chart(radar_chart(prof_a, prof_b, player_a, player_b), use_container_width=True)

        # Top features
        if pred["top_features"]:
            st.subheader("Features décisives")
            feat_df = pd.DataFrame(pred["top_features"])
            feat_df.columns = ["Feature", "Valeur (A-B)", "Importance"]
            st.dataframe(feat_df, use_container_width=True)


def page_results(predictor):
    st.title("Saisie des Résultats RG 2026")

    st.info(
        "Saisissez chaque résultat après le match. "
        "Le modèle se réentraîne automatiquement pour les tours suivants."
    )

    from data_loader import get_player_list
    all_players = get_player_list(predictor.df)
    if not all_players:
        all_players = PLAYER_LIST_DEFAULT

    col1, col2 = st.columns(2)
    with col1:
        winner = st.selectbox("Vainqueur", all_players, key="res_winner")
    with col2:
        loser_opts = [p for p in all_players if p != winner]
        loser = st.selectbox("Perdant", loser_opts, key="res_loser")

    score = st.text_input("Score (ex: 6-4 7-5 6-3)", "6-4 6-3")
    round_num = st.select_slider(
        "Tour",
        options=list(ROUND_LABELS.keys()),
        format_func=lambda x: ROUND_LABELS[x],
        value=1,
        key="res_round",
    )

    if st.button("Enregistrer le résultat", type="primary"):
        with st.spinner("Mise à jour du modèle..."):
            predictor.add_result(winner, loser, score, round_num)
        st.success(f"Résultat enregistré : {winner} bat {loser} ({score})")
        st.cache_resource.clear()
        st.rerun()

    # Résultats déjà saisis
    st.subheader("Résultats enregistrés")
    if predictor._rg2026_results:
        df_res = pd.DataFrame(predictor._rg2026_results)
        df_res["tour"] = df_res["round_number"].map(ROUND_LABELS)
        st.dataframe(
            df_res[["tour", "winner", "loser", "score"]],
            use_container_width=True,
        )
    else:
        st.write("Aucun résultat encore saisi.")


def page_backtest(predictor):
    st.title("Backtesting & Performance Historique")

    if st.button("Lancer le backtesting (RG 2017-2025)", type="primary"):
        with st.spinner("Backtesting en cours (peut prendre 1-2 minutes)..."):
            from backtesting import run_full_backtest
            bt = run_full_backtest(str(Path(predictor.data_path)))

        results = bt["backtest_results"]
        baseline_df = bt["baseline_comparison"]

        # Tableau accuracy par année
        if results:
            st.subheader("Accuracy par édition")
            rows = []
            for r in results:
                rows.append({
                    "Année": r.year,
                    "Accuracy": f"{r.accuracy:.1%}",
                    "Brier Score": f"{r.brier:.4f}",
                    "Log-Loss": f"{r.log_loss_val:.4f}",
                    "N matchs": r.n_matches,
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

            # Graphe évolution
            fig = px.line(
                pd.DataFrame([{"Année": r.year, "Accuracy": r.accuracy} for r in results]),
                x="Année", y="Accuracy", markers=True,
                title="Accuracy XGBoost par édition Roland Garros",
            )
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)

        # Comparaison baselines
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
            df_disp = baseline_df[list(avail.keys())].rename(columns=avail)
            fmt = {c: "{:.1%}" for c in df_disp.columns if c != "Année"}
            for col, f in fmt.items():
                if col in df_disp.columns:
                    df_disp[col] = df_disp[col].map(lambda x: f.format(x) if pd.notna(x) else "N/A")
            st.dataframe(df_disp, use_container_width=True)

            # Graphe comparatif
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
    st.sidebar.markdown("*Prédicteur ML basé sur WElo + XGBoost*")

    page = st.sidebar.radio(
        "Navigation",
        ["Vue d'ensemble", "Match par match", "Saisie résultats", "Backtesting"],
    )

    # Vérification données
    raw_dir = Path(__file__).parent.parent / "data" / "raw"
    csv_files = list(raw_dir.glob("atp_matches_*.csv"))
    if not csv_files:
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
