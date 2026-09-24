import os
import sys
import streamlit as st

# Ensure project root is in sys.path so src module can be imported cleanly
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.prediction.predictor import CricketPredictor

# Set page configuration
st.set_page_config(
    page_title="Cricket ML Predictor",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load CricketPredictor once using Streamlit resource caching
@st.cache_resource
def get_predictor():
    return CricketPredictor()

def main():
    # Application Title
    st.title("🏏 Cricket ML Predictor")
    st.caption("Machine Learning Prediction System for Individual Player Match Performances")
    
    # Initialize Predictor
    try:
        predictor = get_predictor()
    except Exception as e:
        st.error(f"Failed to initialize Prediction Engine: {e}")
        st.stop()

    # Sidebar Navigation & Inputs
    st.sidebar.title("🏏 Navigation")
    st.sidebar.markdown(
        "Forecast individual player performance metrics (Runs, Wickets, Economy, Strike Rates) "
        "using leakage-free ML models trained on 7.14M ball-by-ball delivery records."
    )
    
    st.sidebar.divider()
    
    # Searchable Player Selection
    known_players = sorted(list(predictor.known_players))
    
    selected_player = st.sidebar.selectbox(
        "Select Player",
        options=known_players,
        index=0 if known_players else None,
        format_func=lambda x: x.title(),
        help="Search and select any known player from historical match records."
    )

    predict_btn = st.sidebar.button("🔮 Generate Predictions", type="primary", use_container_width=True)

    # Main Dashboard Content
    if selected_player:
        st.header(f"Player Dashboard: {selected_player.title()}")

        # Generate Prediction via Engine
        res = predictor.predict_player(selected_player)

        if res.get("status") == "error":
            st.error(f"Error generating predictions: {res.get('message')}")
            st.stop()

        # Overview Metadata Cards
        match_id_latest = res.get("prediction_point_match_id", "N/A")
        eval_matches = res.get("historical_matches_used", 0)
        has_batting = res.get("has_batting_history", False)
        has_bowling = res.get("has_bowling_history", False)

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Historical Matches Evaluated", f"{eval_matches:,}")
        with m2:
            st.metric("Last Match ID", f"{match_id_latest}")
        with m3:
            st.metric("Batting Status", "Active" if has_batting else "No History")
        with m4:
            st.metric("Bowling Status", "Active" if has_bowling else "No History")

        st.divider()

        # Predictions Sections
        if predict_btn or True:  # Display automatically and update on button click
            if not has_batting and not has_bowling:
                st.warning(f"Player '{selected_player.title()}' has insufficient historical records for predictions.")
                st.stop()

            # 1. BATTING PREDICTIONS
            if has_batting:
                st.subheader("🏏 Batting Predictions")
                batting = res.get("batting", {})
                
                b1, b2, b3, b4 = st.columns(4)
                with b1:
                    exp_runs = batting.get("expected_runs")
                    st.metric("Expected Runs", f"{exp_runs:.1f}" if exp_runs is not None else "N/A")
                with b2:
                    exp_avg = batting.get("x_batting_average")
                    st.metric("Expected Batting Avg", f"{exp_avg:.1f}" if exp_avg is not None else "N/A")
                with b3:
                    exp_sr = batting.get("x_strike_rate")
                    st.metric("Expected Strike Rate", f"{exp_sr:.1f}" if exp_sr is not None else "N/A")
                with b4:
                    exp_bpb = batting.get("x_balls_per_boundary")
                    st.metric("Expected Balls / Boundary", f"{exp_bpb:.1f}" if exp_bpb is not None else "N/A")
            else:
                st.info(f"ℹ️ {selected_player.title()} has no prior batting history recorded.")

            st.divider()

            # 2. BOWLING PREDICTIONS
            if has_bowling:
                st.subheader("🎯 Bowling Predictions")
                bowling = res.get("bowling", {})

                w1, w2, w3 = st.columns(3)
                with w1:
                    exp_wickets = bowling.get("x_wickets")
                    st.metric("Expected Wickets", f"{exp_wickets:.1f}" if exp_wickets is not None else "N/A")
                with w2:
                    exp_econ = bowling.get("x_economy_rate")
                    st.metric("Expected Economy Rate", f"{exp_econ:.1f}" if exp_econ is not None else "N/A")
                with w3:
                    exp_bsr = bowling.get("x_bowling_strike_rate")
                    st.metric("Expected Bowling SR", f"{exp_bsr:.1f}" if exp_bsr is not None else "N/A")
            else:
                st.info(f"ℹ️ {selected_player.title()} has no prior bowling history recorded.")

            # Disclaimer
            st.caption(
                "⚠️ **Disclaimer:** Predictions are statistical estimates based on historical data "
                "and should not be treated as guaranteed outcomes."
            )

    st.divider()
    # Footer
    st.markdown(
        "<div style='text-align: center; color: gray; padding: 10px;'>"
        "Model-powered prediction using historical cricket performance data."
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
