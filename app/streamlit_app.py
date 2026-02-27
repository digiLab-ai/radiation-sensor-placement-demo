from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from simulator import Box, Detector
from simulator.batch import run_design

# --- digiLab brand  ---
INDIGO = "#16425B"
KEPPEL = "#16D5C2"
KEY_LIME = "#EBF38B"
INK = "#000000"
BG = "#FFFFFF"


def inject_brand_css() -> None:
    st.markdown(
        f"""
        <style>
          :root {{
            --digilab-indigo: {INDIGO};
            --digilab-keppel: {KEPPEL};
            --digilab-keylime: {KEY_LIME};
            --digilab-ink: {INK};
            --digilab-bg: {BG};
          }}
          .stApp {{
            background: var(--digilab-bg);
            color: var(--digilab-ink);
          }}
          .digilab-title {{
            font-size: 2.0rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            margin-bottom: 0.25rem;
          }}
          .digilab-subtitle {{
            color: rgba(0, 0, 0, 0.72);
            margin-top: 0;
          }}
          .digilab-card {{
            background: white;
            border-radius: 16px;
            padding: 16px 16px 8px 16px;
            border: 1px solid rgba(0, 0, 0, 0.10);
            box-shadow: 0 6px 18px rgba(0, 0, 0, 0.06);
          }}
          .digilab-pill {{
            display: inline-block;
            padding: 0.15rem 0.55rem;
            border-radius: 999px;
            background: rgba(22, 66, 91, 0.10);
            border: 1px solid rgba(22, 66, 91, 0.25);
            color: var(--digilab-indigo);
            font-weight: 600;
            font-size: 0.85rem;
          }}
          div[data-testid="stSidebar"] {{
            background: linear-gradient(
              180deg,
              rgba(22, 66, 91, 0.10),
              rgba(22, 213, 194, 0.08) 60%,
              rgba(235, 243, 139, 0.22)
            );
            border-right: 1px solid rgba(0, 0, 0, 0.08);
          }}
          .stButton>button {{
            background: var(--digilab-indigo);
            color: white;
            border-radius: 12px;
            border: 0;
            padding: 0.6rem 0.9rem;
            font-weight: 700;
          }}
          .stDownloadButton>button {{
            background: var(--digilab-keppel);
            color: var(--digilab-ink);
            border-radius: 12px;
            border: 0;
            padding: 0.6rem 0.9rem;
            font-weight: 700;
          }}
          a {{
            color: var(--digilab-indigo);
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def parse_distances(text: str) -> np.ndarray:
    toks = [t for t in text.replace(",", " ").split() if t.strip()]
    d = np.array([float(t) for t in toks], dtype=float)
    if d.size == 0:
        raise ValueError("No distances provided.")
    if np.any(d < 0):
        raise ValueError("Distances must be >= 0.")
    return d


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def main() -> None:
    st.set_page_config(page_title="digiLab Hotspot Detector Simulator", layout="wide")
    inject_brand_css()

    cols = st.columns([1, 5])
    with cols[0]:
        logo_path = Path(__file__).resolve().parents[1] / "assets" / "digilab.png"
        if logo_path.exists():
            st.image(str(logo_path), use_container_width=True)
        else:
            st.markdown('<div class="digilab-pill">LOGO PLACEHOLDER</div>', unsafe_allow_html=True)

    with cols[1]:
        st.markdown('<div class="digilab-title">Hotspot Detector Simulator</div>', unsafe_allow_html=True)
        st.markdown(
            '<p class="digilab-subtitle">Generate synthetic count-rate measurements vs detector distance for a fixed-size sample box, with nuisance parameters and DOE sampling.</p>',
            unsafe_allow_html=True,
        )

    overview_path = Path(__file__).resolve().parents[1] / "assets" / "overview.png"
    if overview_path.exists():
        st.image(str(overview_path), use_container_width=True)

    st.markdown("")

    with st.sidebar:
        st.header("Experiment design")

        distances_text = st.text_area(
            "Candidate distances (m)",
            value="0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25",
            help="Comma or space separated.",
            height=90,
        )

        n_samples = st.number_input("Number of samples", min_value=1, max_value=50000, value=200, step=10)
        strategy = st.selectbox("Sampling strategy", ["lhs", "sobol", "random"], index=0)
        seed = st.number_input("Seed", min_value=0, max_value=2**31-1, value=0, step=1)

        st.divider()
        st.header("Fixed sample box")
        Lx = st.number_input("Box width Lx (m)", 1e-3, 10.0, 0.10, 0.01)
        Ly = st.number_input("Box depth Ly (m)", 1e-3, 10.0, 0.04, 0.01)
        Lz = st.number_input("Box height Lz (m)", 1e-3, 10.0, 0.10, 0.01)

        st.divider()
        st.header("Hotspot bounds")

        def bound_row(lbl: str, lo: float, hi: float, step: float, fmt: str = "%.6f"):
            c1, c2 = st.columns(2)
            with c1:
                a = st.number_input(f"{lbl} low", value=lo, step=step, format=fmt)
            with c2:
                b = st.number_input(f"{lbl} high", value=hi, step=step, format=fmt)
            return float(a), float(b)

        b_width = bound_row("x position width_x_m", 0.0, float(Lx), 0.005)
        b_depth = bound_row("y position depth_y_m", 0.0, float(Ly), 0.002)
        b_height = bound_row("z position height_z_m", 0.0, float(Lz), 0.005)
        b_activity = bound_row("mean_activity_bq", 1e4, 1e6, 1e4, fmt="%.6g")
        b_size = bound_row("size_sigma_m", 0.001, 0.010, 0.0005)

        st.divider()
        st.header("Nuisance parameters")

        use_fov = st.checkbox("Use FOV cone", value=True)
        fov_half_angle_deg = st.slider("FOV half-angle (deg)", 1.0, 89.0, 30.0, 1.0) if use_fov else None

        mu_material_m_inv = st.number_input("Material attenuation μ (1/m)", 0.0, 1e4, 8.0, 0.5)
        noise = st.selectbox("Sensor noise", ["poisson", "none"], index=0)
        distance_offset_m = st.number_input("Distance offset (m)", 0.0, 1.0, 0.0, 0.001, format="%.3f")

        st.divider()
        st.header("Detector")

        area_m2 = st.number_input("Area (m²)", 1e-8, 0.1, 2.5e-4, 1e-5, format="%.6f")
        efficiency = st.slider("Efficiency", 0.0, 1.0, 0.2, 0.01)
        background_cps = st.number_input("Background (cps)", 0.0, 1e6, 1.5, 0.1)
        dwell_s = st.number_input("Dwell time (s)", 0.001, 1e4, 2.0, 0.5)

    current_sidebar_signature = (
        distances_text.strip(),
        int(n_samples),
        str(strategy),
        int(seed),
        float(Lx),
        float(Ly),
        float(Lz),
        tuple(map(float, b_width)),
        tuple(map(float, b_depth)),
        tuple(map(float, b_height)),
        tuple(map(float, b_activity)),
        tuple(map(float, b_size)),
        bool(use_fov),
        None if fov_half_angle_deg is None else float(fov_half_angle_deg),
        float(mu_material_m_inv),
        str(noise),
        float(distance_offset_m),
        float(area_m2),
        float(efficiency),
        float(background_cps),
        float(dwell_s),
    )

    if "last_run_signature" not in st.session_state:
        st.session_state.last_run_signature = None
    if "cached_inputs_df" not in st.session_state:
        st.session_state.cached_inputs_df = None
    if "cached_measurements_df" not in st.session_state:
        st.session_state.cached_measurements_df = None
    if "cached_distances_m" not in st.session_state:
        st.session_state.cached_distances_m = None

    sidebar_changed_after_run = (
        st.session_state.last_run_signature is not None
        and current_sidebar_signature != st.session_state.last_run_signature
    )
    if sidebar_changed_after_run:
        st.session_state.cached_inputs_df = None
        st.session_state.cached_measurements_df = None
        st.session_state.cached_distances_m = None

    status_placeholder = st.empty()
    run_btn = st.button("Run simulation", use_container_width=True)

    if run_btn:
        try:
            distances_m = parse_distances(distances_text)
        except Exception as e:
            st.error(f"Could not parse distances: {e}")
            return

        box = Box(Lx=float(Lx), Ly=float(Ly), Lz=float(Lz))
        detector = Detector(
            area_m2=float(area_m2),
            efficiency=float(efficiency),
            background_cps=float(background_cps),
            dwell_s=float(dwell_s),
        )

        bounds: Dict[str, Tuple[float, float]] = {
            "width_x_m": b_width,
            "depth_y_m": b_depth,
            "height_z_m": b_height,
            "mean_activity_bq": b_activity,
            "size_sigma_m": b_size,
        }

        try:
            inputs_df, measurements_df = run_design(
                distances_m=distances_m,
                box=box,
                detector=detector,
                bounds=bounds,
                n_samples=int(n_samples),
                strategy=str(strategy),
                seed=int(seed),
                mu_material_m_inv=float(mu_material_m_inv),
                fov_half_angle_deg=fov_half_angle_deg,
                distance_offset_m=float(distance_offset_m),
                noise=str(noise),
            )
        except Exception as e:
            st.error(f"Simulation failed: {e}")
            return

        st.session_state.cached_inputs_df = inputs_df
        st.session_state.cached_measurements_df = measurements_df
        st.session_state.cached_distances_m = distances_m
        st.session_state.last_run_signature = current_sidebar_signature
        status_placeholder.empty()
    elif st.session_state.cached_inputs_df is None or st.session_state.cached_measurements_df is None:
        status_placeholder.markdown(
            '<div class="digilab-card">Configure the design in the sidebar, then click <b>Run simulation</b>.</div>',
            unsafe_allow_html=True,
        )
        return
    else:
        status_placeholder.empty()

    inputs_df = st.session_state.cached_inputs_df
    measurements_df = st.session_state.cached_measurements_df
    distances_m = st.session_state.cached_distances_m

    left, right = st.columns([1, 1])

    with left:
        st.subheader("Quantities of Interest")
        st.dataframe(inputs_df, use_container_width=True, height=420)
        st.download_button(
            "Download quantities_of_interest CSV",
            data=df_to_csv_bytes(inputs_df),
            file_name="quantities_of_interest.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with right:
        st.subheader("Measurements dataframe")
        st.dataframe(measurements_df, use_container_width=True, height=420)
        st.download_button(
            "Download measurements CSV",
            data=df_to_csv_bytes(measurements_df),
            file_name="measurements.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.markdown("")
    st.subheader("Quick look")
    n_samples, n_dist = measurements_df.shape
    sample_ids = np.arange(n_samples, dtype=int)
    sample_labels = np.array([f"sample_{i}" for i in sample_ids])

    rng = np.random.default_rng(int(seed) + 2026)
    default_n = min(10, n_samples)
    default_ids = np.sort(rng.choice(sample_ids, size=default_n, replace=False))
    sample_options = [f"sample_{i}" for i in sample_ids]
    default_labels = [f"sample_{i}" for i in default_ids]
    default_selection = [{"sample_label": lbl} for lbl in default_labels]

    if "quick_look_selected_labels" not in st.session_state:
        st.session_state.quick_look_selected_labels = default_labels
    else:
        selected = st.session_state.quick_look_selected_labels
        if any(lbl not in sample_options for lbl in selected):
            st.session_state.quick_look_selected_labels = default_labels

    plot_df = pd.DataFrame(
        {
            "sample_label": np.repeat(sample_labels, n_dist),
            "distance_m": np.tile(distances_m, n_samples),
            "cps": measurements_df.to_numpy().reshape(-1),
        }
    )
    quick_look_left, quick_look_right = st.columns([5, 2])

    with quick_look_right:
        selected_labels = st.multiselect(
            "Samples",
            options=sample_options,
            key="quick_look_selected_labels",
            help="Scrollable list for selecting many samples at once.",
        )

    filtered_plot_df = plot_df[plot_df["sample_label"].isin(selected_labels)]
    if filtered_plot_df.empty:
        with quick_look_left:
            st.info("Select at least one sample to display traces.")
        # st.caption("Default selection is 10 random samples.")
        return

    select_samples = alt.selection_point(
        fields=["sample_label"],
        bind="legend",
        toggle="true",
        clear=False,
        value=default_selection,
    )

    quick_look_chart = (
        alt.Chart(filtered_plot_df)
        .mark_line()
        .encode(
            x=alt.X("distance_m:Q", title="Distance (m)"),
            y=alt.Y("cps:Q", title="Count rate (cps)"),
            color=alt.Color("sample_label:N", title="Samples"),
            opacity=alt.condition(select_samples, alt.value(0.95), alt.value(0.0)),
        )
        .add_params(select_samples)
        .properties(height=380)
    )
    with quick_look_left:
        st.altair_chart(quick_look_chart, use_container_width=True)
    st.caption(
        f"Use the sample list for bulk selection and the legend for quick toggling."
    )


if __name__ == "__main__":
    main()
