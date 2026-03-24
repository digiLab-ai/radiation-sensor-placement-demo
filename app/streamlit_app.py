from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, Optional, Tuple

import altair as alt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulator import Box, Detector, Hotspot, simulate_measured_activity
from simulator.batch import run_design

# --- digiLab brand  ---
INDIGO = "#16425B"
KEPPEL = "#16D5C2"
GREY = "#7A8793"
KEY_LIME = "#EBF38B"
INK = "#000000"
BG = "#FFFFFF"

QOI_FIELDS = [
    "mean_activity_bq",
    "width_x_m",
    "depth_y_m",
    "height_z_m",
    "size_sigma_m",
]

QOI_PUBLIC_NAMES = {
    "mean_activity_bq": "activity_mean_bq",
    "width_x_m": "x_position_m",
    "depth_y_m": "y_position_m",
    "height_z_m": "z_position_m",
    "size_sigma_m": "size_m",
}

QOI_LABELS = {
    "mean_activity_bq": "activity_mean_bq",
    "width_x_m": "x_position_m",
    "depth_y_m": "y_position_m",
    "height_z_m": "z_position_m",
    "size_sigma_m": "size_m",
}


def inject_brand_css() -> None:
    st.markdown(
        f"""
        <style>
          :root {{
            --digilab-indigo: {INDIGO};
            --digilab-keppel: {KEPPEL};
            --digilab-grey: {GREY};
            --digilab-keylime: {KEY_LIME};
            --digilab-ink: {INK};
            --digilab-bg: {BG};
          }}
          .stApp {{
            background:
              radial-gradient(circle at top right, rgba(22, 213, 194, 0.14), transparent 28%),
              radial-gradient(circle at bottom left, rgba(22, 66, 91, 0.12), transparent 30%),
              var(--digilab-bg);
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
          div[data-baseweb="tab-list"] {{
            gap: 0.5rem;
          }}
          button[data-baseweb="tab"] {{
            border-radius: 999px;
            border: 1px solid rgba(22, 66, 91, 0.12);
            background: rgba(255, 255, 255, 0.8);
            padding: 0.35rem 0.95rem;
          }}
          button[data-baseweb="tab"][aria-selected="true"] {{
            background: rgba(22, 66, 91, 0.08);
            border-color: rgba(22, 66, 91, 0.32);
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


def midpoint(bounds: Tuple[float, float]) -> float:
    return float(bounds[0] + (bounds[1] - bounds[0]) / 2.0)


def slider_step(bounds: Tuple[float, float]) -> float:
    lo, hi = bounds
    return max((float(hi) - float(lo)) / 100.0, 1e-6)


def sensor_names(n_sensors: int) -> list[str]:
    return [f"S{i}" for i in range(1, n_sensors + 1)]


def default_sensor_positions(box: Box) -> np.ndarray:
    x_levels = np.linspace(-0.15 * box.Lx, 1.15 * box.Lx, 3)
    y_levels = np.array([-1.1 * box.Ly, -0.55 * box.Ly, -0.15 * box.Ly], dtype=float)
    z_levels = np.linspace(-0.15 * box.Lz, 1.15 * box.Lz, 5)

    positions = []
    for z_index, z_value in enumerate(z_levels):
        for x_index, x_value in enumerate(x_levels):
            y_value = y_levels[(x_index + z_index) % len(y_levels)]
            positions.append((x_value, y_value, z_value))
    return np.asarray(positions, dtype=float)


def sensor_layout_dataframe(box: Box, sensor_positions_m: np.ndarray) -> pd.DataFrame:
    names = sensor_names(len(sensor_positions_m))
    sample_x = np.full(len(sensor_positions_m), box.Lx / 2.0)
    sample_y = np.full(len(sensor_positions_m), box.Ly / 2.0)
    sample_z = np.full(len(sensor_positions_m), box.Lz / 2.0)
    return pd.DataFrame(
        {
            "sensor": names,
            "sensor_x_m": sensor_positions_m[:, 0],
            "sensor_y_m": sensor_positions_m[:, 1],
            "sensor_z_m": sensor_positions_m[:, 2],
            "sample_x_m": sample_x,
            "sample_y_m": sample_y,
            "sample_z_m": sample_z,
        }
    )


def present_qoi_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    ordered_columns = [field for field in QOI_FIELDS if field in df.columns]
    renamed = df.loc[:, ordered_columns].rename(columns=QOI_PUBLIC_NAMES)
    return renamed


def present_measurements_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return df.copy()


def make_hotspot(values: Dict[str, float]) -> Hotspot:
    return Hotspot(
        width_x_m=float(values["width_x_m"]),
        depth_y_m=float(values["depth_y_m"]),
        height_z_m=float(values["height_z_m"]),
        mean_activity_bq=float(values["mean_activity_bq"]),
        size_sigma_m=float(values["size_sigma_m"]),
    )


def truth_dataframe(values: Dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame([{QOI_PUBLIC_NAMES[key]: value for key, value in values.items()}])


def measurement_dataframe(
    *,
    sensor_positions_m: np.ndarray,
    box: Box,
    detector: Detector,
    hotspot: Hotspot,
    mu_material_m_inv: float,
    fov_half_angle_deg: Optional[float],
    distance_offset_m: float,
    noise: str,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 4242)
    expected_cps, measured_cps = simulate_measured_activity(
        None,
        box=box,
        detector=detector,
        hotspot=hotspot,
        sensor_positions_m=sensor_positions_m,
        mu_material_m_inv=mu_material_m_inv,
        fov_half_angle_deg=fov_half_angle_deg,
        distance_offset_m=distance_offset_m,
        noise=noise,  # type: ignore[arg-type]
        rng=rng,
    )
    labels = sensor_names(len(sensor_positions_m))
    measurement_row = {label: value for label, value in zip(labels, measured_cps)}
    return pd.DataFrame([measurement_row])


def read_csv_upload(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        raise ValueError("No CSV file provided.")
    uploaded_file.seek(0)
    return pd.read_csv(uploaded_file)


def extract_qoi_values(
    df: pd.DataFrame,
    qoi_fields: list[str],
    *,
    value_candidates: list[str],
) -> Dict[str, float]:
    if df.empty:
        raise ValueError("Uploaded CSV is empty.")

    allowed_names = {field: field for field in qoi_fields}
    allowed_names.update({public_name: internal_name for internal_name, public_name in QOI_PUBLIC_NAMES.items()})

    wide_matches = [column for column in df.columns if column in allowed_names]
    if wide_matches:
        row = df.iloc[0]
        return {allowed_names[field]: float(row[field]) for field in wide_matches}

    lowered = {str(col).strip().lower(): col for col in df.columns}
    name_col = next(
        (lowered[key] for key in ["quantity", "qoi", "name", "parameter", "variable"] if key in lowered),
        None,
    )
    value_col = next((lowered[key] for key in value_candidates if key in lowered), None)

    if name_col is None or value_col is None:
        raise ValueError(
            "CSV format not recognised. Use either one row with QOI columns, or a long format with quantity/name and value columns."
        )

    values: Dict[str, float] = {}
    for _, row in df[[name_col, value_col]].dropna().iterrows():
        qoi_name = str(row[name_col]).strip()
        if qoi_name in allowed_names:
            values[allowed_names[qoi_name]] = float(row[value_col])
    return values


def gaussian_curve_df(truth: float, mean: float, std: float, bounds: Tuple[float, float]) -> pd.DataFrame:
    sigma = max(float(std), 1e-9)
    x_min = float(bounds[0])
    x_max = float(bounds[1])
    x = np.linspace(x_min, x_max, 300)
    density = (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((x - mean) / sigma) ** 2)
    return pd.DataFrame({"value": x, "density": density})


def comparison_chart(
    quantity: str,
    truth: float,
    mean: float,
    std: float,
    bounds: Tuple[float, float],
) -> alt.Chart:
    curve_df = gaussian_curve_df(truth, mean, std, bounds)
    truth_df = pd.DataFrame({"truth": [truth]})
    mean_df = pd.DataFrame({"mean": [mean]})
    x_scale = alt.Scale(domain=[float(bounds[0]), float(bounds[1])])

    area = (
        alt.Chart(curve_df)
        .mark_area(color=GREY, opacity=0.28)
        .encode(
            x=alt.X("value:Q", title=quantity, scale=x_scale),
            y=alt.Y("density:Q", title="Probability Density"),
        )
    )
    line = (
        alt.Chart(curve_df)
        .mark_line(color=INDIGO, strokeWidth=3)
        .encode(x="value:Q", y="density:Q")
    )
    truth_rule = (
        alt.Chart(truth_df)
        .mark_rule(color=KEPPEL, strokeWidth=3, strokeDash=[8, 6])
        .encode(x="truth:Q")
    )
    mean_rule = (
        alt.Chart(mean_df)
        .mark_rule(color=INDIGO, strokeWidth=2, opacity=0.65)
        .encode(x="mean:Q")
    )

    return (area + line + truth_rule + mean_rule).properties(height=220, title=quantity)


def sensor_schematic_figure(box: Box, sensor_positions_m: np.ndarray) -> go.Figure:
    labels = sensor_names(len(sensor_positions_m))
    sample_center = np.array([box.Lx / 2.0, box.Ly / 2.0, box.Lz / 2.0], dtype=float)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=sensor_positions_m[:, 0],
            y=sensor_positions_m[:, 1],
            z=sensor_positions_m[:, 2],
            mode="markers+text",
            text=labels,
            textposition="top center",
            name="Sensors",
            marker=dict(size=6, color=INDIGO),
            hovertemplate="Sensor %{text}<br>x=%{x:.3f} m<br>y=%{y:.3f} m<br>z=%{z:.3f} m<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[sample_center[0]],
            y=[sample_center[1]],
            z=[sample_center[2]],
            mode="text",
            text=["Sample"],
            textposition="bottom center",
            name="Sample center",
            hovertemplate="Sample center<br>x=%{x:.3f} m<br>y=%{y:.3f} m<br>z=%{z:.3f} m<extra></extra>",
        )
    )

    corners = np.array(
        [
            [0.0, 0.0, 0.0],
            [box.Lx, 0.0, 0.0],
            [box.Lx, box.Ly, 0.0],
            [0.0, box.Ly, 0.0],
            [0.0, 0.0, box.Lz],
            [box.Lx, 0.0, box.Lz],
            [box.Lx, box.Ly, box.Lz],
            [0.0, box.Ly, box.Lz],
        ],
        dtype=float,
    )
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    for start, end in edges:
        fig.add_trace(
            go.Scatter3d(
                x=[corners[start, 0], corners[end, 0]],
                y=[corners[start, 1], corners[end, 1]],
                z=[corners[start, 2], corners[end, 2]],
                mode="lines",
                line=dict(color=KEPPEL, width=5),
                opacity=0.45,
                showlegend=False,
                hoverinfo="skip",
            )
        )

    x_min = min(float(sensor_positions_m[:, 0].min()), 0.0)
    x_max = max(float(sensor_positions_m[:, 0].max()), box.Lx)
    y_min = min(float(sensor_positions_m[:, 1].min()), 0.0)
    y_max = max(float(sensor_positions_m[:, 1].max()), box.Ly)
    z_min = min(float(sensor_positions_m[:, 2].min()), 0.0)
    z_max = max(float(sensor_positions_m[:, 2].max()), box.Lz)

    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=40),
        height=620,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        scene=dict(
            xaxis=dict(title="x (m)", range=[x_min, x_max]),
            yaxis=dict(title="y (m)", range=[y_min, y_max]),
            zaxis=dict(title="z (m)", range=[z_min, z_max]),
            aspectmode="data",
            camera=dict(eye=dict(x=1.45, y=1.35, z=1.1)),
        ),
    )
    return fig


def render_design_results(
    *,
    inputs_df: Optional[pd.DataFrame],
    measurements_df: Optional[pd.DataFrame],
    sensor_positions_m: np.ndarray,
    box: Box,
    seed: int,
    run_requested: bool,
) -> None:
    if inputs_df is None or measurements_df is None:
        if not run_requested:
            st.markdown(
                '<div class="digilab-card">Configure the design in the sidebar, then click <b>Run simulation</b>.</div>',
                unsafe_allow_html=True,
            )
        return

    presented_inputs_df = present_qoi_dataframe(inputs_df)
    presented_measurements_df = present_measurements_dataframe(measurements_df)

    left, right = st.columns([1, 1])

    with left:
        st.subheader("Quantities of Interest")
        st.dataframe(presented_inputs_df, use_container_width=True, height=420)
        st.download_button(
            "Download quantities_of_interest CSV",
            data=df_to_csv_bytes(presented_inputs_df),
            file_name="quantities_of_interest.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with right:
        st.subheader("Measurements dataframe")
        st.dataframe(presented_measurements_df, use_container_width=True, height=420)
        st.download_button(
            "Download measurements CSV",
            data=df_to_csv_bytes(presented_measurements_df),
            file_name="measurements.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.markdown("")
    st.subheader("Sensor placement sketch")
    st.plotly_chart(sensor_schematic_figure(box, sensor_positions_m), use_container_width=True)
    # st.caption("Sample box is shown as the wireframe volume, with sensors labeled S1-S15 and the sample labeled at its center.")


def render_generate_measurement_tab(
    *,
    sensor_positions_m: np.ndarray,
    bounds: Dict[str, Tuple[float, float]],
    box: Box,
    detector: Detector,
    mu_material_m_inv: float,
    fov_half_angle_deg: Optional[float],
    distance_offset_m: float,
    noise: str,
    seed: int,
) -> None:
    st.subheader("Generate measurement")
    st.caption("Tune a single hotspot within the configured bounds, choose available sensors, and export the simulated measurements.")

    controls_col, outputs_col = st.columns([1.4, 1.0], gap="large")

    with controls_col:
        st.markdown("##### Quantities of Interest")
        qoi_values: Dict[str, float] = {}
        for field in QOI_FIELDS:
            lo, hi = bounds[field]
            default_value = float(st.session_state.get(f"measurement_{field}", midpoint((lo, hi))))
            slider_format = "%.3f" if field == "size_sigma_m" else None
            qoi_values[field] = st.slider(
                QOI_LABELS[field],
                min_value=float(lo),
                max_value=float(hi),
                value=min(max(default_value, float(lo)), float(hi)),
                step=slider_step((lo, hi)),
                key=f"measurement_{field}",
                format=slider_format,
            )

    with outputs_col:
        st.markdown("##### Available sensors")
        sensor_choices = sensor_names(len(sensor_positions_m))
        default_selected = sensor_choices if "selected_measurement_sensors" not in st.session_state else st.session_state.selected_measurement_sensors
        selected_sensors = st.multiselect(
            "Sensors",
            options=sensor_choices,
            default=default_selected,
            key="selected_measurement_sensors",
            help="Each sensor corresponds to one candidate detector distance.",
        )

        generate_clicked = st.button("Generate measurement", key="generate_measurement_button", use_container_width=True)

        if generate_clicked:
            if not selected_sensors:
                st.error("Select at least one sensor.")
            else:
                selected_mask = np.array([label in selected_sensors for label in sensor_choices], dtype=bool)
                selected_positions = sensor_positions_m[selected_mask]
                measurement_df = measurement_dataframe(
                    sensor_positions_m=selected_positions,
                    box=box,
                    detector=detector,
                    hotspot=make_hotspot(qoi_values),
                    mu_material_m_inv=mu_material_m_inv,
                    fov_half_angle_deg=fov_half_angle_deg,
                    distance_offset_m=distance_offset_m,
                    noise=noise,
                    seed=seed,
                )
                st.session_state.generated_measurement_df = measurement_df
                st.session_state.generated_truth_qoi = qoi_values

        measurement_df = st.session_state.get("generated_measurement_df")
        if measurement_df is not None:
            st.markdown("##### Measurements")
            st.dataframe(measurement_df, use_container_width=True, height=260)
            file_prefix = st.text_input(
                "CSV filename prefix",
                value="test",
                key="generated_csv_prefix",
            )
            st.download_button(
                "Download measurement CSV",
                data=df_to_csv_bytes(measurement_df),
                file_name=f"{(file_prefix or 'test')}_measurement.csv",
                mime="text/csv",
                use_container_width=True,
            )
            truth_values = st.session_state.get("generated_truth_qoi")
            if truth_values is not None:
                truth_df = truth_dataframe(truth_values)
                st.download_button(
                    "Download truth CSV",
                    data=df_to_csv_bytes(truth_df),
                    file_name=f"{(file_prefix or 'test')}_truth.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
        else:
            st.info("Generate a measurement to populate the downloadable dataframe.")

    st.divider()
    st.subheader("Compare results")
    st.caption("Upload truth, predicted means, and standard deviations as CSV files to compare them.")

    upload_left, upload_mid, upload_right = st.columns(3)
    with upload_left:
        truth_file = st.file_uploader("Truth CSV", type=["csv"], key="comparison_truth_csv")
    with upload_mid:
        mean_file = st.file_uploader("Predicted mean CSV", type=["csv"], key="comparison_mean_csv")
    with upload_right:
        std_file = st.file_uploader("Predicted uncertainty CSV", type=["csv"], key="comparison_std_csv")

    if truth_file is None or mean_file is None or std_file is None:
        st.info("Upload all three CSV files to render the comparison plots.")
        return

    try:
        truth_df = read_csv_upload(truth_file)
        mean_df = read_csv_upload(mean_file)
        std_df = read_csv_upload(std_file)
        truth_qoi = extract_qoi_values(truth_df, QOI_FIELDS, value_candidates=["truth", "value"])
        mean_values = extract_qoi_values(mean_df, QOI_FIELDS, value_candidates=["mean", "prediction", "predicted_mean", "value"])
        std_values = extract_qoi_values(std_df, QOI_FIELDS, value_candidates=["std", "sd", "sigma", "uncertainty", "value"])
    except Exception as exc:
        st.error(f"Could not read comparison CSVs: {exc}")
        return

    missing = [field for field in QOI_FIELDS if field not in truth_qoi or field not in mean_values or field not in std_values]
    if missing:
        st.warning(
            "Missing comparison values for: " + ", ".join(missing)
        )

    available = [field for field in QOI_FIELDS if field in truth_qoi and field in mean_values and field in std_values]
    if not available:
        st.info("No recognised QOI values were found in the uploaded CSV files.")
        return

    for field in available:
        chart = comparison_chart(
            QOI_PUBLIC_NAMES[field],
            truth=float(truth_qoi[field]),
            mean=float(mean_values[field]),
            std=float(std_values[field]),
            bounds=bounds[field],
        )
        st.altair_chart(chart, use_container_width=True)


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
        st.caption("The simulator uses a fixed 3D layout of 15 sensors labeled S1-S15 around the sample volume.")

        n_samples = st.number_input("Number of samples", min_value=1, max_value=50000, value=200, step=10)
        strategy = st.selectbox("Sampling strategy", ["lhs", "sobol", "random"], index=1)
        seed = st.number_input("Seed", min_value=0, max_value=2**31 - 1, value=0, step=1)

        st.divider()
        st.header("Fixed sample box")
        Lx = st.number_input("Box width Lx (m)", 1e-6, 10.0, 0.05, 0.001, format="%.3f")
        Ly = st.number_input("Box depth Ly (m)", 1e-6, 10.0, 0.05, 0.001, format="%.3f")
        Lz = st.number_input("Box height Lz (m)", 1e-6, 10.0, 0.05, 0.001, format="%.3f")

        st.divider()
        st.header("Hotspot bounds")

        def bound_row(label: str, lo: float, hi: float, step: float, fmt: str = "%.6f") -> Tuple[float, float]:
            c1, c2 = st.columns(2)
            with c1:
                low_value = st.number_input(f"{label} low", value=lo, step=step, format=fmt)
            with c2:
                high_value = st.number_input(f"{label} high", value=hi, step=step, format=fmt)
            return float(low_value), float(high_value)

        b_activity = bound_row("activity_mean_bq", 1e4, 1e6, 1e4, fmt="%.6g")
        b_width = bound_row("x_position_m", 0.0, float(Lx), 0.005)
        b_depth = bound_row("y_position_m", 0.0, float(Ly), 0.005)
        b_height = bound_row("z_position_m", 0.0, float(Lz), 0.005)
        b_size = bound_row("size_m", 0.001, 0.010, 0.0005)

        st.divider()
        st.header("Nuisance parameters")

        use_fov = st.checkbox("Use FOV cone", value=True)
        fov_half_angle_deg = st.slider("FOV half-angle (deg)", 1.0, 89.0, 30.0, 1.0) if use_fov else None

        mu_material_m_inv = st.number_input("Material attenuation μ (1/m)", 0.0, 1e4, 8.0, 0.5)
        noise = st.selectbox("Sensor noise", ["none", "poisson"], index=1)
        distance_offset_m = st.number_input("Distance offset (m)", 0.0, 1.0, 0.0, 0.001, format="%.3f")

        st.divider()
        st.header("Detector")

        area_m2 = st.number_input("Area (m²)", 1e-8, 0.1, 2.5e-4, 1e-5, format="%.6f")
        efficiency = st.slider("Efficiency", 0.0, 1.0, 0.2, 0.01)
        background_cps = st.number_input("Background (cps)", 0.0, 1e6, 1.5, 0.1)
        dwell_s = st.number_input("Dwell time (s)", 0.001, 1e4, 2.0, 0.5)

    box = Box(Lx=float(Lx), Ly=float(Ly), Lz=float(Lz))
    sensor_positions_m = default_sensor_positions(box)
    sensor_labels = sensor_names(len(sensor_positions_m))
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

    current_sidebar_signature = (
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
    if "generated_measurement_df" not in st.session_state:
        st.session_state.generated_measurement_df = None
    if "generated_truth_qoi" not in st.session_state:
        st.session_state.generated_truth_qoi = None

    sidebar_changed_after_run = (
        st.session_state.last_run_signature is not None
        and current_sidebar_signature != st.session_state.last_run_signature
    )
    if sidebar_changed_after_run:
        st.session_state.cached_inputs_df = None
        st.session_state.cached_measurements_df = None

    design_tab, measurement_tab = st.tabs(["Sensor Placement Optimisation", "Generate measurement"])

    with design_tab:
        run_btn = st.button("Run simulation", key="run_design_simulation", use_container_width=True)
        if run_btn:
            try:
                inputs_df, measurements_df = run_design(
                    sensor_positions_m=sensor_positions_m,
                    sensor_names=sensor_labels,
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
            except Exception as exc:
                st.error(f"Simulation failed: {exc}")
            else:
                st.session_state.cached_inputs_df = inputs_df
                st.session_state.cached_measurements_df = measurements_df
                st.session_state.last_run_signature = current_sidebar_signature

        render_design_results(
            inputs_df=st.session_state.cached_inputs_df,
            measurements_df=st.session_state.cached_measurements_df,
            sensor_positions_m=sensor_positions_m,
            box=box,
            seed=int(seed),
            run_requested=run_btn,
        )

    with measurement_tab:
        render_generate_measurement_tab(
            sensor_positions_m=sensor_positions_m,
            bounds=bounds,
            box=box,
            detector=detector,
            mu_material_m_inv=float(mu_material_m_inv),
            fov_half_angle_deg=fov_half_angle_deg,
            distance_offset_m=float(distance_offset_m),
            noise=str(noise),
            seed=int(seed),
        )


if __name__ == "__main__":
    main()
