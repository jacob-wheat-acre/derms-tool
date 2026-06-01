"""app.py — IEEE Test Feeders: interactive distribution system analysis platform."""
from __future__ import annotations

import os
import pickle
import subprocess
import sys
import tempfile

import streamlit as st

from feeder_io import FEEDERS

_WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_dss_worker.py")


def _worker_run(args: list[str]) -> tuple[bool, str, str]:
    r = subprocess.run([sys.executable, _WORKER] + args, capture_output=True, text=True)
    return r.returncode == 0, r.stdout, r.stderr


def _load_pkl(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def _tmp_pkl() -> str:
    fd, path = tempfile.mkstemp(suffix=".pkl")
    os.close(fd)
    return path


# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="IEEE Test Feeders",
    layout="wide",
    page_icon="⚡",
)

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("IEEE Test Feeders")

    feeder_name = st.selectbox("IEEE Test Feeder", list(FEEDERS.keys()), key="feeder")
    feeder_cfg  = FEEDERS[feeder_name]
    st.caption(feeder_cfg["caption"])

    st.divider()

    module = st.radio(
        "Module",
        ["⚡ DERMS", "🔥 PSPS", "🔋 VVO"],
        key="module",
    )

    st.divider()

# ── Load feeder base case (cached per feeder) ──────────────────────────────────

cache_key = f"base_{feeder_name}"

if cache_key not in st.session_state:
    with st.spinner(f"Loading {feeder_name} via OpenDSS…"):
        _pkl  = _tmp_pkl()
        _args = ["base", feeder_cfg["master"], _pkl]
        _kv_min = feeder_cfg.get("der_kv_min")
        if _kv_min is not None:
            _args.append(str(_kv_min))
        _ok, _out, _err = _worker_run(_args)
        if not _ok:
            os.unlink(_pkl)
            st.error(f"DSS worker failed:\n{_err}")
            st.stop()
        st.session_state[cache_key] = _load_pkl(_pkl)
        os.unlink(_pkl)

base = st.session_state[cache_key]

# ── Module routing ─────────────────────────────────────────────────────────────

if module == "⚡ DERMS":
    from modules.derms.ui import render as render_derms
    render_derms(feeder_cfg, base, feeder_name, _worker_run, _load_pkl, _tmp_pkl)

elif module == "🔥 PSPS":
    from modules.psps.ui import render as render_psps
    render_psps(feeder_cfg, base, feeder_name)

elif module == "🔋 VVO":
    from modules.vvo.ui import render as render_vvo
    render_vvo(feeder_cfg, base, feeder_name, _worker_run, _load_pkl, _tmp_pkl)
