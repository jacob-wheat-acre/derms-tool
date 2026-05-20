# DERMS Explorer

Interactive distribution feeder analysis using OpenDSS. Visualize voltage profiles, model DER (solar PV) impact, and run hosting capacity sweeps across IEEE test feeders.

**Included feeders:** IEEE 123-Bus · IEEE 13-Bus · IEEE 9500-Node

---

## Prerequisites

- **Python 3.10 or newer** — [python.org/downloads](https://www.python.org/downloads/)
- **Git** — [git-scm.com](https://git-scm.com/)

Verify your Python version:
```
python --version
```

---

## Setup

### Mac

```bash
# 1. Clone the repo
git clone https://github.com/jacob-wheat-acre/derms-tool.git
cd derms-tool

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

### Windows (Command Prompt or PowerShell)

```bat
:: 1. Clone the repo
git clone https://github.com/jacob-wheat-acre/derms-tool.git
cd derms-tool

:: 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate

:: 3. Install dependencies
pip install -r requirements.txt

:: 4. Run the app
streamlit run app.py
```

The app opens automatically in your browser at `http://localhost:8501`.

---

## Running after initial setup

Activate the virtual environment first, then run:

**Mac:**
```bash
source .venv/bin/activate
streamlit run app.py
```

**Windows:**
```bat
.venv\Scripts\activate
streamlit run app.py
```

---

## Troubleshooting

**`python` not found on Mac** — try `python3` instead.

**`opendssdirect` install fails on Windows** — make sure you have the [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) installed (required for native extensions).

**Port 8501 already in use** — run on a different port:
```
streamlit run app.py --server.port 8502
```

**Feeder loads slowly** — the IEEE 9500-Node feeder takes 15–30 seconds on first load. Subsequent feeder switches within the same session are cached.
