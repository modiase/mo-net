import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final, List, Optional

import click
import pandas as pd
import plotly.graph_objects as go
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from loguru import logger

from mnist_numpy.data import DATA_DIR


class State:
    @dataclass(frozen=True, kw_only=True)
    class S:
        current_data: Optional[pd.DataFrame] = None
        current_log_path: Optional[Path] = None
        current_seed: Optional[str] = None
        last_modified: float = 0.0
        background_task: Optional[asyncio.Task] = None

    def __init__(self):
        self._state = self.S()
        self._lock = asyncio.Lock()

    async def mutate(self, next_state: "State.S") -> None:
        async with self._lock:
            self._state = next_state

    async def get(self) -> "State.S":
        async with self._lock:
            return self._state


state: Final[State] = State()


def get_latest_training_log() -> Optional[Path]:
    run_dir = DATA_DIR / "run"
    if not run_dir.exists():
        return None
    training_log_files = list(run_dir.glob("*_training_log.csv"))
    return (
        max(training_log_files, key=lambda p: p.stat().st_mtime)
        if training_log_files
        else None
    )


def get_training_log_by_seed(seed: str) -> Optional[Path]:
    run_dir = DATA_DIR / "run"
    if not run_dir.exists():
        return None
    matching_files = list(run_dir.glob(f"{seed}_*_training_log.csv"))
    return matching_files[0] if matching_files else None


def get_all_available_seeds() -> List[str]:
    run_dir = DATA_DIR / "run"
    if not run_dir.exists():
        return []
    training_log_files = list(run_dir.glob("*_training_log.csv"))
    seeds = [
        file_path.name.split("_")[0]
        for file_path in training_log_files
        if "_" in file_path.name
    ]
    return sorted(seeds, reverse=True)


def load_training_data(log_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(log_path)
        if df.empty:
            return df
        if "monotonic_test_loss" not in df.columns:
            df["monotonic_test_loss"] = df["test_loss"].cummin()
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    except Exception as e:
        logger.error(f"Error loading training data: {e}")
        return pd.DataFrame()


async def update_data():
    while True:
        try:
            current_state = await state.get()

            if current_state.current_seed is None:
                log_path = get_latest_training_log()
                if log_path and log_path != current_state.current_log_path:
                    pass
            else:
                log_path = get_training_log_by_seed(current_state.current_seed)

            if log_path and log_path.exists():
                current_modified = log_path.stat().st_mtime

                if (
                    log_path == current_state.current_log_path
                    and current_modified > current_state.last_modified
                ):
                    await state.mutate(
                        State.S(
                            current_data=load_training_data(log_path),
                            current_log_path=log_path,
                            current_seed=current_state.current_seed,
                            last_modified=current_modified,
                            background_task=current_state.background_task,
                        )
                    )
                    seed_info = (
                        f" (seed {current_state.current_seed})"
                        if current_state.current_seed
                        else " (latest)"
                    )
                    logger.info(f"Updated data from {log_path}{seed_info}")
                elif (
                    current_state.current_seed is None
                    and log_path != current_state.current_log_path
                ):
                    await state.mutate(
                        State.S(
                            current_data=load_training_data(log_path),
                            current_log_path=log_path,
                            current_seed=None,
                            last_modified=log_path.stat().st_mtime,
                            background_task=current_state.background_task,
                        )
                    )
                    logger.info(f"Switched to new latest file: {log_path}")

            await asyncio.sleep(2)
        except Exception as e:
            logger.error(f"Error in background update: {e}")
            await asyncio.sleep(5)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting background data monitoring task")
    task = asyncio.create_task(update_data())
    current_state = await state.get()
    await state.mutate(
        State.S(
            current_data=current_state.current_data,
            current_log_path=current_state.current_log_path,
            current_seed=current_state.current_seed,
            last_modified=current_state.last_modified,
            background_task=task,
        )
    )

    yield

    logger.info("Shutting down background data monitoring task")
    current_state = await state.get()
    if current_state.background_task:
        current_state.background_task.cancel()
        try:
            await current_state.background_task
        except asyncio.CancelledError:
            pass


app = FastAPI(
    title="MNIST Training Monitor",
    description="Real-time visualization of MNIST training progress",
    lifespan=lifespan,
)


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return HTMLResponse(
        content="""
    <!DOCTYPE html>
    <html>
    <head>
        <title>MNIST Training Monitor</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .header {
                text-align: center;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                border-radius: 10px;
                margin-bottom: 30px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            .container {
                max-width: 1400px;
                margin: 0 auto;
            }
            .plot-container {
                background: white;
                border-radius: 10px;
                padding: 20px;
                margin-bottom: 30px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }
            .stat-card {
                background: white;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                border-left: 4px solid #667eea;
            }
            .stat-value {
                font-size: 24px;
                font-weight: bold;
                color: #333;
            }
            .stat-label {
                color: #666;
                font-size: 14px;
                margin-top: 5px;
            }
            .refresh-btn {
                position: fixed;
                top: 20px;
                right: 20px;
                background: #667eea;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                cursor: pointer;
                font-size: 14px;
            }
            .refresh-btn:hover {
                background: #5a6fd8;
            }
            .status {
                text-align: center;
                padding: 10px;
                border-radius: 5px;
                margin-bottom: 20px;
            }
            .status.connected { background-color: #d4edda; color: #155724; }
            .status.error { background-color: #f8d7da; color: #721c24; }
            .nav-bar {
                background: white;
                padding: 15px;
                border-radius: 10px;
                margin-bottom: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                display: flex;
                align-items: center;
                gap: 15px;
                flex-wrap: wrap;
            }
            .nav-item {
                font-size: 14px;
                color: #666;
            }
            .nav-item strong {
                color: #333;
            }
            .nav-links {
                display: flex;
                gap: 10px;
                margin-left: auto;
            }
            .nav-link {
                background: #667eea;
                color: white;
                text-decoration: none;
                padding: 8px 12px;
                border-radius: 5px;
                font-size: 12px;
                transition: background 0.3s;
            }
            .nav-link:hover {
                background: #5a6fd8;
            }
            .seed-selector {
                position: relative;
                display: inline-block;
            }
            .seed-dropdown {
                background: white;
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 5px 8px;
                font-size: 12px;
                cursor: pointer;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>MNIST Training Monitor</h1>
                <p>Real-time visualization of neural network training progress</p>
            </div>
            
            <button class="refresh-btn" onclick="refreshData()">Refresh Data</button>
            
            <div class="nav-bar">
                <div class="nav-item">
                    <strong>Current Log:</strong> <span id="current-log">Loading...</span>
                </div>
                <div class="nav-item">
                    <strong>Seed:</strong> <span id="current-seed">-</span>
                </div>
                <div class="nav-links">
                    <a href="/latest" class="nav-link">Latest</a>
                    <div class="seed-selector">
                        <select id="seed-selector" class="seed-dropdown" onchange="navigateToSeed()">
                            <option value="">Select Seed...</option>
                        </select>
                    </div>
                </div>
            </div>
            
            <div id="status" class="status"></div>
            
            <div class="stats-grid" id="stats-grid">
            </div>
            
            <div class="plot-container">
                <h3>Training Progress - Loss Over Epochs</h3>
                <div id="loss-plot" style="width:100%;height:500px;"></div>
            </div>
            
            <div class="plot-container">
                <h3>Learning Rate Schedule</h3>
                <div id="lr-plot" style="width:100%;height:400px;"></div>
            </div>
            
            <div class="plot-container">
                <h3>Training Timeline</h3>
                <div id="timeline-plot" style="width:100%;height:400px;"></div>
            </div>
        </div>

        <script>
            let refreshInterval;
            
            async function fetchData() {
                try {
                    const response = await fetch('/api/data');
                    return response.ok ? await response.json() : null;
                } catch (error) {
                    console.error('Error fetching data:', error);
                    return null;
                }
            }
            
            function updateStatus(message, type = 'connected') {
                const statusDiv = document.getElementById('status');
                statusDiv.textContent = message;
                statusDiv.className = `status ${type}`;
            }
            
            function updateStats(data) {
                if (!data || data.length === 0) return;
                const latest = data[data.length - 1];
                document.getElementById('stats-grid').innerHTML = `
                    <div class="stat-card">
                        <div class="stat-value">${latest.epoch}</div>
                        <div class="stat-label">Current Epoch</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">${latest.batch_loss.toFixed(6)}</div>
                        <div class="stat-label">Batch Loss</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">${latest.test_loss.toFixed(6)}</div>
                        <div class="stat-label">Test Loss</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">${latest.learning_rate.toFixed(8)}</div>
                        <div class="stat-label">Learning Rate</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">${data.length}</div>
                        <div class="stat-label">Total Epochs</div>
                    </div>
                `;
            }
            
            function plotLossChart(data) {
                if (!data || data.length === 0) return;
                Plotly.newPlot('loss-plot', [
                    {
                        x: data.map(d => d.epoch),
                        y: data.map(d => d.batch_loss),
                        type: 'scatter',
                        mode: 'lines+markers',
                        name: 'Batch Loss',
                        line: { color: '#1f77b4', width: 2 },
                        marker: { size: 4 }
                    },
                    {
                        x: data.map(d => d.epoch),
                        y: data.map(d => d.test_loss),
                        type: 'scatter',
                        mode: 'lines+markers',
                        name: 'Test Loss',
                        line: { color: '#ff7f0e', width: 2 },
                        marker: { size: 4 }
                    },
                    {
                        x: data.map(d => d.epoch),
                        y: data.map(d => d.monotonic_test_loss),
                        type: 'scatter',
                        mode: 'lines',
                        name: 'Monotonic Test Loss',
                        line: { color: '#2ca02c', width: 3, dash: 'dash' }
                    }
                ], {
                    xaxis: { title: 'Epoch' },
                    yaxis: { title: 'Loss', type: 'log' },
                    hovermode: 'x unified',
                    legend: { x: 0, y: 1 },
                    margin: { l: 60, r: 20, t: 20, b: 60 }
                }, {responsive: true});
            }
            
            function plotLearningRate(data) {
                if (!data || data.length === 0) return;
                Plotly.newPlot('lr-plot', [{
                    x: data.map(d => d.epoch),
                    y: data.map(d => d.learning_rate),
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'Learning Rate',
                    line: { color: '#d62728', width: 2 },
                    marker: { size: 4 }
                }], {
                    xaxis: { title: 'Epoch' },
                    yaxis: { title: 'Learning Rate' },
                    hovermode: 'x unified',
                    margin: { l: 60, r: 20, t: 20, b: 60 }
                }, {responsive: true});
            }
            
            function plotTimeline(data) {
                if (!data || data.length === 0 || !data[0].timestamp) return;
                Plotly.newPlot('timeline-plot', [{
                    x: data.map(d => d.timestamp),
                    y: data.map(d => d.test_loss),
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'Test Loss over Time',
                    line: { color: '#9467bd', width: 2 },
                    marker: { size: 4 }
                }], {
                    xaxis: { title: 'Time' },
                    yaxis: { title: 'Test Loss', type: 'log' },
                    hovermode: 'x unified',
                    margin: { l: 60, r: 20, t: 20, b: 60 }
                }, {responsive: true});
            }
            
            async function refreshData() {
                const data = await fetchData();
                if (data) {
                    updateStatus(`Data updated at ${new Date().toLocaleTimeString()}`, 'connected');
                    updateStats(data);
                    plotLossChart(data);
                    plotLearningRate(data);
                    plotTimeline(data);
                    await updateCurrentLogInfo();
                } else {
                    updateStatus('Error loading data', 'error');
                }
            }
            
            async function updateCurrentLogInfo() {
                try {
                    const response = await fetch('/api/status');
                    if (response.ok) {
                        const status = await response.json();
                        if (status.log_file) {
                            const filename = status.log_file.split('/').pop();
                            const seed = filename.split('_')[0];
                            document.getElementById('current-log').textContent = filename;
                            document.getElementById('current-seed').textContent = seed;
                        }
                    }
                } catch (error) {
                    console.error('Error fetching status:', error);
                }
            }
            
            async function loadAvailableSeeds() {
                try {
                    const response = await fetch('/api/seeds');
                    if (response.ok) {
                        const data = await response.json();
                        const selector = document.getElementById('seed-selector');
                        selector.innerHTML = '<option value="">Select Seed...</option>';
                        data.seeds.forEach(seed => {
                            const option = document.createElement('option');
                            option.value = seed;
                            option.textContent = seed;
                            selector.appendChild(option);
                        });
                    }
                } catch (error) {
                    console.error('Error loading seeds:', error);
                }
            }
            
            function navigateToSeed() {
                const seed = document.getElementById('seed-selector').value;
                if (seed) {
                    window.location.href = `/${seed}`;
                }
            }
            
            refreshData();
            loadAvailableSeeds();
            refreshInterval = setInterval(refreshData, 5000);
        </script>
    </body>
    </html>
    """
    )


@app.get("/api/data")
async def get_training_data():
    current_state = await state.get()
    data_to_use = current_state.current_data

    if data_to_use is None or data_to_use.empty:
        if current_state.current_seed is None:
            log_path = get_latest_training_log()
        else:
            log_path = get_training_log_by_seed(current_state.current_seed)

        if not log_path:
            raise HTTPException(status_code=404, detail="No training log files found")

        data_to_use = load_training_data(log_path)
        await state.mutate(
            State.S(
                current_data=data_to_use,
                current_log_path=log_path,
                current_seed=current_state.current_seed,
                last_modified=log_path.stat().st_mtime,
                background_task=current_state.background_task,
            )
        )

    if data_to_use.empty:
        raise HTTPException(status_code=404, detail="No training data available")

    data = data_to_use.to_dict("records")
    for row in data:
        if "timestamp" in row and pd.notna(row["timestamp"]):
            if isinstance(row["timestamp"], pd.Timestamp):
                row["timestamp"] = row["timestamp"].isoformat()

    return JSONResponse(content=data)


@app.get("/api/status")
async def get_status():
    current_state = await state.get()
    status = {
        "log_file": str(current_state.current_log_path)
        if current_state.current_log_path
        else None,
        "has_data": current_state.current_data is not None
        and not current_state.current_data.empty,
        "epochs": len(current_state.current_data)
        if current_state.current_data is not None
        else 0,
        "last_update": datetime.now().isoformat(),
    }

    if current_state.current_data is not None and not current_state.current_data.empty:
        latest = current_state.current_data.iloc[-1]
        status.update(
            {
                "current_epoch": int(latest["epoch"]),
                "current_batch_loss": float(latest["batch_loss"]),
                "current_test_loss": float(latest["test_loss"]),
                "current_learning_rate": float(latest["learning_rate"]),
            }
        )

    return JSONResponse(content=status)


@app.get("/api/plots/loss")
async def get_loss_plot():
    current_state = await state.get()
    if current_state.current_data is None or current_state.current_data.empty:
        raise HTTPException(status_code=404, detail="No data available")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=current_state.current_data["epoch"],
            y=current_state.current_data["batch_loss"],
            mode="lines+markers",
            name="Batch Loss",
            line=dict(color="blue", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=current_state.current_data["epoch"],
            y=current_state.current_data["test_loss"],
            mode="lines+markers",
            name="Test Loss",
            line=dict(color="red", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=current_state.current_data["epoch"],
            y=current_state.current_data["monotonic_test_loss"],
            mode="lines",
            name="Monotonic Test Loss",
            line=dict(color="green", width=3, dash="dash"),
        )
    )
    fig.update_layout(
        title="Training Progress - Loss Over Epochs",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        yaxis_type="log",
        hovermode="x unified",
    )
    return JSONResponse(content=fig.to_dict())


@app.get("/latest")
async def get_latest_dashboard():
    log_path = get_latest_training_log()
    if not log_path:
        raise HTTPException(status_code=404, detail="No training log files found")

    if "_" not in log_path.name:
        raise HTTPException(
            status_code=500, detail="Invalid training log filename format"
        )

    data = load_training_data(log_path)
    current_state = await state.get()
    await state.mutate(
        State.S(
            current_data=data,
            current_log_path=log_path,
            current_seed=None,
            last_modified=log_path.stat().st_mtime,
            background_task=current_state.background_task,
        )
    )
    return await dashboard()


@app.get("/api/latest/data")
async def get_latest_training_data():
    log_path = get_latest_training_log()
    if not log_path:
        raise HTTPException(status_code=404, detail="No training log files found")

    data = load_training_data(log_path)
    if data.empty:
        raise HTTPException(status_code=404, detail="No training data available")

    result = data.to_dict("records")
    for row in result:
        if "timestamp" in row and pd.notna(row["timestamp"]):
            if isinstance(row["timestamp"], pd.Timestamp):
                row["timestamp"] = row["timestamp"].isoformat()

    return JSONResponse(content=result)


@app.get("/{seed}")
async def get_seed_dashboard(seed: str):
    if not seed.isdigit():
        raise HTTPException(status_code=404, detail=f"Invalid seed format: {seed}")

    log_path = get_training_log_by_seed(seed)
    if not log_path:
        available_seeds = get_all_available_seeds()
        raise HTTPException(
            status_code=404,
            detail=f"Training log with seed {seed} not found. Available seeds: {available_seeds}",
        )

    data = load_training_data(log_path)
    current_state = await state.get()
    await state.mutate(
        State.S(
            current_data=data,
            current_log_path=log_path,
            current_seed=seed,
            last_modified=log_path.stat().st_mtime,
            background_task=current_state.background_task,
        )
    )
    return await dashboard()


@app.get("/api/{seed}/data")
async def get_seed_training_data(seed: str):
    if not seed.isdigit():
        raise HTTPException(status_code=404, detail=f"Invalid seed format: {seed}")

    log_path = get_training_log_by_seed(seed)
    if not log_path:
        available_seeds = get_all_available_seeds()
        raise HTTPException(
            status_code=404,
            detail=f"Training log with seed {seed} not found. Available seeds: {available_seeds}",
        )

    data = load_training_data(log_path)
    if data.empty:
        raise HTTPException(status_code=404, detail="No training data available")

    result = data.to_dict("records")
    for row in result:
        if "timestamp" in row and pd.notna(row["timestamp"]):
            if isinstance(row["timestamp"], pd.Timestamp):
                row["timestamp"] = row["timestamp"].isoformat()

    return JSONResponse(content=result)


@app.get("/api/seeds")
async def get_available_seeds():
    return JSONResponse(content={"seeds": get_all_available_seeds()})


@click.command()
@click.option("--host", default="127.0.0.1", help="Host to bind to")
@click.option("--port", default=8000, help="Port to bind to")
@click.option("--reload", is_flag=True, help="Enable auto-reload")
def main(host: str, port: int, reload: bool):
    logger.info(f"Starting MNIST Training Monitor on http://{host}:{port}")
    logger.info(f"Monitoring directory: {DATA_DIR / 'run'}")
    uvicorn.run(
        "mnist_numpy.scripts.serve_plot:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
