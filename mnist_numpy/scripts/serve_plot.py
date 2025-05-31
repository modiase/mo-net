import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import click
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from loguru import logger
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from mnist_numpy.train.server.models import DB_PATH, DbRun, Iteration


@dataclass(frozen=True, kw_only=True)
class State:
    current_data: Optional[pd.DataFrame] = None
    current_run_id: Optional[int] = None
    current_run: Optional[DbRun] = None
    last_row_count: int = 0
    background_task: Optional[asyncio.Task] = None


class StateManager:
    def __init__(self):
        self._state = State()
        self._lock = asyncio.Lock()

    async def mutate(self, **kwargs) -> None:
        async with self._lock:
            self._state = State(**{**self._state.__dict__, **kwargs})

    async def get(self) -> State:
        async with self._lock:
            return self._state


state = StateManager()


def get_session():
    if not DB_PATH.exists():
        raise HTTPException(status_code=503, detail="Database not available")
    return sessionmaker(bind=create_engine(f"sqlite:///{DB_PATH}"))()


def get_run_data(run_id: int | None = None) -> tuple[pd.DataFrame, int, DbRun]:
    session = get_session()
    try:
        run = (
            session.query(DbRun).filter(DbRun.id == run_id).first()
            if run_id
            else session.query(DbRun).order_by(DbRun.updated_at.desc()).first()
        )
        if not run:
            raise HTTPException(status_code=404, detail="No runs found")

        iterations = (
            session.query(Iteration)
            .filter(Iteration.run_id == run.id)
            .order_by(Iteration.timestamp)
            .all()
        )
        if not iterations:
            raise HTTPException(status_code=404, detail="No iterations found")

        data = pd.DataFrame(
            [
                {
                    **{
                        k: getattr(it, k)
                        for k in [
                            "batch_loss",
                            "val_loss",
                            "batch",
                            "epoch",
                            "learning_rate",
                            "timestamp",
                        ]
                    }
                }
                for it in iterations
            ]
        )
        data["monotonic_val_loss"] = data["val_loss"].cummin()
        return data, run.id, run
    finally:
        session.close()


async def update_data():
    while True:
        try:
            current_state = await state.get()
            data, run_id, run = get_run_data(current_state.current_run_id)

            if len(data) > current_state.last_row_count:
                await state.mutate(
                    current_data=data,
                    current_run_id=run_id,
                    current_run=run,
                    last_row_count=len(data),
                )
                logger.info(f"Updated data from database (run {run_id})")

            await asyncio.sleep(2)
        except Exception as e:
            logger.error(f"Error in background update: {e}")
            await asyncio.sleep(5)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting background data monitoring task")
    task = asyncio.create_task(update_data())
    await state.mutate(background_task=task)
    yield
    logger.info("Shutting down background data monitoring task")
    task.cancel()


app = FastAPI(
    title="MNIST Training Monitor",
    description="Real-time visualisation of MNIST training progress",
    lifespan=lifespan,
)


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return HTMLResponse(
        content="""<!DOCTYPE html><html><head><title>MNIST Training Monitor</title><script src="https://cdn.plot.ly/plotly-latest.min.js"></script><style>body{font-family:'Segoe UI',Tahoma,Geneva,Verdana,sans-serif;margin:0;padding:20px;background-color:#f5f5f5}.header{text-align:center;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:white;padding:30px;border-radius:10px;margin-bottom:30px;box-shadow:0 4px 6px rgba(0,0,0,0.1)}.container{max-width:1400px;margin:0 auto}.plot-container{background:white;border-radius:10px;padding:20px;margin-bottom:30px;box-shadow:0 2px 4px rgba(0,0,0,0.1)}.stats-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:20px;margin-bottom:30px}.stat-card{background:white;padding:20px;border-radius:10px;text-align:center;box-shadow:0 2px 4px rgba(0,0,0,0.1);border-left:4px solid #667eea}.stat-card.completed{border-left:4px solid #28a745}.stat-card.running{border-left:4px solid #ffc107}.stat-value{font-size:24px;font-weight:bold;color:#333}.stat-label{color:#666;font-size:14px;margin-top:5px}.refresh-btn{position:fixed;top:20px;right:20px;background:#667eea;color:white;border:none;padding:10px 20px;border-radius:5px;cursor:pointer;font-size:14px}.refresh-btn:hover{background:#5a6fd8}.status{text-align:center;padding:10px;border-radius:5px;margin-bottom:20px}.status.connected{background-color:#d4edda;color:#155724}.status.error{background-color:#f8d7da;color:#721c24}.nav-bar{background:white;padding:15px;border-radius:10px;margin-bottom:20px;box-shadow:0 2px 4px rgba(0,0,0,0.1);display:flex;align-items:center;gap:15px;flex-wrap:wrap}.nav-item{font-size:14px;color:#666}.nav-item strong{color:#333}.nav-links{display:flex;gap:10px;margin-left:auto}.nav-link{background:#667eea;color:white;text-decoration:none;padding:8px 12px;border-radius:5px;font-size:12px;transition:background 0.3s}.nav-link:hover{background:#5a6fd8}.run-dropdown{background:white;border:1px solid #ddd;border-radius:5px;padding:5px 8px;font-size:12px;cursor:pointer}.progress-bar{width:100%;height:20px;background-color:#e0e0e0;border-radius:10px;overflow:hidden;margin-top:10px}.progress-fill{height:100%;background:linear-gradient(90deg,#667eea 0%,#764ba2 100%);transition:width 0.3s ease}.status-badge{padding:4px 8px;border-radius:4px;font-size:12px;font-weight:bold}.status-running{background-color:#fff3cd;color:#856404}.status-completed{background-color:#d4edda;color:#155724}</style></head><body><div class="container"><div class="header"><h1>MNIST Training Monitor</h1><p>Real-time visualisation of neural network training progress</p></div><button class="refresh-btn" onclick="refreshData()">Refresh Data</button><div class="nav-bar"><div class="nav-item"><strong>Current Run:</strong> <span id="current-run">Loading...</span></div><div class="nav-links"><a href="/latest" class="nav-link">Latest</a><a href="/manage" class="nav-link">Manage</a><select id="run-selector" class="run-dropdown" onchange="navigateToRun()"><option value="">Select Run...</option></select></div></div><div id="status" class="status"></div><div class="stats-grid" id="stats-grid"></div><div class="plot-container"><h3>Training Progress - Loss Over Epochs</h3><div id="loss-plot" style="width:100%;height:500px;"></div></div><div class="plot-container"><h3>Learning Rate Schedule</h3><div id="lr-plot" style="width:100%;height:400px;"></div></div><div class="plot-container"><h3>Training Timeline</h3><div id="timeline-plot" style="width:100%;height:400px;"></div></div></div><script>async function fetchData(){try{const response=await fetch('/api/data');return response.ok?await response.json():null}catch(error){console.error('Error fetching data:',error);return null}}function updateStatus(message,type='connected'){const statusDiv=document.getElementById('status');statusDiv.textContent=message;statusDiv.className=`status ${type}`}function updateStats(data){if(!data||data.length===0)return;const latest=data[data.length-1];fetch('/api/status').then(res=>res.json()).then(status=>{const progress=status.progress||0;const totalEpochs=status.total_epochs||'N/A';const isCompleted=status.is_completed||false;const runStatus=isCompleted?'Completed':'Running';const statusClass=isCompleted?'completed':'running';const statusBadge=isCompleted?'<span class="status-badge status-completed">Completed</span>':'<span class="status-badge status-running">Running</span>';document.getElementById('stats-grid').innerHTML=`<div class="stat-card"><div class="stat-value">${latest.epoch}</div><div class="stat-label">Current Epoch</div></div><div class="stat-card"><div class="stat-value">${latest.batch_loss.toFixed(6)}</div><div class="stat-label">Batch Loss</div></div><div class="stat-card"><div class="stat-value">${latest.val_loss.toFixed(6)}</div><div class="stat-label">Validation Loss</div></div><div class="stat-card"><div class="stat-value">${latest.learning_rate.toFixed(8)}</div><div class="stat-label">Learning Rate</div></div><div class="stat-card"><div class="stat-value">${totalEpochs}</div><div class="stat-label">Total Epochs</div></div><div class="stat-card ${statusClass}"><div class="stat-value">${statusBadge}</div><div class="stat-label">Run Status</div></div><div class="stat-card"><div class="stat-value">${(progress*100).toFixed(1)}%</div><div class="stat-label">Progress</div><div class="progress-bar"><div class="progress-fill" style="width:${progress*100}%"></div></div></div>`})}function plotLossChart(data){if(!data||data.length===0)return;Plotly.newPlot('loss-plot',[{x:data.map(d=>d.epoch),y:data.map(d=>d.batch_loss),type:'scatter',mode:'lines+markers',name:'Batch Loss',line:{color:'#1f77b4',width:2},marker:{size:4}},{x:data.map(d=>d.epoch),y:data.map(d=>d.val_loss),type:'scatter',mode:'lines+markers',name:'Validation Loss',line:{color:'#ff7f0e',width:2},marker:{size:4}},{x:data.map(d=>d.epoch),y:data.map(d=>d.monotonic_val_loss),type:'scatter',mode:'lines',name:'Monotonic Validation Loss',line:{color:'#2ca02c',width:3,dash:'dash'}}],{xaxis:{title:'Epoch'},yaxis:{title:'Loss',type:'log'},hovermode:'x unified',legend:{x:0,y:1},margin:{l:60,r:20,t:20,b:60}},{responsive:true})}function plotLearningRate(data){if(!data||data.length===0)return;Plotly.newPlot('lr-plot',[{x:data.map(d=>d.epoch),y:data.map(d=>d.learning_rate),type:'scatter',mode:'lines+markers',name:'Learning Rate',line:{color:'#d62728',width:2},marker:{size:4}}],{xaxis:{title:'Epoch'},yaxis:{title:'Learning Rate'},hovermode:'x unified',margin:{l:60,r:20,t:20,b:60}},{responsive:true})}function plotTimeline(data){if(!data||data.length===0||!data[0].timestamp)return;Plotly.newPlot('timeline-plot',[{x:data.map(d=>d.timestamp),y:data.map(d=>d.val_loss),type:'scatter',mode:'lines+markers',name:'Validation Loss over Time',line:{color:'#9467bd',width:2},marker:{size:4}}],{xaxis:{title:'Time'},yaxis:{title:'Validation Loss',type:'log'},hovermode:'x unified',margin:{l:60,r:20,t:20,b:60}},{responsive:true})}async function refreshData(){const data=await fetchData();if(data){updateStatus(`Data updated at ${new Date().toLocaleTimeString()}`,'connected');updateStats(data);plotLossChart(data);plotLearningRate(data);plotTimeline(data);await updateCurrentInfo()}else{updateStatus('Error loading data','error')}}async function updateCurrentInfo(){try{const response=await fetch('/api/status');if(response.ok){const status=await response.json();document.getElementById('current-run').textContent=status.current_run_id||'None'}}catch(error){console.error('Error fetching status:',error)}}async function loadAvailableRuns(){try{const response=await fetch('/api/runs');if(response.ok){const data=await response.json();const selector=document.getElementById('run-selector');selector.innerHTML='<option value="">Select Run...</option>';data.runs.forEach(run=>{const option=document.createElement('option');option.value=run.id;option.textContent=`Run ${run.id} (Seed ${run.seed})`;selector.appendChild(option)})}}catch(error){console.error('Error loading runs:',error)}}function navigateToRun(){const runId=document.getElementById('run-selector').value;if(runId)window.location.href=`/${runId}`}refreshData();loadAvailableRuns();setInterval(refreshData,5000)</script></body></html>"""
    )


@app.get("/manage", response_class=HTMLResponse)
async def manage_dashboard():
    return HTMLResponse(
        content="""<!DOCTYPE html><html><head><title>Run Management - MNIST Training Monitor</title><style>body{font-family:'Segoe UI',Tahoma,Geneva,Verdana,sans-serif;margin:0;padding:20px;background-color:#f5f5f5}.header{text-align:center;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:white;padding:30px;border-radius:10px;margin-bottom:30px;box-shadow:0 4px 6px rgba(0,0,0,0.1)}.container{max-width:1200px;margin:0 auto}.nav-bar{background:white;padding:15px;border-radius:10px;margin-bottom:20px;box-shadow:0 2px 4px rgba(0,0,0,0.1);display:flex;align-items:center;gap:15px}.nav-link{background:#667eea;color:white;text-decoration:none;padding:8px 12px;border-radius:5px;font-size:12px;transition:background 0.3s}.nav-link:hover{background:#5a6fd8}.table-container{background:white;border-radius:10px;padding:20px;box-shadow:0 2px 4px rgba(0,0,0,0.1)}.table{width:100%;border-collapse:collapse;margin-top:20px}.table th,.table td{padding:12px;text-align:left;border-bottom:1px solid #ddd}.table th{background-color:#f8f9fa;font-weight:600}.status-badge{padding:4px 8px;border-radius:4px;font-size:12px;font-weight:bold}.status-running{background-color:#fff3cd;color:#856404}.status-completed{background-color:#d4edda;color:#155724}.btn{padding:6px 12px;border:none;border-radius:4px;cursor:pointer;font-size:12px;transition:background 0.3s}.btn-complete{background-color:#28a745;color:white}.btn-complete:hover{background-color:#218838}.btn-complete:disabled{background-color:#6c757d;cursor:not-allowed}.btn-view{background-color:#007bff;color:white;margin-right:5px}.btn-view:hover{background-color:#0056b3}.status{text-align:center;padding:10px;border-radius:5px;margin-bottom:20px}.status.success{background-color:#d4edda;color:#155724}.status.error{background-color:#f8d7da;color:#721c24}</style></head><body><div class="container"><div class="header"><h1>Run Management</h1><p>Manage training runs and their completion status</p></div><div class="nav-bar"><a href="/" class="nav-link">← Back to Dashboard</a><a href="/latest" class="nav-link">Latest</a><button class="nav-link" onclick="refreshRuns()" style="border:none;background:#667eea;color:white;cursor:pointer">Refresh</button></div><div id="status" class="status" style="display:none"></div><div class="table-container"><h3>All Training Runs</h3><table class="table"><thead><tr><th>Run ID</th><th>Seed</th><th>Started</th><th>Last Updated</th><th>Current Epoch</th><th>Total Epochs</th><th>Status</th><th>Actions</th></tr></thead><tbody id="runs-table-body"></tbody></table></div></div><script>function showStatus(message,type='success'){const statusDiv=document.getElementById('status');statusDiv.textContent=message;statusDiv.className=`status ${type}`;statusDiv.style.display='block';setTimeout(()=>statusDiv.style.display='none',3000)}async function fetchRuns(){try{const response=await fetch('/api/runs/all');return response.ok?await response.json():null}catch(error){console.error('Error fetching runs:',error);return null}}async function completeRun(runId){if(!confirm(`Are you sure you want to mark run ${runId} as completed?`))return;try{const response=await fetch(`/api/runs/${runId}/complete`,{method:'POST'});if(response.ok){showStatus(`Run ${runId} marked as completed`,'success');refreshRuns()}else{showStatus('Failed to update run status','error')}}catch(error){console.error('Error completing run:',error);showStatus('Error updating run status','error')}}function formatDate(dateString){return new Date(dateString).toLocaleString()}function updateRunsTable(runs){const tbody=document.getElementById('runs-table-body');tbody.innerHTML=runs.map(run=>{const isCompleted=run.completed_at!==null;const statusBadge=isCompleted?'<span class="status-badge status-completed">Completed</span>':'<span class="status-badge status-running">Running</span>';const completeButton=isCompleted?'<button class="btn btn-complete" disabled>Already Completed</button>':`<button class="btn btn-complete" onclick="completeRun(${run.id})">Mark Complete</button>`;return`<tr><td>${run.id}</td><td>${run.seed}</td><td>${formatDate(run.started_at)}</td><td>${formatDate(run.updated_at)}</td><td>${run.current_epoch}</td><td>${run.total_epochs}</td><td>${statusBadge}</td><td><a href="/${run.id}" class="btn btn-view">View</a>${completeButton}</td></tr>`}).join('')}async function refreshRuns(){const runs=await fetchRuns();if(runs){updateRunsTable(runs.runs);showStatus('Runs refreshed successfully','success')}else{showStatus('Failed to load runs','error')}}refreshRuns()</script></body></html>"""
    )


@app.get("/api/data")
async def get_training_data():
    current_state = await state.get()
    if current_state.current_data is None or current_state.current_data.empty:
        data, run_id, run = get_run_data(current_state.current_run_id)
        await state.mutate(
            current_data=data,
            current_run_id=run_id,
            current_run=run,
            last_row_count=len(data),
        )
        current_state = await state.get()

    data = current_state.current_data.to_dict("records")
    for row in data:
        if (
            "timestamp" in row
            and pd.notna(row["timestamp"])
            and isinstance(row["timestamp"], pd.Timestamp)
        ):
            row["timestamp"] = row["timestamp"].isoformat()
    return JSONResponse(content=data)


@app.get("/api/status")
async def get_status():
    current_state = await state.get()
    status = {
        "current_run_id": current_state.current_run_id,
        "has_data": current_state.current_data is not None
        and not current_state.current_data.empty,
        "epochs": len(current_state.current_data)
        if current_state.current_data is not None
        else 0,
        "last_update": datetime.now().isoformat(),
    }

    if current_state.current_run:
        progress = (
            current_state.current_run.current_epoch
            / current_state.current_run.total_epochs
            if current_state.current_run.total_epochs > 0
            else 0
        )
        status.update(
            {
                "progress": progress,
                "total_epochs": current_state.current_run.total_epochs,
                "is_completed": current_state.current_run.completed_at is not None,
            }
        )

    if current_state.current_data is not None and not current_state.current_data.empty:
        latest = current_state.current_data.iloc[-1]
        status.update(
            {
                "current_epoch": int(latest["epoch"]),
                "current_batch_loss": float(latest["batch_loss"]),
                "current_val_loss": float(latest["val_loss"]),
                "current_learning_rate": float(latest["learning_rate"]),
            }
        )
    return JSONResponse(content=status)


@app.get("/latest")
async def get_latest_dashboard():
    data, run_id, run = get_run_data()
    await state.mutate(
        current_data=data,
        current_run_id=run_id,
        current_run=run,
        last_row_count=len(data),
    )
    return await dashboard()


@app.get("/{run_id}")
async def get_run_dashboard(run_id: int):
    data, _, run = get_run_data(run_id)
    await state.mutate(
        current_data=data,
        current_run_id=run_id,
        current_run=run,
        last_row_count=len(data),
    )
    return await dashboard()


@app.get("/api/runs")
async def get_available_runs():
    session = get_session()
    try:
        return JSONResponse(
            content={
                "runs": [
                    {
                        "id": run.id,
                        "seed": run.seed,
                        "updated_at": run.updated_at.isoformat(),
                    }
                    for run in session.query(DbRun)
                    .order_by(DbRun.updated_at.desc())
                    .all()
                ]
            }
        )
    finally:
        session.close()


@app.get("/api/runs/all")
async def get_all_runs():
    session = get_session()
    try:
        return JSONResponse(
            content={
                "runs": [
                    {
                        "id": run.id,
                        "seed": run.seed,
                        "started_at": run.started_at.isoformat(),
                        "updated_at": run.updated_at.isoformat(),
                        "completed_at": run.completed_at.isoformat()
                        if run.completed_at
                        else None,
                        "current_epoch": run.current_epoch,
                        "total_epochs": run.total_epochs,
                        "is_completed": run.completed_at is not None,
                    }
                    for run in session.query(DbRun)
                    .order_by(DbRun.updated_at.desc())
                    .all()
                ]
            }
        )
    finally:
        session.close()


@app.post("/api/runs/{run_id}/complete")
async def complete_run(run_id: int):
    session = get_session()
    try:
        run = session.query(DbRun).filter(DbRun.id == run_id).first()
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")

        if run.completed_at is not None:
            raise HTTPException(status_code=400, detail="Run already completed")

        run.completed_at = datetime.now()
        session.commit()

        return JSONResponse(
            content={
                "message": f"Run {run_id} marked as completed",
                "completed_at": run.completed_at.isoformat(),
            }
        )
    finally:
        session.close()


@click.command()
@click.option("--host", default="127.0.0.1", help="Host to bind to")
@click.option("--port", default=8000, help="Port to bind to")
@click.option("--reload", is_flag=True, help="Enable auto-reload")
def main(host: str, port: int, reload: bool):
    logger.info(
        f"Starting MNIST Training Monitor on http://{host}:{port} using database: {DB_PATH}"
    )
    uvicorn.run(
        "mnist_numpy.scripts.serve_plot:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
