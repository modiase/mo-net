import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from loguru import logger
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from mnist_numpy import ROOT_DIR
from mnist_numpy.train.backends.models import DB_PATH, DbRun, Iteration


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
            logger.debug(f"Updated data from database (run {run_id})")

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

SERVER_DIR: Final[Path] = ROOT_DIR / "server"
templates = Jinja2Templates(directory=SERVER_DIR / "templates")
app.mount("/static", StaticFiles(directory=SERVER_DIR / "static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})


@app.get("/manage", response_class=HTMLResponse)
async def manage_dashboard(request: Request):
    return templates.TemplateResponse("manage.html", {"request": request})


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
async def get_latest_dashboard(request: Request):
    data, run_id, run = get_run_data()
    await state.mutate(
        current_data=data,
        current_run_id=run_id,
        current_run=run,
        last_row_count=len(data),
    )
    return templates.TemplateResponse("dashboard.html", {"request": request})


@app.get("/{run_id}")
async def get_run_dashboard(request: Request, run_id: int):
    data, _, run = get_run_data(run_id)
    await state.mutate(
        current_data=data,
        current_run_id=run_id,
        current_run=run,
        last_row_count=len(data),
    )
    return templates.TemplateResponse("dashboard.html", {"request": request})


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
