import click
import uvicorn
from loguru import logger

from mo_net.db import DB_PATH
from mo_net.log import LogLevel, setup_logging


@click.command()
@click.option("--host", default="127.0.0.1", help="Host to bind to")
@click.option("--port", default=8000, help="Port to bind to")
@click.option("--reload", is_flag=True, help="Enable auto-reload")
def main(host: str, port: int, reload: bool):
    setup_logging(LogLevel.INFO)
    logger.info(
        f"Starting Training Monitor on http://{host}:{port} using database: {DB_PATH}"
    )
    uvicorn.run(
        "mo_net.server.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
