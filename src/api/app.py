from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from src.api.routes import router

WEB_DIR = Path(__file__).resolve().parents[2] / "web"


def create_app() -> FastAPI:
    application = FastAPI(title="CV to Job Guestimator")
    application.include_router(router)

    index_file = WEB_DIR / "index.html"
    if index_file.exists():

        @application.get("/", include_in_schema=False)
        def index() -> FileResponse:
            return FileResponse(index_file)

    return application


app = create_app()
