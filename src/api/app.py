import os

from fastapi import FastAPI

from src.api.routes import router, upload_router


def cv_upload_enabled() -> bool:
    """Whether the endpoints that accept a `candidate_cv` upload are mounted.

    Off unless ALLOW_CV_UPLOAD is explicitly one of the values below. The
    default is the deployed posture, not the convenient one: a public site
    that accepts strangers' CVs would be redacting, storing and sending
    other people's personal data, which this project has no reason to hold
    and no way to be trusted with. Opting in is a local, deliberate act.

    Note the default direction. A missing, empty, misspelled or malformed
    variable leaves uploads OFF, so the failure mode of getting this wrong
    is a locked-down site, never an open one.
    """
    return os.getenv("ALLOW_CV_UPLOAD", "").strip().lower() in {"1", "true", "yes", "on"}


def create_app() -> FastAPI:
    """The API, and only the API.

    This used to serve a static web/index.html at "/" as well. That UI is
    gone: the frontend now lives in the separate portfolio-sonny project,
    which reaches these endpoints from its own server rather than from the
    visitor's browser. So "/" 404s by design, and nothing here renders
    anything -- the one route a deployment needs is /api/match.
    """
    application = FastAPI(title="CV to Job Guestimator")
    application.include_router(router)
    if cv_upload_enabled():
        application.include_router(upload_router)

    return application


app = create_app()
