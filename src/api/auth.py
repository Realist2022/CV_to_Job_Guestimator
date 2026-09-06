"""Bearer-token check for the public API.

The service is reachable from the internet once deployed, and every request
it accepts spends GPU time on Modal. Without this, anyone who finds the URL
can run up that bill directly -- the per-IP rate limiting lives in the
portfolio's Next.js proxy, so calling this service straight past it has no
limit at all.

The portfolio already sends `Authorization: Bearer <key>` on every upstream
call (see its src/lib/cvGuestimator.ts authHeaders); this is the other half
of that handshake, which until now was never checked.
"""

import os
import secrets
import warnings

from fastapi import Header, HTTPException

API_KEY_ENV = "CV_GUESTIMATOR_API_KEY"


def configured_api_key() -> str:
    """The expected bearer token, or "" when the API is left unauthenticated."""
    return os.getenv(API_KEY_ENV, "").strip()


def warn_if_unauthenticated() -> None:
    """Say so, loudly, when the service starts with no key configured.

    Enforcement is opt-in because requiring a key would break every local
    run and the whole test suite. That makes "forgot to set it" the
    dangerous case, so it is at least never silent: this prints at startup,
    and a deployment should attach the secret so the branch is never taken.
    """
    if not configured_api_key():
        warnings.warn(
            f"{API_KEY_ENV} is not set: /api/* is unauthenticated and anyone "
            "who can reach this service can spend its model budget. Set it "
            "for any deployment reachable from outside localhost.",
            stacklevel=2,
        )


def require_api_key(authorization: str | None = Header(default=None)) -> None:
    """FastAPI dependency: reject a request without the configured token.

    A no-op when no key is configured, so local development and the tests
    keep working unchanged.

    Compared with `secrets.compare_digest` rather than `==`: token checks
    that short-circuit on the first wrong byte leak the token's prefix to
    anyone who can time the responses.
    """
    expected = configured_api_key()
    if not expected:
        return

    scheme, _, token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not token:
        # 401 with the challenge header: the caller sent no usable
        # credentials, as opposed to the wrong ones.
        raise HTTPException(
            status_code=401,
            detail="Missing bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not secrets.compare_digest(token, expected):
        # Deliberately 403, not 401: a 401 invites the caller to retry with
        # different credentials, and there is only one valid token here.
        raise HTTPException(status_code=403, detail="Invalid bearer token.")
