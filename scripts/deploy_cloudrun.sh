#!/usr/bin/env bash
#
# Deploy the API to Google Cloud Run.
#
# WHAT RUNS WHERE. Cloud Run runs the FastAPI app only. It has no GPU and no
# Ollama beside it, so the model must come from Modal -- which means
# `modal deploy modal_app/inference.py` is a PREREQUISITE, not a follow-up.
# Without it every request falls back to gemini-flash (see
# configs/pipeline.yaml fallback_models) and silently stops using the
# fine-tuned model, which is the one thing this service exists to show off.
#
# COSTS MONEY. This creates a project, a bucket, secrets and a Cloud Run
# service. Cloud Run scales to zero so idle cost is ~nothing, but each
# request wakes an L4 on Modal. --max-instances caps the blast radius.
#
# Re-running is safe: every create step tolerates "already exists".

set -euo pipefail

# ── settings ────────────────────────────────────────────────────────────────
PROJECT_ID="${PROJECT_ID:-cv-guestimator}"
REGION="${REGION:-australia-southeast1}"   # Sydney: closest region to NZ
SERVICE="${SERVICE:-cv-guestimator-api}"
BUCKET="${BUCKET:-${PROJECT_ID}-artifacts}"
RUNTIME_SA="cv-guestimator-run"
SA_EMAIL="${RUNTIME_SA}@${PROJECT_ID}.iam.gserviceaccount.com"

# Read from your local .env; each becomes a Secret Manager secret.
ENV_FILE="${ENV_FILE:-.env}"

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

# ── preflight ───────────────────────────────────────────────────────────────
say "Preflight"

[ -f "serving/redacted_cv.json" ] || {
  echo "ERROR: serving/redacted_cv.json is missing. The service has no CV to" >&2
  echo "       match against. Build it: python scripts/build_serving_cv.py <cv_id>" >&2
  exit 1
}

[ -f ".gcloudignore" ] || {
  echo "ERROR: .gcloudignore is missing. Without it, 'gcloud run deploy --source'" >&2
  echo "       may upload dataSet/ and redacted_cvs/ -- real CVs and detected PII" >&2
  echo "       -- into Cloud Build. Refusing to deploy." >&2
  exit 1
}

# shellcheck disable=SC1090
read_env() { grep -E "^$1=" "$ENV_FILE" | head -1 | cut -d= -f2- | tr -d '"'"'"'\r'; }

for var in MODAL_INFERENCE_URL MODAL_API_KEY GOOGLE_API_KEY; do
  [ -n "$(read_env "$var")" ] || { echo "ERROR: $var not found in $ENV_FILE" >&2; exit 1; }
done

# The bearer token the portfolio must send. Generated once, then stored in
# Secret Manager here and in the portfolio's CV_GUESTIMATOR_API_KEY there.
API_KEY="$(read_env CV_GUESTIMATOR_API_KEY)"
if [ -z "$API_KEY" ]; then
  API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
  echo "Generated a new CV_GUESTIMATOR_API_KEY. Add this to BOTH .env files:"
  echo "  CV_GUESTIMATOR_API_KEY=${API_KEY}"
fi

# ── project ─────────────────────────────────────────────────────────────────
say "Project: ${PROJECT_ID}"
gcloud projects describe "$PROJECT_ID" >/dev/null 2>&1 \
  || gcloud projects create "$PROJECT_ID" --name="CV to Job Guestimator"

gcloud config set project "$PROJECT_ID" >/dev/null

if ! gcloud billing projects describe "$PROJECT_ID" \
     --format='value(billingEnabled)' 2>/dev/null | grep -q True; then
  echo "ERROR: billing is not enabled on ${PROJECT_ID}." >&2
  echo "  gcloud billing accounts list" >&2
  echo "  gcloud billing projects link ${PROJECT_ID} --billing-account=XXXXXX" >&2
  exit 1
fi

say "Enabling APIs (slow the first time)"
gcloud services enable \
  run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com \
  secretmanager.googleapis.com storage.googleapis.com

# ── artifact bucket ─────────────────────────────────────────────────────────
say "Bucket: gs://${BUCKET}"
gcloud storage buckets describe "gs://${BUCKET}" >/dev/null 2>&1 \
  || gcloud storage buckets create "gs://${BUCKET}" \
       --location="$REGION" --uniform-bucket-level-access

# Run artifacts embed extracted CV and job-listing text, so the bucket must
# never be public. Uniform access above plus no allUsers binding keeps it
# private; this asserts it rather than assuming.
if gcloud storage buckets get-iam-policy "gs://${BUCKET}" --format=json \
   | grep -q '"allUsers"'; then
  echo "ERROR: gs://${BUCKET} is publicly readable. Run artifacts contain" >&2
  echo "       extracted CV content. Remove the allUsers binding first." >&2
  exit 1
fi

# ── runtime identity ────────────────────────────────────────────────────────
say "Service account: ${SA_EMAIL}"
gcloud iam service-accounts describe "$SA_EMAIL" >/dev/null 2>&1 \
  || gcloud iam service-accounts create "$RUNTIME_SA" \
       --display-name="CV Guestimator Cloud Run runtime"

# objectAdmin, not admin: the service creates and reads artifact objects and
# the run counter. It has no business reconfiguring or deleting the bucket.
gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" \
  --member="serviceAccount:${SA_EMAIL}" --role=roles/storage.objectAdmin >/dev/null

# ── secrets ─────────────────────────────────────────────────────────────────
say "Secrets"
put_secret() {
  local name="$1" value="$2"
  gcloud secrets describe "$name" >/dev/null 2>&1 \
    || gcloud secrets create "$name" --replication-policy=automatic >/dev/null
  printf '%s' "$value" | gcloud secrets versions add "$name" --data-file=- >/dev/null
  gcloud secrets add-iam-policy-binding "$name" \
    --member="serviceAccount:${SA_EMAIL}" \
    --role=roles/secretmanager.secretAccessor >/dev/null
  echo "  ${name}"
}

put_secret cv-guestimator-api-key   "$API_KEY"
put_secret modal-inference-url      "$(read_env MODAL_INFERENCE_URL)"
put_secret modal-api-key            "$(read_env MODAL_API_KEY)"
put_secret google-api-key           "$(read_env GOOGLE_API_KEY)"

# ── deploy ──────────────────────────────────────────────────────────────────
say "Building the image"
# Built via cloudbuild.yaml rather than `gcloud run deploy --source .`,
# because that flag only auto-detects a Dockerfile at the repository root.
# This project's lives at docker/Dockerfile.api, and --source would quietly
# fall back to buildpacks and ship something other than what compose builds.
AR_REPO="${REGION}-docker.pkg.dev/${PROJECT_ID}/cv-guestimator"
IMAGE="${AR_REPO}/api:$(date +%Y%m%d-%H%M%S)"

gcloud artifacts repositories describe cv-guestimator --location="$REGION" >/dev/null 2>&1 \
  || gcloud artifacts repositories create cv-guestimator \
       --repository-format=docker --location="$REGION" \
       --description="CV Guestimator API images"

gcloud builds submit --config cloudbuild.yaml \
  --substitutions=_IMAGE="$IMAGE" --region="$REGION"

say "Deploying ${SERVICE} to ${REGION}"
#
# --allow-unauthenticated is deliberate: the gate is the bearer token checked
#   in src/api/auth.py, so the portfolio (on Vercel, outside GCP) can call
#   this without a service-account key. Cloud Run IAM would mean managing one.
# --timeout 900 rides out a Modal cold start, which is minutes. The portfolio
#   gives up at 180s first, so this is headroom rather than the real ceiling.
# --max-instances caps how much GPU spend a bad day can cause.
# --memory 1Gi is generous: spaCy and presidio are installed but never
#   imported on the /api/match path (uploads are off), so nothing large loads.
gcloud run deploy "$SERVICE" \
  --image "$IMAGE" \
  --region "$REGION" \
  --service-account "$SA_EMAIL" \
  --allow-unauthenticated \
  --memory 1Gi \
  --cpu 1 \
  --timeout 900 \
  --min-instances 0 \
  --max-instances 3 \
  --set-env-vars "MODEL_EVALUATION=cv-guestimator-modal,ARTIFACTS_BUCKET=${BUCKET},ARTIFACTS_PREFIX=artifacts" \
  --set-secrets "CV_GUESTIMATOR_API_KEY=cv-guestimator-api-key:latest,MODAL_INFERENCE_URL=modal-inference-url:latest,MODAL_API_KEY=modal-api-key:latest,GOOGLE_API_KEY=google-api-key:latest"

URL="$(gcloud run services describe "$SERVICE" --region "$REGION" --format='value(status.url)')"

say "Deployed"
cat <<EOF
  URL     ${URL}
  Bucket  gs://${BUCKET}/artifacts

Smoke test (should be 401 -- proving the gate is on):
  curl -s -o /dev/null -w '%{http_code}\\n' -X POST ${URL}/api/match

Then in portfolio-sonny/.env:
  CV_GUESTIMATOR_API_URL=${URL}
  CV_GUESTIMATOR_API_KEY=<the same key stored in cv-guestimator-api-key>
EOF
