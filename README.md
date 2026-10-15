# Clusture AI — Brain MRI Segmentation Showcase

A starter full-stack demo for connecting a Next.js website to a remote FastAPI backend that runs an existing nnU-Net Docker inference command.

## What this project includes

- Next.js upload interface for FLAIR, T1, T1ce and T2 `.nii` / `.nii.gz` files
- FastAPI backend with asynchronous job tracking
- One-job-at-a-time inference lock (appropriate for a single GPU demo server)
- Upload validation and configurable size limits
- Job polling and downloadable prediction NIfTI
- Docker-command adapter configured by environment variable
- Clear setup instructions and safety notes

## Important limitation

The `rixez/brats21nnunet` image's exact command-line interface must be confirmed from the command that already worked on your server. The backend is intentionally configured using `NNUNET_COMMAND_TEMPLATE`; do not guess this value. The app cannot run inference until you configure it to match your working command and mount paths.

This is an experimental research showcase, not a medical device or diagnostic service. Use only permitted, de-identified test data. Do not expose the API publicly without authentication, HTTPS, and a security review.

## Architecture

Browser (Next.js) -> FastAPI -> Docker inference command -> NIfTI prediction -> browser download.

For a remote GPU server, deploy `backend/` on that server. Deploy `frontend/` locally or on Vercel, with `NEXT_PUBLIC_API_BASE_URL` set to the backend's HTTPS URL.

---

## A. Backend setup on the remote Linux/GPU server

### 1. Copy backend folder to the server

From your laptop, from the directory containing this project:

```bash
scp -r backend xeon4@YOUR_SERVER:/home/xeon4/clusture-showcase-backend
```

Replace the SSH username/host with your actual server details. Alternatively, clone/copy the project using your existing workflow.

### 2. Create Python environment

On the remote server:

```bash
cd /home/xeon4/clusture-showcase-backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set:
- `API_KEY`: a long random secret for the demo API
- `CORS_ORIGINS`: your frontend origin (e.g. `http://localhost:3000`)
- `HOST_INPUT_ROOT` and `HOST_OUTPUT_ROOT`: directories visible to Docker on the host
- `NNUNET_COMMAND_TEMPLATE`: the exact known-working Docker command with placeholders

### 3. Configure the inference command

The command template is a JSON array, not a shell string. It supports:
- `{input_dir}` — absolute input directory on the host
- `{output_dir}` — absolute output directory on the host
- `{case_id}` — generated case identifier

**Do not invent a command for your image.** First run your known-working command manually. Then adapt its input/output paths to the placeholders. Example shape only (not a working command for every image):

```dotenv
NNUNET_COMMAND_TEMPLATE=["docker","run","--rm","--gpus","all","-v","{input_dir}:/INPUT:ro","-v","{output_dir}:/OUTPUT","rixez/brats21nnunet","<REPLACE_WITH_THE_REST_OF_YOUR_VERIFIED_COMMAND>"]
```

Replace the placeholder with the real command/arguments used by your existing image. The command must write a prediction `.nii` or `.nii.gz` file inside `{output_dir}`. Never paste credentials or secrets into the command.

The backend creates per-job host folders under `DATA_ROOT`, and the folders must be accessible to the Docker daemon. For Docker on the same server, absolute host paths are appropriate.

### 4. Start backend

```bash
source .venv/bin/activate
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

This binds locally only. For public access, put it behind a reverse proxy with HTTPS and authentication, or use a VPN/tunnel for a private demo. Do not expose port 8000 directly to the public internet.

Check health:

```bash
curl http://127.0.0.1:8000/health
```

The API requires `X-API-Key` on upload, job status and result endpoints. `/health` is unauthenticated for local health checks.

### 5. Test the backend

```bash
curl -H "X-API-Key: YOUR_API_KEY" http://127.0.0.1:8000/api/jobs
```

Use the website to submit a test case. Watch the backend terminal logs.

---

## B. Frontend setup

On your laptop:

```bash
cd frontend
npm install
cp .env.local.example .env.local
```

Set `NEXT_PUBLIC_API_BASE_URL` to the API URL, and `NEXT_PUBLIC_DEMO_API_KEY` to the same API key. This is acceptable only for a private demo; a key embedded in a browser is visible to users. For a public deployment, replace this with proper user authentication and a server-side Next.js proxy/session.

Start:

```bash
npm run dev
```

Open `http://localhost:3000`.

For Vercel, configure the same environment variables in project settings. Do not use a browser-exposed API key for a public clinical service.

---

## C. What the demo does and does not do

- The UI submits all four modalities as one case.
- Backend validates extensions and size, stages files in a unique job folder, then launches the configured command in a worker thread.
- Job status is `queued`, `running`, `completed`, or `failed`.
- When completed, the backend locates the first NIfTI prediction in the job output folder.
- The UI downloads the raw predicted segmentation.

Not included yet:
- DICOM ingestion/PACS integration
- clinical report generation
- tumor classification (nnU-Net segmentation is not tumor-type classification)
- validated tumor volumes or anatomical localization
- ground-truth metrics unless you explicitly add a ground-truth mask
- production authentication, user accounts, database, audit logging, encrypted storage lifecycle, monitoring, and regulatory compliance

## D. Operational cautions

1. Start with a small, de-identified, authorized research case.
2. Check modality geometry, orientation and spacing before inference. This starter checks file types, not full medical-image geometry.
3. The command's output path and filename convention must be verified.
4. Avoid simultaneous jobs on a single GPU; this starter serializes inference in one backend process.
5. If the backend process restarts, in-memory job status is lost. This is acceptable only for an initial demo; production needs persistent job storage/queue.
6. Keep uploads private and remove data when no longer needed.
7. Label all outputs as experimental AI assistance requiring qualified radiologist review.
