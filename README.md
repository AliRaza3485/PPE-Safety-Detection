# PPE Safety Detection 🦺

Real-time detection of **Personal Protective Equipment** (hard hats, masks,
safety vests) on worksite imagery using a YOLO object-detection model, served
through a FastAPI backend with a Next.js frontend.

The model also flags **violations** — a person detected without a required
piece of PPE (e.g. `NO-Hardhat`, `NO-Safety Vest`).

---

## 📁 Project structure

```
ppe-safety-detection/
├── data/                 # dataset (DVC tracked)
│   └── data.yaml         # YOLO dataset config (paths + class names)
├── notebooks/            # exploration, training experiments
├── src/
│   ├── train.py          # train / fine-tune the YOLO model
│   ├── predict.py        # run inference from the CLI
│   └── utils.py          # shared helpers (classes, drawing, model loading)
├── api/
│   └── main.py           # FastAPI inference service
├── frontend/             # Next.js app (upload image → view detections)
├── models/               # trained weights (best.pt) + metrics.json
├── params.yaml           # hyperparameters (DVC tracked)
├── dvc.yaml              # DVC training pipeline
├── requirements.txt
├── Dockerfile            # containerizes the API
└── README.md
```

---

## 🚀 Quickstart

### 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Get the data (DVC)

```bash
dvc pull        # downloads images/labels from the configured remote
```

The dataset follows the standard YOLO layout referenced in
[`data/data.yaml`](data/data.yaml):

```
data/
├── images/{train,val}/*.jpg
└── labels/{train,val}/*.txt
```

### 3. Train

```bash
# via DVC (recommended — versions the model + metrics)
dvc repro

# or directly
python -m src.train --data data/data.yaml --epochs 50 --imgsz 640
```

The best checkpoint is copied to `models/best.pt` and metrics to
`models/metrics.json`.

### 4. Predict from the CLI

```bash
python -m src.predict --source path/to/image.jpg --weights models/best.pt
```

### 5. Run the API

```bash
uvicorn api.main:app --reload --port 8000
# Swagger UI at http://localhost:8000/docs
```

| Method | Endpoint    | Description                                   |
|--------|-------------|-----------------------------------------------|
| GET    | `/health`   | Service + model status                        |
| POST   | `/predict`  | Upload an image → JSON detections + summary   |

### 6. Run the frontend

```bash
cd frontend
npm install
npm run dev        # http://localhost:3000
```

Set `NEXT_PUBLIC_API_URL` (defaults to `http://localhost:8000`) to point the
UI at your API.

---

## 🐳 Docker (API)

```bash
docker build -t ppe-api .
docker run -p 8000:8000 ppe-api
```

---

## 🏷️ Classes

`Hardhat`, `Mask`, `NO-Hardhat`, `NO-Mask`, `NO-Safety Vest`, `Person`,
`Safety Cone`, `Safety Vest`, `machinery`, `vehicle`

> Classes prefixed with `NO-` are treated as **safety violations**.
