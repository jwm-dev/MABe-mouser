# MABe Mouser

Interactive 3D visualization tool for the MABe 2025 Mouse Behavior Challenge.

## Quick Start

### Create Virtual Environment (Recommended)
```bash
cd $PROJECT_ROOT
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate   # Windows
```

### Automated Setup

Run the setup script to configure everything automatically:

```bash
python run.py
```

This will:
- Install all dependencies (Python + Node.js)
- Download competition data
- Start both backend and frontend servers

Opens [http://localhost:3000](http://localhost:3000) when ready.

### Kaggle API Setup

Before running `run.py`, you need Kaggle API credentials:

1. Go to https://www.kaggle.com/settings/account
2. Click "Create New Token" to download `kaggle.json`
3. Place the file in the correct location:
   - **Linux/Mac**: `~/.kaggle/kaggle.json` (or `~/.config/kaggle/kaggle.json`)
   - **Windows**: `C:\Users\<username>\.kaggle\kaggle.json`

4. Set permissions (Linux/Mac only):
   ```bash
   chmod 600 ~/.kaggle/kaggle.json
   ```

## Controls

- **Space**: Play/pause
- **Left/Right arrows**: Skip frames
- **R**: Reset camera
- **Mouse drag**: Rotate camera
- **Mouse wheel**: Zoom

## Architecture

**Backend**: FastAPI + Polars for efficient parquet streaming  
**Frontend**: React + TypeScript + deck.gl for 3D rendering

## Manual Setup (Advanced)

If you prefer to set up manually:

### 1. Install Dependencies
```bash
pip install -r requirements.txt
cd frontend
npm install
cd ..
```

### 2. Download Data
```bash
kaggle competitions download -c [competition-name]
# Extract to MABe-mouse-behavior-detection/
```

### 3. Start Servers
```bash
# Terminal 1 (Backend)
python -m src.app

# Terminal 2 (Frontend)
cd frontend
npm run dev
```

## Troubleshooting

**Port conflicts**: Backend uses port 8000, frontend uses 5173  
**Kaggle API errors**: Verify `kaggle.json` credentials and permissions  
**Module not found**: Ensure virtual environment is activated
