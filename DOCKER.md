# 🐳 Docker Deployment - Quick Reference

## One-Command Deployment

```bash
chmod +x deploy.sh
./deploy.sh
```

That's it! The script will:
- ✅ Check prerequisites (Docker, GPU support)
- ✅ Build frontend and backend images
- ✅ Start services with Docker Compose
- ✅ Verify health checks
- ✅ Display access URLs

## Manual Deployment

### 1. Build Images
```bash
chmod +x docker/build.sh
./docker/build.sh
```

### 2. Configure Dataset Path
Edit `docker-compose.yml` line 14:
```yaml
volumes:
  - ./MABe-mouse-behavior-detection:/app/data:ro
```

### 3. Start Services
```bash
docker-compose up -d
```

### 4. Access Application
- Frontend: http://localhost:87000
- API: http://localhost:87001/docs

## Portainer Deployment

1. Open Portainer UI
2. Go to **Stacks** → **Add stack**
3. Name: `mabe-mouser`
4. Upload `docker-compose.portainer.yml`
5. Update dataset path in the YAML
6. Build images first: `./docker/build.sh`
7. Deploy stack

## GPU Configuration

**Both GPUs:**
```yaml
environment:
  - CUDA_VISIBLE_DEVICES=0,1
deploy:
  resources:
    reservations:
      devices:
        - count: 2
```

**Single GPU:**
```yaml
environment:
  - CUDA_VISIBLE_DEVICES=0
deploy:
  resources:
    reservations:
      devices:
        - count: 1
```

## Useful Commands

```bash
# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Restart services
docker-compose restart

# Check status
docker-compose ps

# Monitor GPU usage
docker exec -it mabe-mouser-backend nvidia-smi

# Update and redeploy
git pull
./docker/build.sh
docker-compose up -d
```

## Troubleshooting

**Backend won't start:**
```bash
docker-compose logs backend
# Check dataset path in docker-compose.yml
```

**GPUs not detected:**
```bash
# Test GPU access
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi

# Install NVIDIA Container Toolkit if needed
# See DEPLOYMENT.md for instructions
```

**Frontend 404 errors:**
```bash
docker-compose build --no-cache frontend
docker-compose up -d frontend
```

## Full Documentation

See [DEPLOYMENT.md](DEPLOYMENT.md) for complete deployment guide including:
- Prerequisites and setup
- Detailed configuration options
- GPU setup instructions
- Monitoring and logging
- Performance tuning
- HTTPS setup
- Backup and restore

## Architecture

```
┌─────────────────────────────────┐
│   Frontend (Nginx)              │
│   Port 80                       │
│   - React app                   │
│   - Static assets               │
└──────────┬──────────────────────┘
           │ /api/* proxy
┌──────────▼──────────────────────┐
│   Backend (FastAPI)             │
│   Port 8000                     │
│   - REST API                    │
│   - GPU acceleration (CUDA)     │
│   - Dataset access              │
└─────────────────────────────────┘
```

## Requirements

- Docker 20.10+
- Docker Compose 2.0+
- NVIDIA drivers (for GPU)
- NVIDIA Container Toolkit (for GPU)
- 16GB+ RAM recommended
- 2+ GPU cores (optional but recommended)

## Ports

- `87000` - Frontend (HTTP)
- `87001` - Backend API
- `443` - HTTPS (if configured)

Change ports in `docker-compose.yml` if needed.
