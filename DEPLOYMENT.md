# 🚀 MABe Mouser - Docker Deployment Guide

Complete guide for deploying MABe Mouser on a GPU server with Docker.

## 📋 Table of Contents
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Portainer Deployment](#portainer-deployment)
- [Configuration](#configuration)
- [GPU Setup](#gpu-setup)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)
- [Updates](#updates)

---

## Prerequisites

### Required
- **Docker**: Version 20.10 or later
- **Docker Compose**: Version 2.0 or later
- **NVIDIA GPU Drivers**: Latest stable version
- **NVIDIA Container Toolkit**: For GPU support
- **Dataset**: MABe mouse behavior detection dataset

### Optional
- **Portainer**: For GUI-based container management
- **Reverse Proxy**: Nginx, Caddy, or Traefik for HTTPS

---

## Quick Start (Command Line)

### 1. Clone Repository

```bash
# SSH to your server
ssh user@10.0.0.1

# Clone to a permanent location
cd /opt
git clone https://github.com/your-username/mabe-mouser.git
cd mabe-mouser
```

### 2. Prepare Dataset

```bash
# Option A: Copy from local machine
scp -r /path/to/MABe-mouse-behavior-detection user@10.0.0.1:/opt/mabe-mouser/

# Option B: Download from Kaggle (requires Kaggle API token)
# First, set up Kaggle API credentials (see Kaggle API Setup section below)
kaggle competitions download -c mabe-22-mouse-reach
unzip mabe-22-mouse-reach.zip -d MABe-mouse-behavior-detection
```

#### Kaggle API Setup

To download the dataset directly on the server, you need to transfer your Kaggle API token:

```bash
# On your LOCAL machine, copy your Kaggle API token to the server
scp ~/.kaggle/kaggle.json user@10.0.0.1:~/.kaggle/

# If the .kaggle directory doesn't exist on the server, create it first:
ssh user@10.0.0.1 "mkdir -p ~/.kaggle"
scp ~/.kaggle/kaggle.json user@10.0.0.1:~/.kaggle/

# Set proper permissions on the server
ssh user@10.0.0.1 "chmod 600 ~/.kaggle/kaggle.json"

# Install Kaggle CLI on the server
ssh user@10.0.0.1
pip install kaggle

# Now you can download the dataset
cd /opt/mabe-mouser
kaggle competitions download -c mabe-22-mouse-reach
unzip mabe-22-mouse-reach.zip -d MABe-mouse-behavior-detection
```

**Note:** If you don't have `~/.kaggle/kaggle.json` locally:
1. Go to https://www.kaggle.com
2. Click on your profile picture → Account
3. Scroll to "API" section → Click "Create New API Token"
4. This downloads `kaggle.json` to your Downloads folder
5. Move it: `mv ~/Downloads/kaggle.json ~/.kaggle/`

### 3. Configure Dataset Path

Edit `docker-compose.yml`:

```yaml
volumes:
  # Update this line (around line 14):
  - ./MABe-mouse-behavior-detection:/app/data:ro
  # Or use absolute path:
  - /opt/mabe-mouser/MABe-mouse-behavior-detection:/app/data:ro
```

### 4. Build Images

```bash
# Make build script executable
chmod +x docker/build.sh

# Build both frontend and backend images
./docker/build.sh
```

This will:
- Build the backend with CUDA support
- Build the frontend and create optimized production bundle
- Tag images as `mabe-mouser-backend:latest` and `mabe-mouser-frontend:latest`

### 5. Start the Stack

```bash
# Start in detached mode
docker-compose up -d

# View logs
docker-compose logs -f

# Check status
docker-compose ps
```

### 6. Access the Application

Open your browser to:
- **Application**: `http://10.0.0.1:87000` (or your server IP)
- **API Docs**: `http://10.0.0.1:87001/docs`

---

## Portainer Deployment

If you're using Portainer for container management:

### 1. Access Portainer
Navigate to your Portainer instance (usually `http://10.0.0.1:9000`)

### 2. Create New Stack

1. Go to **Stacks** → **Add stack**
2. Name: `mabe-mouser`
3. Build method: **Upload** or **Web editor**

### 3. Upload Configuration

**Option A: Upload File**
- Upload `docker-compose.portainer.yml`

**Option B: Copy/Paste**
- Open `docker-compose.portainer.yml` in a text editor
- Copy entire contents
- Paste into Portainer web editor

### 4. Configure Environment

Before deploying, update the dataset path in the YAML:

```yaml
volumes:
  # Change this line to your dataset location:
  - /path/to/your/MABe-dataset:/app/data:ro
```

### 5. Build Images First

Since Portainer uses pre-built images, you need to build them first on the server:

```bash
# SSH to server
ssh user@10.0.0.1
cd /opt/mabe-mouser

# Build images
./docker/build.sh
```

### 6. Deploy Stack

Click **Deploy the stack** in Portainer

### 7. Verify Deployment

1. Go to **Containers**
2. You should see:
   - `mabe-mouser-backend` (healthy)
   - `mabe-mouser-frontend` (healthy)
3. Check logs for any errors

---

## Configuration

### GPU Configuration

**Use All GPUs:**
```yaml
environment:
  - CUDA_VISIBLE_DEVICES=0,1  # Both GPUs
```

**Use Single GPU:**
```yaml
environment:
  - CUDA_VISIBLE_DEVICES=0  # Only GPU 0
```

**Adjust GPU Count:**
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1  # Change this number
          capabilities: [gpu]
```

### Port Configuration

**Default Ports:**
```yaml
services:
  frontend:
    ports:
      - "87000:80"  # Frontend at http://10.0.0.1:87000
  backend:
    ports:
      - "87001:8000"  # Backend API at http://10.0.0.1:87001
```

**To Change Ports:**
```yaml
services:
  frontend:
    ports:
      - "YOUR_PORT:80"  # Change YOUR_PORT to desired port
  backend:
    ports:
      - "YOUR_PORT:8000"  # Change YOUR_PORT to desired port
```

**Note:** If you change backend port, you must also update `docker/nginx.conf`:
```nginx
location /api/ {
    proxy_pass http://backend:8000;  # Update this if needed
```

### Memory Limits

Add memory limits if needed:

```yaml
services:
  backend:
    deploy:
      resources:
        limits:
          memory: 16G  # Limit to 16GB RAM
        reservations:
          memory: 8G   # Reserve 8GB RAM
```

### Environment Variables

Available environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `DATA_DIR` | `/app/data/train_tracking` | Path to tracking data |
| `METADATA_CSV` | `/app/data/train.csv` | Path to metadata CSV |
| `CUDA_VISIBLE_DEVICES` | `0,1` | Which GPUs to use |

---

## GPU Setup

### Verify NVIDIA Drivers

```bash
nvidia-smi
```

You should see your GPU information. If not, install NVIDIA drivers:

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y nvidia-driver-535  # Adjust version as needed
sudo reboot
```

### Install NVIDIA Container Toolkit

```bash
# Add NVIDIA repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Test GPU Access

```bash
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

You should see the same GPU information as running `nvidia-smi` directly.

---

## Monitoring

### View Logs

```bash
# All services
docker-compose logs -f

# Backend only
docker-compose logs -f backend

# Frontend only
docker-compose logs -f frontend

# Last 100 lines
docker-compose logs --tail=100
```

### Check Container Status

```bash
# List running containers
docker-compose ps

# Detailed info
docker-compose ps -a
```

### Monitor GPU Usage

```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# GPU usage in backend container
docker exec -it mabe-mouser-backend nvidia-smi

# Continuous monitoring
docker exec -it mabe-mouser-backend watch -n 1 nvidia-smi
```

### Health Checks

```bash
# Check backend health
curl http://localhost:87001/api/files

# Check frontend
curl http://localhost:87000/

# Via docker
docker inspect mabe-mouser-backend | grep -A 10 Health
docker inspect mabe-mouser-frontend | grep -A 10 Health
```

### Resource Usage

```bash
# Container resource usage
docker stats

# Specific container
docker stats mabe-mouser-backend mabe-mouser-frontend
```

---

## Troubleshooting

### Backend Won't Start

**Check Logs:**
```bash
docker-compose logs backend
```

**Common Issues:**

1. **Dataset not found:**
   ```
   ⚠️  Warning: Data directory not found
   ```
   **Solution:** Update dataset path in `docker-compose.yml`

2. **Port already in use:**
   ```
   Error: bind: address already in use
   ```
   **Solution:** Change port mapping in `docker-compose.yml` (default ports are 87000 and 87001)

3. **Python dependency errors:**
   ```
   ModuleNotFoundError: No module named 'X'
   ```
   **Solution:** Rebuild with no cache:
   ```bash
   docker-compose build --no-cache backend
   docker-compose up -d
   ```

### Frontend Shows 404

**Rebuild Frontend:**
```bash
docker-compose build --no-cache frontend
docker-compose up -d frontend
```

**Check nginx config:**
```bash
docker exec -it mabe-mouser-frontend cat /etc/nginx/conf.d/default.conf
```

### GPUs Not Detected

**Test GPU access:**
```bash
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

**If fails:**
1. Install NVIDIA Container Toolkit (see [GPU Setup](#gpu-setup))
2. Restart Docker: `sudo systemctl restart docker`
3. Rebuild images: `./docker/build.sh`

### API Calls Failing

**Check backend logs:**
```bash
docker-compose logs -f backend
```

**Test API directly:**
```bash
# List files
curl http://localhost:87001/api/files

# Get overview stats
curl http://localhost:87001/api/analytics/overview
```

**Common Issues:**
- Dataset path incorrect
- Parquet files corrupted
- Memory issues (increase limits)

### Slow Performance

**Check resource usage:**
```bash
docker stats
```

**Solutions:**
1. **Increase memory limits** in docker-compose.yml
2. **Enable GPU acceleration** (check CUDA_VISIBLE_DEVICES)
3. **Use SSD** for dataset storage
4. **Increase worker processes** in backend startup

---

## Updates

### Update Code

```bash
cd /opt/mabe-mouser
git pull origin main
```

### Rebuild and Restart

```bash
# Rebuild images
./docker/build.sh

# Restart services
docker-compose down
docker-compose up -d
```

### Update Without Downtime

```bash
# Build new images
docker-compose build

# Rolling update (one service at a time)
docker-compose up -d --no-deps --build backend
docker-compose up -d --no-deps --build frontend
```

### Clean Rebuild

```bash
# Stop services
docker-compose down

# Remove old images
docker rmi mabe-mouser-backend:latest
docker rmi mabe-mouser-frontend:latest

# Rebuild from scratch
./docker/build.sh

# Start services
docker-compose up -d
```

---

## Backup and Restore

### Backup Configuration

```bash
# Create backup directory
mkdir -p /backup/mabe-mouser

# Backup configuration files
tar -czf /backup/mabe-mouser/config-$(date +%Y%m%d).tar.gz \
    docker-compose.yml \
    docker/ \
    backend/ \
    frontend/

# Backup images
docker save mabe-mouser-backend:latest | gzip > /backup/mabe-mouser/backend-image.tar.gz
docker save mabe-mouser-frontend:latest | gzip > /backup/mabe-mouser/frontend-image.tar.gz
```

### Restore from Backup

```bash
# Extract configuration
tar -xzf /backup/mabe-mouser/config-YYYYMMDD.tar.gz

# Load images
docker load < /backup/mabe-mouser/backend-image.tar.gz
docker load < /backup/mabe-mouser/frontend-image.tar.gz

# Start services
docker-compose up -d
```

---

## Performance Tuning

### Increase Backend Workers

Edit `docker/Dockerfile.backend`, change last line:

```dockerfile
# Single worker (default)
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Multiple workers for better performance
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### Enable HTTP/2 and Compression

Add to `docker/nginx.conf`:

```nginx
server {
    listen 80 http2;  # Enable HTTP/2
    
    # Gzip compression (already included)
    gzip on;
    gzip_comp_level 6;
    gzip_types text/plain text/css text/xml text/javascript application/javascript application/json;
}
```

### Use Production ASGI Server

For even better performance, use Gunicorn with Uvicorn workers:

```dockerfile
# In Dockerfile.backend
CMD ["gunicorn", "backend.main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
```

---

## HTTPS Setup (Optional)

### Option 1: Caddy (Easiest)

Create `Caddyfile`:

```caddy
mabe.yourdomain.com {
    reverse_proxy localhost:80
}
```

Run Caddy:
```bash
docker run -d \
    --name caddy \
    -p 80:80 \
    -p 443:443 \
    -v $PWD/Caddyfile:/etc/caddy/Caddyfile \
    -v caddy_data:/data \
    caddy
```

### Option 2: Nginx with Let's Encrypt

Use Certbot to get SSL certificate, then update nginx config.

---

## Support

For issues or questions:
1. Check logs: `docker-compose logs -f`
2. Review this guide
3. Check GitHub issues
4. Create new issue with logs and configuration

---

## License

[Your License Here]

---

**Last Updated:** October 6, 2025
