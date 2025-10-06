#!/bin/bash
set -e

echo "🚀 MABe Mouser - Quick Deploy Script"
echo "===================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   echo -e "${RED}❌ Do not run this script as root${NC}"
   exit 1
fi

# Check for Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed${NC}"
    echo "Please install Docker first: https://docs.docker.com/engine/install/"
    exit 1
fi

# Check for Docker Compose
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo -e "${RED}❌ Docker Compose is not installed${NC}"
    echo "Please install Docker Compose: https://docs.docker.com/compose/install/"
    exit 1
fi

# Check for NVIDIA runtime (optional warning)
if docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✅ NVIDIA GPU runtime detected${NC}"
else
    echo -e "${YELLOW}⚠️  NVIDIA GPU runtime not detected${NC}"
    echo "   GPU acceleration will be unavailable"
    echo "   To enable: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    echo ""
fi

# Check for dataset
echo "📁 Checking for dataset..."
if [ -d "./MABe-mouse-behavior-detection" ]; then
    echo -e "${GREEN}✅ Dataset found${NC}"
else
    echo -e "${YELLOW}⚠️  Dataset not found in current directory${NC}"
    echo ""
    echo "Please ensure your dataset is located at:"
    echo "  $(pwd)/MABe-mouse-behavior-detection"
    echo ""
    echo "Or update the volume path in docker-compose.yml"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "🏗️  Building Docker images..."
echo "This may take 5-10 minutes on first build..."
echo ""

# Make build script executable
chmod +x docker/build.sh

# Build images
if ! ./docker/build.sh; then
    echo -e "${RED}❌ Build failed${NC}"
    exit 1
fi

echo ""
echo "🎬 Starting services..."
echo ""

# Start services
if ! docker-compose up -d; then
    echo -e "${RED}❌ Failed to start services${NC}"
    exit 1
fi

echo ""
echo "⏳ Waiting for services to be healthy..."
sleep 5

# Check backend health
for i in {1..30}; do
    if curl -s http://localhost:8000/api/files > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Backend is healthy${NC}"
        break
    fi
    if [ $i -eq 30 ]; then
        echo -e "${RED}❌ Backend health check timeout${NC}"
        echo "Check logs with: docker-compose logs backend"
        exit 1
    fi
    sleep 2
done

# Check frontend health
for i in {1..15}; do
    if curl -s http://localhost:87000/ > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Frontend is healthy${NC}"
        break
    fi
    if [ $i -eq 15 ]; then
        echo -e "${RED}❌ Frontend health check timeout${NC}"
        echo "Check logs with: docker-compose logs frontend"
        exit 1
    fi
    sleep 2
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${GREEN}✅ Deployment Complete!${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📱 Access the application:"
echo "   🌐 Frontend: http://localhost:87000"
echo "   📊 API Docs: http://localhost:87001/docs"
echo ""
echo "🔧 Useful commands:"
echo "   View logs:    docker-compose logs -f"
echo "   Stop:         docker-compose down"
echo "   Restart:      docker-compose restart"
echo "   Status:       docker-compose ps"
echo ""

# Get server IP if available
SERVER_IP=$(hostname -I | awk '{print $1}' 2>/dev/null || echo "your-server-ip")
if [ "$SERVER_IP" != "your-server-ip" ] && [ "$SERVER_IP" != "127.0.0.1" ]; then
    echo "🌍 Remote access:"
    echo "   http://${SERVER_IP}:87000"
    echo "   http://${SERVER_IP}:87001/docs"
    echo ""
fi

echo "📖 Full documentation: DEPLOYMENT.md"
echo ""
