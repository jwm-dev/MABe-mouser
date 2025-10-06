#!/bin/bash
set -e

echo "🏗️  Building MABe Mouser Docker images..."
echo ""

# Check if docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if NVIDIA runtime is available (optional, will warn but not fail)
if docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU runtime detected"
else
    echo "⚠️  NVIDIA GPU runtime not detected - GPU acceleration will be unavailable"
    echo "   To enable GPU support, install NVIDIA Container Toolkit"
fi

echo ""
echo "📦 Building backend image..."
docker build -f docker/Dockerfile.backend -t mabe-mouser-backend:latest .

echo ""
echo "🎨 Building frontend image..."
docker build -f docker/Dockerfile.frontend -t mabe-mouser-frontend:latest .

echo ""
echo "✅ Build complete!"
echo ""
echo "📋 Next steps:"
echo ""
echo "1. Update dataset path in docker-compose.yml (line 14):"
echo "   volumes:"
echo "     - /path/to/your/dataset:/app/data:ro"
echo ""
echo "2. Start the stack:"
echo "   docker-compose up -d"
echo ""
echo "3. View logs:"
echo "   docker-compose logs -f"
echo ""
echo "4. Access the application:"
echo "   http://localhost (or http://your-server-ip)"
echo ""
echo "5. Stop the stack:"
echo "   docker-compose down"
echo ""
