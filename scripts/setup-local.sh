#!/bin/bash
set -e

echo "🔧 MXTOAI Local Setup"
echo "===================="

# Check for required tools
echo "Checking for required tools..."

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker Desktop."
    echo "   Download from: https://www.docker.com/products/docker-desktop"
    exit 1
fi

# Check Docker Compose (try both old and new commands)
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose."
    echo "   Usually comes with Docker Desktop."
    exit 1
fi

echo "✅ Docker and Docker Compose are installed"

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p attachments downloads email_attachments

# Copy environment files if they don't exist
if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        echo "📝 Copying .env.example to .env..."
        cp .env.example .env
        echo "⚠️  Please edit .env file and configure your environment variables."
    else
        echo "❌ .env.example not found. Please create it first."
        exit 1
    fi
else
    echo "✅ .env file already exists"
fi

# Copy model configuration if it doesn't exist
if [ ! -f model.config.toml ]; then
    if [ -f model.config.example.toml ]; then
        echo "📝 Copying model configuration example..."
        cp model.config.example.toml model.config.toml
        echo "⚠️  Please edit model.config.toml and configure your AI model credentials."
    else
        echo "❌ model.config.example.toml not found. Please create it first."
        exit 1
    fi
else
    echo "✅ Model configuration already exists"
fi

# Make scripts executable
echo "Making scripts executable..."
chmod +x scripts/start-local.sh scripts/setup-local.sh

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your configuration"
echo "2. Edit model.config.toml with your AI model credentials"
echo "3. Validate configuration: ./scripts/validate-env.sh"
echo "4. Run: ./scripts/start-local.sh"
echo ""
echo "For more information, see README.md and ENV_VARIABLES.md"
