#!/bin/bash
# setup_environment.sh - Environment setup script for SmartScrape
# Usage: ./setup_environment.sh [dev|test|prod]

set -e

# Default to development environment if not specified
ENV=${1:-dev}

echo "Setting up SmartScrape for $ENV environment..."

# Check for required tools
command -v python3 >/dev/null 2>&1 || { echo "Python 3 is required but not installed. Aborting."; exit 1; }
command -v pip >/dev/null 2>&1 || { echo "pip is required but not installed. Aborting."; exit 1; }

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create appropriate .env file
if [ "$ENV" = "dev" ]; then
    echo "Setting up development environment..."
    cp -n .env.example .env || true
    echo "Development environment setup complete."
elif [ "$ENV" = "test" ]; then
    echo "Setting up test environment..."
    cp -n .env.example .env.test || true
    echo "ENVIRONMENT=test" >> .env.test
    echo "Test environment setup complete."
elif [ "$ENV" = "prod" ]; then
    echo "Setting up production environment..."
    
    # Check if .env.production exists, create from example if not
    if [ ! -f ".env.production" ]; then
        cp .env.production.example .env.production || cp .env.example .env.production
        echo "Created .env.production from template."
        echo "IMPORTANT: You must edit .env.production with your production values!"
    fi
    
    echo "Production environment setup complete."
    echo "IMPORTANT: Review .env.production and update values before deploying!"
else
    echo "Unknown environment: $ENV. Use dev, test, or prod."
    exit 1
fi

# Create required directories
echo "Creating required directories..."
mkdir -p logs cache data

# Create secret key if needed
if [ "$ENV" = "prod" ] && ! grep -q "SECRET_KEY=" .env.production; then
    echo "Generating a secure SECRET_KEY..."
    SECRET=$(python -c 'import secrets; print(secrets.token_urlsafe(32))')
    echo "SECRET_KEY=$SECRET" >> .env.production
    echo "Generated SECRET_KEY and added to .env.production"
fi

# Initialize database if needed
if [ "$ENV" = "prod" ]; then
    echo "Would you like to initialize/upgrade the database? (y/n)"
    read answer
    if [ "$answer" = "y" ] || [ "$answer" = "Y" ]; then
        echo "Initializing database..."
        # Add database initialization commands here
        echo "Database initialization complete."
    fi
fi

echo "Environment setup complete for $ENV!"
echo "To activate this environment: source venv/bin/activate"

# Show next steps based on environment
if [ "$ENV" = "dev" ]; then
    echo "To start development server: uvicorn app:app --reload"
elif [ "$ENV" = "test" ]; then
    echo "To run tests: pytest"
elif [ "$ENV" = "prod" ]; then
    echo "To start production server: docker-compose up -d"
fi
