#!/bin/bash

# SmartScrape Service Startup Script
# This script starts Redis, Celery workers, and the FastAPI application

set -e  # Exit on any error

echo "🚀 Starting SmartScrape Services..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if a service is running
check_service() {
    local service_name=$1
    local port=$2
    local max_attempts=${3:-30}
    local attempt=1
    
    print_status "Checking if $service_name is ready on port $port..."
    
    while [ $attempt -le $max_attempts ]; do
        if nc -z localhost $port 2>/dev/null; then
            print_success "$service_name is ready on port $port"
            return 0
        fi
        print_status "Waiting for $service_name... (attempt $attempt/$max_attempts)"
        sleep 1
        ((attempt++))
    done
    
    print_error "$service_name failed to start on port $port"
    return 1
}

# Function to check if Redis is running
check_redis() {
    if redis-cli ping >/dev/null 2>&1; then
        print_success "Redis is already running"
        return 0
    fi
    return 1
}

# Function to start Redis
start_redis() {
    if check_redis; then
        return 0
    fi
    
    print_status "Starting Redis server..."
    
    # Try to start Redis as a background service
    if command -v brew >/dev/null 2>&1; then
        # macOS with Homebrew
        if brew services list | grep redis | grep started >/dev/null; then
            print_success "Redis is already running via brew services"
        else
            print_status "Starting Redis via brew services..."
            brew services start redis
            sleep 2
        fi
    elif command -v systemctl >/dev/null 2>&1; then
        # Linux with systemd
        print_status "Starting Redis via systemctl..."
        sudo systemctl start redis
        sleep 2
    else
        # Manual start
        print_status "Starting Redis manually..."
        redis-server --daemonize yes --port 6379
        sleep 2
    fi
    
    # Verify Redis is running
    if check_redis; then
        print_success "Redis started successfully"
        return 0
    else
        print_error "Failed to start Redis"
        return 1
    fi
}

# Function to start Celery worker
start_celery() {
    print_status "Starting Celery worker..."
    
    # Kill any existing Celery processes
    pkill -f "celery worker" 2>/dev/null || true
    sleep 1
    
    # Start Celery worker in background
    celery -A core.celery_config worker --loglevel=info --detach \
        --pidfile=/tmp/celery_worker.pid \
        --logfile=logs/celery_worker.log
    
    if [ $? -eq 0 ]; then
        print_success "Celery worker started successfully"
        return 0
    else
        print_error "Failed to start Celery worker"
        return 1
    fi
}

# Function to start FastAPI application
start_fastapi() {
    local port=${1:-5000}
    local host=${2:-"0.0.0.0"}
    
    print_status "Starting FastAPI application on $host:$port..."
    
    # Kill any existing FastAPI processes on the port
    lsof -ti:$port | xargs kill -9 2>/dev/null || true
    sleep 1
    
    # Start FastAPI with uvicorn
    python app.py --host $host --port $port &
    FASTAPI_PID=$!
    echo $FASTAPI_PID > /tmp/fastapi.pid
    
    # Wait for FastAPI to be ready
    if check_service "FastAPI" $port; then
        print_success "FastAPI application started successfully (PID: $FASTAPI_PID)"
        return 0
    else
        print_error "Failed to start FastAPI application"
        return 1
    fi
}

# Function to create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    mkdir -p logs
    mkdir -p cache
    mkdir -p data
    mkdir -p /tmp
    print_success "Directories created"
}

# Function to show service status
show_status() {
    echo ""
    print_status "Service Status Summary:"
    echo "========================"
    
    # Redis status
    if check_redis >/dev/null 2>&1; then
        print_success "✅ Redis: Running"
    else
        print_error "❌ Redis: Not running"
    fi
    
    # Celery status
    if [ -f /tmp/celery_worker.pid ] && kill -0 $(cat /tmp/celery_worker.pid) 2>/dev/null; then
        print_success "✅ Celery Worker: Running (PID: $(cat /tmp/celery_worker.pid))"
    else
        print_error "❌ Celery Worker: Not running"
    fi
    
    # FastAPI status
    if [ -f /tmp/fastapi.pid ] && kill -0 $(cat /tmp/fastapi.pid) 2>/dev/null; then
        print_success "✅ FastAPI: Running (PID: $(cat /tmp/fastapi.pid))"
    else
        print_error "❌ FastAPI: Not running"
    fi
    
    echo ""
}

# Function to stop all services
stop_services() {
    print_status "Stopping all services..."
    
    # Stop FastAPI
    if [ -f /tmp/fastapi.pid ]; then
        kill $(cat /tmp/fastapi.pid) 2>/dev/null || true
        rm -f /tmp/fastapi.pid
    fi
    
    # Stop Celery
    if [ -f /tmp/celery_worker.pid ]; then
        kill $(cat /tmp/celery_worker.pid) 2>/dev/null || true
        rm -f /tmp/celery_worker.pid
    fi
    pkill -f "celery worker" 2>/dev/null || true
    
    # Stop Redis (only if we started it)
    if command -v brew >/dev/null 2>&1; then
        brew services stop redis 2>/dev/null || true
    fi
    
    print_success "All services stopped"
}

# Main execution
case "${1:-start}" in
    start)
        # Parse port argument
        PORT=${2:-5000}
        HOST=${3:-"0.0.0.0"}
        
        print_status "Starting SmartScrape services..."
        print_status "FastAPI will run on $HOST:$PORT"
        
        create_directories
        
        # Start services in order
        if start_redis && start_celery && start_fastapi $PORT $HOST; then
            show_status
            print_success "🎉 All services started successfully!"
            print_status "Access the application at: http://localhost:$PORT"
            print_status "API documentation at: http://localhost:$PORT/docs"
            
            # Keep the script running and show logs
            print_status "Monitoring services... (Press Ctrl+C to stop all services)"
            trap stop_services EXIT INT TERM
            
            # Follow logs
            if [ -f logs/celery_worker.log ]; then
                tail -f logs/celery_worker.log &
            fi
            
            # Wait for FastAPI process
            if [ -f /tmp/fastapi.pid ]; then
                wait $(cat /tmp/fastapi.pid)
            fi
        else
            print_error "Failed to start services"
            stop_services
            exit 1
        fi
        ;;
    stop)
        stop_services
        ;;
    status)
        show_status
        ;;
    restart)
        stop_services
        sleep 2
        exec $0 start ${2:-5000} ${3:-"0.0.0.0"}
        ;;
    *)
        echo "Usage: $0 {start|stop|status|restart} [port] [host]"
        echo "  start   - Start all services (default port: 5000, host: 0.0.0.0)"
        echo "  stop    - Stop all services"
        echo "  status  - Show service status"
        echo "  restart - Restart all services"
        echo ""
        echo "Examples:"
        echo "  $0 start          # Start on port 5000"
        echo "  $0 start 8080     # Start on port 8080"
        echo "  $0 start 8080 127.0.0.1  # Start on port 8080, localhost only"
        exit 1
        ;;
esac
