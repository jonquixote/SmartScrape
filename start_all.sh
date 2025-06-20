#!/bin/bash
# start_all.sh - Unified startup script for SmartScrape services
# Starts Redis, Celery workers, and the FastAPI server in the correct order

set -e  # Exit on error

# Configuration
DEFAULT_HOST="127.0.0.1"
DEFAULT_PORT="5000"
DEFAULT_REDIS_PORT="6379"
DEFAULT_LOG_LEVEL="info"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}[SMARTSCRAPE]${NC} $1"
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if a port is in use
port_in_use() {
    lsof -i :$1 >/dev/null 2>&1
}

# Function to wait for service to be ready
wait_for_service() {
    local host=$1
    local port=$2
    local service_name=$3
    local max_attempts=30
    local attempt=1
    
    print_status "Waiting for $service_name to be ready on $host:$port..."
    
    while [ $attempt -le $max_attempts ]; do
        if nc -z $host $port 2>/dev/null; then
            print_status "$service_name is ready!"
            return 0
        fi
        
        echo -n "."
        sleep 1
        attempt=$((attempt + 1))
    done
    
    print_error "$service_name failed to start within $max_attempts seconds"
    return 1
}

# Function to start Redis
start_redis() {
    print_header "Starting Redis..."
    
    # Check if Redis is already running
    if port_in_use $DEFAULT_REDIS_PORT; then
        print_warning "Redis is already running on port $DEFAULT_REDIS_PORT"
        return 0
    fi
    
    # Try Docker first
    if command_exists docker; then
        print_status "Starting Redis using Docker..."
        docker run -d --name smartscrape-redis -p $DEFAULT_REDIS_PORT:6379 redis:alpine >/dev/null 2>&1 || {
            # Clean up existing container if it exists
            docker rm -f smartscrape-redis >/dev/null 2>&1 || true
            docker run -d --name smartscrape-redis -p $DEFAULT_REDIS_PORT:6379 redis:alpine >/dev/null 2>&1
        }
        
        # Wait for Redis to be ready
        wait_for_service localhost $DEFAULT_REDIS_PORT "Redis (Docker)"
        return 0
    fi
    
    # Try Homebrew Redis on macOS
    if [[ "$OSTYPE" == "darwin"* ]] && command_exists brew; then
        if brew services list | grep redis | grep started >/dev/null; then
            print_status "Redis is already running via Homebrew"
            return 0
        fi
        
        print_status "Starting Redis using Homebrew..."
        brew services start redis >/dev/null 2>&1 || {
            print_warning "Failed to start Redis with Homebrew, trying direct redis-server"
            redis-server &
        }
        
        wait_for_service localhost $DEFAULT_REDIS_PORT "Redis (Homebrew)"
        return 0
    fi
    
    # Try system Redis
    if command_exists redis-server; then
        print_status "Starting Redis server directly..."
        redis-server --daemonize yes --port $DEFAULT_REDIS_PORT
        wait_for_service localhost $DEFAULT_REDIS_PORT "Redis (System)"
        return 0
    fi
    
    print_error "Redis is not available. Please install Redis using:"
    print_error "  macOS: brew install redis"
    print_error "  Docker: docker run -d -p 6379:6379 redis:alpine"
    print_error "  Ubuntu: sudo apt install redis-server"
    return 1
}

# Function to start Celery worker
start_celery() {
    print_header "Starting Celery worker..."
    
    # Check if we're in the right directory
    if [ ! -f "core/celery_config.py" ]; then
        print_error "core/celery_config.py not found. Please run from SmartScrape root directory"
        return 1
    fi
    
    # Start Celery worker in background
    print_status "Starting Celery worker..."
    celery -A core.celery_config worker --loglevel=$DEFAULT_LOG_LEVEL --detach --pidfile=logs/celery.pid --logfile=logs/celery.log || {
        print_error "Failed to start Celery worker"
        return 1
    }
    
    # Give Celery a moment to start
    sleep 2
    print_status "Celery worker started"
    return 0
}

# Function to start the FastAPI server
start_server() {
    local host=$1
    local port=$2
    local debug=$3
    
    print_header "Starting SmartScrape server..."
    
    # Check if port is available
    if port_in_use $port; then
        print_error "Port $port is already in use"
        return 1
    fi
    
    # Start the server
    print_status "Starting FastAPI server on $host:$port..."
    
    if [ "$debug" == "true" ]; then
        python app.py --host $host --port $port --reload --debug --log-level debug
    else
        python app.py --host $host --port $port --log-level $DEFAULT_LOG_LEVEL
    fi
}

# Function to stop services
stop_services() {
    print_header "Stopping SmartScrape services..."
    
    # Stop Celery worker
    if [ -f "logs/celery.pid" ]; then
        print_status "Stopping Celery worker..."
        celery -A core.celery_config control shutdown || true
        rm -f logs/celery.pid
    fi
    
    # Stop Docker Redis if it's running
    if command_exists docker && docker ps | grep smartscrape-redis >/dev/null; then
        print_status "Stopping Redis Docker container..."
        docker stop smartscrape-redis >/dev/null 2>&1 || true
        docker rm smartscrape-redis >/dev/null 2>&1 || true
    fi
    
    # Stop Homebrew Redis on macOS
    if [[ "$OSTYPE" == "darwin"* ]] && command_exists brew; then
        if brew services list | grep redis | grep started >/dev/null; then
            print_status "Stopping Redis via Homebrew..."
            brew services stop redis >/dev/null 2>&1 || true
        fi
    fi
    
    print_status "Services stopped"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "OPTIONS:"
    echo "  --host HOST      Host to bind to (default: $DEFAULT_HOST)"
    echo "  --port PORT      Port to bind to (default: $DEFAULT_PORT)"
    echo "  --debug          Enable debug mode"
    echo "  --stop           Stop all services"
    echo "  --status         Show service status"
    echo "  --help           Show this help message"
    echo ""
    echo "EXAMPLES:"
    echo "  $0                           # Start with defaults"
    echo "  $0 --port 8000 --debug      # Start on port 8000 with debug"
    echo "  $0 --stop                   # Stop all services"
    echo "  $0 --status                 # Check service status"
}

# Function to show service status
show_status() {
    print_header "SmartScrape Service Status"
    
    # Check Redis
    if port_in_use $DEFAULT_REDIS_PORT; then
        print_status "Redis: Running on port $DEFAULT_REDIS_PORT"
    else
        print_warning "Redis: Not running"
    fi
    
    # Check Celery
    if [ -f "logs/celery.pid" ] && kill -0 $(cat logs/celery.pid) 2>/dev/null; then
        print_status "Celery: Running (PID: $(cat logs/celery.pid))"
    else
        print_warning "Celery: Not running"
    fi
    
    # Check SmartScrape server
    if port_in_use $DEFAULT_PORT; then
        print_status "SmartScrape Server: Running on port $DEFAULT_PORT"
    else
        print_warning "SmartScrape Server: Not running"
    fi
}

# Main execution
main() {
    local host=$DEFAULT_HOST
    local port=$DEFAULT_PORT
    local debug=false
    local stop=false
    local status=false
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --host)
                host="$2"
                shift 2
                ;;
            --port)
                port="$2"
                shift 2
                ;;
            --debug)
                debug=true
                shift
                ;;
            --stop)
                stop=true
                shift
                ;;
            --status)
                status=true
                shift
                ;;
            --help)
                show_usage
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # Handle stop command
    if [ "$stop" == "true" ]; then
        stop_services
        exit 0
    fi
    
    # Handle status command
    if [ "$status" == "true" ]; then
        show_status
        exit 0
    fi
    
    # Create logs directory if it doesn't exist
    mkdir -p logs
    
    print_header "Starting SmartScrape Services"
    print_status "Host: $host"
    print_status "Port: $port"
    print_status "Debug: $debug"
    echo ""
    
    # Start services in order
    if ! start_redis; then
        print_error "Failed to start Redis"
        exit 1
    fi
    
    if ! start_celery; then
        print_error "Failed to start Celery"
        exit 1
    fi
    
    # Start the server (this will block)
    if ! start_server $host $port $debug; then
        print_error "Failed to start server"
        exit 1
    fi
}

# Handle Ctrl+C gracefully
trap 'print_header "Shutting down..."; stop_services; exit 0' INT TERM

# Run main function
main "$@"
