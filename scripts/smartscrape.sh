#!/bin/bash

# SmartScrape Master Script
# This script provides a unified interface to manage the entire SmartScrape system

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SERVICES_SCRIPT="$SCRIPT_DIR/start_services.sh"
HEALTH_SCRIPT="$SCRIPT_DIR/health_check.py"
INTEGRATION_SCRIPT="$SCRIPT_DIR/integration_test.py"

# Default settings
DEFAULT_PORT=5000
DEFAULT_HOST="0.0.0.0"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Logging functions
log() {
    echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

info() {
    echo -e "${PURPLE}[INFO]${NC} $1"
}

# Function to show banner
show_banner() {
    echo -e "${BLUE}"
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║                     SmartScrape Master                       ║"
    echo "║               Intelligent Web Scraping System                ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Function to check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    local missing_deps=()
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        missing_deps+=("python3")
    fi
    
    # Check Redis
    if ! command -v redis-server &> /dev/null && ! command -v redis-cli &> /dev/null; then
        missing_deps+=("redis")
    fi
    
    # Check required Python packages
    if ! python3 -c "import fastapi, uvicorn, celery, redis" &> /dev/null; then
        warning "Some Python dependencies may be missing. Run: pip install -r requirements.txt"
    fi
    
    if [ ${#missing_deps[@]} -gt 0 ]; then
        error "Missing dependencies: ${missing_deps[*]}"
        echo ""
        echo "Please install missing dependencies:"
        for dep in "${missing_deps[@]}"; do
            case $dep in
                "python3")
                    echo "  - Install Python 3: https://python.org/downloads/"
                    ;;
                "redis")
                    echo "  - Install Redis:"
                    echo "    macOS: brew install redis"
                    echo "    Ubuntu: sudo apt-get install redis-server"
                    ;;
            esac
        done
        return 1
    else
        success "All prerequisites are available"
        return 0
    fi
}

# Function to install/update dependencies
install_dependencies() {
    log "Installing/updating Python dependencies..."
    cd "$PROJECT_ROOT"
    
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
        success "Dependencies installed"
    else
        warning "requirements.txt not found"
    fi
}

# Function to start all services
start_system() {
    local port=${1:-$DEFAULT_PORT}
    local host=${2:-$DEFAULT_HOST}
    
    log "Starting SmartScrape system on $host:$port..."
    
    # Make scripts executable
    chmod +x "$SERVICES_SCRIPT"
    
    # Start services
    "$SERVICES_SCRIPT" start "$port" "$host"
}

# Function to stop all services
stop_system() {
    log "Stopping SmartScrape system..."
    
    if [ -x "$SERVICES_SCRIPT" ]; then
        "$SERVICES_SCRIPT" stop
    else
        warning "Services script not executable, attempting manual cleanup..."
        pkill -f "celery worker" 2>/dev/null || true
        pkill -f "python.*app.py" 2>/dev/null || true
        if command -v brew &> /dev/null; then
            brew services stop redis 2>/dev/null || true
        fi
    fi
    
    success "System stopped"
}

# Function to restart the system
restart_system() {
    local port=${1:-$DEFAULT_PORT}
    local host=${2:-$DEFAULT_HOST}
    
    log "Restarting SmartScrape system..."
    stop_system
    sleep 3
    start_system "$port" "$host"
}

# Function to show system status
show_status() {
    log "Checking system status..."
    
    if [ -x "$SERVICES_SCRIPT" ]; then
        "$SERVICES_SCRIPT" status
    else
        warning "Services script not available, showing basic status..."
        
        # Check Redis
        if redis-cli ping &> /dev/null; then
            success "✅ Redis: Running"
        else
            error "❌ Redis: Not running"
        fi
        
        # Check processes
        if pgrep -f "celery worker" &> /dev/null; then
            success "✅ Celery: Running"
        else
            error "❌ Celery: Not running"
        fi
        
        if pgrep -f "python.*app.py" &> /dev/null; then
            success "✅ FastAPI: Running"
        else
            error "❌ FastAPI: Not running"
        fi
    fi
}

# Function to run health checks
run_health_check() {
    local port=${1:-$DEFAULT_PORT}
    
    log "Running comprehensive health check..."
    
    if [ -f "$HEALTH_SCRIPT" ]; then
        python3 "$HEALTH_SCRIPT" --fastapi-url "http://localhost:$port" --exit-code
    else
        warning "Health check script not found, running basic check..."
        
        if curl -s "http://localhost:$port/health" > /dev/null; then
            success "Basic health check passed"
        else
            error "Basic health check failed"
            return 1
        fi
    fi
}

# Function to run integration tests
run_integration_tests() {
    local port=${1:-$DEFAULT_PORT}
    
    log "Running integration tests..."
    
    if [ -f "$INTEGRATION_SCRIPT" ]; then
        python3 "$INTEGRATION_SCRIPT" \
            --base-url "http://localhost:$port" \
            --wait-for-server \
            --max-wait 60
    else
        warning "Integration test script not found, skipping..."
        return 0
    fi
}

# Function to run all tests
run_all_tests() {
    local port=${1:-$DEFAULT_PORT}
    
    log "Running all test suites..."
    
    # Run health check
    if run_health_check "$port"; then
        success "Health check passed"
    else
        error "Health check failed"
        return 1
    fi
    
    # Run integration tests
    if run_integration_tests "$port"; then
        success "Integration tests passed"
    else
        warning "Some integration tests failed"
    fi
    
    success "All tests completed"
}

# Function to setup development environment
setup_dev() {
    log "Setting up development environment..."
    
    cd "$PROJECT_ROOT"
    
    # Install dependencies
    install_dependencies
    
    # Create necessary directories
    mkdir -p logs cache data
    
    # Make scripts executable
    find scripts -name "*.sh" -exec chmod +x {} \;
    
    success "Development environment setup complete"
}

# Function to deploy for production
deploy_production() {
    local port=${1:-$DEFAULT_PORT}
    local host=${2:-"0.0.0.0"}
    
    log "Deploying for production..."
    
    # Set environment
    export ENVIRONMENT=production
    
    # Install dependencies
    install_dependencies
    
    # Start system
    start_system "$port" "$host"
    
    # Wait for system to be ready
    sleep 10
    
    # Run health check
    if run_health_check "$port"; then
        success "Production deployment successful"
        info "System is running at http://$host:$port"
    else
        error "Production deployment failed health check"
        return 1
    fi
}

# Function to show logs
show_logs() {
    local service=${1:-"all"}
    
    case "$service" in
        "fastapi"|"api")
            if [ -f "services/fastapi.log" ]; then
                tail -f services/fastapi.log
            else
                warning "FastAPI log not found"
            fi
            ;;
        "celery")
            if [ -f "logs/celery_worker.log" ]; then
                tail -f logs/celery_worker.log
            else
                warning "Celery log not found"
            fi
            ;;
        "all"|*)
            log "Showing all available logs..."
            if [ -f "services/fastapi.log" ]; then
                echo "=== FastAPI Logs ==="
                tail -20 services/fastapi.log
            fi
            if [ -f "logs/celery_worker.log" ]; then
                echo "=== Celery Logs ==="
                tail -20 logs/celery_worker.log
            fi
            ;;
    esac
}

# Function to show help
show_help() {
    echo "SmartScrape Master Script"
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  start [port] [host]    Start the SmartScrape system (default: port 5000, host 0.0.0.0)"
    echo "  stop                   Stop all services"
    echo "  restart [port] [host]  Restart the system"
    echo "  status                 Show service status"
    echo "  health [port]          Run health checks"
    echo "  test [port]            Run integration tests"
    echo "  test-all [port]        Run all tests (health + integration)"
    echo "  setup-dev              Setup development environment"
    echo "  deploy [port] [host]   Deploy for production"
    echo "  logs [service]         Show logs (service: fastapi, celery, all)"
    echo "  install-deps           Install/update dependencies"
    echo "  check-prereq           Check prerequisites"
    echo "  help                   Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 start                      # Start on default port 5000"
    echo "  $0 start 8080                 # Start on port 8080"
    echo "  $0 start 8080 127.0.0.1       # Start on port 8080, localhost only"
    echo "  $0 test                       # Run tests against port 5000"
    echo "  $0 deploy 80 0.0.0.0          # Deploy for production on port 80"
    echo "  $0 logs fastapi               # Show FastAPI logs"
    echo ""
}

# Main script logic
main() {
    # Change to project root
    cd "$PROJECT_ROOT"
    
    case "${1:-help}" in
        "start")
            show_banner
            check_prerequisites && start_system "${2:-$DEFAULT_PORT}" "${3:-$DEFAULT_HOST}"
            ;;
        "stop")
            stop_system
            ;;
        "restart")
            show_banner
            restart_system "${2:-$DEFAULT_PORT}" "${3:-$DEFAULT_HOST}"
            ;;
        "status")
            show_status
            ;;
        "health")
            run_health_check "${2:-$DEFAULT_PORT}"
            ;;
        "test")
            run_integration_tests "${2:-$DEFAULT_PORT}"
            ;;
        "test-all")
            run_all_tests "${2:-$DEFAULT_PORT}"
            ;;
        "setup-dev")
            show_banner
            setup_dev
            ;;
        "deploy")
            show_banner
            check_prerequisites && deploy_production "${2:-$DEFAULT_PORT}" "${3:-$DEFAULT_HOST}"
            ;;
        "logs")
            show_logs "$2"
            ;;
        "install-deps")
            install_dependencies
            ;;
        "check-prereq")
            check_prerequisites
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        *)
            error "Unknown command: $1"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
