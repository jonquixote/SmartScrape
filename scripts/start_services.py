#!/usr/bin/env python3
"""
SmartScrape Services Startup Script
Starts Redis, Database, and Web Server in the correct order
"""

import os
import sys
import time
import subprocess
import signal
import psutil
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def check_redis_running():
    """Check if Redis is running"""
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        r.ping()
        return True
    except:
        return False

def start_redis():
    """Start Redis server"""
    if check_redis_running():
        print("✅ Redis is already running")
        return None
    
    print("🚀 Starting Redis server...")
    try:
        # Try Docker first
        result = subprocess.run(['docker', 'run', '-d', '--name', 'smartscrape-redis', 
                               '-p', '6379:6379', 'redis:alpine'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Redis started via Docker")
            time.sleep(2)
            return 'docker'
    except:
        pass
    
    try:
        # Try local Redis
        redis_process = subprocess.Popen(['redis-server', '--daemonize', 'yes'])
        time.sleep(2)
        if check_redis_running():
            print("✅ Redis started locally")
            return redis_process
    except:
        pass
    
    try:
        # Try brew services (macOS)
        result = subprocess.run(['brew', 'services', 'start', 'redis'], 
                              capture_output=True, text=True)
        time.sleep(3)
        if check_redis_running():
            print("✅ Redis started via Homebrew")
            return 'brew'
    except:
        pass
    
    print("❌ Failed to start Redis. Please install Redis manually:")
    print("   brew install redis && brew services start redis")
    print("   OR")
    print("   docker run -d -p 6379:6379 redis:alpine")
    return None

def start_celery():
    """Start Celery worker"""
    print("🚀 Starting Celery worker...")
    try:
        celery_process = subprocess.Popen([
            sys.executable, '-m', 'celery', '-A', 'core.celery_config', 'worker', 
            '--loglevel=info', '--detach'
        ], cwd=project_root)
        time.sleep(2)
        print("✅ Celery worker started")
        return celery_process
    except Exception as e:
        print(f"⚠️ Celery worker failed to start: {e}")
        print("   This is optional - main server will still work")
        return None

def start_web_server(port=5000):
    """Start the web server"""
    print(f"🚀 Starting SmartScrape web server on port {port}...")
    try:
        server_process = subprocess.Popen([
            sys.executable, 'app.py', '--port', str(port)
        ], cwd=project_root)
        time.sleep(3)
        print(f"✅ SmartScrape server started on http://localhost:{port}")
        return server_process
    except Exception as e:
        print(f"❌ Failed to start web server: {e}")
        return None

def cleanup_services():
    """Cleanup services on exit"""
    print("\n🧹 Cleaning up services...")
    
    # Stop any running processes
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = ' '.join(proc.info['cmdline'] or [])
            if 'smartscrape' in cmdline.lower() or 'celery' in cmdline.lower():
                proc.terminate()
        except:
            pass
    
    # Stop Docker Redis if we started it
    try:
        subprocess.run(['docker', 'stop', 'smartscrape-redis'], 
                      capture_output=True)
        subprocess.run(['docker', 'rm', 'smartscrape-redis'], 
                      capture_output=True)
    except:
        pass
    
    print("✅ Cleanup complete")

def main():
    """Main startup function"""
    print("🚀 SmartScrape Services Startup")
    print("=" * 50)
    
    # Register cleanup handler
    def signal_handler(sig, frame):
        cleanup_services()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start services in order
        redis_proc = start_redis()
        if not redis_proc:
            print("❌ Cannot start without Redis")
            return 1
        
        celery_proc = start_celery()
        server_proc = start_web_server(port=5000)
        
        if not server_proc:
            print("❌ Failed to start web server")
            return 1
        
        print("\n" + "=" * 50)
        print("🎉 All services started successfully!")
        print("🌐 SmartScrape API: http://localhost:5000")
        print("📊 Health Check: http://localhost:5000/health")
        print("📈 Metrics: http://localhost:5000/metrics/performance")
        print("⚙️ Configuration: http://localhost:5000/config/current")
        print("\nPress Ctrl+C to stop all services")
        print("=" * 50)
        
        # Wait for user interrupt
        try:
            server_proc.wait()
        except KeyboardInterrupt:
            pass
        
    except Exception as e:
        print(f"❌ Startup failed: {e}")
        return 1
    
    finally:
        cleanup_services()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
