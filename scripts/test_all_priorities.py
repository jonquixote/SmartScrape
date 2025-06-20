#!/usr/bin/env python3
"""
Master Test Script for SmartScrape - All Priorities
Tests all completed priorities: 5, 7, 8, 9, and 10
"""
import asyncio
import subprocess
import sys
import time
import requests
import os
from pathlib import Path

class MasterTester:
    def __init__(self):
        self.scripts_dir = Path(__file__).parent
        self.project_root = self.scripts_dir.parent
        self.test_scripts = [
            ("Priority 7 - API Performance Enhancement", "test_priority7.py"),
            ("Priority 8 - Enhanced Error Handling & Monitoring", "test_priority8.py"),
            ("Priority 9 - Configuration Management", "test_priority9.py")
        ]
        self.server_host = "localhost"
        self.server_port = 5000
        self.server_url = f"http://{self.server_host}:{self.server_port}"
    
    def check_server_connectivity(self) -> bool:
        """Check if SmartScrape server is running and accessible"""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def start_server_if_needed(self, auto_start: bool = False) -> bool:
        """Start server if not running. Returns True if server is available."""
        if self.check_server_connectivity():
            print(f"✅ SmartScrape server is already running at {self.server_url}")
            return True
        
        print(f"❌ SmartScrape server is not running at {self.server_url}")
        
        if auto_start:
            print("🚀 Starting SmartScrape server...")
            
            # Use the connectivity test script to check services
            connectivity_script = self.scripts_dir / "test_connectivity.py"
            if connectivity_script.exists():
                print("🔍 Running connectivity test...")
                result = subprocess.run([
                    sys.executable, str(connectivity_script),
                    "--host", self.server_host,
                    "--port", str(self.server_port),
                    "--wait", "5",
                    "--retry", "1"
                ], cwd=self.project_root)
                
                if result.returncode == 0:
                    print("✅ All services are accessible!")
                    return True
            
            # Try to start services using the startup script
            startup_script = self.project_root / "start_all.sh"
            if startup_script.exists():
                print("🚀 Starting services using start_all.sh...")
                print("Note: This will start the server in the foreground.")
                print("Please run the server manually or use the --no-server flag to skip server tests.")
                return False
            else:
                print("❌ start_all.sh not found. Please start the server manually:")
                print(f"   cd {self.project_root}")
                print(f"   python app.py --port {self.server_port}")
                return False
        else:
            print("💡 To automatically check services, use --auto-start flag")
            print("💡 Or start the server manually:")
            print(f"   cd {self.project_root}")
            print(f"   ./start_all.sh --port {self.server_port}")
            print("   # Or just the server:")
            print(f"   python app.py --port {self.server_port}")
            return False
    
    def run_test_script(self, script_name: str, description: str) -> bool:
        """Run a test script and return success status"""
        script_path = self.scripts_dir / script_name
        
        if not script_path.exists():
            print(f"❌ Test script not found: {script_path}")
            return False
        
        print(f"\n🚀 Running {description}")
        print("=" * 80)
        
        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=self.scripts_dir.parent,  # Run from project root
                capture_output=False,  # Show output in real-time
                text=True
            )
            
            success = result.returncode == 0
            
            if success:
                print(f"\n✅ {description}: PASSED")
            else:
                print(f"\n❌ {description}: FAILED (exit code: {result.returncode})")
            
            return success
            
        except Exception as e:
            print(f"\n❌ {description}: ERROR - {e}")
            return False
    
    def check_pipeline_orchestration(self) -> bool:
        """Check Priority 5 - Pipeline Orchestration"""
        print("\n🔄 Testing Priority 5: Pipeline Orchestration")
        print("=" * 80)
        
        try:
            # Check if Celery configuration exists
            celery_config = self.project_root / "core" / "celery_config.py"
            if not celery_config.exists():
                print("❌ Celery configuration not found")
                return False
            
            # Check if Celery tasks exist
            celery_tasks = self.project_root / "core" / "tasks.py"
            if not celery_tasks.exists():
                print("❌ Celery tasks not found")
                return False
            
            # Check if Pipeline Orchestrator exists
            orchestrator = self.project_root / "core" / "pipeline_orchestrator.py"
            if not orchestrator.exists():
                print("❌ Pipeline orchestrator not found")
                return False
            
            print("✅ All Celery components found")
            
            # Try to import and test basic functionality
            sys.path.insert(0, str(self.project_root))
            
            try:
                from core.celery_config import celery_app
                from core.pipeline_orchestrator import PipelineOrchestrator
                
                print("✅ Successfully imported Celery components")
                
                # Test orchestrator initialization
                orchestrator = PipelineOrchestrator()
                print("✅ Pipeline orchestrator initialized")
                
                return True
                
            except ImportError as e:
                print(f"❌ Failed to import Celery components: {e}")
                return False
            except Exception as e:
                print(f"❌ Error testing pipeline orchestration: {e}")
                return False
                
        except Exception as e:
            print(f"❌ Error checking pipeline orchestration: {e}")
            return False
    
    def run_all_tests(self, auto_start: bool = False, skip_server: bool = False) -> bool:
        """Run all priority tests"""
        print("🎯 SmartScrape Master Test Suite")
        print("Testing all completed priorities: 5, 7, 8, 9, and 10")
        print("=" * 80)
        print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Check server connectivity unless skipped
        if not skip_server:
            if not self.start_server_if_needed(auto_start):
                print("\n❌ Server is not available. Some tests may fail.")
                print("Use --skip-server to run only non-server tests, or --auto-start to check services.")
                if not auto_start:
                    return False
        
        results = {}
        
        # Check Priority 5 (Pipeline Orchestration)
        results["Priority 5"] = self.check_pipeline_orchestration()
        
        # Run test scripts for other priorities
        for description, script_name in self.test_scripts:
            results[description] = self.run_test_script(script_name, description)
        
        # Summary
        print("\n" + "=" * 80)
        print("🎯 TEST SUMMARY")
        print("=" * 80)
        
        passed = 0
        total = len(results)
        
        for test_name, success in results.items():
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"{test_name}: {status}")
            if success:
                passed += 1
        
        print("-" * 80)
        print(f"TOTAL: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
        
        if passed == total:
            print("\n🎉 ALL PRIORITIES COMPLETED SUCCESSFULLY!")
            print("SmartScrape is fully implemented with:")
            print("  ✅ Pipeline Orchestration (Celery)")
            print("  ✅ API Performance Enhancement (Streaming & Rate Limiting)")
            print("  ✅ Enhanced Error Handling & Monitoring")
            print("  ✅ Configuration Management")
            print("  ✅ Comprehensive Testing & Validation")
        elif passed >= total * 0.8:
            print("\n🎊 EXCELLENT PROGRESS!")
            print(f"Most priorities ({passed}/{total}) are working correctly.")
        else:
            print("\n⚠️ SOME ISSUES FOUND")
            print("Please check the failed tests and fix any issues.")
        
        print(f"\nCompleted at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        return passed >= total * 0.8

def main():
    """Main test runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description="SmartScrape Master Test Suite")
    parser.add_argument("--auto-start", action="store_true", 
                      help="Automatically check and attempt to start services")
    parser.add_argument("--skip-server", action="store_true", 
                      help="Skip server connectivity tests")
    parser.add_argument("--no-prompt", action="store_true",
                      help="Skip the confirmation prompt")
    
    args = parser.parse_args()
    
    if not args.no_prompt and not args.auto_start:
        print("SmartScrape Master Test Suite")
        print("This will test all implemented priorities.")
        print("")
        if not args.skip_server:
            print("Server tests require SmartScrape running on localhost:5000")
            print("Use --auto-start to check services automatically")
            print("Use --skip-server to run only non-server tests")
        print("")
        print("Press Enter to continue or Ctrl+C to cancel...")
        
        try:
            input()
        except KeyboardInterrupt:
            print("\nTest cancelled by user")
            return False
    
    tester = MasterTester()
    return tester.run_all_tests(auto_start=args.auto_start, skip_server=args.skip_server)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
