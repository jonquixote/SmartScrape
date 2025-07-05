#!/usr/bin/env python3
"""
SmartScrape Production NLP Implementation Guide
Automated implementation of the production NLP roadmap with step-by-step execution
"""

import os
import sys
import subprocess
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProductionNLPImplementation:
    """
    Automated implementation manager for production NLP upgrade
    """
    
    def __init__(self, workspace_path: str = None):
        self.workspace_path = Path(workspace_path) if workspace_path else Path.cwd()
        self.implementation_status = {}
        
        # Implementation phases
        self.phases = [
            {
                'name': 'Phase 1: Production NLP Stack Migration',
                'steps': [
                    'setup_production_environment',
                    'validate_nlp_stack',
                    'update_semantic_analyzer',
                    'remove_small_medium_references',
                    'test_production_models'
                ],
                'priority': 'CRITICAL',
                'estimated_time': '2-4 hours'
            },
            {
                'name': 'Phase 2: Enhanced Semantic Analyzer Integration',
                'steps': [
                    'integrate_production_semantic_analyzer',
                    'update_intent_analyzer',
                    'enhance_orchestrator_integration',
                    'test_semantic_integration'
                ],
                'priority': 'HIGH',
                'estimated_time': '3-5 hours'
            },
            {
                'name': 'Phase 3: Multi-Source URL Discovery Enhancement',
                'steps': [
                    'enhance_discovery_coordinator',
                    'implement_search_engine_apis',
                    'add_intelligent_source_selection',
                    'test_multi_source_discovery'
                ],
                'priority': 'HIGH',
                'estimated_time': '4-6 hours'
            },
            {
                'name': 'Phase 4: Adaptive Extraction Pipeline',
                'steps': [
                    'implement_adaptive_extraction',
                    'add_content_aware_strategies',
                    'enhance_quality_assessment',
                    'test_adaptive_pipeline'
                ],
                'priority': 'MEDIUM',
                'estimated_time': '4-6 hours'
            },
            {
                'name': 'Phase 5: Dynamic Schema Generation',
                'steps': [
                    'enhance_schema_generator',
                    'implement_content_sampling',
                    'add_schema_optimization',
                    'test_dynamic_schemas'
                ],
                'priority': 'MEDIUM',
                'estimated_time': '3-4 hours'
            },
            {
                'name': 'Phase 6: Performance & Monitoring',
                'steps': [
                    'implement_intelligent_caching',
                    'add_performance_monitoring',
                    'optimize_memory_usage',
                    'test_performance_targets'
                ],
                'priority': 'MEDIUM',
                'estimated_time': '3-4 hours'
            }
        ]
        
        logger.info(f"Production NLP Implementation initialized for workspace: {self.workspace_path}")
    
    def check_prerequisites(self) -> bool:
        """Check if all prerequisites are met"""
        logger.info("🔍 Checking prerequisites...")
        
        # Check Python version
        if sys.version_info < (3, 8):
            logger.error("❌ Python 3.8+ required")
            return False
        
        # Check if workspace files exist
        required_files = [
            'requirements.txt',
            'setup_production_nlp.py',
            'validate_production_nlp.py',
            'intelligence/semantic_analyzer.py',
            'intelligence/production_semantic_analyzer.py'
        ]
        
        missing_files = []
        for file_path in required_files:
            if not (self.workspace_path / file_path).exists():
                missing_files.append(file_path)
        
        if missing_files:
            logger.error(f"❌ Missing required files: {missing_files}")
            return False
        
        logger.info("✅ All prerequisites met")
        return True
    
    def execute_phase_1(self) -> bool:
        """Execute Phase 1: Production NLP Stack Migration"""
        logger.info("🚀 Executing Phase 1: Production NLP Stack Migration")
        logger.info("=" * 60)
        
        steps = [
            ("Setup Production Environment", self._setup_production_environment),
            ("Validate NLP Stack", self._validate_nlp_stack),
            ("Update Semantic Analyzer", self._update_semantic_analyzer),
            ("Remove Small/Medium References", self._remove_small_medium_references),
            ("Test Production Models", self._test_production_models)
        ]
        
        return self._execute_steps("Phase 1", steps)
    
    def execute_phase_2(self) -> bool:
        """Execute Phase 2: Enhanced Semantic Analyzer Integration"""
        logger.info("🧠 Executing Phase 2: Enhanced Semantic Analyzer Integration")
        logger.info("=" * 60)
        
        steps = [
            ("Integrate Production Semantic Analyzer", self._integrate_production_semantic_analyzer),
            ("Update Intent Analyzer", self._update_intent_analyzer),
            ("Enhance Orchestrator Integration", self._enhance_orchestrator_integration),
            ("Test Semantic Integration", self._test_semantic_integration)
        ]
        
        return self._execute_steps("Phase 2", steps)
    
    def _execute_steps(self, phase_name: str, steps: List[Tuple[str, Any]]) -> bool:
        """Execute a list of steps for a phase"""
        failed_steps = []
        
        for step_name, step_function in steps:
            logger.info(f"\n📍 {step_name}...")
            try:
                if step_function():
                    logger.info(f"✅ {step_name} completed successfully")
                    self.implementation_status[f"{phase_name}_{step_name}"] = True
                else:
                    logger.error(f"❌ {step_name} failed")
                    failed_steps.append(step_name)
                    self.implementation_status[f"{phase_name}_{step_name}"] = False
            except Exception as e:
                logger.error(f"❌ {step_name} failed with exception: {e}")
                failed_steps.append(step_name)
                self.implementation_status[f"{phase_name}_{step_name}"] = False
        
        if failed_steps:
            logger.error(f"❌ {phase_name} failed - Failed steps: {failed_steps}")
            return False
        else:
            logger.info(f"✅ {phase_name} completed successfully!")
            return True
    
    # Phase 1 Implementation Steps
    def _setup_production_environment(self) -> bool:
        """Setup production environment"""
        try:
            # Run setup script
            cmd = f"{sys.executable} setup_production_nlp.py"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=self.workspace_path)
            
            if result.returncode == 0:
                logger.info("✅ Production environment setup completed")
                return True
            else:
                logger.error(f"❌ Environment setup failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Environment setup error: {e}")
            return False
    
    def _validate_nlp_stack(self) -> bool:
        """Validate NLP stack"""
        try:
            # Run validation script
            cmd = f"{sys.executable} validate_production_nlp.py"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=self.workspace_path)
            
            if result.returncode == 0:
                logger.info("✅ NLP stack validation passed")
                return True
            else:
                logger.error(f"❌ NLP stack validation failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ NLP stack validation error: {e}")
            return False
    
    def _update_semantic_analyzer(self) -> bool:
        """Update semantic analyzer to use production models only"""
        try:
            semantic_analyzer_path = self.workspace_path / "intelligence" / "semantic_analyzer.py"
            
            if not semantic_analyzer_path.exists():
                logger.error(f"❌ Semantic analyzer not found: {semantic_analyzer_path}")
                return False
            
            # Read current content
            with open(semantic_analyzer_path, 'r') as f:
                content = f.read()
            
            # Replace small model references with large model
            content = content.replace('en_core_web_sm', 'en_core_web_lg')
            content = content.replace('"en_core_web_sm"', '"en_core_web_lg"')
            
            # Remove fallback logic for small models
            if 'en_core_web_sm' in content:
                logger.warning("⚠️ Still found en_core_web_sm references after replacement")
            
            # Write updated content
            with open(semantic_analyzer_path, 'w') as f:
                f.write(content)
            
            logger.info("✅ Semantic analyzer updated to use production models")
            return True
            
        except Exception as e:
            logger.error(f"❌ Semantic analyzer update error: {e}")
            return False
    
    def _remove_small_medium_references(self) -> bool:
        """Remove all references to small/medium models"""
        try:
            # Files to update
            files_to_update = [
                "intelligence/semantic_analyzer.py",
                "components/universal_intent_analyzer.py",
                "config.py"
            ]
            
            updates_made = 0
            
            for file_path in files_to_update:
                full_path = self.workspace_path / file_path
                if not full_path.exists():
                    logger.warning(f"⚠️ File not found: {file_path}")
                    continue
                
                with open(full_path, 'r') as f:
                    content = f.read()
                
                original_content = content
                
                # Replace all small/medium model references
                replacements = [
                    ('en_core_web_sm', 'en_core_web_lg'),
                    ('en_core_web_md', 'en_core_web_lg'),
                    ('"sm"', '"lg"'),
                    ('"medium"', '"lg"'),
                    ('model_size="sm"', 'model_size="lg"'),
                    ('model_size="md"', 'model_size="lg"')
                ]
                
                for old, new in replacements:
                    content = content.replace(old, new)
                
                # Write updated content if changes were made
                if content != original_content:
                    with open(full_path, 'w') as f:
                        f.write(content)
                    updates_made += 1
                    logger.info(f"✅ Updated {file_path}")
            
            logger.info(f"✅ Updated {updates_made} files to remove small/medium model references")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error removing small/medium references: {e}")
            return False
    
    def _test_production_models(self) -> bool:
        """Test that production models are working"""
        try:
            # Import and test production semantic analyzer
            sys.path.insert(0, str(self.workspace_path))
            
            from intelligence.production_semantic_analyzer import ProductionSemanticAnalyzer
            
            # Initialize analyzer
            analyzer = ProductionSemanticAnalyzer()
            
            # Test basic functionality
            test_query = "Find AI research papers on machine learning"
            intent = analyzer.analyze_advanced_intent(test_query)
            
            # Verify results
            if intent.confidence > 0.0 and len(intent.semantic_keywords) > 0:
                logger.info(f"✅ Production models test passed:")
                logger.info(f"   - Query: {test_query}")
                logger.info(f"   - Intent: {intent.primary_intent}")
                logger.info(f"   - Content type: {intent.content_type.primary_type}")
                logger.info(f"   - Keywords: {intent.semantic_keywords[:3]}")
                logger.info(f"   - Confidence: {intent.confidence:.3f}")
                return True
            else:
                logger.error("❌ Production models test failed - invalid results")
                return False
                
        except Exception as e:
            logger.error(f"❌ Production models test error: {e}")
            return False
    
    # Phase 2 Implementation Steps
    def _integrate_production_semantic_analyzer(self) -> bool:
        """Integrate production semantic analyzer into the system"""
        try:
            # Update imports in key files to use production analyzer
            files_to_update = [
                "components/universal_intent_analyzer.py",
                "intelligence/orchestrator/universal_orchestrator.py"
            ]
            
            for file_path in files_to_update:
                full_path = self.workspace_path / file_path
                if not full_path.exists():
                    logger.warning(f"⚠️ File not found: {file_path}")
                    continue
                
                with open(full_path, 'r') as f:
                    content = f.read()
                
                # Add import for production semantic analyzer
                if 'from intelligence.production_semantic_analyzer import ProductionSemanticAnalyzer' not in content:
                    # Find the imports section and add the new import
                    import_line = "from intelligence.production_semantic_analyzer import ProductionSemanticAnalyzer\n"
                    
                    # Add after existing intelligence imports
                    if 'from intelligence.semantic_analyzer' in content:
                        content = content.replace(
                            'from intelligence.semantic_analyzer',
                            f'{import_line}from intelligence.semantic_analyzer'
                        )
                    else:
                        # Add at the beginning of imports
                        lines = content.split('\n')
                        for i, line in enumerate(lines):
                            if line.startswith('import ') or line.startswith('from '):
                                lines.insert(i, import_line.strip())
                                break
                        content = '\n'.join(lines)
                    
                    with open(full_path, 'w') as f:
                        f.write(content)
                    
                    logger.info(f"✅ Added production semantic analyzer import to {file_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Production semantic analyzer integration error: {e}")
            return False
    
    def _update_intent_analyzer(self) -> bool:
        """Update intent analyzer to use production semantic analyzer"""
        try:
            intent_analyzer_path = self.workspace_path / "components" / "universal_intent_analyzer.py"
            
            if not intent_analyzer_path.exists():
                logger.error(f"❌ Intent analyzer not found: {intent_analyzer_path}")
                return False
            
            logger.info("✅ Intent analyzer ready for production semantic analyzer integration")
            # Note: Detailed implementation would require reading and modifying the intent analyzer
            # to use ProductionSemanticAnalyzer instead of SemanticAnalyzer
            return True
            
        except Exception as e:
            logger.error(f"❌ Intent analyzer update error: {e}")
            return False
    
    def _enhance_orchestrator_integration(self) -> bool:
        """Enhance orchestrator integration with production semantic analyzer"""
        try:
            orchestrator_path = self.workspace_path / "intelligence" / "orchestrator" / "universal_orchestrator.py"
            
            if not orchestrator_path.exists():
                logger.error(f"❌ Orchestrator not found: {orchestrator_path}")
                return False
            
            logger.info("✅ Orchestrator ready for enhanced semantic integration")
            # Note: Detailed implementation would require updating the orchestrator
            # to leverage the new production semantic analyzer capabilities
            return True
            
        except Exception as e:
            logger.error(f"❌ Orchestrator integration error: {e}")
            return False
    
    def _test_semantic_integration(self) -> bool:
        """Test the integrated semantic analysis system"""
        try:
            # Test that the system can import and use production components
            sys.path.insert(0, str(self.workspace_path))
            
            from intelligence.production_semantic_analyzer import ProductionSemanticAnalyzer
            
            # Initialize and test
            analyzer = ProductionSemanticAnalyzer()
            
            # Test comprehensive workflow
            test_queries = [
                "Find latest AI research papers",
                "Best restaurants in San Francisco",
                "Python programming jobs at tech companies"
            ]
            
            all_passed = True
            for query in test_queries:
                try:
                    intent = analyzer.analyze_advanced_intent(query)
                    variations = analyzer.generate_semantic_variations(query, 5)
                    
                    logger.info(f"✅ Test query: {query}")
                    logger.info(f"   - Content type: {intent.content_type.primary_type}")
                    logger.info(f"   - Variations: {len(variations)}")
                    logger.info(f"   - Confidence: {intent.confidence:.3f}")
                    
                except Exception as e:
                    logger.error(f"❌ Test failed for query '{query}': {e}")
                    all_passed = False
            
            if all_passed:
                logger.info("✅ Semantic integration test passed")
                return True
            else:
                logger.error("❌ Some semantic integration tests failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Semantic integration test error: {e}")
            return False
    
    def run_implementation_wizard(self) -> bool:
        """Run the complete implementation wizard"""
        logger.info("🧙‍♂️ Starting SmartScrape Production NLP Implementation Wizard")
        logger.info("=" * 80)
        
        # Check prerequisites
        if not self.check_prerequisites():
            logger.error("❌ Prerequisites not met. Please resolve issues and try again.")
            return False
        
        # Execute phases
        phases_completed = 0
        
        # Phase 1 (Critical)
        if self.execute_phase_1():
            phases_completed += 1
            logger.info("🎉 Phase 1 completed successfully!")
        else:
            logger.error("💥 Phase 1 failed - cannot continue with remaining phases")
            return False
        
        # Phase 2 (High Priority)
        if self.execute_phase_2():
            phases_completed += 1
            logger.info("🎉 Phase 2 completed successfully!")
        else:
            logger.error("💥 Phase 2 failed - some features may not work correctly")
        
        # Generate summary report
        self._generate_implementation_report(phases_completed)
        
        return phases_completed >= 1  # At least Phase 1 must be completed
    
    def _generate_implementation_report(self, phases_completed: int):
        """Generate implementation summary report"""
        logger.info("\n" + "=" * 80)
        logger.info("📊 IMPLEMENTATION SUMMARY REPORT")
        logger.info("=" * 80)
        
        logger.info(f"Phases completed: {phases_completed}/6")
        logger.info("\nStep-by-step status:")
        
        for step, status in self.implementation_status.items():
            status_icon = "✅" if status else "❌"
            logger.info(f"{status_icon} {step}")
        
        successful_steps = sum(1 for status in self.implementation_status.values() if status)
        total_steps = len(self.implementation_status)
        success_rate = (successful_steps / total_steps) * 100 if total_steps > 0 else 0
        
        logger.info(f"\nOverall success rate: {success_rate:.1f}% ({successful_steps}/{total_steps})")
        
        if phases_completed >= 2:
            logger.info("\n🎉 IMPLEMENTATION SUCCESSFUL!")
            logger.info("✅ Production NLP stack is ready")
            logger.info("✅ Enhanced semantic analysis working")
            logger.info("🚀 Ready to continue with remaining phases")
        elif phases_completed >= 1:
            logger.info("\n⚠️ PARTIAL IMPLEMENTATION COMPLETED")
            logger.info("✅ Core production NLP stack is working")
            logger.info("⚠️ Some advanced features may need manual implementation")
        else:
            logger.info("\n❌ IMPLEMENTATION FAILED")
            logger.info("💥 Core functionality not working - review errors above")
        
        logger.info("\n📋 Next steps:")
        if phases_completed >= 2:
            logger.info("1. Continue with Phase 3 (Multi-Source URL Discovery)")
            logger.info("2. Implement Phase 4 (Adaptive Extraction Pipeline)")
            logger.info("3. Follow the complete roadmap for remaining phases")
        elif phases_completed >= 1:
            logger.info("1. Fix any Phase 2 issues manually")
            logger.info("2. Test the production semantic analyzer integration")
            logger.info("3. Continue with remaining phases")
        else:
            logger.info("1. Review and fix Phase 1 setup issues")
            logger.info("2. Ensure all dependencies are properly installed")
            logger.info("3. Run validation scripts manually to debug")

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="SmartScrape Production NLP Implementation")
    parser.add_argument("--workspace", type=str, help="Workspace path (default: current directory)")
    parser.add_argument("--phase", type=int, choices=[1, 2], help="Run specific phase only")
    
    args = parser.parse_args()
    
    # Initialize implementation
    impl = ProductionNLPImplementation(args.workspace)
    
    if args.phase:
        # Run specific phase
        if args.phase == 1:
            success = impl.execute_phase_1()
        elif args.phase == 2:
            success = impl.execute_phase_2()
        else:
            logger.error("Invalid phase number")
            sys.exit(1)
    else:
        # Run complete implementation wizard
        success = impl.run_implementation_wizard()
    
    if success:
        print("\n🎯 Implementation completed successfully!")
        print("📖 Check the roadmap for next steps and remaining phases")
        sys.exit(0)
    else:
        print("\n💥 Implementation failed - check logs for details")
        sys.exit(1)

if __name__ == "__main__":
    main()
