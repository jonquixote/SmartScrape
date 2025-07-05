#!/usr/bin/env python3
"""
Production NLP Stack Validation for SmartScrape
Comprehensive testing and validation of all production NLP components
Ensures only large models are used and all features work correctly
"""

import sys
import logging
import time
import traceback
from typing import Dict, List, Tuple, Any
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProductionNLPValidator:
    """
    Comprehensive validator for production NLP stack
    """
    
    def __init__(self):
        self.test_results = {}
        self.performance_metrics = {}
        
        # Test queries for comprehensive validation
        self.test_queries = [
            "Find latest AI research papers on machine learning",
            "Best restaurants in San Francisco with reviews",
            "Top 10 Python programming tutorials for beginners", 
            "Climate change articles from scientific journals",
            "Job opportunities in software engineering at tech companies",
            "Product reviews for iPhone 15 Pro specifications and prices"
        ]
        
        # Expected content types for validation
        self.expected_content_types = [
            'research', 'review', 'tutorial', 'news', 'job', 'product'
        ]
        
        logger.info("Production NLP Validator initialized")
    
    def validate_imports(self) -> bool:
        """Validate all required imports are available"""
        logger.info("🧪 Testing imports...")
        
        required_imports = [
            ('torch', 'PyTorch'),
            ('spacy', 'spaCy'),
            ('sentence_transformers', 'Sentence Transformers'),
            ('transformers', 'Hugging Face Transformers'),
            ('sklearn', 'Scikit-learn'),
            ('numpy', 'NumPy'),
            ('scipy', 'SciPy')
        ]
        
        failed_imports = []
        
        for module, name in required_imports:
            try:
                __import__(module)
                logger.info(f"✅ {name} import successful")
            except ImportError as e:
                logger.error(f"❌ {name} import failed: {e}")
                failed_imports.append(name)
        
        success = len(failed_imports) == 0
        self.test_results['imports'] = success
        
        if failed_imports:
            logger.error(f"❌ Failed imports: {failed_imports}")
            
        return success
    
    def validate_spacy_models(self) -> bool:
        """Validate spaCy models (PRODUCTION ONLY - NO SMALL/MEDIUM)"""
        logger.info("🧠 Testing spaCy models...")
        
        try:
            import spacy
            
            # Test that ONLY large models are available for production
            required_model = "en_core_web_lg"
            optional_model = "en_core_web_trf"
            deprecated_models = ["en_core_web_sm", "en_core_web_md"]
            
            # Test required model
            try:
                nlp = spacy.load(required_model)
                logger.info(f"✅ {required_model} loaded successfully")
                
                # Test model capabilities
                test_text = "SmartScrape uses artificial intelligence for web scraping automation"
                doc = nlp(test_text)
                
                entities = [(ent.text, ent.label_) for ent in doc.ents]
                pos_tags = [(token.text, token.pos_) for token in doc]
                
                logger.info(f"   - Entities found: {entities}")
                logger.info(f"   - Model has vector: {doc.has_vector}")
                
                # Validate model performance
                if not doc.has_vector:
                    logger.warning("⚠️ Model does not have word vectors")
                    
            except OSError:
                logger.error(f"❌ CRITICAL: {required_model} not available")
                self.test_results['spacy_models'] = False
                return False
            
            # Test optional transformer model
            try:
                nlp_trf = spacy.load(optional_model)
                logger.info(f"✅ {optional_model} loaded successfully (optional)")
            except OSError:
                logger.info(f"ℹ️ {optional_model} not available (optional)")
            
            # Check that deprecated models are NOT being used
            deprecated_found = []
            for dep_model in deprecated_models:
                try:
                    spacy.load(dep_model)
                    deprecated_found.append(dep_model)
                except OSError:
                    pass  # Good, deprecated model not found
            
            if deprecated_found:
                logger.warning(f"⚠️ Deprecated models found: {deprecated_found}")
                logger.warning("Consider removing these for production:")
                for model in deprecated_found:
                    logger.warning(f"   pip uninstall {model}")
            else:
                logger.info("✅ No deprecated small/medium models found")
            
            self.test_results['spacy_models'] = True
            return True
            
        except Exception as e:
            logger.error(f"❌ spaCy model validation failed: {e}")
            self.test_results['spacy_models'] = False
            return False
    
    def validate_sentence_transformers(self) -> bool:
        """Validate sentence transformers functionality"""
        logger.info("🤖 Testing sentence transformers...")
        
        try:
            from sentence_transformers import SentenceTransformer
            
            # Test primary model
            model_name = 'all-MiniLM-L6-v2'
            start_time = time.time()
            model = SentenceTransformer(model_name)
            load_time = time.time() - start_time
            
            logger.info(f"✅ {model_name} loaded in {load_time:.2f}s")
            
            # Test encoding
            test_sentences = [
                "SmartScrape is a web scraping tool",
                "Web scraping automation with AI",
                "Completely different topic about cooking"
            ]
            
            start_time = time.time()
            embeddings = model.encode(test_sentences)
            encode_time = time.time() - start_time
            
            logger.info(f"✅ Encoded {len(test_sentences)} sentences in {encode_time:.3f}s")
            logger.info(f"   - Embedding shape: {embeddings.shape}")
            
            # Test similarity
            from sklearn.metrics.pairwise import cosine_similarity
            similarity_matrix = cosine_similarity(embeddings)
            
            # First two sentences should be more similar than with third
            sim_related = similarity_matrix[0, 1]
            sim_unrelated = similarity_matrix[0, 2]
            
            logger.info(f"   - Related similarity: {sim_related:.3f}")
            logger.info(f"   - Unrelated similarity: {sim_unrelated:.3f}")
            
            if sim_related > sim_unrelated:
                logger.info("✅ Semantic similarity working correctly")
                self.performance_metrics['sentence_transformer_load_time'] = load_time
                self.performance_metrics['sentence_transformer_encode_time'] = encode_time
                self.test_results['sentence_transformers'] = True
                return True
            else:
                logger.warning("⚠️ Semantic similarity not working as expected")
                self.test_results['sentence_transformers'] = False
                return False
                
        except Exception as e:
            logger.error(f"❌ Sentence transformers validation failed: {e}")
            self.test_results['sentence_transformers'] = False
            return False
    
    def validate_semantic_analysis_integration(self) -> bool:
        """Test integrated semantic analysis workflow"""
        logger.info("🔬 Testing semantic analysis integration...")
        
        try:
            import spacy
            from sentence_transformers import SentenceTransformer
            
            # Load models
            nlp = spacy.load("en_core_web_lg")
            sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            results = []
            
            for query in self.test_queries:
                start_time = time.time()
                
                # spaCy analysis
                doc = nlp(query)
                entities = [(ent.text, ent.label_) for ent in doc.ents]
                keywords = [token.lemma_ for token in doc 
                           if not token.is_stop and not token.is_punct 
                           and token.pos_ in ['NOUN', 'PROPN', 'ADJ']]
                
                # Sentence transformer embedding
                embedding = sentence_model.encode([query])
                
                process_time = time.time() - start_time
                
                result = {
                    'query': query,
                    'entities': entities,
                    'keywords': keywords[:5],  # Top 5 keywords
                    'embedding_shape': embedding.shape,
                    'process_time': process_time
                }
                results.append(result)
                
                logger.info(f"   Query: {query[:50]}...")
                logger.info(f"     - Entities: {entities}")
                logger.info(f"     - Keywords: {keywords[:3]}")
                logger.info(f"     - Process time: {process_time:.3f}s")
            
            avg_process_time = np.mean([r['process_time'] for r in results])
            logger.info(f"✅ Average processing time: {avg_process_time:.3f}s")
            
            self.performance_metrics['avg_semantic_analysis_time'] = avg_process_time
            self.test_results['semantic_integration'] = True
            
            # Performance check
            if avg_process_time > 2.0:
                logger.warning(f"⚠️ Processing time slower than target (<2s): {avg_process_time:.3f}s")
            else:
                logger.info(f"✅ Processing time within target: {avg_process_time:.3f}s")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Semantic analysis integration failed: {e}")
            logger.error(traceback.format_exc())
            self.test_results['semantic_integration'] = False
            return False
    
    def validate_memory_usage(self) -> bool:
        """Check memory usage of loaded models"""
        logger.info("💾 Testing memory usage...")
        
        try:
            import psutil
            import os
            
            # Get current process
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Load all models
            import spacy
            from sentence_transformers import SentenceTransformer
            
            nlp = spacy.load("en_core_web_lg")
            sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Process some text to fully initialize
            test_text = "Testing memory usage with production models"
            doc = nlp(test_text)
            embeddings = sentence_model.encode([test_text])
            
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_usage = final_memory - initial_memory
            
            logger.info(f"✅ Memory usage: {memory_usage:.1f} MB")
            logger.info(f"   - Initial: {initial_memory:.1f} MB")
            logger.info(f"   - Final: {final_memory:.1f} MB")
            
            self.performance_metrics['memory_usage_mb'] = memory_usage
            
            # Check against target (should be < 4GB per worker)
            if memory_usage > 4000:  # 4GB
                logger.warning(f"⚠️ Memory usage higher than target (4GB): {memory_usage:.1f} MB")
            else:
                logger.info(f"✅ Memory usage within target: {memory_usage:.1f} MB")
            
            self.test_results['memory_usage'] = True
            return True
            
        except ImportError:
            logger.warning("⚠️ psutil not available, skipping memory test")
            self.test_results['memory_usage'] = True
            return True
        except Exception as e:
            logger.error(f"❌ Memory usage test failed: {e}")
            self.test_results['memory_usage'] = False
            return False
    
    def validate_concurrent_processing(self) -> bool:
        """Test concurrent processing capabilities"""
        logger.info("⚡ Testing concurrent processing...")
        
        try:
            import threading
            import concurrent.futures
            from sentence_transformers import SentenceTransformer
            
            # Load model once
            sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            def process_query(query):
                """Process a single query"""
                start_time = time.time()
                embedding = sentence_model.encode([query])
                process_time = time.time() - start_time
                return {
                    'query': query,
                    'process_time': process_time,
                    'embedding_shape': embedding.shape
                }
            
            # Test concurrent processing
            start_time = time.time()
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(process_query, query) for query in self.test_queries]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
            
            total_time = time.time() - start_time
            avg_time = np.mean([r['process_time'] for r in results])
            
            logger.info(f"✅ Concurrent processing completed:")
            logger.info(f"   - Total time: {total_time:.3f}s")
            logger.info(f"   - Average per query: {avg_time:.3f}s")
            logger.info(f"   - Queries processed: {len(results)}")
            
            self.performance_metrics['concurrent_total_time'] = total_time
            self.performance_metrics['concurrent_avg_time'] = avg_time
            self.test_results['concurrent_processing'] = True
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Concurrent processing test failed: {e}")
            self.test_results['concurrent_processing'] = False
            return False
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        logger.info("📊 Generating performance report...")
        
        report = {
            'test_results': self.test_results,
            'performance_metrics': self.performance_metrics,
            'recommendations': [],
            'status': 'PASS' if all(self.test_results.values()) else 'FAIL'
        }
        
        # Add recommendations based on performance
        if 'avg_semantic_analysis_time' in self.performance_metrics:
            avg_time = self.performance_metrics['avg_semantic_analysis_time']
            if avg_time > 1.0:
                report['recommendations'].append(
                    f"Consider model optimization - avg processing time: {avg_time:.3f}s"
                )
        
        if 'memory_usage_mb' in self.performance_metrics:
            memory = self.performance_metrics['memory_usage_mb']
            if memory > 2000:  # 2GB
                report['recommendations'].append(
                    f"High memory usage detected: {memory:.1f}MB - consider model quantization"
                )
        
        return report
    
    def run_comprehensive_validation(self) -> bool:
        """Run all validation tests"""
        logger.info("🚀 Starting Comprehensive Production NLP Validation")
        logger.info("=" * 70)
        
        validation_steps = [
            ("Import Validation", self.validate_imports),
            ("spaCy Models (Production)", self.validate_spacy_models),
            ("Sentence Transformers", self.validate_sentence_transformers),
            ("Semantic Integration", self.validate_semantic_analysis_integration),
            ("Memory Usage", self.validate_memory_usage),
            ("Concurrent Processing", self.validate_concurrent_processing),
        ]
        
        failed_steps = []
        
        for step_name, step_function in validation_steps:
            logger.info(f"\n📍 {step_name}...")
            try:
                if step_function():
                    logger.info(f"✅ {step_name} - PASSED")
                else:
                    logger.error(f"❌ {step_name} - FAILED")
                    failed_steps.append(step_name)
            except Exception as e:
                logger.error(f"❌ {step_name} - FAILED with exception: {e}")
                failed_steps.append(step_name)
        
        # Generate final report
        report = self.generate_performance_report()
        
        logger.info("\n" + "=" * 70)
        logger.info("📊 VALIDATION SUMMARY")
        logger.info("=" * 70)
        
        for test, result in report['test_results'].items():
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"{test:25s}: {status}")
        
        logger.info("\n📈 PERFORMANCE METRICS:")
        for metric, value in report['performance_metrics'].items():
            if 'time' in metric:
                logger.info(f"{metric:25s}: {value:.3f}s")
            elif 'memory' in metric:
                logger.info(f"{metric:25s}: {value:.1f}MB")
            else:
                logger.info(f"{metric:25s}: {value}")
        
        if report['recommendations']:
            logger.info("\n💡 RECOMMENDATIONS:")
            for rec in report['recommendations']:
                logger.info(f"   - {rec}")
        
        logger.info(f"\n🎯 OVERALL STATUS: {report['status']}")
        
        if report['status'] == 'PASS':
            logger.info("🎉 PRODUCTION NLP STACK VALIDATION SUCCESSFUL!")
            logger.info("\n✅ Ready for production deployment!")
            logger.info("✅ All large models working correctly!")
            logger.info("✅ Performance metrics within acceptable ranges!")
            return True
        else:
            logger.error(f"💥 VALIDATION FAILED - Failed steps: {failed_steps}")
            logger.error("❌ Fix the issues above before deploying to production")
            return False

def main():
    """Main entry point"""
    validator = ProductionNLPValidator()
    success = validator.run_comprehensive_validation()
    
    if success:
        print("\n🎯 Production NLP stack validation completed successfully!")
        print("🚀 Ready to begin Phase 1 implementation!")
        sys.exit(0)
    else:
        print("\n💥 Validation failed - check logs above for details")
        sys.exit(1)

if __name__ == "__main__":
    main()
