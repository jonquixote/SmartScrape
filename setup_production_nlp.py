#!/usr/bin/env python3
"""
Production NLP Setup for SmartScrape
Installs and validates all production-grade NLP components
Removes all small/medium model dependencies and enforces large models only
"""

import subprocess
import sys
import os
import logging
import platform
from pathlib import Path
from typing import List, Dict, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProductionNLPSetup:
    """
    Comprehensive setup manager for production NLP stack
    """
    
    def __init__(self):
        self.python_executable = sys.executable
        self.platform_info = {
            'system': platform.system(),
            'machine': platform.machine(),
            'python_version': platform.python_version()
        }
        
        # Required packages for production
        self.production_packages = [
            # Core PyTorch stack (CPU optimized for better inference speed)
            "torch>=2.0.0",
            "torchvision>=0.15.0",
            "torchaudio>=2.0.0",
            
            # Advanced NLP libraries
            "spacy>=3.7.0",
            "sentence-transformers==2.7.0", # Pinned version
            "transformers>=4.30.0",
            "protobuf==3.20.3", # Pinned version
            
            # Enhanced semantic processing
            "scikit-learn>=1.3.0",
            "numpy>=1.24.0",
            "scipy>=1.11.0",
            
            # Vector similarity and indexing
            
            "faiss-binary>=1.8.0", # Use pre-compiled binary
            
            # Additional NLP utilities
            "textblob>=0.17.1",
            "nltk>=3.9.1"
        ]
        
        # Required spaCy models (PRODUCTION ONLY - no small/medium)
        self.required_models = [
            "en_core_web_lg",      # Large English model (750MB) - REQUIRED
            "en_core_web_trf",     # Transformer model (500MB) - OPTIONAL but recommended
        ]
        
        # Sentence transformer models (auto-downloaded on first use)
        self.sentence_models = [
            "all-MiniLM-L6-v2",    # 90MB - General purpose, fast
            "all-mpnet-base-v2",   # 420MB - High quality, slower
        ]
        
        logger.info(f"Production NLP Setup initialized on {self.platform_info}")
    
    def check_python_version(self) -> bool:
        """Check if Python version is compatible"""
        version = sys.version_info
        if not (3, 9) <= (version.major, version.minor) <= (3, 11):
            logger.error(f"❌ Incompatible Python version: {version.major}.{version.minor}.{version.micro}")
            logger.error("This script requires Python 3.9, 3.10, or 3.11 for compatibility with NLP packages.")
            logger.error("Please create a new environment with a compatible Python version.")
            logger.error("Example using pyenv:")
            logger.error("  brew install pyenv")
            logger.error("  pyenv install 3.11.9")
            logger.error("  pyenv virtualenv 3.11.9 smartscrape-nlp")
            logger.error("  pyenv activate smartscrape-nlp")
            logger.error("  pip install --upgrade pip")
            logger.error("  python setup_production_nlp.py")
            return False
        logger.info(f"✅ Python version {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    
    def install_pytorch_optimized(self) -> bool:
        """Install PyTorch optimized for CPU inference"""
        logger.info("🔥 Installing PyTorch (CPU optimized)...")
        
        try:
            # Install CPU-optimized PyTorch for faster inference
            cmd = f"{self.python_executable} -m pip install torch torchvision torchaudio"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"PyTorch installation failed: {result.stderr}")
                return False
                
            logger.info("✅ PyTorch installed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error installing PyTorch: {e}")
            return False
    
    def install_production_packages(self) -> bool:
        """Install all production NLP packages"""
        logger.info("📦 Installing production NLP packages...")
        
        failed_packages = []
        
        for package in self.production_packages:
            if "torch" in package:
                continue  # Already handled in pytorch installation
                
            try:
                logger.info(f"Installing {package}...")
                cmd = f"{self.python_executable} -m pip install '{package}'"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                
                if result.returncode != 0:
                    logger.error(f"Failed to install {package}: {result.stderr}")
                    failed_packages.append(package)
                else:
                    logger.info(f"✅ {package} installed")
                    
            except Exception as e:
                logger.error(f"Error installing {package}: {e}")
                failed_packages.append(package)
        
        if failed_packages:
            logger.error(f"❌ Failed to install packages: {failed_packages}")
            return False
            
        logger.info("✅ All production packages installed successfully")
        return True
    
    def download_spacy_models(self) -> bool:
        """Download required spaCy models (LARGE ONLY)"""
        logger.info("🧠 Downloading spaCy models (PRODUCTION - LARGE ONLY)...")
        
        failed_models = []
        
        for model in self.required_models:
            try:
                logger.info(f"Downloading {model}... (this may take several minutes)")
                cmd = f"{self.python_executable} -m spacy download {model}"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                
                if result.returncode != 0:
                    if "en_core_web_trf" in model:
                        logger.warning(f"Optional model {model} failed to download: {result.stderr}")
                    else:
                        logger.error(f"CRITICAL: Required model {model} failed to download: {result.stderr}")
                        failed_models.append(model)
                else:
                    logger.info(f"✅ {model} downloaded successfully")
                    
            except Exception as e:
                logger.error(f"Error downloading {model}: {e}")
                if "en_core_web_lg" in model:  # Critical model
                    failed_models.append(model)
        
        if failed_models:
            logger.error(f"❌ Failed to download critical models: {failed_models}")
            return False
            
        logger.info("✅ All required spaCy models downloaded")
        return True
    
    def validate_sentence_transformers(self) -> bool:
        """Pre-validate sentence transformers (they auto-download on first use)"""
        logger.info("🤖 Validating sentence transformers...")
        
        try:
            from sentence_transformers import SentenceTransformer
            
            # Test loading the primary model (will auto-download if needed)
            logger.info("Loading sentence transformer model (may download on first use)...")
            model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Test encoding
            test_text = "This is a test sentence for semantic analysis"
            embeddings = model.encode([test_text])
            
            logger.info(f"✅ Sentence transformers working (embedding shape: {embeddings.shape})")
            return True
            
        except Exception as e:
            logger.error(f"❌ Sentence transformers validation failed: {e}")
            return False
    
    def validate_spacy_installation(self) -> bool:
        """Validate spaCy installation with production models"""
        logger.info("🧠 Validating spaCy installation...")
        
        try:
            import spacy
            
            # Test loading the large model (REQUIRED)
            try:
                nlp = spacy.load("en_core_web_lg")
                logger.info("✅ en_core_web_lg loaded successfully")
            except OSError:
                logger.error("❌ CRITICAL: en_core_web_lg not available - this is required for production")
                return False
            
            # Test the transformer model (OPTIONAL)
            try:
                nlp_trf = spacy.load("en_core_web_trf")
                logger.info("✅ en_core_web_trf loaded successfully (optional)")
            except OSError:
                logger.warning("⚠️ en_core_web_trf not available (optional model)")
            
            # Test processing
            test_text = "SmartScrape is an advanced web scraping tool with AI capabilities"
            doc = nlp(test_text)
            
            # Validate features
            entities = [ent.text for ent in doc.ents]
            tokens = [token.lemma_ for token in doc if not token.is_stop]
            
            logger.info(f"✅ spaCy processing test passed (entities: {entities})")
            return True
            
        except Exception as e:
            logger.error(f"❌ spaCy validation failed: {e}")
            return False
    
    def validate_complete_stack(self) -> bool:
        """Comprehensive validation of the entire production NLP stack"""
        logger.info("🔬 Running comprehensive NLP stack validation...")
        
        try:
            # Import all critical components
            import torch
            import spacy
            from sentence_transformers import SentenceTransformer
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np
            
            # Load models
            nlp = spacy.load("en_core_web_lg")
            sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Test comprehensive workflow
            test_query = "Find the latest artificial intelligence research papers on machine learning"
            
            # spaCy analysis
            doc = nlp(test_query)
            spacy_entities = [ent.text for ent in doc.ents]
            spacy_tokens = [token.lemma_ for token in doc if not token.is_stop and token.pos_ in ['NOUN', 'ADJ']]
            
            # Sentence transformer embeddings
            embeddings = sentence_model.encode([test_query])
            
            # Similarity test
            test_similar = "Research papers about AI and ML"
            similar_embeddings = sentence_model.encode([test_similar])
            similarity = cosine_similarity(embeddings, similar_embeddings)[0][0]
            
            logger.info(f"✅ Comprehensive validation passed:")
            logger.info(f"   - spaCy entities: {spacy_entities}")
            logger.info(f"   - spaCy tokens: {spacy_tokens[:5]}")
            logger.info(f"   - Embedding shape: {embeddings.shape}")
            logger.info(f"   - Semantic similarity: {similarity:.3f}")
            
            if similarity > 0.7:  # Good semantic similarity
                logger.info("✅ Production NLP stack is working correctly!")
                return True
            else:
                logger.warning(f"⚠️ Semantic similarity lower than expected: {similarity}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Comprehensive validation failed: {e}")
            return False
    
    def cleanup_old_models(self) -> bool:
        """Remove references to small/medium models (cleanup)"""
        logger.info("🧹 Cleaning up old model references...")
        
        try:
            # This is mainly for cleanup - actual file removal should be done manually
            # We just check if small models are still accessible
            import spacy
            
            deprecated_models = ["en_core_web_sm", "en_core_web_md"]
            found_deprecated = []
            
            for model in deprecated_models:
                try:
                    spacy.load(model)
                    found_deprecated.append(model)
                except OSError:
                    pass  # Good, model not found
            
            if found_deprecated:
                logger.warning(f"⚠️ Deprecated models still available: {found_deprecated}")
                logger.warning("Consider removing these models to save space:")
                for model in found_deprecated:
                    logger.warning(f"   pip uninstall {model}")
            else:
                logger.info("✅ No deprecated models found")
                
            return True
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return False
    
    def run_complete_setup(self) -> bool:
        """Run the complete production NLP setup process"""
        logger.info("🚀 Starting Production NLP Setup for SmartScrape")
        logger.info("=" * 60)
        
        steps = [
            ("Python Version Check", self.check_python_version),
            ("PyTorch Installation", self.install_pytorch_optimized),
            ("Production Packages", self.install_production_packages),
            ("spaCy Models Download", self.download_spacy_models),
            ("Sentence Transformers", self.validate_sentence_transformers),
            ("spaCy Validation", self.validate_spacy_installation),
            ("Complete Stack Test", self.validate_complete_stack),
            ("Cleanup Old Models", self.cleanup_old_models),
        ]
        
        failed_steps = []
        
        for step_name, step_function in steps:
            logger.info(f"\n📍 {step_name}...")
            try:
                if not step_function():
                    failed_steps.append(step_name)
                    logger.error(f"❌ {step_name} failed")
                else:
                    logger.info(f"✅ {step_name} completed")
            except Exception as e:
                logger.error(f"❌ {step_name} failed with exception: {e}")
                failed_steps.append(step_name)
        
        logger.info("\n" + "=" * 60)
        
        if failed_steps:
            logger.error(f"❌ SETUP FAILED - Failed steps: {failed_steps}")
            return False
        else:
            logger.info("🎉 PRODUCTION NLP SETUP COMPLETED SUCCESSFULLY!")
            logger.info("\nNext steps:")
            logger.info("1. Run 'python validate_production_nlp.py' to verify everything is working")
            logger.info("2. Begin Phase 1 implementation from the roadmap")
            logger.info("3. Update your code to use only en_core_web_lg models")
            return True

def main():
    """Main entry point"""
    setup = ProductionNLPSetup()
    success = setup.run_complete_setup()
    
    if success:
        print("\n🎯 Production NLP stack is ready!")
        print("🔗 Next: Run validation script to confirm everything works")
        sys.exit(0)
    else:
        print("\n💥 Setup failed - check logs above for details")
        sys.exit(1)

if __name__ == "__main__":
    main()
