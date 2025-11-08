"""
Verify that the project setup is correct
"""

import os
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_dependencies():
    """Check if all required packages are installed"""
    required_packages = [
        'torch', 'torchvision', 'facenet_pytorch', 'PIL', 
        'numpy', 'cv2', 'sklearn', 'matplotlib', 'yaml'
    ]
    
    missing = []
    for package in required_packages:
        try:
            if package == 'PIL':
                __import__('PIL')
            elif package == 'cv2':
                __import__('cv2')
            elif package == 'sklearn':
                __import__('sklearn')
            elif package == 'facenet_pytorch':
                __import__('facenet_pytorch')
            else:
                __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        logger.error(f"Missing packages: {', '.join(missing)}")
        logger.error("Run: pip install -r requirements.txt")
        return False
    else:
        logger.info("✓ All dependencies installed")
        return True


def check_directories():
    """Check if all required directories exist"""
    required_dirs = [
        'data/raw',
        'data/processed',
        'models/pretrained',
        'models/checkpoints',
        'models/saved',
        'logs/tensorboard',
        'src',
        'scripts'
    ]
    
    missing = []
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing.append(dir_path)
    
    if missing:
        logger.error(f"Missing directories: {', '.join(missing)}")
        logger.error("Run: python setup.py")
        return False
    else:
        logger.info("✓ All directories exist")
        return True


def check_files():
    """Check if all required files exist"""
    required_files = [
        'config.yaml',
        'requirements.txt',
        'README.md',
        'src/model.py',
        'src/train.py',
        'src/server.py',
        'src/client.py',
        'scripts/prepare_data.py',
        'scripts/evaluate.py'
    ]
    
    missing = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing.append(file_path)
    
    if missing:
        logger.error(f"Missing files: {', '.join(missing)}")
        return False
    else:
        logger.info("✓ All required files exist")
        return True


def check_data_directories():
    """Check if member data directories exist"""
    try:
        from src.utils import load_config
        config = load_config()
        members = config.get('client', {}).get('member_names', ['omarmej', 'abir', 'omarbr', 'jihene'])
    except Exception:
        members = ['omarmej', 'abir', 'omarbr', 'jihene']
    
    all_exist = True
    
    for member in members:
        raw_dir = Path(f'data/raw/{member}')
        if not raw_dir.exists():
            logger.warning(f"⚠ {raw_dir} does not exist (this is okay if you haven't added data yet)")
        else:
            logger.info(f"✓ {raw_dir} exists")
    
    return True


def main():
    """Run all checks"""
    logger.info("Verifying project setup...")
    logger.info("=" * 50)
    
    checks = [
        ("Dependencies", check_dependencies),
        ("Directories", check_directories),
        ("Files", check_files),
        ("Data Directories", check_data_directories),
    ]
    
    results = []
    for name, check_func in checks:
        logger.info(f"\nChecking {name}...")
        result = check_func()
        results.append(result)
    
    logger.info("\n" + "=" * 50)
    if all(results):
        logger.info("✓ Setup verification passed!")
        logger.info("\nNext steps:")
        logger.info("1. Collect face images")
        logger.info("2. Place them in data/raw/[your_name]/ (omarmej, abir, omarbr, or jihene)")
        logger.info("3. Run: python scripts/prepare_data.py --member [your_name]")
        logger.info("4. Start training with centralized server!")
    else:
        logger.error("✗ Setup verification failed!")
        logger.error("Please fix the issues above and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()

