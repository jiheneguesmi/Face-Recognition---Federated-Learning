"""
Setup script for the federated learning project
"""

from src.utils import load_config, create_directories
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_project():
    """
    Setup project directories and structure
    """
    try:
        config = load_config()
        create_directories(config)
        logger.info("Project setup completed successfully!")
        logger.info("\nNext steps:")
        logger.info("1. Collect face images and place them in data/raw/[your_name]/")
        logger.info("   (omarmej, abir, omarbr, or jihene)")
        logger.info("2. Run: python scripts/prepare_data.py --member [your_name]")
        logger.info("3. Start training with centralized server!")
    except Exception as e:
        logger.error(f"Error during setup: {e}")
        raise

if __name__ == "__main__":
    setup_project()

