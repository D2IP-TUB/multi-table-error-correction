#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unified EC-at-Scale Pipeline Entry Point

This module serves as the main entry point for the error correction pipeline.
It dispatches to multi-classifier mode based on configuration.

Supported modes:
- "multi": Multiple zone-specific classifiers (main_multi_clf.py logic, default)

Usage:
    python main.py  (uses config file)
    python main.py --config /path/to/config.ini
    python main.py --strategy multi  (override strategy)
"""

import logging
import sys
from pathlib import Path

from config.parse_config import read_ecs_config
from utils.app_logger import setup_logging


def main():
    """
    Main entry point for unified EC-at-Scale pipeline.

    Reads configuration and dispatches to appropriate implementation.
    """
    # Parse command-line arguments
    config_file = None
    strategy_override = None

    for i, arg in enumerate(sys.argv[1:]):
        if arg == "--config" and i + 1 < len(sys.argv) - 1:
            config_file = sys.argv[i + 2]
        elif arg == "--strategy" and i + 1 < len(sys.argv) - 1:
            strategy_override = sys.argv[i + 2]

    # Read configuration
    if config_file:
        config = read_ecs_config(config_file)
    else:
        config = read_ecs_config()

    # Setup logging
    setup_logging(config.experiment.log_level)
    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("EC-at-Scale: Unified Error Correction Pipeline")
    logger.info("=" * 80)

    # Get classification strategy from config or command line
    strategy = strategy_override or config.training.classification_strategy

    logger.info(f"Classification Strategy: {strategy}")
    logger.info(f"Zoning Strategy: {config.zoning.strategy}")

    # Validate strategy
    if strategy not in ["multi"]:
        raise ValueError(
            f"Unknown classification strategy: {strategy}. "        )
        
    if strategy == "multi":
        logger.info("Using MULTI-ZONE CLASSIFIER mode")
        logger.info("  - Separate classifier per zone")
        logger.info(f"  - Zone detection: {config.zoning.strategy}")
        logger.info("  - Specialized per-zone error correction")
        run_multi_classifier(config)

    logger.info("=" * 80)
    logger.info("Pipeline completed successfully")
    logger.info("=" * 80)



def run_multi_classifier(config):
    """
    Run multi-zone classifier mode.

    Creates zones (rule-based or clustering-based) and trains separate
    classifiers per zone for specialized error handling.
    """
    # Import here to avoid circular dependencies
    import main_multi_clf

    main_multi_clf.main()


if __name__ == "__main__":
    main()
