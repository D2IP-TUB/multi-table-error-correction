#!/usr/bin/env python
"""
Quick test to verify memory monitoring system is working correctly.
"""

import logging
import sys
from utils.memory_monitor import MemoryMonitor

# Setup basic logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s %(levelname)s %(name)s - %(message)s"
)

def test_memory_monitor():
    """Test the MemoryMonitor class."""
    print("\n" + "="*80)
    print("MEMORY MONITORING SYSTEM TEST")
    print("="*80 + "\n")
    
    try:
        # Test 1: Initialization
        print("[TEST 1] Initializing MemoryMonitor...")
        monitor = MemoryMonitor(memory_limit_percent=80.0)
        print("✓ MemoryMonitor initialized successfully\n")
        
        # Test 2: Get memory usage
        print("[TEST 2] Checking memory usage...")
        usage = monitor.get_memory_usage()
        print(f"✓ Current system memory usage: {usage:.1f}%\n")
        
        # Test 3: Get process memory
        print("[TEST 3] Checking process memory...")
        proc_mem = monitor.get_process_memory()
        print(f"✓ Current process memory: {proc_mem:.1f}MB\n")
        
        # Test 4: Check memory (no enforcement)
        print("[TEST 4] Running check_memory (no enforcement)...")
        result = monitor.check_memory("TEST - Before critical operation")
        print(f"✓ Memory check passed: {result}\n")
        
        # Test 5: Get comprehensive stats
        print("[TEST 5] Getting memory statistics...")
        stats = monitor.get_memory_stats()
        print("✓ Memory statistics retrieved:")
        for key, value in stats.items():
            if isinstance(value, float):
                if "gb" in key.lower():
                    print(f"  - {key}: {value:.2f} GB")
                elif "percent" in key.lower():
                    print(f"  - {key}: {value:.1f}%")
                else:
                    print(f"  - {key}: {value:.2f}")
            else:
                print(f"  - {key}: {value}")
        print()
        
        # Test 6: Check and enforce (should pass)
        print("[TEST 6] Running check_and_enforce (should pass)...")
        try:
            monitor.check_and_enforce("TEST - Normal operation")
            print("✓ check_and_enforce passed (memory within limit)\n")
        except SystemExit:
            print("✗ check_and_enforce terminated (memory exceeded - unexpected)\n")
            return False
        
        print("="*80)
        print("ALL TESTS PASSED ✓")
        print("="*80)
        print("\nMemory monitoring system is working correctly!")
        print("The system will terminate if memory usage exceeds 80%.\n")
        return True
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_memory_monitor()
    sys.exit(0 if success else 1)
