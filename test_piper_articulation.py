#!/usr/bin/env python3
"""Test script to verify Piper Articulation loading."""

import sys
sys.path.insert(0, '/home/mojie/taskdog')

try:
    from custom_envs.tasks.deeprobotics_m20_pro.piper_env_cfg import DeeproboticsM20ProPiperEnvCfg
    
    print("[TEST] Loading DeeproboticsM20ProPiperEnvCfg...")
    cfg = DeeproboticsM20ProPiperEnvCfg()
    print("[TEST] ✓ Config loaded successfully")
    
    # Check if piper was added to scene
    if hasattr(cfg.scene, 'piper'):
        print("[TEST] ✓ Piper found in scene")
        print(f"[TEST] Piper config type: {type(cfg.scene.piper)}")
    else:
        print("[TEST] ✗ Piper NOT found in scene")
    
    # Check robot
    if hasattr(cfg.scene, 'robot'):
        print("[TEST] ✓ Robot found in scene")
    else:
        print("[TEST] ✗ Robot NOT found in scene")
    
    print("\n[TEST] ✓ Basic configuration checks passed!")
    
except Exception as e:
    print(f"[TEST] ✗ Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
