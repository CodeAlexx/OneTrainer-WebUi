#!/usr/bin/env python3
"""
Test script to verify all imports work correctly.
"""
import sys
from pathlib import Path

# Add parent directory to path (like run.py does)
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

print("="*60)
print("OneTrainer Web UI Import Test")
print("="*60)
print(f"Python version: {sys.version}")
print(f"Root directory: {root_dir}")
print("="*60)

# Test imports
try:
    print("\n1. Testing trainer service import...")
    from web_ui.backend.services.trainer_service import TrainerService, get_trainer_service
    print("   ✓ trainer_service imports OK")

    print("\n2. Testing API imports...")
    from web_ui.backend.api import training, config, samples, system
    print("   ✓ API modules import OK")

    print("\n3. Testing FastAPI app import...")
    from web_ui.backend.main import app
    print("   ✓ FastAPI app imports OK")

    print("\n4. Testing OneTrainer core imports...")
    from modules.util.config.TrainConfig import TrainConfig
    from modules.util.commands.TrainCommands import TrainCommands
    from modules.util.callbacks.TrainCallbacks import TrainCallbacks
    from modules.util.TrainProgress import TrainProgress
    print("   ✓ OneTrainer core modules import OK")

    print("\n5. Testing trainer service singleton...")
    service1 = get_trainer_service()
    service2 = TrainerService.get_instance()
    service3 = TrainerService()
    assert service1 is service2 is service3, "Singleton pattern failed!"
    print("   ✓ Singleton pattern works correctly")

    print("\n6. Testing trainer service state...")
    state = service1.get_state()
    assert isinstance(state, dict), "State should be a dict"
    assert 'is_training' in state, "State should have is_training"
    assert 'status' in state, "State should have status"
    print(f"   ✓ Initial state: {state}")

    print("\n" + "="*60)
    print("ALL TESTS PASSED ✓")
    print("="*60)
    print("\nThe web UI backend is ready to run!")
    print("Start the server with: python web_ui/run.py")
    print("="*60)

except ImportError as e:
    print(f"\n✗ Import failed: {e}")
    print("\nTroubleshooting:")
    print("1. Make sure you're in the OneTrainer directory")
    print("2. Check that all required packages are installed:")
    print("   pip install fastapi uvicorn websockets psutil pydantic")
    sys.exit(1)

except Exception as e:
    print(f"\n✗ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
