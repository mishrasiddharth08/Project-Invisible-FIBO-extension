"""Run host-isolated tests; never boots Forge or modifies its settings."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
suite = unittest.TestSuite()
for name in ("test_contracts.py", "test_generation.py", "test_progress.py", "test_model_setup.py", "test_degrid_preview.py"):
    suite.addTests(unittest.defaultTestLoader.discover(str(Path(__file__).parent), pattern=name))
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(not result.wasSuccessful())
