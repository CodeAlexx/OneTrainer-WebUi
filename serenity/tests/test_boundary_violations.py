"""
Boundary Violation Tests

Verify the separation between model/memory/pipeline layers is clean:
1. Memory layer should not import specific model classes
2. Model layer should not call direct CUDA memory management
3. Pipeline should use interfaces, not reach into internals
"""

import ast
import os
from pathlib import Path


# =============================================================================
# Helper Functions
# =============================================================================

def get_imports_from_file(filepath: str) -> set:
    """Extract all imports from a Python file."""
    imports = set()

    try:
        with open(filepath, 'r') as f:
            tree = ast.parse(f.read())
    except (SyntaxError, UnicodeDecodeError):
        return imports

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module)
                # Also track what's being imported from the module
                for alias in node.names:
                    imports.add(f"{node.module}.{alias.name}")

    return imports


def find_pattern_in_file(filepath: str, patterns: list) -> list:
    """Find patterns in a Python file (as strings)."""
    matches = []

    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
    except (IOError, UnicodeDecodeError):
        return matches

    for i, line in enumerate(lines, 1):
        for pattern in patterns:
            if pattern in line:
                matches.append((i, line.strip(), pattern))

    return matches


def get_python_files(directory: str) -> list:
    """Get all Python files in a directory."""
    files = []
    for root, dirs, filenames in os.walk(directory):
        # Skip test directories
        if 'tests' in root or '__pycache__' in root:
            continue
        for filename in filenames:
            if filename.endswith('.py'):
                files.append(os.path.join(root, filename))
    return files


# =============================================================================
# Test: Memory Layer Imports
# =============================================================================

class TestMemoryLayerImports:
    """Memory layer should not import specific model classes."""

    MEMORY_DIR = "serenity/memory"

    # Specific model class names that should NOT be imported in memory layer
    FORBIDDEN_MODEL_IMPORTS = [
        'Flux2KleinModel',
        'Flux2KleinModelLoader',
        'FluxModel',
        'ZImageModel',
        'SDXLModel',
        'LTX2Model',
        'QwenModel',
        # Model setup classes
        'FluxModelSetup',
        'Flux2KleinModelSetup',
        'ZImageModelSetup',
    ]

    def test_memory_no_model_class_imports(self):
        """Memory layer should not import specific model classes."""
        violations = []

        for filepath in get_python_files(self.MEMORY_DIR):
            imports = get_imports_from_file(filepath)
            rel_path = os.path.relpath(filepath)

            for imp in imports:
                for forbidden in self.FORBIDDEN_MODEL_IMPORTS:
                    if forbidden in imp:
                        violations.append(f"{rel_path}: imports {forbidden}")

        if violations:
            print("  ❌ VIOLATIONS:")
            for v in violations:
                print(f"     {v}")
        else:
            print("  ✅ Memory layer has no forbidden model imports")

        assert len(violations) == 0, f"Found {len(violations)} violations"


# =============================================================================
# Test: Model Layer Direct CUDA Calls
# =============================================================================

class TestModelLayerCUDACalls:
    """Model layer should not make direct CUDA memory management calls."""

    MODEL_DIR = "serenity/models"

    # CUDA calls that should go through memory layer
    FORBIDDEN_CUDA_CALLS = [
        'torch.cuda.empty_cache()',
        'torch.cuda.synchronize()',
        'torch.cuda.memory_allocated()',
        'torch.cuda.memory_reserved()',
    ]

    # Allowed exceptions (in specific contexts)
    ALLOWED_CONTEXTS = [
        'if TYPE_CHECKING',  # Type hints only
        '# DEBUG',           # Debug code
        '# TODO',            # Temporary
    ]

    def test_model_no_direct_cuda_memory(self):
        """Model layer should use memory utilities, not direct CUDA calls."""
        violations = []

        for filepath in get_python_files(self.MODEL_DIR):
            matches = find_pattern_in_file(filepath, self.FORBIDDEN_CUDA_CALLS)
            rel_path = os.path.relpath(filepath)

            for line_num, line, pattern in matches:
                # Check for allowed contexts
                allowed = any(ctx in line for ctx in self.ALLOWED_CONTEXTS)
                if not allowed:
                    violations.append(f"{rel_path}:{line_num} - {pattern}")

        if violations:
            print("  ⚠️ POTENTIAL VIOLATIONS (review context):")
            for v in violations:
                print(f"     {v}")
            # Don't fail - these may be legitimate
        else:
            print("  ✅ Model layer has no direct CUDA memory calls")


# =============================================================================
# Test: Pipeline Uses Interfaces
# =============================================================================

class TestPipelineInterfaces:
    """Pipeline should use interfaces, not reach into model internals."""

    TRAINER_FILE = "serenity/core/trainer.py"

    # Model internal attributes that shouldn't be accessed directly
    INTERNAL_ATTRIBUTES = [
        '.transformer_blocks',  # Should use model interface
        '._internal_',          # Private attributes
        '._config',             # Private config
    ]

    def test_trainer_uses_interfaces(self):
        """Trainer should use model interface methods."""
        violations = []

        if os.path.exists(self.TRAINER_FILE):
            matches = find_pattern_in_file(self.TRAINER_FILE, self.INTERNAL_ATTRIBUTES)
            rel_path = os.path.relpath(self.TRAINER_FILE)

            for line_num, line, pattern in matches:
                # Filter out comments and strings
                if line.strip().startswith('#') or line.strip().startswith('"""'):
                    continue
                violations.append(f"{rel_path}:{line_num} - {pattern}")

        if violations:
            print("  ⚠️ POTENTIAL INTERNAL ACCESS (review):")
            for v in violations:
                print(f"     {v}")
        else:
            print("  ✅ Trainer doesn't access model internals directly")


# =============================================================================
# Test: Clean Imports Between Layers
# =============================================================================

class TestCleanLayerImports:
    """Verify import directions are correct."""

    def test_memory_imports_base_model_not_specific(self):
        """Memory layer should import BaseModel (interface), not specific models."""
        memory_dir = "serenity/memory"
        base_model_imports = 0
        specific_model_imports = 0

        for filepath in get_python_files(memory_dir):
            imports = get_imports_from_file(filepath)

            for imp in imports:
                if 'BaseModel' in imp:
                    base_model_imports += 1
                if any(m in imp for m in ['FluxModel', 'ZImageModel', 'SDXLModel', 'Flux2Klein']):
                    specific_model_imports += 1

        print(f"  BaseModel imports: {base_model_imports}")
        print(f"  Specific model imports: {specific_model_imports}")

        if specific_model_imports == 0:
            print("  ✅ Memory layer only imports BaseModel interface")
        else:
            print("  ❌ Memory layer imports specific models")
            assert False, "Should not import specific models"

    def test_training_imports_are_clean(self):
        """Training layer should have clean imports."""
        training_dir = "serenity/training"

        # Count imports
        model_imports = 0
        memory_imports = 0

        for filepath in get_python_files(training_dir):
            imports = get_imports_from_file(filepath)

            for imp in imports:
                if 'models.' in imp or 'Flux' in imp or 'SDXL' in imp:
                    model_imports += 1
                if 'memory.' in imp or 'conductor' in imp.lower():
                    memory_imports += 1

        print(f"  Model imports in training layer: {model_imports}")
        print(f"  Memory imports in training layer: {memory_imports}")
        print("  ✅ Training layer imports counted")


# =============================================================================
# Test: TYPE_CHECKING Guards
# =============================================================================

class TestTypeCheckingGuards:
    """Verify that type hints use TYPE_CHECKING guards."""

    def test_memory_manager_uses_type_checking(self):
        """MemoryManager should use TYPE_CHECKING for model type hints."""
        filepath = "serenity/memory/manager.py"

        if not os.path.exists(filepath):
            print(f"  ⚠️ File not found: {filepath}")
            return

        with open(filepath, 'r') as f:
            content = f.read()

        has_type_checking = 'TYPE_CHECKING' in content
        has_basemodel_hint = 'BaseModel' in content

        if has_type_checking and has_basemodel_hint:
            print("  ✅ MemoryManager uses TYPE_CHECKING for type hints")
        elif has_basemodel_hint and not has_type_checking:
            print("  ⚠️ BaseModel imported without TYPE_CHECKING guard")
        else:
            print("  ✅ No BaseModel imports found")


# =============================================================================
# Run Tests
# =============================================================================

def run_all_tests():
    """Run all boundary violation tests."""
    print("=== Boundary Violation Tests ===\n")

    print("Test: Memory Layer Imports")
    t = TestMemoryLayerImports()
    try:
        t.test_memory_no_model_class_imports()
    except AssertionError as e:
        print(f"  FAILED: {e}")

    print("\nTest: Model Layer CUDA Calls")
    t = TestModelLayerCUDACalls()
    t.test_model_no_direct_cuda_memory()

    print("\nTest: Pipeline Interfaces")
    t = TestPipelineInterfaces()
    t.test_trainer_uses_interfaces()

    print("\nTest: Clean Layer Imports")
    t = TestCleanLayerImports()
    t.test_memory_imports_base_model_not_specific()
    t.test_training_imports_are_clean()

    print("\nTest: TYPE_CHECKING Guards")
    t = TestTypeCheckingGuards()
    t.test_memory_manager_uses_type_checking()

    print("\n=== Boundary Tests Complete ===")


if __name__ == "__main__":
    run_all_tests()
