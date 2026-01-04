#!/usr/bin/env python3
"""
TensorBoard launcher that patches the duplicate plugins issue.
This is a workaround for: https://github.com/tensorflow/tensorboard/issues/6852
"""
import sys
from tensorboard import default

# Patch to remove duplicate plugins before launching
original_get_plugins = default.get_plugins

def get_plugins_patched():
    plugins = original_get_plugins()
    
    # Deduplicate by plugin_name
    seen = set()
    unique_plugins = []
    for plugin in plugins:
        name = getattr(plugin, 'plugin_name', None) or str(type(plugin))
        if name not in seen:
            seen.add(name)
            unique_plugins.append(plugin)
        else:
            print(f"TensorBoard: Removing duplicate plugin: {name}", file=sys.stderr)
    
    return unique_plugins

default.get_plugins = get_plugins_patched

# Now run tensorboard normally
from tensorboard.main import run_main
sys.exit(run_main())
