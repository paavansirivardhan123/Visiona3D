import re
import ast
import os

WORKSPACE = r"e:\git_projects\Projects\Visiona3D"
CONFIG_FILE = os.path.join(WORKSPACE, "core", "config.py")

# 1. Parse defined attributes in Config
with open(CONFIG_FILE, "r", encoding="utf-8-sig") as f:
    config_tree = ast.parse(f.read())

defined_keys = set()
for node in ast.walk(config_tree):
    if isinstance(node, ast.ClassDef) and node.name == "Config":
        for item in node.body:
            if isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        defined_keys.add(target.id)

# 2. Find all Config.ATTR references in all python files
referenced_keys = {}
for root, dirs, files in os.walk(WORKSPACE):
    if any(ex in root for ex in [".git", "__pycache__", ".venv", "audit"]):
        continue
    for f in files:
        if f.endswith(".py"):
            path = os.path.join(root, f)
            rel_path = os.path.relpath(path, WORKSPACE)
            try:
                with open(path, "r", encoding="utf-8-sig") as fp:
                    content = fp.read()
                # AST parse for Config.ATTR
                tree = ast.parse(content, filename=rel_path)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Attribute):
                        if isinstance(node.value, ast.Name) and node.value.id == "Config":
                            attr = node.attr
                            referenced_keys.setdefault(attr, []).append(rel_path)
                    # Also check getattr(Config, 'ATTR', ...)
                    elif isinstance(node, ast.Call):
                        if isinstance(node.func, ast.Name) and node.func.id == "getattr":
                            if len(node.args) >= 2 and isinstance(node.args[0], ast.Name) and node.args[0].id == "Config":
                                if isinstance(node.args[1], ast.Constant):
                                    attr = str(node.args[1].value)
                                    referenced_keys.setdefault(attr, []).append(rel_path)
            except Exception as e:
                print(f"Error reading {rel_path}: {e}")

print("=== CONFIG AUDIT ===")
print(f"Total Config keys defined: {len(defined_keys)}")
print(f"Total Config keys referenced in code: {len(referenced_keys)}")

missing_in_config = set(referenced_keys.keys()) - defined_keys
unused_in_code = defined_keys - set(referenced_keys.keys())

print(f"\nKeys REFERENCED in code but NOT DEFINED in core/config.py ({len(missing_in_config)}):")
for k in sorted(missing_in_config):
    print(f"  - Config.{k} (referenced in: {set(referenced_keys[k])})")

print(f"\nKeys DEFINED in core/config.py but NEVER REFERENCED in code ({len(unused_in_code)}):")
for k in sorted(unused_in_code):
    print(f"  - Config.{k}")
