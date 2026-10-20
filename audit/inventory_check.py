import os
import re
import ast
import subprocess

WORKSPACE = r"e:\git_projects\Projects\Visiona3D"

EXCLUDE_DIRS = {".git", "__pycache__", ".venv", "venv", ".early.coverage", "audit"}
EXCLUDE_EXTS = {".pt", ".pth", ".bin", ".mp4", ".avi", ".mkv", ".mov", ".lock"}

def get_file_tree_and_loc():
    folder_stats = {}
    file_list = []
    
    for root, dirs, files in os.walk(WORKSPACE):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        rel_root = os.path.relpath(root, WORKSPACE)
        if rel_root == ".":
            rel_root = ""
        
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in EXCLUDE_EXTS:
                continue
            full_path = os.path.join(root, f)
            rel_path = os.path.join(rel_root, f) if rel_root else f
            
            loc = 0
            try:
                with open(full_path, "r", encoding="utf-8-sig", errors="ignore") as fp:
                    loc = sum(1 for _ in fp)
            except Exception:
                pass
            
            folder = rel_root if rel_root else "(root)"
            if folder not in folder_stats:
                folder_stats[folder] = {"files": 0, "loc": 0, "file_names": []}
            folder_stats[folder]["files"] += 1
            folder_stats[folder]["loc"] += loc
            folder_stats[folder]["file_names"].append((rel_path, loc))
            file_list.append((rel_path, loc))
            
    return folder_stats, file_list

def find_imports():
    all_imports = set()
    file_imports = {}
    
    for root, dirs, files in os.walk(WORKSPACE):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for f in files:
            if f.endswith(".py"):
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, WORKSPACE)
                file_imports[rel_path] = set()
                try:
                    with open(full_path, "r", encoding="utf-8-sig", errors="ignore") as fp:
                        tree = ast.parse(fp.read(), filename=rel_path)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for alias in node.names:
                                top_level = alias.name.split(".")[0]
                                all_imports.add(top_level)
                                file_imports[rel_path].add(top_level)
                        elif isinstance(node, ast.ImportFrom):
                            if node.module:
                                top_level = node.module.split(".")[0]
                                all_imports.add(top_level)
                                file_imports[rel_path].add(top_level)
                except Exception as e:
                    print(f"Error parsing AST in {rel_path}: {e}")
    return all_imports, file_imports

def check_secrets():
    cmd_env = ["git", "--no-pager", "log", "--all", "--full-history", "--", ".env"]
    res_env = subprocess.run(cmd_env, cwd=WORKSPACE, capture_output=True, text=True, stdin=subprocess.DEVNULL)
    is_env_in_history = bool(res_env.stdout.strip())
    
    cmd_tracked = ["git", "ls-files"]
    res_tracked = subprocess.run(cmd_tracked, cwd=WORKSPACE, capture_output=True, text=True, stdin=subprocess.DEVNULL)
    tracked_files = res_tracked.stdout.splitlines()
    is_env_tracked = ".env" in tracked_files
    
    patterns = [
        re.compile(r"gsk_[a-zA-Z0-9]{20,}"),
        re.compile(r"sk-[a-zA-Z0-9]{20,}"),
        re.compile(r"AIza[0-9A-Za-z-_]{35}"),
        re.compile(r"(?:api_key|apikey|token|secret)\s*[:=]\s*['\"][a-zA-Z0-9_\-]{16,}['\"]", re.IGNORECASE)
    ]
    
    findings = []
    for tf in tracked_files:
        ext = os.path.splitext(tf)[1].lower()
        if ext in EXCLUDE_EXTS or ext in {".png", ".jpg", ".jpeg", ".lock"}:
            continue
        full_path = os.path.join(WORKSPACE, tf)
        if not os.path.exists(full_path) or os.path.isdir(full_path):
            continue
        try:
            with open(full_path, "r", encoding="utf-8-sig", errors="ignore") as fp:
                for idx, line in enumerate(fp, 1):
                    for pat in patterns:
                        if pat.search(line):
                            findings.append((tf, f"line {idx}", "tracked file"))
                            break
        except Exception:
            pass
            
    # Check all commits using git grep across tree
    res_commits = subprocess.run(["git", "--no-pager", "rev-list", "--all"], cwd=WORKSPACE, capture_output=True, text=True, stdin=subprocess.DEVNULL)
    commits = res_commits.stdout.splitlines()
    
    for c in commits:
        res_grep = subprocess.run(["git", "--no-pager", "grep", "-i", "-E", "(gsk_[a-zA-Z0-9]{15,}|sk-[a-zA-Z0-9]{20,}|AIza[0-9A-Za-z-_]{35})", c],
                                  cwd=WORKSPACE, capture_output=True, text=True, stdin=subprocess.DEVNULL)
        if res_grep.stdout.strip():
            for match_line in res_grep.stdout.splitlines():
                findings.append((c[:7], match_line.split(":", 1)[0], "git commit tree"))
                
    return is_env_tracked, is_env_in_history, findings

if __name__ == "__main__":
    print("=== STEP 1: INVENTORY & STATS ===", flush=True)
    folder_stats, file_list = get_file_tree_and_loc()
    total_files = sum(s["files"] for s in folder_stats.values())
    total_loc = sum(s["loc"] for s in folder_stats.values())
    
    print(f"\nTotal files (excluding weights, media, venv, git, audit): {total_files}", flush=True)
    print(f"Total lines of code: {total_loc}\n", flush=True)
    print(f"{'Folder':<35} | {'Files':<6} | {'LOC':<8}", flush=True)
    print("-" * 55, flush=True)
    for folder, s in sorted(folder_stats.items()):
        print(f"{folder:<35} | {s['files']:<6} | {s['loc']:<8}", flush=True)
        
    print("\n--- File Tree Detail ---", flush=True)
    for path, loc in sorted(file_list):
        print(f"  {path:<45} : {loc} lines", flush=True)
        
    print("\n=== DEPENDENCY ANALYSIS ===", flush=True)
    all_imports, file_imports = find_imports()
    print(f"All top-level imports in codebase: {sorted(all_imports)}", flush=True)
    
    pyproj_list = [
        ("numpy", "numpy"), ("opencv-python", "cv2"), ("pyttsx3", "pyttsx3"),
        ("torch", "torch"), ("torchvision", "torchvision"), ("ultralytics", "ultralytics"),
        ("python-dotenv", "dotenv"), ("langchain", "langchain"), ("langchain-groq", "langchain_groq"),
        ("speechrecognition", "speech_recognition"), ("pyaudio", "pyaudio"),
        ("timm", "timm"), ("transformers", "transformers"), ("pynput", "pynput")
    ]
    
    internal_modules = {"core", "kinematics", "perception", "audio", "agents", "services", "tools"}
    stdlib_modules = {
        "os", "sys", "time", "threading", "queue", "json", "math", "collections",
        "dataclasses", "typing", "datetime", "re", "ast", "subprocess", "hashlib",
        "abc", "copy", "traceback", "logging", "tempfile", "shutil", "heapq", "inspect",
        "unittest", "winsound", "wave", "io"
    }
    
    third_party_imports = {imp for imp in all_imports if imp not in internal_modules and imp not in stdlib_modules}
    print(f"\nThird-party imports detected: {sorted(third_party_imports)}", flush=True)
    
    print("\nChecking pyproject.toml declared dependencies:", flush=True)
    for pkg, mod in pyproj_list:
        if mod in third_party_imports:
            print(f"  [USED] {pkg} (imported as '{mod}')", flush=True)
        else:
            print(f"  [UNUSED / UNREFERENCED IN CODE] {pkg} (imported as '{mod}')", flush=True)
            
    declared_mods = {mod for _, mod in pyproj_list}
    undeclared = third_party_imports - declared_mods
    print(f"\nUndeclared third-party dependencies imported in code: {sorted(undeclared)}", flush=True)
    
    print("\n=== SECRETS CHECK ===", flush=True)
    env_tracked, env_history, secret_findings = check_secrets()
    print(f"Is .env tracked in git: {env_tracked}", flush=True)
    print(f"Is .env in git commit history: {env_history}", flush=True)
    print(f"Secret findings count: {len(secret_findings)}", flush=True)
    if secret_findings:
        for item, loc_desc, src in secret_findings:
            print(f"  Found potential secret token pattern in: {item} ({loc_desc}, {src})", flush=True)
    else:
        print("  No API keys, tokens, or secrets found in tracked files or commit history.", flush=True)
