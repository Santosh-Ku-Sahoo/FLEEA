import os
import subprocess
import sys
from pathlib import Path

def run_command(command, cwd=None):
    print(f"\n> Running: {command}")
    try:
        subprocess.check_call(command, shell=True, cwd=cwd)
    except subprocess.CalledProcessError as e:
        print(f"Error: Command failed with exit code {e.returncode}")
        sys.exit(1)

def main():
    root_dir = Path(__file__).resolve().parent.parent
    web_dir = root_dir / "web"

    print("=== FLEEA Production Deployment Tool ===")

    # 1. Install Python dependencies
    print("\n[1/4] Installing Python dependencies...")
    run_command("pip install -r requirements.txt", root_dir)

    # 2. Build Frontend
    print("\n[2/4] Building React frontend...")
    if not (web_dir / "node_modules").exists():
        print("node_modules not found, running npm install...")
        run_command("npm install", web_dir)
    
    run_command("npm run build", web_dir)

    # 3. Verify Build
    dist_dir = web_dir / "dist"
    if not dist_dir.exists():
        print("Error: build failed, dist folder not found.")
        sys.exit(1)
    print(f"Frontend built successfully at {dist_dir}")

    # 4. Success Message
    print("\n=== DEPLOYMENT READY ===")
    print("The frontend is now built and will be served by the Flask backend.")
    print("\nTo start FLEEA in production mode:")
    print("python -m interfaces.voice_ui_server")
    print("\nYour dashboard will be available at: http://localhost:5050")

if __name__ == "__main__":
    main()
