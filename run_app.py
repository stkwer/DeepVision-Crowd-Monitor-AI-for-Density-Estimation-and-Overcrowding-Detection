#!/usr/bin/env python3
"""
People Counter AI - Presentation & Launch Script
Streamlines running the Streamlit app with proper setup
"""

import subprocess
import sys
import os
from pathlib import Path
import platform

def check_python_version():
    """Verify Python version compatibility"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        sys.exit(1)
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected")

def install_dependencies():
    """Install required packages from requirements.txt"""
    req_file = Path(__file__).parent / "requirements.txt"
    
    if not req_file.exists():
        print("❌ requirements.txt not found")
        sys.exit(1)
    
    print("\n📦 Installing dependencies...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(req_file), "-q"],
        capture_output=False
    )
    
    if result.returncode == 0:
        print("✓ Dependencies installed successfully")
    else:
        print("❌ Failed to install dependencies")
        sys.exit(1)

def launch_streamlit_app():
    """Launch the Streamlit application"""
    app_path = Path(__file__).parent / "streamlit" / "app.py"
    
    if not app_path.exists():
        print(f"❌ App file not found: {app_path}")
        sys.exit(1)
    
    print(f"\n🚀 Launching People Counter AI...")
    print(f"📍 App: {app_path}")
    print("\n" + "="*60)
    print("🌐 Opening in browser at http://localhost:8501")
    print("💡 Press Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(app_path), "--logger.level=info"],
            cwd=str(Path(__file__).parent)
        )
    except KeyboardInterrupt:
        print("\n\n✓ Application stopped")
        sys.exit(0)

def main():
    """Main entry point"""
    os.system('cls' if platform.system() == 'Windows' else 'clear')
    
    print("=" * 60)
    print("👥 PEOPLE COUNTER AI - PRESENTATION MODE")
    print("=" * 60)
    
    check_python_version()
    install_dependencies()
    launch_streamlit_app()

if __name__ == "__main__":
    main()
