#!/usr/bin/env python3
"""
MABe Mouser - Main Entry Point

This script handles all bootstrapping, dependency management, data acquisition,
and server startup for the MABe Mouser application.

Usage:
    python run.py
"""

import os
import sys
import subprocess
import shutil
import json
import time
import webbrowser
import signal
import zipfile
from pathlib import Path
from typing import Optional, Tuple

# ANSI color codes for pretty output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(msg: str):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{msg:^60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}\n")

def print_success(msg: str):
    print(f"{Colors.OKGREEN}✓ {msg}{Colors.ENDC}")

def print_info(msg: str):
    print(f"{Colors.OKCYAN}ℹ {msg}{Colors.ENDC}")

def print_warning(msg: str):
    print(f"{Colors.WARNING}⚠ {msg}{Colors.ENDC}")

def print_error(msg: str):
    print(f"{Colors.FAIL}✗ {msg}{Colors.ENDC}")

# Global process tracking for cleanup
backend_process: Optional[subprocess.Popen] = None
frontend_process: Optional[subprocess.Popen] = None

def cleanup_processes(signum=None, frame=None):
    """Clean shutdown of all spawned processes"""
    print_info("\nShutting down servers...")
    
    if backend_process:
        print_info("Stopping backend server...")
        backend_process.terminate()
        try:
            backend_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            backend_process.kill()
    
    if frontend_process:
        print_info("Stopping frontend server...")
        frontend_process.terminate()
        try:
            frontend_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            frontend_process.kill()
    
    print_success("Servers stopped. Goodbye!")
    sys.exit(0)

# Register cleanup handlers
signal.signal(signal.SIGINT, cleanup_processes)
signal.signal(signal.SIGTERM, cleanup_processes)

def check_python_version() -> bool:
    """Ensure Python 3.8+"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print_error(f"Python 3.8+ required, you have {version.major}.{version.minor}")
        return False
    print_success(f"Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_node_installed() -> bool:
    """Check if Node.js is installed"""
    try:
        result = subprocess.run(['node', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            version = result.stdout.strip()
            print_success(f"Node.js {version}")
            return True
    except FileNotFoundError:
        pass
    
    print_error("Node.js not found")
    print_info("Please install Node.js from: https://nodejs.org/")
    return False

def get_kaggle_config_path() -> Path:
    """Get the Kaggle API config directory"""
    if os.name == 'nt':  # Windows
        return Path.home() / '.kaggle'
    else:  # Unix-like
        return Path.home() / '.kaggle'

def check_kaggle_token() -> Tuple[bool, Optional[Path]]:
    """Check if Kaggle API token exists and is valid"""
    config_dir = get_kaggle_config_path()
    token_path = config_dir / 'kaggle.json'
    
    if not token_path.exists():
        return False, config_dir
    
    try:
        with open(token_path) as f:
            data = json.load(f)
            if 'username' in data and 'key' in data:
                print_success(f"Kaggle API token found: {token_path}")
                return True, config_dir
    except Exception as e:
        print_warning(f"Invalid Kaggle token: {e}")
    
    return False, config_dir

def setup_kaggle_token(config_dir: Path) -> bool:
    """Guide user through Kaggle API token setup"""
    print_header("Kaggle API Token Setup")
    
    print_info("The MABe Mouser app requires data from the Kaggle competition.")
    print_info("To download it automatically, you need a Kaggle API token.\n")
    
    print(f"{Colors.BOLD}Steps to get your token:{Colors.ENDC}")
    print("1. Go to: https://www.kaggle.com/settings/account")
    print("2. Log in (or create an account if you don't have one)")
    print("3. Scroll to 'API' section")
    print("4. Click 'Create New Token'")
    print("5. This will download 'kaggle.json'\n")
    
    input(f"{Colors.OKCYAN}Press Enter when you have downloaded kaggle.json...{Colors.ENDC}")
    
    # Look for kaggle.json in common locations
    search_paths = [
        Path.home() / 'Downloads' / 'kaggle.json',
        Path.home() / 'kaggle.json',
        Path.cwd() / 'kaggle.json',
    ]
    
    found_token = None
    for path in search_paths:
        if path.exists():
            found_token = path
            print_success(f"Found kaggle.json at: {path}")
            break
    
    if not found_token:
        print_warning("Could not auto-locate kaggle.json")
        token_path_input = input(f"{Colors.OKCYAN}Enter the full path to kaggle.json: {Colors.ENDC}").strip()
        found_token = Path(token_path_input)
        
        if not found_token.exists():
            print_error(f"File not found: {found_token}")
            return False
    
    # Validate the token
    try:
        with open(found_token) as f:
            data = json.load(f)
            if 'username' not in data or 'key' not in data:
                print_error("Invalid kaggle.json format")
                return False
    except Exception as e:
        print_error(f"Error reading kaggle.json: {e}")
        return False
    
    # Install the token
    config_dir.mkdir(parents=True, exist_ok=True)
    dest_path = config_dir / 'kaggle.json'
    shutil.copy2(found_token, dest_path)
    
    # Set proper permissions (Unix only)
    if os.name != 'nt':
        os.chmod(dest_path, 0o600)
    
    print_success(f"Kaggle token installed to: {dest_path}")
    return True

def check_kaggle_data() -> Tuple[bool, Path]:
    """Check if Kaggle competition data exists"""
    data_dir = Path(__file__).parent / 'MABe-mouse-behavior-detection'
    
    if not data_dir.exists():
        return False, data_dir
    
    # Check for essential files
    required_files = ['train.csv', 'test.csv', 'sample_submission.csv']
    required_dirs = ['train_tracking', 'train_annotation']
    
    for file in required_files:
        if not (data_dir / file).exists():
            print_warning(f"Missing required file: {file}")
            return False, data_dir
    
    for dir in required_dirs:
        if not (data_dir / dir).exists():
            print_warning(f"Missing required directory: {dir}")
            return False, data_dir
    
    print_success(f"Kaggle data found: {data_dir}")
    return True, data_dir

def download_kaggle_data(data_dir: Path) -> bool:
    """Download and extract Kaggle competition data"""
    print_header("Downloading Competition Data")
    
    try:
        # Import kaggle here to check if it's installed
        try:
            import kaggle
        except ImportError:
            print_info("Installing Kaggle API...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'kaggle'])
            import kaggle
        
        # Create temp directory for download
        temp_dir = Path(__file__).parent / '.kaggle_download'
        temp_dir.mkdir(exist_ok=True)
        
        print_info("Downloading MABe22 Mouse Behavior Detection dataset...")
        print_info("This may take several minutes depending on your connection...")
        
        # Download the competition files
        os.chdir(temp_dir)
        subprocess.check_call([
            sys.executable, '-m', 'kaggle', 'competitions', 'download',
            '-c', 'mabe22-mouse-behavior-recognition'
        ])
        
        # Find the zip file
        zip_files = list(temp_dir.glob('*.zip'))
        if not zip_files:
            print_error("No zip file downloaded")
            return False
        
        zip_path = zip_files[0]
        print_success(f"Downloaded: {zip_path.name}")
        
        # Extract
        print_info("Extracting data...")
        data_dir.mkdir(parents=True, exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        
        print_success(f"Data extracted to: {data_dir}")
        
        # Cleanup
        shutil.rmtree(temp_dir)
        
        return True
        
    except Exception as e:
        print_error(f"Failed to download data: {e}")
        return False

def install_python_dependencies() -> bool:
    """Install/update Python dependencies"""
    print_header("Python Dependencies")
    
    requirements_file = Path(__file__).parent / 'requirements.txt'
    
    if not requirements_file.exists():
        print_warning("requirements.txt not found, skipping Python dependencies")
        return True
    
    try:
        print_info("Installing/updating Python packages...")
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', '-q', '-r', str(requirements_file)
        ])
        print_success("Python dependencies up to date")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to install Python dependencies: {e}")
        return False

def install_frontend_dependencies() -> bool:
    """Install/update frontend dependencies"""
    print_header("Frontend Dependencies")
    
    frontend_dir = Path(__file__).parent / 'frontend'
    package_json = frontend_dir / 'package.json'
    
    if not package_json.exists():
        print_warning("package.json not found")
        return False
    
    try:
        print_info("Installing frontend packages (this may take a minute)...")
        subprocess.check_call(
            ['npm', 'install'],
            cwd=frontend_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE
        )
        print_success("Frontend dependencies up to date")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to install frontend dependencies: {e}")
        return False

def start_backend() -> Optional[subprocess.Popen]:
    """Start the FastAPI backend server"""
    print_header("Starting Backend Server")
    
    try:
        backend_dir = Path(__file__).parent
        print_info("Starting FastAPI server on http://localhost:8000...")
        
        # Don't capture output - let it show in terminal
        process = subprocess.Popen(
            [sys.executable, '-m', 'uvicorn', 'backend.main:app', '--host', '0.0.0.0', '--port', '8000'],
            cwd=backend_dir
        )
        
        # Wait a bit and check if it started successfully
        time.sleep(2)
        if process.poll() is not None:
            print_error("Backend failed to start - check output above")
            return None
        
        print_success("Backend server started")
        return process
        
    except Exception as e:
        print_error(f"Failed to start backend: {e}")
        return None

def start_frontend() -> Optional[subprocess.Popen]:
    """Start the Vite frontend dev server"""
    print_header("Starting Frontend Server")
    
    try:
        frontend_dir = Path(__file__).parent / 'frontend'
        print_info("Starting Vite dev server on http://localhost:3000...")
        
        # Don't capture output - let it show in terminal
        process = subprocess.Popen(
            ['npm', 'run', 'dev'],
            cwd=frontend_dir
        )
        
        # Wait a bit and check if it started successfully
        time.sleep(3)
        if process.poll() is not None:
            print_error("Frontend failed to start - check output above")
            return None
        
        print_success("Frontend server started")
        return process
        
    except Exception as e:
        print_error(f"Failed to start frontend: {e}")
        return None

def open_browser():
    """Open the app in the default browser"""
    print_info("Opening browser...")
    time.sleep(1)  # Give servers a moment to stabilize
    
    url = "http://localhost:3000"
    try:
        webbrowser.open(url)
        print_success(f"Opened {url}")
    except Exception as e:
        print_warning(f"Could not open browser automatically: {e}")
        print_info(f"Please manually open: {url}")

def main():
    """Main entry point"""
    print_header("🐱 MABe Mouser - Startup")
    
    # 1. Check system requirements
    print_header("Checking System Requirements")
    if not check_python_version():
        return 1
    if not check_node_installed():
        return 1
    
    # 2. Setup Kaggle API token
    has_token, config_dir = check_kaggle_token()
    if not has_token:
        if not setup_kaggle_token(config_dir):
            print_error("Kaggle token setup failed. Cannot continue.")
            return 1
    
    # 3. Check/download competition data
    has_data, data_dir = check_kaggle_data()
    if not has_data:
        print_warning("Competition data not found")
        response = input(f"{Colors.OKCYAN}Download it now? [Y/n]: {Colors.ENDC}").strip().lower()
        if response in ['', 'y', 'yes']:
            if not download_kaggle_data(data_dir):
                print_error("Data download failed. Cannot continue.")
                return 1
        else:
            print_error("Data required to run the app. Exiting.")
            return 1
    
    # 4. Install dependencies
    if not install_python_dependencies():
        return 1
    if not install_frontend_dependencies():
        return 1
    
    # 5. Start servers
    global backend_process, frontend_process
    
    backend_process = start_backend()
    if not backend_process:
        return 1
    
    frontend_process = start_frontend()
    if not frontend_process:
        cleanup_processes()
        return 1
    
    # 6. Open browser
    open_browser()
    
    # 7. Keep running
    print_header("🚀 MABe Mouser is Running!")
    print_success("Backend:  http://localhost:8000")
    print_success("Frontend: http://localhost:3000")
    print_info("\nPress Ctrl+C to stop the servers\n")
    
    try:
        # Monitor processes
        while True:
            time.sleep(1)
            
            if backend_process and backend_process.poll() is not None:
                print_error("Backend server crashed!")
                break
            
            if frontend_process and frontend_process.poll() is not None:
                print_error("Frontend server crashed!")
                break
    except KeyboardInterrupt:
        pass
    
    cleanup_processes()
    return 0

if __name__ == '__main__':
    sys.exit(main())
