#!/usr/bin/env python3
"""
Startup script for Flood Damage Assessment Tool
Kills existing processes and starts the application cleanly
"""

import os
import sys
import signal
import subprocess
import time

def kill_existing_processes():
    """Kill any existing processes using our ports"""
    print("Cleaning up existing processes...")
    
    # Kill processes using port 8000 (backend)
    try:
        result = subprocess.run(['lsof', '-ti:8000'], capture_output=True, text=True)
        if result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                try:
                    print(f"   Killing process {pid} on port 8000")
                    os.kill(int(pid), signal.SIGTERM)
                    time.sleep(1)  # Give it time to die gracefully
                    # Force kill if still alive
                    try:
                        os.kill(int(pid), signal.SIGKILL)
                    except ProcessLookupError:
                        pass  # Already dead
                except (ValueError, ProcessLookupError):
                    pass
    except FileNotFoundError:
        pass  # lsof not found, skip
    
    # Kill processes using port 3000 (frontend)
    try:
        result = subprocess.run(['lsof', '-ti:3000'], capture_output=True, text=True)
        if result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                try:
                    print(f"   Killing process {pid} on port 3000")
                    os.kill(int(pid), signal.SIGTERM)
                    time.sleep(1)
                    try:
                        os.kill(int(pid), signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                except (ValueError, ProcessLookupError):
                    pass
    except FileNotFoundError:
        pass
    
    # Kill any existing app.py processes
    try:
        result = subprocess.run(['pgrep', '-f', 'python.*app.py'], capture_output=True, text=True)
        if result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                try:
                    print(f"   Killing app.py process {pid}")
                    os.kill(int(pid), signal.SIGTERM)
                    time.sleep(1)
                    try:
                        os.kill(int(pid), signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                except (ValueError, ProcessLookupError):
                    pass
    except FileNotFoundError:
        pass
    
    print("Cleanup completed")
    time.sleep(2)  # Give system time to clean up

def main():
    print("Starting Flood Damage Assessment Tool")
    print("=" * 50)
    
    # Kill existing processes
    kill_existing_processes()
    
    # Import and start the FastAPI app
    print("Starting backend server...")
    
    # Change to backend directory
    backend_dir = os.path.join(os.path.dirname(__file__), 'backend')
    os.chdir(backend_dir)
    
    # Import the app after changing directory
    sys.path.insert(0, backend_dir)
    from app import app
    
    # Start the server
    import uvicorn
    
    print("Starting server on http://localhost:8000")
    print("Frontend should be accessed at http://localhost:3000")
    print("\nOpen your browser to http://localhost:3000 to use the app")
    print("Remember to start the frontend server in another terminal:")
    print("   cd frontend && python -m http.server 3000")
    print("\nPress Ctrl+C to stop the server")
    
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
    except KeyboardInterrupt:
        print("\nShutting down server...")
        kill_existing_processes()
        print("Server stopped cleanly")

if __name__ == "__main__":
    main()