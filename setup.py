import subprocess
import sys
import os

def install_dependencies():
    """Install all required dependencies."""
    print("Installing dependencies for Memory Chatbot...")
    
    # Read requirements from file
    with open("requirements.txt", "r") as f:
        requirements = f.read().splitlines()
    
    # Filter out comments and empty lines
    requirements = [r for r in requirements if r and not r.startswith("#")]
    
    # Install each requirement
    for req in requirements:
        print(f"Installing {req}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", req])
        except subprocess.CalledProcessError:
            print(f"Failed to install {req}")
    
    print("Dependencies installed successfully!")

def setup_qdrant():
    """Set up Qdrant if not already running."""
    try:
        # Check if Docker is installed
        subprocess.check_call(["docker", "--version"], stdout=subprocess.DEVNULL)
        
        # Check if Qdrant container is running
        result = subprocess.run(
            ["docker", "ps", "--filter", "name=qdrant", "--format", "{{.Names}}"],
            capture_output=True,
            text=True
        )
        
        if "qdrant" not in result.stdout:
            print("Starting Qdrant container...")
            subprocess.check_call([
                "docker", "run", "-d",
                "--name", "qdrant",
                "-p", "6333:6333",
                "-p", "6334:6334",
                "-v", "qdrant_data:/qdrant/storage",
                "qdrant/qdrant"
            ])
            print("Qdrant container started!")
        else:
            print("Qdrant container is already running.")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Docker not found or error starting Qdrant container.")
        print("You can still use the application with FAISS fallback.")

def create_directories():
    """Create necessary directories."""
    os.makedirs("data", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    print("Created necessary directories.")

if __name__ == "__main__":
    create_directories()
    install_dependencies()
    setup_qdrant()
    
    print("\nSetup complete! You can now run the application with:")
    print("streamlit run app.py")
