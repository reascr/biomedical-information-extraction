#!/bin/bash

# Submit a job on ucloud

LOCAL_PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#LOCAL_PROJECT_DIR="/Users/rea/Library/Mobile Documents/com~apple~CloudDocs/IT_and_Cognition/MSc_IT_and_Cognition/Masterthesis/Biomedical_Information_Extraction"# Local directory path
SERVER_PROJECT_DIR="/home/ucloud/" # Server project directory path
RESULTS_DIR_SERVER="$SERVER_PROJECT_DIR/results/" # path to training results on server
echo "Enter Port:"
read SSH_PORT

SSH_KEY_PATH="~/.ssh/id_rsa" 

echo "Copying from local: $LOCAL_PROJECT_DIR to remote: $SERVER_PROJECT_DIR"
scp -P $SSH_PORT -i "$SSH_KEY_PATH" -r "$LOCAL_PROJECT_DIR" ucloud@ssh.cloud.sdu.dk:"$SERVER_PROJECT_DIR"

ssh -i "$SSH_KEY_PATH" -p $SSH_PORT ucloud@ssh.cloud.sdu.dk << 'EOF' # SSH into the server and run commands

    # Install Python 3.8 (server has python version 3.12.3. We need to downgrade to python 3.10.12)
    export DEBIAN_FRONTEND=noninteractive  # Disable interactive prompts
    sudo apt update -y
    sudo apt install -y software-properties-common
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt install -y python3.10 python3.10-venv python3.10-dev

    # Navigate to the project directory
    cd /home/ucloud/biomedical-information-extraction 

    # Set up virtual environment with Python 3.10
    python3.10 -m venv venv  # Create virtual environment with Python 3.10
    source venv/bin/activate  # Activate the virtual environment

    # Install required dependencies from the requirements.txt
    pip install -r requirements.txt

    # Run the script in the background using nohup (write output to both output.log and terminal)
    echo "Running the script in the background..."
    # nohup python3 scripts/optuna_NER.py 2>&1 | tee output.log &
    echo "Job started in the background." # If not using this script to connect to the server, make sure to activate the venv before running the code

EOF

