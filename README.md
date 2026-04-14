# Installation and Usage
_notes by Julian Miguel Roxas_

Open a terminal (Command Prompt or PowerShell on Windows, Terminal on Mac) and run:

1. git clone https://github.com/abunjingbunjing/speech-command-rl.git
2. cd speech-command-rl
3. python -m venv venv

4. environement activation
for windows:

venv\Scripts\activate

for mac/linux:

source venv/bin/activate

6. In your Windows POWERSHELL:
curl -o speech_commands_v0.02.tar.gz https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz

7. mkdir data\speech_commands
8. tar -xzf speech_commands_v0.02.tar.gz -C data\speech_commands
9. pip install -r requirements.txt
10. bash run.sh

All output files will be saved in experiments/results/

# Data

Download the dataset file via browser

Click this link to download directly:
👉 https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz
Your browser will download a file called "speech_commands_v0.02.tar.gz"
It is about 2.3GB so it may take a few minutes depending on your internet speed
By default it saves to your Downloads folder.

To download manually:

wget https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz
tar -xzf speech_commands_v0.02.tar.gz -C ./speech_commands
