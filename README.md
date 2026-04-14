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

5. pip install -r requirements.txt
6. bash run.sh

All output files will be saved in experiments/results/

# Data

The dataset is downloaded automatically when you run the notebook.
It requires an internet connection and approximately 2.3GB of free space.

To download manually:

wget https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz
tar -xzf speech_commands_v0.02.tar.gz -C ./speech_commands
