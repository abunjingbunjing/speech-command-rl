# Model Card: Speech Command Recognition + RL Threshold Tuning

## Model details
- Architecture: 3-layer Spectrogram CNN
- Task: Class Speech Command Classification
- Framework: PyTorch

## Training data
- Dataset: Google Speech Commands v2
- Classes: yes, no, up, down, left, right, on, off, stop, go
- Split: 70% train / 15% val / 15% test

## Performance
- Test Accuracy: 0.84
- Macro F1: 0.84
- RL final threshold: 0.594

## Intended use
Educational demonstration of CNN + RL pipeline for speech recognition.
NOT for deployment in safety-critical systems.

## Limitations
- Trained on English commands only
- May underperform for non-native speakers
- Small dataset per class (800 samples)
