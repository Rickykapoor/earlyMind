import os
from pathlib import Path
from src.config import cfg
from src.data.eeg_loader import process_eeg_dataset

print("Test script running!")

cfg.paths.makedirs()

# Mock dummy EDF files in raw directory
import numpy as np
cfg.paths.eeg_raw.mkdir(parents=True, exist_ok=True)
dummy_edf = cfg.paths.eeg_raw / 'eeg10.edf'
# If the real file isn't there, we can't easily mock an EDF. But we already have the real files locally!

if len(list(cfg.paths.eeg_raw.glob("*.edf"))) == 0:
    print("No EDF files found locally, skipping test!")
else:
    results = process_eeg_dataset(cfg.paths.eeg_raw, cfg.paths.eeg_processed)
    print("Test finished.")
    if (cfg.paths.eeg_processed / 'labels.csv').exists():
        print(f"File EXISTS: {cfg.paths.eeg_processed / 'labels.csv'}")
    else:
        print("CRITICAL BUG: File DOES NOT EXIST!")
