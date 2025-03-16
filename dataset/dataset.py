import torch
import json
import numpy as np
from torch.utils.data import Dataset

class ResumeDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        self.categories = sorted(set(r['category'] for r in data['resumes']))
        self.texts = [r['text'] for r in data['resumes']]
        self.labels = [self.categories.index(r['category']) for r in data['resumes']]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]
