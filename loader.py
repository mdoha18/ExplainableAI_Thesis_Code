import torch
from torch.utils.data import Dataset


class AudioSpectrogramDataset(Dataset):
    """Dataset class that loads preprocessed spectrograms and labels from a .pt file."""
    def __init__(self, data_file, max_length=300):
        self.data = torch.load(data_file)
        self.max_length = max_length

        self.spectrograms = self.data['spectrograms']
        self.labels = self.data['labels']

        self.spectrograms = [self.pad_spectrogram(spectrogram) for spectrogram in self.spectrograms]

    def pad_spectrogram(self, spectrogram):
        if spectrogram.size(2) < self.max_length:
            spectrogram = torch.nn.functional.pad(spectrogram, (0, self.max_length - spectrogram.size(2)))
        return spectrogram[:, :, :self.max_length]

    def __len__(self):
        return len(self.spectrograms)

    def get_label_path(self, idx):
        return self.samples[idx]

    def __getitem__(self, idx):
        spectrogram = self.spectrograms[idx]
        label = self.labels[idx]
        return spectrogram, label

