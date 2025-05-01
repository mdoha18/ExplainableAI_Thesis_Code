import torch
import os

# Example data
spectrograms = [torch.randn(1, 128, 300) for _ in range(100)]
labels = [torch.randint(0, 35, (1,)).item() for _ in range(100)]

# Save the dataset
data = {'spectrograms': spectrograms, 'labels': labels}
save_path = '/Users/mekholadoha/Desktop/2D_SpeechCommands/pytorch-grad-cam/dataset.pt'
torch.save(data, save_path)

# Confirm the dataset is saved
if os.path.exists(save_path):
    print(f'Dataset successfully saved to {save_path}')
else:
    print('Failed to save the dataset.')