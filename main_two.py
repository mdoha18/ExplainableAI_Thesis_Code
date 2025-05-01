import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from loader import AudioSpectrogramDataset
from resnet import ResNetAudio


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / total_samples
    accuracy = 100 * total_correct / total_samples
    return avg_loss, accuracy


def determine_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(model, train_loader, val_loader, criterion, optimizer, epochs=10, model_save_path='./model_saves'):
    device = determine_device()
    model.to(device)
    print(f'Code will run on {device}')
    os.makedirs(model_save_path, exist_ok=True) 

    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        train_loss, train_acc = evaluate(model, train_loader, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

        model_path = os.path.join(model_save_path, f'model_twoD_epoch{epoch+1}.pth')
        torch.save(model.state_dict(), model_path)

    final_model_path = os.path.join(model_save_path, 'model_2d.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved at {final_model_path}")


def main():
    dataset_path = '/home3/s4790650/test_dataset/speech_commands'
    dataset = AudioSpectrogramDataset(root_dir=dataset_path, max_length=300)
    num_train = int(0.6 * len(dataset))
    num_val = int(0.2 * len(dataset))
    num_test = len(dataset) - num_train - num_val
    train_dataset, val_dataset, test_dataset = random_split(dataset, [num_train, num_val, num_test])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = ResNetAudio(num_classes=35)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train(model, train_loader, val_loader, criterion, optimizer, model_save_path='/home3/s4790650/model_saves')

    test_loss, test_acc = evaluate(model, test_loader, criterion, determine_device())
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%')


if __name__ == '__main__':
    main()
