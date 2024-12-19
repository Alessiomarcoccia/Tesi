import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from typing import Any, Callable, Dict, Optional, Tuple
from torch import Tensor
from torchvision.datasets import UCF101

from model import resnet50

class CustomUCF101(UCF101):
    def __init__(
        self,
        root: str,
        annotation_path: str,
        frames_per_clip: int,
        step_between_clips: int = 1,
        frame_rate: Optional[int] = None,
        fold: int = 1,
        train: bool = True,
        transform: Optional[Callable] = None,
        _precomputed_metadata: Optional[Dict[str, Any]] = None,
        num_workers: int = 1,
        _video_width: int = 0,
        _video_height: int = 0,
        _video_min_dimension: int = 0,
        _audio_samples: int = 0,
        output_format: str = "THWC",
    ) -> None:
        # Initialize the parent UCF101 class
        super().__init__(
            root=root,
            annotation_path=annotation_path,
            frames_per_clip=frames_per_clip,
            step_between_clips=step_between_clips,
            frame_rate=frame_rate,
            fold=fold,
            train=train,
            transform=transform,
            _precomputed_metadata=_precomputed_metadata,
            num_workers=num_workers,
            _video_width=_video_width,
            _video_height=_video_height,
            _video_min_dimension=_video_min_dimension,
            _audio_samples=_audio_samples,
            output_format=output_format,
        )

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        """
        Override __getitem__ to ignore audio and only return video and label.
        """
        video, audio, info, video_idx = self.video_clips.get_clip(idx)
        label = self.samples[self.indices[video_idx]][1]

        # Apply the transform if specified
        if self.transform is not None:
            video = self.transform(video)

        # Return only video and label (ignoring audio)
        return video, label

# Calculate accuracy (a classification metric)
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100
    return acc


def main():
    # Define hyperparameters
    batch_size = 4
    learning_rate = 0.001
    num_epochs = 10
    num_classes = 101  # UCF101 has 101 action classes
    path1 = "processed_train_dataset_clip16NoAudio.pt"
    path2 = "processed_val_dataset_clip16NoAudio.pt"
    train_dataset = torch.load(path1)
    val_dataset = torch.load(path2)
    print("Train dataset loaded!")# val_dataset = torch.load("processed_val_dataset.pt")
    print("Validation dataset loaded!")

    # for i, path in enumerate(train_dataset.video_clips.video_paths):
    #   # Convert Windows-style paths to Linux-style paths
    #   linux_path = path.replace("\\", "/")
    #   # Update the base directory to match your Colab structure
    #   updated_path = linux_path.replace("UCF101_subset", "/content/drive/MyDrive/UCF101_subset")
    #   train_dataset.video_clips.video_paths[i] = updated_path

    # for i, path in enumerate(val_dataset.video_clips.video_paths):
    #   # Convert Windows-style paths to Linux-style paths
    #   linux_path = path.replace("\\", "/")
    #   # Update the base directory to match your Colab structure
    #   updated_path = linux_path.replace("UCF101_subset", "/content/drive/MyDrive/UCF101_subset")
    #   val_dataset.video_clips.video_paths[i] = updated_path


    # random_index = random.randint(0, len(train_dataset) - 1)
    # sample = train_dataset[random_index] #video, label
    # print(f"Returned data structure: {type(sample)}")
    # print(f"Sample shape: {len(sample)}")
    # print(f"Video shape: {sample[0].shape}")
    # print(f"Video type: {type(sample[0])}")
    # print(f"Label: {sample[1]}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    print("Train loader created!")
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    print("Dataloaders created!")

    print(f"Number of batches in train loader: {len(train_loader)}")
    print(f"Number of batches in val loader: {len(val_loader)}")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = resnet50()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate,momentum=0.9)

    # for epoch in range(num_epochs):
    #   model.train()
    #   running_loss = 0.0
    #   correct = 0
    #   total = 0

    #   for inputs, labels in train_loader:
    #       inputs, labels = inputs.to(device), labels.to(device)

    #       optimizer.zero_grad()
    #       outputs = model(inputs)
    #       loss = criterion(outputs, labels)
    #       loss.backward()
    #       optimizer.step()

    #       running_loss += loss.item()
    #       _, predicted = outputs.max(1)
    #       total += labels.size(0)
    #       correct += (predicted == labels).sum().item()

    #   train_acc = 100.0 * correct / total
    #   train_loss = running_loss / len(train_loader)

    #   # Validation loop
    #   model.eval()
    #   val_loss = 0.0
    #   val_correct = 0
    #   val_total = 0

    #   with torch.no_grad():
    #       for inputs, labels in val_loader:
    #           inputs, labels = inputs.to(device), labels.to(device)
    #           outputs = model(inputs)
    #           loss = criterion(outputs, labels)

    #           val_loss += loss.item()
    #           _, predicted = outputs.max(1)
    #           val_total += labels.size(0)
    #           val_correct += (predicted == labels).sum().item()

    #   val_acc = 100.0 * val_correct / val_total
    #   val_loss /= len(val_loader)

    #   print(f"Epoch [{epoch + 1}/{num_epochs}] "
    #         f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% "
    #         f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    for epoch in tqdm(range(num_epochs)):
        print(f"Epoch: {epoch}\n-------")
        ### Training
        train_loss, train_acc = 0,0
        # Add a loop to loop through training batches
        for batch, (X, y) in enumerate(train_loader):
            X,y = X.to(device), y.to(device)
            model.train()

            #1.Forward pass
            y_pred = model(X)

            train_acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))

            #2.Calculate loss per batch
            loss = criterion(y_pred,y)
            train_loss += loss #accumulatively add up the loss per epoch

            # 3. Optimizer zero grad
            optimizer.zero_grad()

            # 4. Loss backward
            loss.backward()

            # 5. Optimizer step
            optimizer.step()

            if batch % 50 == 0:
              print(f"Looked at {batch * len(X)}/{len(train_loader.dataset)} samples")

        # Divide total train loss by length of train dataloader (average loss per batch per epoch)
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        train_acc /= len(train_loader)
        train_accuracies.append(train_acc)

        # Print out what's happening
        print(f"Train loss: {train_loss:.5f}")

        ### Testing
        # Setup variables for accumulatively adding up loss and accuracy
        val_loss, val_acc = 0, 0
        model.eval()
        with torch.inference_mode():
          for X, y in val_loader:
            X,y = X.to(device), y.to(device)
            # 1. Forward pass
            val_pred = model(X)

            # 2. Calculate loss (accumulatively)
            val_loss += criterion(val_pred, y) # accumulatively add up the loss per epoch

            # 3. Calculate accuracy (preds need to be same as y_true)
            val_acc += accuracy_fn(y_true=y, y_pred=val_pred.argmax(dim=1))

          # Calculations on test metrics need to happen inside torch.inference_mode()
          # Divide total test loss by length of test dataloader (per batch)
          val_loss /= len(val_loader)
          val_losses.append(val_loss)

          # Divide total accuracy by length of test dataloader (per batch)
          val_acc /= len(val_loader)
          val_accuracies.append(val_acc)


        ## Print out what's happening
        print(f"\nTrain loss: {train_loss:.5f} | Test loss: {val_loss:.5f}, Test acc: {val_acc:.2f}%\n")

    print(f"Train losses: {train_losses}")
    print(f"Train accuracies: {train_accuracies}")
    print(f"Val losses: {val_losses}")
    print(f"Val accuracies: {val_accuracies}")

     # Save the trained model
    torch.save(model.state_dict(), 'ResnetCustom.pth')


    epochs = range(1, num_epochs + 1)

    train_losses = [loss.detach().cpu().numpy() if isinstance(loss, torch.Tensor) else loss for loss in train_losses]
    val_losses = [loss.detach().cpu().numpy() if isinstance(loss, torch.Tensor) else loss for loss in val_losses]
    train_accuracies = [acc.cpu().numpy() if isinstance(acc, torch.Tensor) else acc for acc in train_accuracies]
    val_accuracies = [acc.cpu().numpy() if isinstance(acc, torch.Tensor) else acc for acc in val_accuracies]

    # Plot Loss
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Training Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()