import torch
import torch.nn as nn
import torch.optim as optim

from src.dataset import get_data_loaders
from src.models import MLP, CNN, AttentionModel

def train_model(model, train_loader, criterion, optimizer, epochs=5):
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            # Clear previous gradients.
            optimizer.zero_grad()

            # Forward propagation.
            outputs = model(images)

            # Calculate loss.
            loss = criterion(outputs, labels)

            # Backpropagation.
            loss.backward()

            # Update model parameters.
            optimizer.step()

            running_loss += loss.item()

            # Calculate training accuracy.
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Loss: {epoch_loss:.4f} "
            f"Accuracy: {epoch_accuracy:.2f}%"
        )


def evaluate_model(model, test_loader, criterion):
    model.eval()

    correct = 0
    total = 0
    test_loss = 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)

            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    average_loss = test_loss / len(test_loader)

    print(f"Test Loss: {average_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.2f}%")

    return accuracy

def run_training_experiment():
    # Load the Fashion-MNIST datasets.
    train_loader, val_loader, test_loader = get_data_loaders(
        batch_size=128
    )

    # Create the three models.
    models = {
        "MLP": MLP(),
        "CNN": CNN(),
        "Attention": AttentionModel()
    }

    criterion = nn.CrossEntropyLoss()

    for model_name, model in models.items():
        print("\n" + "=" * 60)
        print(f"TRAINING {model_name}")
        print("=" * 60)

        # Create Adam optimizer.
        optimizer = optim.Adam(
            model.parameters(),
            lr=0.001
        )

        # Train for 5 epochs.
        train_model(
            model,
            train_loader,
            criterion,
            optimizer,
            epochs=5
        )

        # Evaluate on the test dataset.
        print(f"\n{model_name} TEST RESULTS")
        evaluate_model(model, test_loader, criterion)


if __name__ == "__main__":
    run_training_experiment()