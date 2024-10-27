import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import NeuralNet
from utils import tokenize, stem, bag_of_words
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    filename=f'training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def load_and_preprocess_data():
    with open("data/intents.json", "r", encoding="utf-8") as f:
        intents = json.load(f)

    all_words = []
    tags = []
    xy = []

    for intent in intents["intents"]:
        tag = intent["tag"]
        tags.append(tag)
        for pattern in intent["patterns"]:
            w = tokenize(pattern.lower())
            all_words.extend(w)
            xy.append((w, tag))

    ignore_words = ["?", "!", ".", ",", "'s", "the", "a", "an"]
    all_words = [stem(w) for w in all_words if w not in ignore_words]
    all_words = sorted(set(all_words))
    tags = sorted(set(tags))

    logging.info(f"Unique words in dataset: {len(all_words)}")
    logging.info(f"Tags: {tags}")

    return all_words, tags, xy


def create_training_data(all_words, tags, xy):
    X = []
    y = []

    for pattern_sentence, tag in xy:
        bag = bag_of_words(pattern_sentence, all_words)
        X.append(bag)
        label = tags.index(tag)
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_val, y_train, y_val


class ChatDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.n_samples = len(X_data)
        self.x_data = torch.FloatTensor(X_data)
        self.y_data = torch.LongTensor(y_data)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples


def train_model(X_train, X_val, y_train, y_val, all_words, tags):
    batch_size = 16
    hidden_size = 64  # Increased hidden size for more complexity
    output_size = len(tags)
    input_size = len(X_train[0])
    learning_rate = 0.001
    num_epochs = 3000  # Increased epochs

    # Calculate class weights to handle imbalanced data
    class_weights = np.bincount(y_train)
    class_weights = torch.FloatTensor(1.0 / (class_weights + 1)).to(torch.device("cpu"))

    # Model parameters and dataset preparation
    train_dataset = ChatDataset(X_train, y_train)
    val_dataset = ChatDataset(X_val, y_val)

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralNet(input_size, hidden_size, output_size).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=20, verbose=True
    )

    best_val_loss = float("inf")
    patience_counter = 0
    patience_limit = 50

    logging.info("Starting training...")

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0

        for words, labels in train_loader:
            words, labels = words.to(device), labels.to(device)

            outputs = model(words)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        # Validation loop
        model.eval()
        total_val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for words, labels in val_loader:
                words, labels = words.to(device), labels.to(device)

                outputs = model(words)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1

        if patience_counter >= patience_limit:
            logging.info(f"Early stopping triggered at epoch {epoch+1}")
            break

        if (epoch + 1) % 100 == 0:
            logging.info(
                f"Epoch [{epoch+1}/{num_epochs}], "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Val Loss: {avg_val_loss:.4f}, "
                f"Val Accuracy: {val_accuracy:.2f}%"
            )

    # Save the best model
    model.load_state_dict(best_model_state)
    data = {
        "model_state": model.state_dict(),
        "input_size": input_size,
        "hidden_size": hidden_size,
        "output_size": output_size,
        "all_words": all_words,
        "tags": tags,
    }

    FILE = "data/model/chatbot_model.pth"
    torch.save(data, FILE)
    logging.info(f"Training complete. Best model saved to {FILE}")

    return model


if __name__ == "__main__":
    all_words, tags, xy = load_and_preprocess_data()
    X_train, X_val, y_train, y_val = create_training_data(all_words, tags, xy)
    train_model(X_train, X_val, y_train, y_val, all_words, tags)
