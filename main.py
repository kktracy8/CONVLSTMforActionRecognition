import copy
import random
import ConvLSTM
import caller
import torch
import torch.optim as optim
from salad50 import SaladDataset
from caller import Model
from Loss import CrossEntropyLoss
import pandas as pd
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    train_loss_history, train_acc_history, valid_loss_history, valid_acc_history = run()

def run():
    input_size = 1  # or 3 if RGB
    attention = False  # can set to False or spatial if applicable
    output_size = 56 #CHANGE to 5 if consolidating
    stride = (1, 1, 1)
    padding = 1
    kernel_size = (7, 7) #try smaller if needed
    # dropout_rate = 0.3  # Dropout rate added
    batch_size = 2
    epochs = 4
    patience = 5  # Early stopping patience
    learning_rate = 0.001
    ids = [i for i in range(45)]

    path1 = './rgb'
    path2 = './activityAnnotations'
    path3 = './timestamps'

    dataset = SaladDataset(path1, path2, path3)

    # Initialize Model with Dropout
    model = Model(input_size, output_size, kernel_size, stride, padding, device, attention)

    # Optimizer, Scheduler, and Scaler
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
    scaler = torch.cuda.amp.GradScaler()

    train_loss_history, train_acc_history = [], []
    valid_loss_history, valid_acc_history = [], []
    best_acc, best_model = 0.0, None
    best_loss = float('inf')
    patience_counter = 0  # Early stopping counter

    for epoch in range(epochs):
        # Shuffle IDs
        ids = sorted(ids, key=lambda x: random.random())
        split_index = int(len(ids) * 0.7)
        val_index = int(len(ids) * 0.15)
        train_id = ids[:split_index]
        val_id = ids[split_index:split_index + val_index]

        # Load Data
        data = dataset.generate_arrays(train_id)
        val = dataset.generate_arrays(val_id)

        # Training Loop
        e_loss, e_acc, total_train_samples = 0, 0, 0
        for num, (frames, labels) in enumerate(data):
            frames = torch.tensor(frames).unsqueeze(0).unsqueeze(2).to(device)
            labels = torch.tensor(labels).to(device)

            i = 200
            while i <= labels.shape[0]:
                if epoch == (epochs - 1):
                    if num == 0 and i == 200:
                        savehooks = True
                    else:
                        savehooks = False
                else:
                    savehooks = False 
                window = frames[:, i - 200:i, :, :, :]
                targe = labels[i - 200:i]

                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    output = model.forward(window,save_hooks=savehooks).squeeze(0)
                    acc, loss = CrossEntropyLoss(output, targe.long())

                scaler.scale(loss).backward()
                if savehooks and len(model.conv_lstm.hook_data['grad_output']) > 0:
                    np.save('hook_data_grad_output.npy', np.array(model.conv_lstm.hook_data['grad_output']))
                scaler.step(optimizer)
                scaler.update()

                e_loss += loss.item()
                e_acc += acc.item() * targe.size(0)
                total_train_samples += targe.size(0)
                i += 125

            del frames, labels
            torch.cuda.empty_cache()
            if num == len(train_id) - 1:
                break

        train_loss_history.append(e_loss / total_train_samples)
        train_acc_history.append(e_acc / total_train_samples)

        # Validation Loop
        v_loss, v_acc, total_val_samples = 0, 0, 0
        for num, (frames, labels) in enumerate(val):
            frames = torch.tensor(frames).unsqueeze(0).unsqueeze(2).to(device)
            labels = torch.tensor(labels).to(device)

            i = 175
            while i <= labels.shape[0]:
                window = frames[:, i - 175:i, :, :, :]
                targe = labels[i - 175:i]

                with torch.no_grad():
                    output = model.forward(window,save_hooks=False).squeeze(0)
                    acc, loss = CrossEntropyLoss(output, targe.long())
                    v_loss += loss.item()
                    v_acc += acc.item() * targe.size(0)
                    total_val_samples += targe.size(0)
                i += 125

            del frames, labels
            torch.cuda.empty_cache()
            if num == len(val_id) - 1:
                break

        valid_loss_history.append(v_loss / total_val_samples)
        valid_acc_history.append(v_acc / total_val_samples)

        print(f"Epoch {epoch + 1}: Train Loss {train_loss_history[-1]:.4f}, Validation Loss {valid_loss_history[-1]:.4f}, Validation Accuracy {valid_acc_history[-1]:.4f}")

        # Early Stopping Logic
        if valid_loss_history[-1] < best_loss:
            best_loss = valid_loss_history[-1]
            best_model = copy.deepcopy(model)
            patience_counter = 0  # Reset counter if validation improves
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

        # Scheduler Step
        scheduler.step(valid_loss_history[-1])

    print("Training complete.")
    print(f"Best Validation Loss: {best_loss:.4f}")

    # # Save Best Model
    # torch.save(best_model.state_dict(), "best_model.pth")

    return train_loss_history, train_acc_history, valid_loss_history, valid_acc_history


if __name__ == '__main__':
    random.seed(123)
    train_loss_history, train_acc_history, valid_loss_history, valid_acc_history = run()
    # Save history to CSV
    history = {
        "epoch": list(range(1, len(train_loss_history) + 1)),
        "train_loss": train_loss_history,
        "train_accuracy": train_acc_history,
        "valid_loss": valid_loss_history,
        "valid_accuracy": valid_acc_history,
    }
    df_history = pd.DataFrame(history)
    df_history.to_csv("./training_history.csv", index=False)
    print("Training history saved to training_history.csv")

    # Plot Loss Curve
    plt.figure(figsize=(10, 6))
    plt.plot(history["epoch"], history["train_loss"], label="Train Loss")
    plt.plot(history["epoch"], history["valid_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig("./loss_curve.png")
    plt.show()

    # Plot Accuracy Curve
    plt.figure(figsize=(10, 6))
    plt.plot(history["epoch"], history["train_accuracy"], label="Train Accuracy")
    plt.plot(history["epoch"], history["valid_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig("./accuracy_curve.png")
    plt.show()
