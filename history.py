import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np


class History:

    def __init__(self, history=None):
        self.history = history

        if self.history == None:
            self.load_history()

    def concat_history(self, data, value_name):
        rows = []
        for epoch_idx, epoch_values in enumerate(data):
            for batch_idx, value in enumerate(epoch_values):
                rows.append({
                    "epoch": epoch_idx + 1,
                    "batch": batch_idx + 1,
                    value_name: value
                })
        return pd.DataFrame(rows)

    def save_full_history(self, save_dir="training_logs"):

        os.makedirs(save_dir, exist_ok=True)

        train_loss = self.concat_history(
            self.history['train_loss'], "train_loss")
        train_loss.to_csv(os.path.join(
            save_dir, "train_loss_batches.csv"), index=False)

        train_acc = self.concat_history(
            self.history['train_accuracy'], "train_accuracy")
        train_acc.to_csv(os.path.join(
            save_dir, "train_accuracy_batches.csv"), index=False)

        valid_loss = self.concat_history(['valid_loss'], "valid_loss")
        valid_loss.to_csv(os.path.join(
            save_dir, "valid_loss_batches.csv"), index=False)

        valid_acc = self.concat_history(
            self.history['valid_accuracy'], "valid_accuracy")
        valid_acc.to_csv(os.path.join(
            save_dir, "valid_accuracy_batches.csv"), index=False)

        valid_err = pd.DataFrame({
            "epoch": list(range(1, len(self.history['valid_error']) + 1)),
            "valid_error": self.history['valid_error']
        })
        valid_err.to_csv(os.path.join(
            save_dir, "valid_error_epochs.csv"), index=False)

        self.history_df = {'train_loss': train_loss,
                           'train_acc': train_acc,
                           'valid_loss': valid_loss,
                           'valid_acc': valid_acc,
                           'valid_err': valid_err}

        print(f"history saved inside folder: {save_dir}")

    def load_history(self, log_dir="training_logs"):

        train_loss = pd.read_csv(f"{log_dir}/train_loss_batches.csv")
        valid_loss = pd.read_csv(f"{log_dir}/valid_loss_batches.csv")
        train_acc = pd.read_csv(f"{log_dir}/train_accuracy_batches.csv")
        valid_acc = pd.read_csv(f"{log_dir}/valid_accuracy_batches.csv")
        valid_err = pd.read_csv(f"{log_dir}/valid_error_epochs.csv")

        print(f"history loaded from folder: {log_dir}")

        self.history_df = {'train_loss': train_loss,
                           'train_acc': train_acc,
                           'valid_loss': valid_loss,
                           'valid_acc': valid_acc,
                           'valid_err': valid_err}

    def plot_epoch(self):

        train_loss_epoch = self.history_df["train_loss"].groupby("epoch")[
            "train_loss"].mean()
        train_acc_epoch = self.history_df["train_acc"].groupby("epoch")[
            "train_accuracy"].mean()
        valid_loss_epoch = self.history_df["valid_loss"].groupby("epoch")[
            "valid_loss"].mean()
        valid_acc_epoch = self.history_df["valid_acc"].groupby("epoch")[
            "valid_accuracy"].mean()

        epochs = train_loss_epoch.index

        plt.figure()
        plt.plot(epochs, train_acc_epoch.values)
        plt.plot(epochs, valid_acc_epoch.values)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Training vs Validation Accuracy")
        plt.legend(["Train", "Validation"])
        plt.show()

        plt.figure()
        plt.plot(epochs, train_loss_epoch.values)
        plt.plot(epochs, valid_loss_epoch.values)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training vs Validation Loss")
        plt.legend(["Train", "Validation"])
        plt.show()

    def plot_epoch_with_batches(self):

        train_acc = self.history_df["train_acc"]["train_accuracy"].values
        valid_acc = self.history_df["valid_acc"]["valid_accuracy"].values

        train_loss = self.history_df["train_loss"]["train_loss"].values
        valid_loss = self.history_df["valid_loss"]["valid_loss"].values

        total_epochs = self.history_df["train_loss"]["epoch"].nunique()

        train_batches_per_epoch = len(train_acc) / total_epochs
        valid_batches_per_epoch = len(valid_acc) / total_epochs

        x_train = np.arange(len(train_acc)) / train_batches_per_epoch
        x_valid = np.arange(len(valid_acc)) / valid_batches_per_epoch

        plt.figure(figsize=(7, 5))
        plt.plot(x_train, train_acc, alpha=0.6, label="Train (batch)")
        plt.plot(x_valid, valid_acc, label="Validation (batch)")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Training vs Validation Accuracy with Batch Detail")
        plt.legend()
        plt.show()

        plt.figure(figsize=(7, 5))
        plt.plot(x_train, train_loss, alpha=0.6, label="Train (batch)")
        plt.plot(x_valid, valid_loss, label="Validation (batch)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training vs Validation Loss with Batch Detail")
        plt.legend()
        plt.show()
