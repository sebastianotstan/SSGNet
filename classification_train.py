#!/usr/bin/env python3
"""
Train and evaluate a binary classifier on BreastMNIST (or similar foldered datasets)
using a ResNet50 backbone with TensorFlow Keras.

- Uses ImageDataGenerator with resnet.preprocess_input
- Loads from (train/val/test) directories
- Supports checkpointing, early stopping, LR reduction
- Resumes from best checkpoint if present
- Saves final model to checkpoints/{percentage}_x.keras
- Evaluates on test set with accuracy, classification report, confusion matrix

Example:
    python train_eval_resnet.py \
        --train_dir uncond_breastmnist/original/train \
        --val_dir   uncond_breastmnist/original/val \
        --test_dir  uncond_breastmnist/original/test \
        --batch_size 64 --epochs 100 \
        --checkpoint checkpoints/best_model_og.keras \
        --percentage 0
"""

import os
import argparse
import numpy as np
import tensorflow as tf

from tensorflow.keras import regularizers
from tensorflow.keras.layers import Flatten, Dense, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import resnet
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from sklearn.metrics import classification_report, confusion_matrix


def parse_args():
    parser = argparse.ArgumentParser(description="ResNet50 Binary Classifier (Keras)")
    parser.add_argument("--train_dir", type=str, required=True, help="Path to training directory")
    parser.add_argument("--val_dir",   type=str, required=True, help="Path to validation directory")
    parser.add_argument("--test_dir",  type=str, required=True, help="Path to test directory")
    parser.add_argument("--img_size",  type=int, nargs=2, default=[224, 224], help="Image size H W")
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size")
    parser.add_argument("--val_batch_size", type=int, default=20, help="Validation batch size")
    parser.add_argument("--test_batch_size", type=int, default=20, help="Test batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-3, help="Initial learning rate")
    parser.add_argument("--patience", type=int, default=3, help="ReduceLROnPlateau patience")
    parser.add_argument("--stop_patience", type=int, default=6, help="EarlyStopping patience")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model_og.keras", help="Checkpoint file path")
    parser.add_argument("--percentage", type=int, default=0, help="Tag for final model filename (e.g., synthetic %)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use (if available)")
    return parser.parse_args()


def set_seed(seed: int = 42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def configure_gpus():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for g in gpus:
                tf.config.experimental.set_memory_growth(g, True)
            # Optionally limit visible GPUs to the first N
        except RuntimeError as e:
            print(f"[GPU Config] RuntimeError: {e}")


def make_generators(train_dir, val_dir, test_dir, img_size, batch_sizes, seed=42):
    train_datagen = ImageDataGenerator(preprocessing_function=resnet.preprocess_input)
    val_datagen   = ImageDataGenerator(preprocessing_function=resnet.preprocess_input)
    test_datagen  = ImageDataGenerator(preprocessing_function=resnet.preprocess_input)

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=tuple(img_size),
        batch_size=batch_sizes["train"],
        class_mode="binary",
        shuffle=True,
        seed=seed,
    )

    val_gen = val_datagen.flow_from_directory(
        val_dir,
        target_size=tuple(img_size),
        batch_size=batch_sizes["val"],
        class_mode="binary",
        shuffle=True,
        seed=seed,
    )

    test_gen = test_datagen.flow_from_directory(
        test_dir,
        target_size=tuple(img_size),
        batch_size=batch_sizes["test"],
        class_mode="binary",
        shuffle=False,
    )

    return train_gen, val_gen, test_gen


def build_model(img_shape, lr=5e-3):
    base_model = ResNet50(
        include_top=False,
        pooling="avg",
        weights="imagenet",
        input_shape=img_shape
    )

    # Fine-tune only the last 2 layers of ResNet (as per original notebook)
    for layer in base_model.layers[:-2]:
        layer.trainable = False

    # NOTE: pooling='avg' makes Flatten unnecessary, but we keep the head close to the original
    model = Sequential([
        base_model,
        Flatten(),
        BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001),
        Dense(
            224 * 224,
            kernel_regularizer=regularizers.l2(0.016),
            activity_regularizer=regularizers.l1(0.006),
            bias_regularizer=regularizers.l1(0.006)
        ),
        LeakyReLU(alpha=0.01),
        Dense(500),
        LeakyReLU(alpha=0.01),
        Dense(200),
        LeakyReLU(alpha=0.01),
        Dense(50),
        LeakyReLU(alpha=0.01),
        Dense(1, activation="sigmoid")  # binary output
    ])

    model.compile(optimizer=Adam(learning_rate=lr), loss="binary_crossentropy", metrics=["accuracy"])
    return model


def ensure_dir(path: str):
    d = os.path.dirname(path) if os.path.splitext(path)[1] else path
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def train_and_eval(args):
    set_seed(args.seed)
    configure_gpus()

    # Ensure checkpoint directory exists
    ensure_dir(args.checkpoint)
    ensure_dir("checkpoints")

    # Generators
    batch_sizes = {
        "train": args.batch_size,
        "val": args.val_batch_size,
        "test": args.test_batch_size
    }
    train_gen, val_gen, test_gen = make_generators(
        args.train_dir, args.val_dir, args.test_dir,
        img_size=args.img_size, batch_sizes=batch_sizes, seed=args.seed
    )

    # Model (resume if checkpoint exists)
    if os.path.exists(args.checkpoint):
        print(f"[Resume] Loading model from checkpoint: {args.checkpoint}")
        model = load_model(args.checkpoint)
    else:
        print("[Start] No checkpoint found, building a new model.")
        img_shape = (args.img_size[0], args.img_size[1], 3)
        model = build_model(img_shape, lr=args.lr)

    model.summary()

    # Callbacks
    checkpoint_cb = ModelCheckpoint(
        filepath=args.checkpoint,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
    early_stopping_cb = EarlyStopping(
        monitor="val_loss",
        patience=args.stop_patience,
        restore_best_weights=True,
        verbose=1
    )
    reduce_lr_cb = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.1,
        patience=args.patience,
        min_lr=5e-5,
        verbose=1
    )

    # Train
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        verbose=1,
        shuffle=False,
        callbacks=[checkpoint_cb, early_stopping_cb, reduce_lr_cb],
    )

    # Save final model
    final_model_path = f"checkpoints/{args.percentage}_x.keras"
    model.save(final_model_path)
    print(f"[Saved] Final model saved to: {final_model_path}")

    # Evaluate (reload best checkpoint for evaluation to be safe)
    if os.path.exists(args.checkpoint):
        model = load_model(args.checkpoint)
        print(f"[Evaluate] Using best checkpoint: {args.checkpoint}")

    evaluate_model_with_classwise_accuracy(model, test_gen)


def evaluate_model_with_classwise_accuracy(model, test_generator):
    print("[Predict] Running inference on test set...")
    predictions = model.predict(test_generator, verbose=1)

    # Threshold
    predicted_classes = (predictions >= 0.5).astype(int).ravel()

    # Ground truth
    true_classes = test_generator.classes
    class_labels = list(test_generator.class_indices.keys())

    # Accuracy
    accuracy = np.mean(predicted_classes == true_classes)
    print(f"\nOverall Accuracy: {accuracy:.4f}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(true_classes, predicted_classes, target_names=class_labels, digits=4))

    # Confusion matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(true_classes, predicted_classes))


if __name__ == "__main__":
    args = parse_args()
    train_and_eval(args)
