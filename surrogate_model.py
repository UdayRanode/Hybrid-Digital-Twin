"""
===============================================================================
Hybrid Digital Twin for CO₂ Hydrogenation to Light Olefins
-------------------------------------------------------------------------------
Author     : IITISoC 2026 Team
Project    : Hybrid Digital Twin using Physics-Based Kinetics and Neural Network
Version    : 2.0
Python     : 3.10+

Description
-----------
This module implements a Hybrid Digital Twin for predicting reactor performance
during CO₂ hydrogenation to light olefins.

The workflow consists of:

    Input Conditions
          │
          ▼
  Feature Engineering
          │
          ▼
 Neural Network Surrogate
          │
          ▼
  Anderson–Schulz–Flory Distribution
          │
          ▼
 Carbon/Hydrogen Balances
          │
          ▼
 Outlet Stream Prediction
          │
          ▼
 Aspen Plus COM Interface

When this file is imported (e.g., by Aspen), no model training is executed.

Training, evaluation, plotting and model saving are only performed when this
file is executed directly.

===============================================================================
"""

# =============================================================================
# SECTION 1 : IMPORTS
# =============================================================================

import os
import pickle
import random
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from tensorflow import keras
from tensorflow.keras import Model, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import layers
from tensorflow.keras.layers import (
    BatchNormalization,
    Dense,
    Dropout,
    Input,
)
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings("ignore")

# =============================================================================
# SECTION 2 : GLOBAL CONFIGURATION
# =============================================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "saved_models"
FIGURE_DIR = PROJECT_ROOT / "figures"

MODEL_DIR.mkdir(exist_ok=True)
FIGURE_DIR.mkdir(exist_ok=True)

DATASET_FILE = DATA_DIR / "co2_hydrogenation_dataset_3k.csv"
MODEL_FILE = MODEL_DIR / "hybrid_surrogate.keras"
SCALER_FILE = MODEL_DIR / "scalers.pkl"

# =============================================================================
# SECTION 3 : INPUT / OUTPUT DEFINITIONS
# =============================================================================

INPUT_COLUMNS = [
    "T_C",
    "P_bar",
    "H2_CO2_ratio",
    "Keq",
    "log_Keq",
    "log_T_K",
    "T_K_sq",
    "T_P_interact",
]

OUTPUT_COLUMNS = [
    "X_CO2_pct",
    "alpha_outlet",
    "S_C2_C4_pct",
]

# =============================================================================
# SECTION 4 : PHYSICAL CONSTANTS
# =============================================================================

R = 8.314462618  # J/mol/K
EPSILON = 1e-12

# =============================================================================
# SECTION 5 : TRAINING HYPERPARAMETERS
# =============================================================================

TRAIN_TEST_SPLIT = 0.20
VALIDATION_SPLIT = 0.20
BATCH_SIZE = 64
EPOCHS = 500
INITIAL_LEARNING_RATE = 1e-3

# =============================================================================
# SECTION 6 : FEATURE ENGINEERING & DATA PREPARATION
# =============================================================================

def load_dataset(dataset_path):
    """Load kinetic dataset and retain only thermodynamically valid samples."""
    df = pd.read_csv(dataset_path)

    if "thermo_ok" in df.columns:
        df = df[df["thermo_ok"] == True].reset_index(drop=True)

    print("=" * 60)
    print("Dataset Loaded Successfully")
    print(f"Total Samples : {len(df)}")
    print("=" * 60)

    return df


def engineer_features(df):
    """Add engineered features used by the neural-network surrogate."""
    df = df.copy()

    if "T_K" not in df.columns:
        df["T_K"] = df["T_C"] + 273.15
    if "P_bar" not in df.columns:
        raise KeyError("P_bar column not found.")
    if "Keq" not in df.columns:
        df["Keq"] = 10.0 ** (3.933 - 4076.0 / (df["T_K"] - 39.64))

    df["log_Keq"] = np.log(df["Keq"] + EPSILON)
    df["log_T_K"] = np.log(df["T_K"])
    df["T_K_sq"] = df["T_K"] ** 2
    df["T_P_interact"] = df["T_K"] * df["P_bar"]

    return df


def build_training_arrays(df):
    """Convert dataframe into input/output matrices."""
    X_raw = df[INPUT_COLUMNS].values.astype(np.float32)
    y_raw = df[OUTPUT_COLUMNS].values.astype(np.float32)
    return X_raw, y_raw


def transform_targets(y_raw):
    """Apply log1p transform only to CO2 conversion."""
    y_log = y_raw.copy()
    y_log[:, 0] = np.log1p(y_raw[:, 0])
    return y_log


def split_dataset(
    X_raw,
    y_log,
    y_raw,
    test_size=0.20,
    validation_fraction=0.20,
    random_state=SEED,
):
    """Train / Validation / Test split."""
    (
        X_train,
        X_temp,
        y_train,
        y_temp,
        y_train_original,
        y_temp_original,
    ) = train_test_split(
        X_raw,
        y_log,
        y_raw,
        test_size=test_size + validation_fraction,
        random_state=random_state,
    )

    val_ratio = validation_fraction / (test_size + validation_fraction)
    (
        X_valid,
        X_test,
        y_valid,
        y_test,
        y_valid_original,
        y_test_original,
    ) = train_test_split(
        X_temp,
        y_temp,
        y_temp_original,
        test_size=1.0 - val_ratio,
        random_state=random_state,
    )

    print(
        f"\nDataset Split"
        f"\n-------------"
        f"\nTraining   : {len(X_train)}"
        f"\nValidation : {len(X_valid)}"
        f"\nTesting    : {len(X_test)}"
    )

    return (
        X_train,
        X_valid,
        X_test,
        y_train,
        y_valid,
        y_test,
        y_train_original,
        y_valid_original,
        y_test_original,
    )


def normalize_dataset(
    X_train,
    X_valid,
    X_test,
    y_train,
    y_valid,
):
    """Fit MinMax scalers using training data only."""
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_norm = scaler_X.fit_transform(X_train).astype(np.float32)
    X_valid_norm = scaler_X.transform(X_valid).astype(np.float32)
    X_test_norm = scaler_X.transform(X_test).astype(np.float32)

    y_train_norm = scaler_y.fit_transform(y_train).astype(np.float32)
    y_valid_norm = scaler_y.transform(y_valid).astype(np.float32)

    y_min = scaler_y.data_min_.astype(np.float32)
    y_max = scaler_y.data_max_.astype(np.float32)

    return (
        scaler_X,
        scaler_y,
        X_train_norm,
        X_valid_norm,
        X_test_norm,
        y_train_norm,
        y_valid_norm,
        y_min,
        y_max,
    )


def prepare_training_data(dataset_path=DATASET_FILE):
    """Executes the complete preprocessing pipeline."""
    df = load_dataset(dataset_path)
    df = engineer_features(df)
    X_raw, y_raw = build_training_arrays(df)
    y_log = transform_targets(y_raw)

    (
        X_train,
        X_valid,
        X_test,
        y_train,
        y_valid,
        y_test,
        y_train_original,
        y_valid_original,
        y_test_original,
    ) = split_dataset(X_raw, y_log, y_raw)

    (
        scaler_X,
        scaler_y,
        X_train_norm,
        X_valid_norm,
        X_test_norm,
        y_train_norm,
        y_valid_norm,
        y_min,
        y_max,
    ) = normalize_dataset(X_train, X_valid, X_test, y_train, y_valid)

    return (
        df,
        scaler_X,
        scaler_y,
        X_train_norm,
        X_valid_norm,
        X_test_norm,
        y_train_norm,
        y_valid_norm,
        y_test,
        y_test_original,
        y_min,
        y_max,
    )

# =============================================================================
# SECTION 7 : MODEL ARCHITECTURE
# =============================================================================

def build_model(n_inputs=len(INPUT_COLUMNS)):
    """Hybrid Digital Twin Neural Network Multi-Output Architecture."""
    inputs = keras.Input(shape=(n_inputs,), name="reactor_inputs")

    x = layers.Dense(
        128,
        kernel_initializer="he_uniform",
        kernel_regularizer=regularizers.l2(1e-4),
        name="hidden_1",
    )(inputs)
    x = layers.BatchNormalization(name="bn_1")(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.10)(x)

    x = layers.Dense(
        256,
        kernel_initializer="he_uniform",
        kernel_regularizer=regularizers.l2(1e-4),
        name="hidden_2",
    )(x)
    x = layers.BatchNormalization(name="bn_2")(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.10)(x)

    x = layers.Dense(
        128,
        kernel_initializer="he_uniform",
        kernel_regularizer=regularizers.l2(1e-4),
        name="hidden_3",
    )(x)
    x = layers.BatchNormalization(name="bn_3")(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.10)(x)

    x = layers.Dense(
        64,
        kernel_initializer="he_uniform",
        kernel_regularizer=regularizers.l2(1e-4),
        name="hidden_4",
    )(x)
    x = layers.BatchNormalization(name="bn_4")(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.05)(x)

    # Heads
    xco2_head = layers.Dense(32, activation="relu", name="xco2_head")(x)
    xco2_output = layers.Dense(1, name="out_xco2")(xco2_head)

    alpha_head = layers.Dense(16, activation="relu", name="alpha_head")(x)
    alpha_output = layers.Dense(1, name="out_alpha")(alpha_head)

    sc24_head = layers.Dense(16, activation="relu", name="sc24_head")(x)
    sc24_output = layers.Dense(1, name="out_sc24")(sc24_head)

    outputs = layers.Concatenate(name="all_outputs")(
        [xco2_output, alpha_output, sc24_output]
    )

    model = keras.Model(
        inputs=inputs, outputs=outputs, name="HybridDigitalTwinSurrogate"
    )

    return model

# =============================================================================
# SECTION 8 : CUSTOM LOSS FUNCTION
# =============================================================================

OUTPUT_WEIGHTS = tf.constant([3.0, 1.0, 1.0], dtype=tf.float32)


@tf.function
def weighted_mse(y_true, y_pred):
    """Weighted Mean Squared Error (XCO2 given 3x weight)."""
    squared_error = tf.square(y_true - y_pred)
    weighted_error = squared_error * OUTPUT_WEIGHTS
    return tf.reduce_mean(weighted_error)

# =============================================================================
# SECTION 9 : MODEL INITIALIZATION
# =============================================================================

def initialize_model():
    """Create surrogate model and optimizer."""
    model = build_model()
    optimizer = keras.optimizers.Adam(learning_rate=INITIAL_LEARNING_RATE)
    return model, optimizer

# =============================================================================
# SECTION 10 : TRAINING STEP
# =============================================================================

@tf.function
def train_step(model, optimizer, X_batch, y_batch):
    """Single gradient-descent step."""
    with tf.GradientTape() as tape:
        predictions = model(X_batch, training=True)
        loss = weighted_mse(y_batch, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# =============================================================================
# SECTION 11 : VALIDATION STEP
# =============================================================================

@tf.function
def validation_step(model, X_batch, y_batch):
    """Validation loss calculation."""
    predictions = model(X_batch, training=False)
    return weighted_mse(y_batch, predictions)

# =============================================================================
# SECTION 12A : MODEL TRAINING ENGINE
# =============================================================================

def train_model(
    model,
    optimizer,
    X_train,
    y_train,
    X_valid,
    y_valid,
):
    """Train the Hybrid Digital Twin surrogate model."""
    X_train = tf.constant(X_train, dtype=tf.float32)
    y_train = tf.constant(y_train, dtype=tf.float32)
    X_valid = tf.constant(X_valid, dtype=tf.float32)
    y_valid = tf.constant(y_valid, dtype=tf.float32)

    PATIENCE = 60
    LR_PATIENCE = 25
    LR_FACTOR = 0.50
    MIN_LR = 1e-6

    current_lr = INITIAL_LEARNING_RATE
    n_samples = X_train.shape[0]
    n_batches = int(np.ceil(n_samples / BATCH_SIZE))

    best_validation_loss = np.inf
    best_weights = None
    wait = 0
    lr_wait = 0

    history = {"train_loss": [], "validation_loss": []}

    print("\n" + "=" * 70)
    print("Training Hybrid Digital Twin Surrogate")
    print("=" * 70)
    print(f"Epochs        : {EPOCHS}")
    print(f"Batch Size    : {BATCH_SIZE}")
    print(f"Learning Rate : {INITIAL_LEARNING_RATE}")
    print(f"Patience      : {PATIENCE}")
    print("=" * 70)

    for epoch in range(1, EPOCHS + 1):
        indices = tf.random.shuffle(tf.range(n_samples))
        X_shuffle = tf.gather(X_train, indices)
        y_shuffle = tf.gather(y_train, indices)

        batch_losses = []
        for batch in range(n_batches):
            start = batch * BATCH_SIZE
            stop = (batch + 1) * BATCH_SIZE
            X_batch = X_shuffle[start:stop]
            y_batch = y_shuffle[start:stop]

            loss = train_step(model, optimizer, X_batch, y_batch)
            batch_losses.append(float(loss))

        train_loss = np.mean(batch_losses)
        validation_loss = float(validation_step(model, X_valid, y_valid))

        history["train_loss"].append(train_loss)
        history["validation_loss"].append(validation_loss)

        if validation_loss < best_validation_loss - 1e-6:
            best_validation_loss = validation_loss
            best_weights = model.get_weights()
            wait = 0
            lr_wait = 0
        else:
            wait += 1
            lr_wait += 1

        if lr_wait >= LR_PATIENCE:
            current_lr = max(current_lr * LR_FACTOR, MIN_LR)
            optimizer.learning_rate.assign(current_lr)
            lr_wait = 0

        if epoch == 1 or epoch % 25 == 0 or epoch == EPOCHS:
            print(
                f"Epoch {epoch:4d}/{EPOCHS} | "
                f"Train = {train_loss:.6f} | "
                f"Validation = {validation_loss:.6f} | "
                f"LR = {current_lr:.2e}"
            )

        if wait >= PATIENCE:
            print("\nEarly stopping triggered.")
            print(f"Best Validation Loss : {best_validation_loss:.6f}")
            break

    if best_weights is not None:
        model.set_weights(best_weights)

    print("\nBest model restored.")
    print(f"Validation Loss : {best_validation_loss:.6f}")
    print("=" * 70)

    return history, best_validation_loss

# =============================================================================
# SECTION 12B : MODEL EVALUATION
# =============================================================================

def inverse_transform_predictions(predictions_normalized, scaler_y):
    """Convert normalized predictions back to physical values."""
    predictions_log = scaler_y.inverse_transform(predictions_normalized)
    predictions_physical = predictions_log.copy()
    predictions_physical[:, 0] = np.expm1(predictions_log[:, 0])
    return predictions_physical


def evaluate_model(model, X_test_norm, y_test_original, scaler_y):
    """Evaluate surrogate model on the test dataset."""
    print("\n" + "=" * 70)
    print("TEST SET EVALUATION")
    print("=" * 70)

    predictions_normalized = model.predict(X_test_norm, verbose=0)
    predictions_physical = inverse_transform_predictions(
        predictions_normalized, scaler_y
    )

    metrics = {}
    output_names = ["X_CO2 (%)", "alpha", "S_C2_C4 (%)"]

    for i, output in enumerate(output_names):
        actual = y_test_original[:, i]
        predicted = predictions_physical[:, i]

        r2 = r2_score(actual, predicted)
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))

        metrics[output] = {"R2": r2, "MAE": mae, "RMSE": rmse}

        print(f"\n{output}")
        print("-" * len(output))
        print(f"R²   : {r2:.4f}")
        print(f"MAE  : {mae:.4f}")
        print(f"RMSE : {rmse:.4f}")

    return predictions_physical, metrics


def print_model_summary(metrics):
    """Print concise model performance summary."""
    print("\n" + "=" * 70)
    print("MODEL PERFORMANCE SUMMARY")
    print("=" * 70)
    for variable, values in metrics.items():
        print(
            f"{variable:15s}"
            f" R²={values['R2']:.4f}"
            f"   MAE={values['MAE']:.4f}"
            f"   RMSE={values['RMSE']:.4f}"
        )
    print("=" * 70)

# =============================================================================
# SECTION 13 : MODEL PERSISTENCE
# =============================================================================

def save_surrogate_model(model, model_path=MODEL_FILE):
    """Save trained neural-network surrogate."""
    model_path = Path(model_path)
    model.save(model_path)
    print(f"\n✓ Model saved to:\n{model_path}")


def save_scalers(
    scaler_X,
    scaler_y,
    input_columns,
    output_columns,
    scaler_path=SCALER_FILE,
):
    """Save preprocessing objects required for inference."""
    scaler_path = Path(scaler_path)
    with open(scaler_path, "wb") as f:
        pickle.dump(
            {
                "scaler_X": scaler_X,
                "scaler_y": scaler_y,
                "input_columns": input_columns,
                "output_columns": output_columns,
            },
            f,
        )
    print(f"✓ Scalers saved to:\n{scaler_path}")


def load_surrogate_model(model_path=MODEL_FILE):
    """Load trained surrogate model."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found:\n{model_path}")

    model = keras.models.load_model(
        model_path, custom_objects={"weighted_mse": weighted_mse}
    )
    return model


def load_scalers(scaler_path=SCALER_FILE):
    """Load fitted preprocessing objects."""
    scaler_path = Path(scaler_path)
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler file not found:\n{scaler_path}")

    with open(scaler_path, "rb") as f:
        scaler_data = pickle.load(f)
    return scaler_data


def load_surrogate():
    """Convenience function to load model and scalers."""
    model = load_surrogate_model()
    scaler_data = load_scalers()
    scaler_X = scaler_data["scaler_X"]
    scaler_y = scaler_data["scaler_y"]
    return model, scaler_X, scaler_y


def save_training_results(model, scaler_X, scaler_y):
    """Save every artifact required for deployment."""
    save_surrogate_model(model)
    save_scalers(scaler_X, scaler_y, INPUT_COLUMNS, OUTPUT_COLUMNS)
    print("\nDeployment package created successfully.")

# =============================================================================
# SECTION 14A : SURROGATE INFERENCE ENGINE
# =============================================================================

def build_inference_features(T_C, P_bar, H2_CO2_ratio):
    """Construct complete feature vector matching INPUT_COLUMNS (8 features)."""
    T_K = T_C + 273.15
    Keq = 10.0 ** (3.933 - 4076.0 / (T_K - 39.64))
    log_Keq = np.log(Keq + EPSILON)
    log_T = np.log(T_K)
    T_sq = T_K ** 2
    TP = T_K * P_bar

    x = np.array(
        [[T_C, P_bar, H2_CO2_ratio, Keq, log_Keq, log_T, T_sq, TP]],
        dtype=np.float32,
    )
    return x


def normalize_input(x_raw, scaler_X):
    """Normalize inference input."""
    return scaler_X.transform(x_raw).astype(np.float32)


def surrogate_predict(model, x_normalized):
    """Execute neural-network inference."""
    prediction = model.predict(x_normalized, verbose=0)
    return prediction

# =============================================================================
# SECTION 14B : PHYSICS POST-PROCESSING (ASF & MATERIAL BALANCES)
# =============================================================================

def compute_asf_distribution(alpha, max_carbon_number=12):
    """
    Compute mole fraction distribution using Anderson-Schulz-Flory (ASF).
    W_n = n * (1 - alpha)^2 * alpha^(n - 1)
    """
    n = np.arange(1, max_carbon_number + 1)
    w_n = n * ((1 - alpha) ** 2) * (alpha ** (n - 1))
    # Convert weight fractions to carbon mole distribution
    x_n = w_n / n
    x_n = x_n / np.sum(x_n)
    return x_n, w_n


def post_process_hydrocarbons(
    co2_conv_pct, alpha, s_c2_c4_pct, co2_in_kmol_hr, h2_in_kmol_hr
):
    """
    Perform carbon/hydrogen balances to derive detailed outlet molar flow rates.
    """
    co2_converted = (co2_conv_pct / 100.0) * co2_in_kmol_hr

    # ASF distribution up to C12
    x_n, _ = compute_asf_distribution(alpha, max_carbon_number=12)

    # Scale hydrocarbon product yield to match converted CO2
    # CO2 -> Hydrocarbons + H2O via reverse water-gas shift / FT pathways
    ch4_moles = co2_converted * x_n[0]
    c2_c4_moles = co2_converted * np.sum(x_n[1:4]) * (s_c2_c4_pct / 100.0)
    c5_plus_moles = co2_converted * np.sum(x_n[4:])

    # Unreacted species & water production
    co2_out = co2_in_kmol_hr - co2_converted
    h2o_out = co2_converted * 2.0  # Approx stoichiometric H2O formation
    h2_out = max(0.0, h2_in_kmol_hr - (co2_converted * 3.0))

    return {
        "CO2_out": co2_out,
        "H2_out": h2_out,
        "H2O_out": h2o_out,
        "CH4_out": ch4_moles,
        "C2_C4_out": c2_c4_moles,
        "C5_plus_out": c5_plus_moles,
    }

# =============================================================================
# SECTION 15 : ASPEN PLUS COM INTERFACE
# =============================================================================

def run_aspen_interface(T_C, P_bar, H2_CO2_ratio, co2_in=100.0):
    """
    Execution entry point when called from Aspen Plus COM interface or User Model.
    """
    model, scaler_X, scaler_y = load_surrogate()

    x_raw = build_inference_features(T_C, P_bar, H2_CO2_ratio)
    x_norm = normalize_input(x_raw, scaler_X)

    pred_norm = surrogate_predict(model, x_norm)
    pred_phys = inverse_transform_predictions(pred_norm, scaler_y)[0]

    co2_conv = float(pred_phys[0])
    alpha = float(pred_phys[1])
    s_c24 = float(pred_phys[2])

    h2_in = co2_in * H2_CO2_ratio
    outlet_flows = post_process_hydrocarbons(
        co2_conv, alpha, s_c24, co2_in, h2_in
    )

    return {
        "CO2_Conversion_%": co2_conv,
        "ASF_Alpha": alpha,
        "C2_C4_Selectivity_%": s_c24,
        "Outlet_Flows_kmol_hr": outlet_flows,
    }

# =============================================================================
# SECTION 16 : PLOTTING & VISUALIZATION
# =============================================================================

def plot_training_history(history, figure_dir=FIGURE_DIR):
    """Plot training and validation loss curves."""
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="Train Loss", linewidth=2)
    plt.plot(history["validation_loss"], label="Validation Loss", linewidth=2)
    plt.xlabel("Epochs")
    plt.ylabel("Weighted MSE Loss")
    plt.title("Surrogate Model Training Performance")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(figure_dir / "training_history.png", dpi=300)
    plt.close()


def plot_parity(y_test_original, predictions_physical, figure_dir=FIGURE_DIR):
    """Generate parity plots for predictions vs ground truth."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    targets = ["CO2 Conversion (%)", "ASF Alpha", "C2-C4 Selectivity (%)"]

    for i, title in enumerate(targets):
        actual = y_test_original[:, i]
        pred = predictions_physical[:, i]

        axes[i].scatter(actual, pred, alpha=0.5, color="navy", edgecolors="k")
        min_val = min(np.min(actual), np.min(pred))
        max_val = max(np.max(actual), np.max(pred))
        axes[i].plot(
            [min_val, max_val], [min_val, max_val], "r--", label="Ideal"
        )

        axes[i].set_xlabel(f"Actual {title}")
        axes[i].set_ylabel(f"Predicted {title}")
        axes[i].set_title(title)
        axes[i].grid(True, linestyle="--", alpha=0.5)
        axes[i].legend()

    plt.tight_layout()
    plt.savefig(figure_dir / "parity_plots.png", dpi=300)
    plt.close()

# =============================================================================
# SECTION 17 : MAIN EXECUTION BLOCK
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("HYBRID DIGITAL TWIN FOR CO2 HYDROGENATION")
    print("=" * 70)

    # 1. Prepare Data
    (
        df,
        scaler_X,
        scaler_y,
        X_train_norm,
        X_valid_norm,
        X_test_norm,
        y_train_norm,
        y_valid_norm,
        y_test,
        y_test_original,
        y_min,
        y_max,
    ) = prepare_training_data()

    # 2. Build and Initialize Model
    model, optimizer = initialize_model()

    # 3. Train Model
    history, best_val_loss = train_model(
        model,
        optimizer,
        X_train_norm,
        y_train_norm,
        X_valid_norm,
        y_valid_norm,
    )

    # 4. Evaluate Model
    predictions_physical, metrics = evaluate_model(
        model, X_test_norm, y_test_original, scaler_y
    )
    print_model_summary(metrics)

    # 5. Save Artifacts
    save_training_results(model, scaler_X, scaler_y)

    # 6. Generate Figures
    plot_training_history(history)
    plot_parity(y_test_original, predictions_physical)

    # 7. Run Inference Test
    print("\nExecuting Sample Inference Test...")
    res = run_aspen_interface(T_C=300.0, P_bar=30.0, H2_CO2_ratio=3.0)
    print(f"Sample Output:\n{res}")

    print("\n✓ Process completed successfully.")