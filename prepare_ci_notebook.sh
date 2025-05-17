#!/bin/bash
# Create a CI version of the notebook with verbose logging
set -x  # Print each command as it's executed

# Check if running in CI environment
echo "Checking for CI environment..."
if [ -n "$CI" ] || [ -n "$GITHUB_ACTIONS" ] || [ -n "$BINDER_SERVICE_HOST" ]; then
    echo "CI environment detected - creating minimal CI notebook..."
    
    # Create a minimal CI version of the notebook
    echo "Creating CI_MLFLOW.ipynb..."
    cat > CI_MLFLOW.ipynb << 'EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Minimal Notebook for CI Testing\n",
    "\n",
    "This is a minimal version of the Galaxy CNN notebook created specifically for CI testing."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Basic imports\n",
    "import os\n",
    "import sys\n",
    "import numpy as np\n",
    "import h5py\n",
    "import tensorflow as tf\n",
    "import mlflow\n",
    "import tempfile\n",
    "from datetime import datetime\n",
    "\n",
    "print(\"TensorFlow version:\", tf.__version__)\n",
    "\n",
    "# Create a minimal environment\n",
    "print(\"CI Testing Environment\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Create minimal synthetic dataset\n",
    "temp_dir = tempfile.mkdtemp(prefix=\"ci_data_\")\n",
    "print(f\"Created temp directory: {temp_dir}\")\n",
    "\n",
    "# Create minimal datasets\n",
    "def create_dataset(filename, samples=5, size=8):\n",
    "    with h5py.File(filename, 'w') as f:\n",
    "        f.create_dataset('image', data=np.random.rand(samples, 5, size, size).astype(np.float32))\n",
    "        f.create_dataset('specz_redshift', data=np.random.rand(samples, 1).astype(np.float32))\n",
    "        for band in ['g', 'r', 'i', 'z', 'y']:\n",
    "            f.create_dataset(f\"{band}_cmodel_mag\", data=np.random.rand(samples, 1).astype(np.float32))\n",
    "    return filename\n",
    "\n",
    "TRAIN_PATH = create_dataset(os.path.join(temp_dir, \"train.h5\"), 5)\n",
    "VAL_PATH = create_dataset(os.path.join(temp_dir, \"val.h5\"), 3)\n",
    "TEST_PATH = create_dataset(os.path.join(temp_dir, \"test.h5\"), 3)\n",
    "\n",
    "print(\"Created minimal synthetic datasets\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Define minimal model\n",
    "def create_model(input_shape=(5, 8, 8)):\n",
    "    # Define inputs\n",
    "    input_cnn = tf.keras.layers.Input(shape=input_shape)\n",
    "    input_nn = tf.keras.layers.Input(shape=(5,))\n",
    "    \n",
    "    # Super simple CNN - just one conv layer\n",
    "    x = tf.keras.layers.Conv2D(4, kernel_size=3, activation='relu', padding='same', \n",
    "                            data_format='channels_first')(input_cnn)\n",
    "    x = tf.keras.layers.Flatten()(x)\n",
    "    \n",
    "    # Simple NN branch\n",
    "    y = tf.keras.layers.Dense(4, activation='relu')(input_nn)\n",
    "    \n",
    "    # Combine and output\n",
    "    combined = tf.keras.layers.Concatenate()([x, y])\n",
    "    output = tf.keras.layers.Dense(1)(combined)\n",
    "    \n",
    "    # Create model\n",
    "    model = tf.keras.models.Model(inputs=[input_cnn, input_nn], outputs=output)\n",
    "    model.compile(optimizer='adam', loss='mse')\n",
    "    \n",
    "    return model\n",
    "\n",
    "model = create_model()\n",
    "print(\"Created minimal model\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Simple data generator (minimal version)\n",
    "class SimpleGenerator(tf.keras.utils.Sequence):\n",
    "    def __init__(self, path, batch_size=2):\n",
    "        self.path = path\n",
    "        self.batch_size = batch_size\n",
    "        with h5py.File(path, 'r') as f:\n",
    "            self.n_samples = f['image'].shape[0]\n",
    "    \n",
    "    def __len__(self):\n",
    "        return max(1, self.n_samples // self.batch_size)\n",
    "    \n",
    "    def __getitem__(self, idx):\n",
    "        with h5py.File(self.path, 'r') as f:\n",
    "            start_idx = idx * self.batch_size\n",
    "            end_idx = min((idx + 1) * self.batch_size, self.n_samples)\n",
    "            \n",
    "            X_img = f['image'][start_idx:end_idx]\n",
    "            \n",
    "            # Get numerical features\n",
    "            X_num = np.zeros((end_idx - start_idx, 5))\n",
    "            for i, band in enumerate(['g', 'r', 'i', 'z', 'y']):\n",
    "                X_num[:, i] = f[f'{band}_cmodel_mag'][start_idx:end_idx, 0]\n",
    "            \n",
    "            y = f['specz_redshift'][start_idx:end_idx]\n",
    "            \n",
    "            return [X_img, X_num], y\n",
    "\n",
    "# Create data generators\n",
    "train_gen = SimpleGenerator(TRAIN_PATH)\n",
    "val_gen = SimpleGenerator(VAL_PATH)\n",
    "test_gen = SimpleGenerator(TEST_PATH)\n",
    "\n",
    "print(\"Created data generators\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# MLflow setup\n",
    "mlflow_dir = os.path.abspath(\"ci_mlflow\")\n",
    "os.makedirs(mlflow_dir, exist_ok=True)\n",
    "mlflow.set_tracking_uri(f\"file://{mlflow_dir}\")\n",
    "mlflow.set_experiment(\"CI_Test\")\n",
    "\n",
    "# Train model (minimal)\n",
    "with mlflow.start_run(run_name=\"CI_Run\"):\n",
    "    # Log params\n",
    "    mlflow.log_param(\"image_size\", 8)\n",
    "    mlflow.log_param(\"batch_size\", 2)\n",
    "    mlflow.log_param(\"epochs\", 1)\n",
    "    \n",
    "    # Train for just 1 epoch\n",
    "    history = model.fit(\n",
    "        train_gen,\n",
    "        epochs=1,\n",
    "        validation_data=val_gen,\n",
    "        verbose=1\n",
    "    )\n",
    "    \n",
    "    # Evaluate\n",
    "    test_loss = model.evaluate(test_gen, verbose=1)\n",
    "    mlflow.log_metric(\"test_loss\", test_loss)\n",
    "    \n",
    "    print(\"Training and evaluation complete\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "print(\"CI notebook execution successful!\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF
    
    ls -la
    echo "Created CI_MLFLOW.ipynb - a minimal version for CI testing"
    
    # Create a config file with explicit notebook name
    echo "Creating .ci_notebook_config..."
    echo "NOTEBOOK=CI_MLFLOW.ipynb" > .ci_notebook_config
    cat .ci_notebook_config
    
    # Create environment marker
    touch .ci_environment
    
    echo "CI notebook setup complete"
    ls -la
else
    echo "Local environment - no CI notebook needed"
fi