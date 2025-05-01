import os
import sys
import ctypes
import glob
import logging
# Toggle detailed logs via environment variable:
VERBOSE = os.getenv('GPU_SETUP_VERBOSE', '0') == '1'
# Silence TensorFlow C++ logs (only errors):
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
# Silence Python-side TensorFlow logging and absl:
logging.getLogger('tensorflow').setLevel(logging.ERROR)
try:
    import absl.logging as absllogging
    absllogging.set_verbosity(absllogging.ERROR)  # Fixed line
except ImportError:
    pass
# Configure environment for CUDA and TensorFlow GPU growth:
conda_prefix = os.path.dirname(os.path.dirname(sys.executable))
conda_lib = os.path.join(conda_prefix, 'lib')
os.environ['CUDA_HOME'] = conda_prefix
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Just use one GPU for now
# Update LD_LIBRARY_PATH:
new_ld = f"{conda_lib}:/usr/lib/x86_64-linux-gnu:/usr/lib64:/usr/lib:" + os.environ.get('LD_LIBRARY_PATH', '')
os.environ['LD_LIBRARY_PATH'] = new_ld
if VERBOSE:
    print(f"Set LD_LIBRARY_PATH: {new_ld}", flush=True)
# Pre-load CUDA libraries (only verbose):
patterns = [
    'libcudart.so',
    'libcublas.so',
    'libcufft.so',
    'libcudnn.so',
    'libcusparse.so*'
]
for pat in patterns:
    for lib_path in glob.glob(os.path.join(conda_lib, pat)):
        if VERBOSE:
            print(f"Loading library: {lib_path}", flush=True)
        try:
            ctypes.CDLL(lib_path)
        except Exception:
            if VERBOSE:
                print(f"  → failed to load {lib_path}", flush=True)
# Summary output:
print("✅ Libraries pre-loaded successfully", flush=True)
# Now import TensorFlow (core) to silence its logs:
try:
    import tensorflow as tfcore
    tfcore.get_logger().setLevel('ERROR')
except ImportError:
    pass
# Import and configure TensorFlow v1 compatibility:
import tensorflow.compat.v1 as tf
# IMPORTANT: Keep the v2 behavior
# try:
#     tf.disable_v2_behavior()
# except Exception:
#     pass
# Try to enable eager execution
try:
    tf.enable_eager_execution()
except Exception as e:
    print(f"Note: Could not enable eager execution: {e}")
# Suppress TF v1 deprecation warnings:
try:
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except Exception:
    pass
# Set memory growth on available GPUs:
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # First, enable memory growth for all GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # Limit to using just the first GPU for now
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        print(f"✅ GPU setup complete: GPU 0 enabled and configured for training", flush=True)
    except RuntimeError as e:
        print(f"GPU setup error: {e}")
else:
    print("No GPUs found. Running on CPU.")
# Alias for standard imports:
sys.modules['tensorflow_gpu'] = tf