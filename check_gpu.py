#!/usr/bin/env python3
"""Quick script to check GPU availability for TensorFlow"""

import tensorflow as tf

print("=" * 60)
print("TensorFlow GPU Check")
print("=" * 60)
print(f"TensorFlow version: {tf.__version__}")
print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")

gpus = tf.config.list_physical_devices('GPU')
print(f"\nGPU devices found: {len(gpus)}")

if gpus:
    print("\n✅ GPU is available!")
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu.name}")
        print(f"  Details: {gpu}")

    try:
        print(f"\nGPU Name: {tf.test.gpu_device_name()}")
    except:
        pass

    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("\n✅ Memory growth enabled for all GPUs")
    except:
        pass
else:
    print("\n❌ No GPU devices found!")
    print("\nTo enable GPU support:")
    print("1. Install CUDA toolkit (version 12.x for TF 2.20)")
    print("2. Install cuDNN (version 9.x)")
    print("3. Reinstall TensorFlow: pip install tensorflow[and-cuda]")
    print("\nOr check: https://www.tensorflow.org/install/gpu")

print("=" * 60)

