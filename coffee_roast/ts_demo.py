import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.losses import BinaryCrossentropy
from tensorflow.python.keras.layers import Dense
from lab_coffee_utils import load_coffee_data
X, Y = load_coffee_data()
print('{X.shape}, {Y.shape}')
