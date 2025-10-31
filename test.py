import tensorflow as tf
import numpy as np
from utils import *
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import BatchNormalization, Flatten, Dense, Lambda
from matplotlib import pyplot as plt


# -----------------------------
# Helper functions for BER
# -----------------------------
def qpsk_mod(bits):
    bits = bits.reshape(-1, 2)
    symbols = (2 * bits[:, 0] - 1) + 1j * (2 * bits[:, 1] - 1)
    symbols /= np.sqrt(2)  # normalize power
    return symbols


def qpsk_demod(symbols):
    bits = np.zeros((len(symbols), 2))
    bits[:, 0] = symbols.real > 0
    bits[:, 1] = symbols.imag > 0
    return bits.flatten()


# ------------------------------------------
#  Load and generate simulation data
# ------------------------------------------
path = 'train_set/example/test'
H, H_est = mat_load(path)

H_input = np.expand_dims(np.concatenate([np.real(H_est), np.imag(H_est)], 1), 1)
H = np.squeeze(H)

# ------------------------------------------
#  Construct BFNN model
# ------------------------------------------
imperfect_CSI = Input(name='imperfect_CSI', shape=(H_input.shape[1:4]), dtype=tf.float32)
perfect_CSI = Input(name='perfect_CSI', shape=(H.shape[1],), dtype=tf.complex64)
SNR_input = Input(name='SNR_input', shape=(1,), dtype=tf.float32)

temp = BatchNormalization()(imperfect_CSI)
temp = Flatten()(temp)
temp = BatchNormalization()(temp)
temp = Dense(256, activation='relu')(temp)
temp = BatchNormalization()(temp)
temp = Dense(128, activation='relu')(temp)
phase = Dense(Nt)(temp)
V_RF = Lambda(trans_Vrf, dtype=tf.complex64, output_shape=(Nt,))(phase)
rate = Lambda(Rate_func, dtype=tf.float32, output_shape=(1,))([perfect_CSI, V_RF, SNR_input])

model = Model(inputs=[imperfect_CSI, perfect_CSI, SNR_input], outputs=rate)
model.compile(optimizer='adam', loss=lambda y_true, y_pred: y_pred)
model.summary()

# Load trained weights
model.load_weights('C:/Users/SAIKARTHIK/BF-design-with-DL-master/train_set/20db/20db.h5')

# -----------------------------
# SNR range
# -----------------------------
snr_db_range = range(-20, 25, 5)

# -----------------------------
# Compute Spectral Efficiency
# -----------------------------
rate = []
fd_rate = []

for snr_db in snr_db_range:
    snr_lin = 10 ** (snr_db / 10)
    SNR = np.array([[snr_lin]] * H.shape[0])

    # BFNN rate
    y = model.evaluate(x=[H_input, H, SNR], y=H, batch_size=10000)
    rate.append(-y)

    # FD rate
    rate_samples = np.log2(1 + snr_lin * np.sum(np.abs(H) ** 2, axis=1))
    fd_rate.append(np.mean(rate_samples))

# -----------------------------
# Compute BER (QPSK, single-antenna approximation)
# -----------------------------
num_bits = 10000
tx_bits = np.random.randint(0, 2, num_bits)
tx_symbols = qpsk_mod(tx_bits)

bf_ber = []
fd_ber = []

for snr_db in snr_db_range:
    snr_lin = 10 ** (snr_db / 10)
    SNR = np.array([[snr_lin]] * H.shape[0])

    # BFNN beamformer gain (scalar approximation)
    bf_v = model.predict([H_input, H, SNR], batch_size=10000)
    bf_gain = np.abs(bf_v[0]).mean()
    rx_symbols = tx_symbols * bf_gain + (1 / np.sqrt(2 * snr_lin)) * (
                np.random.randn(*tx_symbols.shape) + 1j * np.random.randn(*tx_symbols.shape))
    rx_bits = qpsk_demod(rx_symbols)
    bf_ber.append(np.mean(rx_bits != tx_bits))

    # FD beamformer gain (scalar approximation)
    fd_gain = np.linalg.norm(H[0])  # use first channel sample
    rx_symbols_fd = tx_symbols * fd_gain + (1 / np.sqrt(2 * snr_lin)) * (
                np.random.randn(*tx_symbols.shape) + 1j * np.random.randn(*tx_symbols.shape))
    rx_bits_fd = qpsk_demod(rx_symbols_fd)
    fd_ber.append(np.mean(rx_bits_fd != tx_bits))

# -----------------------------
# Plot Spectral Efficiency
# -----------------------------
plt.figure()
plt.plot(snr_db_range, rate, marker='o', label='BFNN')
plt.plot(snr_db_range, fd_rate, marker='x', label='FD')
plt.title("Spectral Efficiency Comparison")
plt.xlabel("SNR (dB)")
plt.ylabel("Spectral Efficiency (bits/s/Hz)")
plt.legend()
plt.grid(True)
plt.show()

