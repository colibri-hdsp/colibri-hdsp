r"""
Demo Tensor.
===================================================

This is a pytorch implementation of paper "Fast matrix inversion in compressive spectral imaging based on a tensorial representation"
https://www.spiedigitallibrary.org/journals/journal-of-electronic-imaging/volume-33/issue-1/013034/Fast-matrix-inversion-in-compressive-spectral-imaging-based-on-a/10.1117/1.JEI.33.1.013034.short#_=_
"""



# %%
# Select Working Directory and Device
# -----------------------------------------------
import os
import sys

from skimage.transform import resize

sys.path.append(os.getcwd())
print("Current Working Directory " , os.getcwd())

# Libraries
import scipy.io as sio
import matplotlib.pyplot as plt

from colibri_hdsp.optics.tensorial import *
torch.set_default_dtype(torch.float64)

manual_device = "cpu"
# Check GPU support
print("GPU support: ", torch.cuda.is_available())

if manual_device:
    device = manual_device
else:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# %%
# Parameters
# -----------------------------------------------
M = 32
N = M
L = 4
S = 2

# %%
# Data
# -----------------------------------------------

d = sio.loadmat('examples/data/lego.mat')['hyperimg']
d = resize(d[..., ::int(d.shape[-1] / L)], [M, N, L])
d = torch.tensor(d, dtype=torch.float32).permute(2, 0, 1)[None]

plt.figure()
plt.imshow(d[0, 0])
plt.title('Spectral Image')
plt.show()

# %%
# Coded Apertures
# -----------------------------------------------

CAs = torch.zeros((S, M, N))
for s in range(S):
    CAs[s] = torch.round(torch.rand(M, N))

# %%
# Sensing and computation of P and Q
# -----------------------------------------------

b = sensingH(d, CAs)
P = computeP(CAs, L)
Q = computeQ(CAs, L)

plt.figure()
plt.imshow(b[0, 0])
plt.title('Measurement')

plt.figure()
plt.imshow(P[0, 0])
plt.title('P_11')

plt.figure()
plt.imshow(Q[0, 0])
plt.title('Q_11')
plt.show()

# %%
# We now check HH*b or Pb
# -----------------------------------------------

HHstarb1 = sensingH(sensingHt(b, CAs), CAs)
HHstarb2 = IMVM(P, b)

plt.figure(figsize=(10, 5))
plt.subplot(121), plt.imshow(HHstarb1[0, 0]), plt.title('HH*   (1)')
plt.subplot(122), plt.imshow(HHstarb2[0, 0]), plt.title('HH*   (2)')
plt.show()

error = torch.norm(HHstarb1 - HHstarb2)
print(f'HH* error: {error}')

# %%
# We now check H*Hb or Qd
# -----------------------------------------------

HstarHd1 = sensingHt(sensingH(d, CAs), CAs)
HstarHd2 = IMVMS(Q, d)

plt.figure(figsize=(10, 5))
plt.subplot(121), plt.imshow(HstarHd1[0, 0]), plt.title('H*H   (1)')
plt.subplot(122), plt.imshow(HstarHd2[0, 0]), plt.title('H*H   (2)')
plt.show()

error = torch.norm(HstarHd1 - HstarHd2)
print(f'H*H error: {error}')

# %%
# We now check (rhoI + HH*)^{-1} or (rhoI + P)^{-1}
# InvertedMeasurement: (rhoI + HH*)^{-1}(rhoI + HH*)b = b
# -----------------------------------------------

rho = 1
[Pinv, rhoIP] = ComputePinv(P, rho)
InvertedMeasurement = IMVM(Pinv, rho * b + HHstarb1)

plt.figure(figsize=(10, 5))
plt.subplot(121), plt.imshow(P[0, 0]), plt.title('(I+P)_{11}^{-1}')
plt.subplot(122), plt.imshow(InvertedMeasurement[0, 0]), plt.title('Inverted Measurement')
plt.show()

error = torch.norm(b - InvertedMeasurement)
print(f'Inversion error: {error}')

# %%
# We now check (rhoI + H*H)^{-1} or (rhoI + Q)^{-1}
# InvertedMeasurement: (rhoI + HH*)^{-1}(rhoI + HH*)b = b
# -----------------------------------------------

rho = 1
[Qinv, rhoIQ] = ComputeQinv(Q, rho)
InvertedImage = IMVMS(Qinv, rho * d + HstarHd1)

plt.figure(figsize=(10, 5))
plt.subplot(121), plt.imshow(Q[0, 0]), plt.title('(I+Q)_{11}^{-1}')
plt.subplot(122), plt.imshow(InvertedImage[0, 0]), plt.title('Inverted Image')
plt.show()

error = torch.norm(d - InvertedImage)
print(f'Inversion error: {error}')

# %%
# Employing the Woodbury Lemma
# -----------------------------------------------

Y = sensingHt(b, CAs)
invHtH1 = Y - sensingHt(IMVM(Pinv, sensingH(Y, CAs)), CAs)
invHtH2 = IMVMS(Qinv, Y)

plt.figure(figsize=(10, 5))
plt.subplot(121), plt.imshow(invHtH1[0, 0]), plt.title('E-H*(I+P)_{11}^{-1}Hb')
plt.subplot(122), plt.imshow(invHtH2[0, 0]), plt.title('(I+Q)_{11}^{-1}b')
plt.show()

error = torch.norm(invHtH1 - invHtH2)
print(f'Inversion error: {error}')
