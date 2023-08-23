import tensorflow as tf
import numpy as np
from functional import forward_color_cassi, backward_color_cassi, forward_dd_cassi, backward_dd_cassi, forward_cassi, backward_cassi

class CASSI(tf.keras.layers.Layer):
    """
    Layer that performs the forward and backward operator of coded aperture snapshot spectral imager (CASSI), more information refer to: Compressive Coded Aperture Spectral Imaging: An Introduction: https://doi.org/10.1109/MSP.2013.2278763

    """

    def __init__(self, mode, trainable=False, ca_regularizer=None, initial_ca=None, seed=None):
        """
        :param mode: String, mode of the coded aperture, it can be "base", "dd" or "color"
        :param trainable: Boolean, if True the coded aperture is trainable
        :param ca_regularizer: Regularizer function applied to the coded aperture
        :param initial_ca: Initial coded aperture with shape (1, M, N, 1)
        :param seed: Random seed
        """
        super(CASSI, self).__init__(name='cassi')
        self.seed = seed
        self.trainable = trainable
        self.ca_regularizer = ca_regularizer
        self.initial_ca = initial_ca

        if mode == "base":
            self.forward = forward_cassi
            self.backward = backward_cassi
        elif mode == "dd":
            self.forward = forward_dd_cassi
            self.backward = backward_dd_cassi
        elif mode == "color":
            self.forward = forward_color_cassi
            self.backward = backward_color_cassi

        self.mode = mode

    def build(self, input_shape):
        """
        Build method of the layer, it creates the coded aperture according to the input shape
        :param input_shape: Shape of the input tensor (1, M, N, L)
        :return: None
        """
        super(CASSI, self).build(input_shape)
        self.M, self.N, self.L = input_shape  # Extract spectral image shape

        if self.mode == 'base':
            shape = (1, self.M, self.N, 1)
        elif self.mode == 'dd':
            shape = (1, self.M, self.N + self.L - 1, 1)
        elif self.mode == 'color':
            shape = (1, self.M, self.N, self.L)
        else:
            raise ValueError(f"the mode {self.mode} is not valid")

        if self.initial_ca is None:
            initializer = tf.random_uniform_initializer(minval=0, maxval=1, seed=self.seed)
        else:
            assert self.initial_ca.shape != shape, f"the start CA shape should be {shape}"
            initializer = tf.constant_initializer(self.initial_ca)

        self.ca = self.add_weight(name='coded_apertures', shape=shape, initializer=initializer,
                                    trainable=self.trainable, regularizer=self.ca_regularizer)

    def __call__(self, x, type_calculation="forward"):
        """
        Call method of the layer, it performs the forward or backward operator according to the type_calculation
        :param x: Input tensor with shape (1, M, N, L)
        :param type_calculation: String, it can be "forward", "backward" or "forward_backward"
        :return: Output tensor with shape (1, M, N + L - 1, 1) if type_calculation is "forward", (1, M, N, L) if type_calculation is "backward, or (1, M, N, L) if type_calculation is "forward_backward
        :raises ValueError: If type_calculation is not "forward", "backward" or "forward_backward"
        """
        if type_calculation == "forward":
            return self.forward(x, self.ca)

        elif type_calculation == "backward":
            return self.backward(x, self.ca)
        elif type_calculation == "forward_backward":
            return self.backward(self.forward(x, self.ca), self.ca)

        else:
            raise ValueError("type_calculation must be forward, backward or forward_backward")


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import tensorflow as tf
    import scipy.io as sio
    import os

    # load a mat file

    cube = sio.loadmat(os.path.join('deepoptix', 'examples', 'data', 'spectral_image.mat'))['img']  # (M, N, L)
    ca = np.random.rand(1, cube.shape[0], cube.shape[1], 1)  # custom ca (1, M, N, 1)

    # load optical encoder

    mode = 'dd'
    cassi = CASSI(mode)
    cassi.build(cube.shape)  # this is only for the demo

    # encode the cube

    cube_tf = tf.convert_to_tensor(cube)[None]  # None add a new dimension
    measurement = cassi(cube_tf, type_calculation="forward")
    backward = cassi(measurement, type_calculation="backward")
    direct_backward = cassi(cube_tf)
    measurement2 = cassi(backward, type_calculation="forward_backward")

    # Print information about tensors

    print('cube shape: ', cube_tf.shape)
    print('measurement shape: ', measurement.shape)
    print('backward shape: ', backward.shape)

    # visualize the measurement

    plt.figure(figsize=(10, 10))

    plt.subplot(221)
    plt.title('cube')
    plt.imshow(cube[..., 0])

    plt.subplot(222)
    plt.title('measurement')
    plt.imshow(measurement[0, ..., 0])

    plt.subplot(223)
    plt.title('backward')
    plt.imshow(backward[0, ..., 0])

    plt.subplot(224)
    plt.title('measurement2')
    plt.imshow(measurement2[0, ..., 0])

    plt.tight_layout()
    plt.show()
