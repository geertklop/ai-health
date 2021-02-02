from tensorflow.keras.layers import (
    Conv2D,
    Layer,
    Activation,
    MaxPooling2D,
    Conv2DTranspose,
    concatenate,
)


class DoubleConv(Layer):
    def __init__(self, filters, padding="same"):
        super(DoubleConv, self).__init__()

        # conv layer
        self.conv1 = Conv2D(filters, kernel_size=(3, 3), padding=padding)
        self.activation1 = Activation("relu")

        self.conv2 = Conv2D(filters, kernel_size=(3, 3), padding=padding)
        self.activation2 = Activation("relu")

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.activation1(x)
        x = self.conv2(x)
        x = self.activation2(x)

        return x


class ContractionModule(Layer):
    """Double convolution with MaxPooling"""

    def __init__(self, filters, padding):
        super(ContractionModule, self).__init__()

        self.double_conv = DoubleConv(filters, padding)
        self.pool = MaxPooling2D((2, 2))

    def call(self, inputs):
        conv = self.double_conv(inputs)
        pool = self.pool(conv)

        return pool, conv


class ExpansionModule(Layer):
    """Upconvolutions followed by double convolution"""

    def __init__(self, filters, padding):
        super(ExpansionModule, self).__init__()
        self._padding = padding

        self.up = Conv2DTranspose(filters, (2, 2), strides=2, padding=padding)
        self.conv = DoubleConv(filters, padding)

    def call(self, inputs):
        x = inputs[0]
        x_enc = inputs[1]

        x = self.up(x)

        # No padding, means cropping encoder result
        if self._padding == "valid":
            x_enc = self._calculate_crop(x, x_enc)

        x = concatenate([x, x_enc])
        output = self.conv(x)

        return output

    def _crop_encoder_result(self, x, x_enc):
        return x_enc
