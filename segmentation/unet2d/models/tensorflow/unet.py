import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, Layer
from tensorflow.keras.models import Model

from .parts import ContractionModule, DoubleConv, ExpansionModule


class UNet(Model):
    def __init__(
        self,
        name: str = "UNet",
        out_channels: int = 3,
        init_filters: int = 64,
        padding: str = "same",
    ):
        super(UNet, self).__init__(name=name)
        self._out_channels = out_channels

        # Build model
        self._init_model(init_filters, padding)

    def call(self, inputs: Input) -> Layer:
        x = inputs

        x, enc1 = self.down1(x)
        x, enc2 = self.down2(x)
        x, enc3 = self.down3(x)
        x, enc4 = self.down4(x)

        x = self.bottleneck(x)

        x = self.up1([x, enc4])
        x = self.up2([x, enc3])
        x = self.up3([x, enc2])
        x = self.up4([x, enc1])

        outputs = self.out(x)

        return outputs

    def _init_model(self, init_filters: int = 3, padding: str = "same"):
        filters = init_filters

        # Contracting path
        self.down1 = ContractionModule(filters, padding=padding)
        self.down2 = ContractionModule(filters * 2, padding=padding)
        self.down3 = ContractionModule(filters * 4, padding=padding)
        self.down4 = ContractionModule(filters * 8, padding=padding)

        # Connecting layer
        self.bottleneck = DoubleConv(filters * 16, padding=padding)

        # Expansion path
        self.up1 = ExpansionModule(filters * 8, padding=padding)
        self.up2 = ExpansionModule(filters * 4, padding=padding)
        self.up3 = ExpansionModule(filters * 2, padding=padding)
        self.up4 = ExpansionModule(filters, padding=padding)

        # Output layer
        self.out = Conv2D(self._out_channels, kernel_size=(1, 1), activation="sigmoid")


if __name__ == "__main__":
    model = UNet()
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    model.build((None, 128, 128, 1))
    model.summary()
