import tensorflow as tf

class ResidualBlock(tf.keras.models.Model):

    def __init__(self, filters, kernel_size, channels_in, channels_out):
        super().__init__()

        self.c1 = tf.keras.layers.Conv2D(filters, kernel_size, padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.Activation("relu")

        self.c2 = tf.keras.layers.Conv2D(filters, kernel_size, padding="same")
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.relu2 = tf.keras.layers.Activation("relu")

        self.shortcut = self.shortcut_method(channels_in, channels_out)

        self.addition = tf.keras.layers.Add()
        self.relu3 = tf.keras.layers.Activation("relu")

        return

    def call(self, x, training=None):
        y = self.c1(x)
        y = self.bn1(y, training=training)
        y = self.relu1(y)

        y = self.c2(y)
        y = self.bn2(y, training=training)
        y = self.relu2(y)

        y = self.addition([self.shortcut(x), y])
        y = self.relu3(y)

        return y

    def shortcut_method(self, channels_in, channels_out):
        if channels_in != channels_out:
            return tf.keras.layers.Conv2D(channels_out, (1, 1), padding="same")
        else:
            return lambda x : x

class Resnet34(tf.keras.models.Model):

    def __init__(self):
        super().__init__()

        self.conv1 = tf.keras.layers.Conv2D(64, (7, 7), strides=(2, 2), padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.Activation("relu")

        self.pool1 = tf.keras.layers.MaxPool2D((3, 3), (2, 2), padding="same")

        self.block2 = [ResidualBlock(64, (3, 3), 64, 64) for _ in range(3)]
        self.conv2 = self.block2[-1].relu3

        self.block3_pre_conv = tf.keras.layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same")
        self.block3 = [ResidualBlock(128, (3, 3), 64, 128) for _ in range(4)]
        self.conv3 = self.block3[-1].relu3

        self.block4_pre_conv = tf.keras.layers.Conv2D(256, (3, 3), strides=(2, 2), padding="same")
        self.block4 = [ResidualBlock(256, (3, 3), 128, 256) for _ in range(6)]
        self.conv4 = self.block4[-1].relu3

        self.block5_pre_conv = tf.keras.layers.Conv2D(512, (3, 3), strides=(2, 2), padding="same")
        self.block5 = [ResidualBlock(512, (3, 3), 256, 512) for _ in range(3)]
        self.conv5 = self.block5[-1].relu3

        return

    def call(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu1(y)
        y = self.pool1(y)

        for b in self.block2:
            y = b(y)

        y = self.block3_pre_conv(y)
        for b in self.block3:
            y = b(y)

        y = self.block4_pre_conv(y)
        for b in self.block4:
            y = b(y)

        y = self.block5_pre_conv(y)
        for b in self.block5:
            y = b(y)

        return y

    def model(self):
        x = tf.keras.layers.Input(shape=(224, 224, 3))
        return tf.keras.models.Model(inputs=[x], outputs=self.call(x))

class Resnet34_FPN(tf.keras.models.Model):

    def __init__(self):
        super().__init__()

        self.conv1 = tf.keras.layers.Conv2D(64, (7, 7), strides=(2, 2), padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.Activation("relu")

        self.pool1 = tf.keras.layers.MaxPool2D((3, 3), (2, 2), padding="same")

        self.block2 = [ResidualBlock(64, (3, 3), 64, 64) for _ in range(3)]
        self.conv2 = self.block2[-1].relu3

        self.block3_pre_conv = tf.keras.layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same")
        self.block3 = [ResidualBlock(128, (3, 3), 64, 128) for _ in range(4)]
        self.conv3 = self.block3[-1].relu3

        self.block4_pre_conv = tf.keras.layers.Conv2D(256, (3, 3), strides=(2, 2), padding="same")
        self.block4 = [ResidualBlock(256, (3, 3), 128, 256) for _ in range(6)]
        self.conv4 = self.block4[-1].relu3

        self.block5_pre_conv = tf.keras.layers.Conv2D(512, (3, 3), strides=(2, 2), padding="same")
        self.block5 = [ResidualBlock(512, (3, 3), 256, 512) for _ in range(3)]
        self.conv5 = self.block5[-1].relu3

        self.M5 = tf.keras.layers.Conv2D(256, (1, 1), strides=(1, 1), padding="same", name="M5")
        self.P5 = tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), padding="same", name="P5")
        self.upsampling5 = tf.keras.layers.UpSampling2D((2, 2), interpolation="nearest")

        self.P6 = tf.keras.layers.MaxPool2D((1, 1), (2, 2), padding="same", name="P6")

        self.pre_M4_conv = tf.keras.layers.Conv2D(256, (1, 1), strides=(1, 1), padding="same")
        self.M4 = tf.keras.layers.Add()
        self.P4 = tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), padding="same", name="P4")
        self.upsampling4 = tf.keras.layers.UpSampling2D((2, 2), interpolation="nearest")

        self.pre_M3_conv = tf.keras.layers.Conv2D(256, (1, 1), strides=(1, 1), padding="same")
        self.M3 = tf.keras.layers.Add()
        self.P3 = tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), padding="same", name="P3")
        self.upsampling3 = tf.keras.layers.UpSampling2D((2, 2), interpolation="nearest")

        self.pre_M2_conv = tf.keras.layers.Conv2D(256, (1, 1), strides=(1, 1), padding="same")
        self.M2 = tf.keras.layers.Add()
        self.P2 = tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), padding="same", name="P2")

        return

    def call(self, x, training=None):
        y = self.conv1(x)
        y = self.bn1(y, training=training)
        y = self.relu1(y)
        y = self.pool1(y)

        for b in self.block2:
            y = b(y)
        y2 = y

        y = self.block3_pre_conv(y)
        for b in self.block3:
            y = b(y)
        y3 = y

        y = self.block4_pre_conv(y)
        for b in self.block4:
            y = b(y)
        y4 = y

        y = self.block5_pre_conv(y)
        for b in self.block5:
            y = b(y)
        y5 = y

        m5 = self.M5(y)
        p5 = self.P5(m5)

        p6 = self.P6(p5)

        y1 = self.upsampling5(p5)
        m4 = self.M4([y1, self.pre_M4_conv(y4)])
        p4 = self.P4(m4)

        y1 = self.upsampling4(m4)
        m3 = self.M3([y1, self.pre_M3_conv(y3)])
        p3 = self.P3(m3)

        y1 = self.upsampling3(m3)
        m2 = self.M2([y1, self.pre_M2_conv(y2)])
        p2 = self.P2(m2)

        return p2, p3, p4, p5, p6

    def model(self):
        x = tf.keras.layers.Input(shape=(224, 224, 3))
        return tf.keras.models.Model(inputs=[x], outputs=self.call(x))

if __name__ == "__main__":
    m = Resnet34_FPN()
    m.build((1, 224, 224, 3))
    #print(m.model().summary())
    import numpy as np
    p2, p3, p4, p5, p6 = m.model().predict(np.random.rand(3, 224, 224, 3))