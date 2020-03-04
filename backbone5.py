import tensorflow as tf
import numpy as np

def shortcut_method(self, channels_in, channels_out):
    if channels_in != channels_out:
        return tf.keras.layers.Conv2D(channels_out, (1, 1), padding="same")
    else:
        return lambda x: x

def residual_block(input, filters, kernel_size, channels_in, channels_out):
    c1 = tf.keras.layers.Conv2D(filters, kernel_size, padding="same")(input)
    bn1 = tf.keras.layers.BatchNormalization()(c1)
    relu1 = tf.keras.layers.Activation("relu")(bn1)

    c2 = tf.keras.layers.Conv2D(filters, kernel_size, padding="same")(relu1)
    bn2 = tf.keras.layers.BatchNormalization()(c2)
    relu2 = tf.keras.layers.Activation("relu")(bn2)

    shortcut = shortcut_method(channels_in, channels_out)

    addition = tf.keras.layers.Add()(shortcut(input), relu2)
    relu3 = tf.keras.layers.Activation("relu")(addition)

    return relu3

def Resnet34_FPN(model, input):
    conv1 = tf.keras.layers.Conv2D(64, (7, 7), strides=(2, 2), padding="same")(input)
    bn1 = tf.keras.layers.BatchNormalization()(conv1)
    relu1 = tf.keras.layers.Activation("relu")(bn1)

    pool1 = tf.keras.layers.MaxPool2D((3, 3), (2, 2), padding="same")(relu1)

    b2 = pool1
    for _ in range(3):
        b2 = residual_block(b2, 64, (3, 3), 64, 64)
    conv2 = b2

    block3_pre_conv = tf.keras.layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same")(conv2)
    b3 = block3_pre_conv
    for _ in range(4):
        b3 = residual_block(b3, 128, (3, 3), 64, 128)
    conv3 = b3

    block4_pre_conv = tf.keras.layers.Conv2D(256, (3, 3), strides=(2, 2), padding="same")(conv3)
    b4 = block4_pre_conv
    for _ in range(6):
        b4 = residual_block(b4, 256, (3, 3), 128, 256)
    conv4 = b4

    block5_pre_conv = tf.keras.layers.Conv2D(512, (3, 3), strides=(2, 2), padding="same")(conv4)
    b5 = block5_pre_conv
    for _ in range(3):
        b5 = residual_block(b5, 512, (3, 3), 256, 512)
    conv5 = b5

    P5 = tf.keras.layers.Conv2D(256, (1, 1), strides=(1, 1), padding="same", name="P5")(conv5)
    upsampling5 = tf.keras.layers.UpSampling2D((2, 2), interpolation="nearest")(P5)

    pre_M4_conv = tf.keras.layers.Conv2D(256, (1, 1), strides=(1, 1), padding="same")(conv4)
    M4 = tf.keras.layers.Add()([upsampling5, pre_M4_conv])
    P4 = tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), padding="same", name="P4")(M4)
    upsampling4 = tf.keras.layers.UpSampling2D((2, 2), interpolation="nearest")(P4)

    pre_M3_conv = tf.keras.layers.Conv2D(256, (1, 1), strides=(1, 1), padding="same")(conv3)
    M3 = tf.keras.layers.Add()([upsampling4, pre_M3_conv])
    P3 = tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), padding="same", name="P3")(M3)
    upsampling3 = tf.keras.layers.UpSampling2D((2, 2), interpolation="nearest")(P3)

    pre_M2_conv = tf.keras.layers.Conv2D(256, (1, 1), strides=(1, 1), padding="same")(conv2)
    M2 = tf.keras.layers.Add()([upsampling3, pre_M2_conv])
    P2 = tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), padding="same", name="P2")(M2)

    return P2, P3, P4, P5

class Resnet34_FPN(tf.keras.models.Model):

    def __init__(self, input):
        super().__init__()

        self.conv1 = tf.keras.layers.Conv2D(64, (7, 7), strides=(2, 2), padding="same")(input)
        self.bn1 = tf.keras.layers.BatchNormalization()(self.conv1)
        self.relu1 = tf.keras.layers.Activation("relu")(self.relu1)

        self.pool1 = tf.keras.layers.MaxPool2D((3, 3), (2, 2), padding="same")(self.relu1)

        self.b2 = self.pool1
        for _ in range(3):
            self.b2 = residual_block(self.b2, 64, (3, 3), 64, 64)
        self.conv2 = self.b2

        self.block3_pre_conv = tf.keras.layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same")(self.conv2)
        self.b3 = self.block3_pre_conv
        for _ in range(4):
            self.b3 = residual_block(self.b3, 128, (3, 3), 64, 128)
        self.conv3 = self.b3

        self.block4_pre_conv = tf.keras.layers.Conv2D(256, (3, 3), strides=(2, 2), padding="same")
        self.b4 = self.block4_pre_conv
        for _ in range(6):
            self.b4 = residual_block(self.b4, 256, (3, 3), 128, 256)
        self.conv4 = self.b4

        self.block5_pre_conv = tf.keras.layers.Conv2D(512, (3, 3), strides=(2, 2), padding="same")
        self.b5 = self.block5_pre_conv
        for _ in range(3):
            self.b5 = residual_block(self.b5, 512, (3, 3), 256, 512)
        self.conv5 = self.b5

        self.P5 = tf.keras.layers.Conv2D(256, (1, 1), strides=(1, 1), padding="same", name="P5")
        self.upsampling5 = tf.keras.layers.UpSampling2D((2, 2), interpolation="nearest")

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

    def call(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
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

        p5 = self.P5(y)

        y1 = self.upsampling5(p5)
        m4 = self.M4([y1, self.pre_M4_conv(y4)])
        p4 = self.P4(m4)

        y1 = self.upsampling4(m4)
        m3 = self.M3([y1, self.pre_M3_conv(y3)])
        p3 = self.P3(m3)

        y1 = self.upsampling3(m3)
        m2 = self.M2([y1, self.pre_M2_conv(y2)])
        p2 = self.P2(m2)

        return p2, p3, p4, p5

    def model(self):
        x = tf.keras.layers.Input(shape=(224, 224, 3))
        return tf.keras.models.Model(inputs=[x], outputs=self.call(x))

if __name__ == "__main__":
    m = Resnet34_FPN(np.zeros(224, 244, 3))