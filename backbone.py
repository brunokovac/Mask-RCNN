import tensorflow as tf
import config

class ResidualBlock(tf.keras.models.Model):

    def __init__(self, name, filters, kernel_size, channels_in=None, downsize=False):
        super().__init__()
        if not channels_in:
            channels_in = filters

        c1_strides = (2, 2) if downsize else (1, 1)
        self.c1 = tf.keras.layers.Conv2D(filters, kernel_size, c1_strides, padding="same", name=name+"-conv1")
        self.bn1 = tf.keras.layers.BatchNormalization(name=name+"-bn1")
        self.relu1 = tf.keras.layers.Activation("relu")

        self.c2 = tf.keras.layers.Conv2D(filters, kernel_size, padding="same", name=name+"-conv2")
        self.bn2 = tf.keras.layers.BatchNormalization(name=name+"-bn2")
        self.relu2 = tf.keras.layers.Activation("relu")

        self.shortcut = self.shortcut_method(channels_in, filters, name)

        self.addition = tf.keras.layers.Add()
        self.relu3 = tf.keras.layers.Activation("relu")

        return

    def call(self, x, training):
        y = self.c1(x)
        y = self.bn1(y, training=training)
        y = self.relu1(y)

        y = self.c2(y)
        y = self.bn2(y, training=training)
        y = self.relu2(y)

        y = self.addition([self.shortcut(x), y])
        y = self.relu3(y)

        return y

    def shortcut_method(self, channels_in, channels_out, name):
        if channels_in != channels_out:
            return tf.keras.layers.Conv2D(channels_out, (1, 1), strides=(2, 2), name=name+"-conv-shortcut")
        else:
            return lambda x : x

class Resnet34_FPN(tf.keras.models.Model):

    def __init__(self):
        super().__init__()

        self.conv1 = tf.keras.layers.Conv2D(64, (7, 7), strides=(2, 2), padding="same", name="block1-conv")
        self.bn1 = tf.keras.layers.BatchNormalization(name="block1-bn")
        self.relu1 = tf.keras.layers.Activation("relu")

        self.pool1 = tf.keras.layers.MaxPool2D((3, 3), (2, 2), padding="same", name="block1-pool")

        self.block2 = [ResidualBlock("block2-{}".format(i), 64, (3, 3)) for i in range(3)]
        self.conv2 = self.block2[-1].relu3

        self.block3 = [ResidualBlock("block3", 128, (3, 3), 64, downsize=True)]
        self.block3.extend([ResidualBlock("block3-{}".format(i), 128, (3, 3)) for i in range(3)])
        self.conv3 = self.block3[-1].relu3

        self.block4 = [ResidualBlock("block4", 256, (3, 3), 128, downsize=True)]
        self.block4.extend([ResidualBlock("block4-{}".format(i), 256, (3, 3)) for i in range(5)])
        self.conv4 = self.block4[-1].relu3

        self.block5 = [ResidualBlock("block5", 512, (3, 3), 256, downsize=True)]
        self.block5.extend([ResidualBlock("block5-{}".format(i), 512, (3, 3)) for i in range(2)])
        self.conv5 = self.block5[-1].relu3

        self.M5 = tf.keras.layers.Conv2D(config.FPN_NUM_CHANNELS, (1, 1), strides=(1, 1), padding="same", name="M5")
        self.P5 = tf.keras.layers.Conv2D(config.FPN_NUM_CHANNELS, (3, 3), strides=(1, 1), padding="same", name="P5")
        self.upsampling5 = tf.keras.layers.UpSampling2D((2, 2), interpolation="nearest")

        self.P6 = tf.keras.layers.MaxPool2D((1, 1), (2, 2), name="P6")

        self.pre_M4_conv = tf.keras.layers.Conv2D(config.FPN_NUM_CHANNELS, (1, 1), strides=(1, 1), padding="same", name="pre_M4_conv")
        self.M4 = tf.keras.layers.Add()
        self.P4 = tf.keras.layers.Conv2D(config.FPN_NUM_CHANNELS, (3, 3), strides=(1, 1), padding="same", name="P4")
        self.upsampling4 = tf.keras.layers.UpSampling2D((2, 2), interpolation="nearest")

        self.pre_M3_conv = tf.keras.layers.Conv2D(config.FPN_NUM_CHANNELS, (1, 1), strides=(1, 1), padding="same", name="pre_M3_conv")
        self.M3 = tf.keras.layers.Add()
        self.P3 = tf.keras.layers.Conv2D(config.FPN_NUM_CHANNELS, (3, 3), strides=(1, 1), padding="same", name="P3")
        self.upsampling3 = tf.keras.layers.UpSampling2D((2, 2), interpolation="nearest")

        self.pre_M2_conv = tf.keras.layers.Conv2D(config.FPN_NUM_CHANNELS, (1, 1), strides=(1, 1), padding="same", name="pre_M2_conv")
        self.M2 = tf.keras.layers.Add()
        self.P2 = tf.keras.layers.Conv2D(config.FPN_NUM_CHANNELS, (3, 3), strides=(1, 1), padding="same", name="P2")

        #pretrained
        self.resnet = tf.keras.applications.ResNet50(include_top=False, weights="imagenet")
        self.conv2 = self.resnet.get_layer('conv2_block2_out').output
        self.conv3 = self.resnet.get_layer('conv3_block4_out').output
        self.conv4 = self.resnet.get_layer('conv4_block6_out').output
        self.conv5 = self.resnet.get_layer('conv5_block3_out').output
        self.resnet_outputs = tf.keras.models.Model(inputs=self.resnet.input, outputs=[self.conv2, self.conv3, self.conv4, self.conv5])
        '''self.resnet.trainable = False
        self.resnet_outputs.trainable = False'''

        #return self.compile(tf.keras.optimizers.SGD(1))

    def call(self, x, training):
        y = self.conv1(x)
        y = self.bn1(y, training=training)
        y = self.relu1(y)
        y = self.pool1(y)
        print(self.bn1.moving_variance)

        for b in self.block2:
            y = b(y, training)
        y2 = y

        for b in self.block3:
            y = b(y, training)
        y3 = y

        for b in self.block4:
            y = b(y, training)
        y4 = y

        for b in self.block5:
            y = b(y, training)
        y5 = y

        y2, y3, y4, y5 = self.resnet_outputs(x, training=training)

        m5 = self.M5(y5)
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
        return tf.keras.models.Model(inputs=[x], outputs=self.call(x, True))

if __name__ == "__main__":
    m = Resnet34_FPN()
    m.build((1, 224, 224, 3))
    print(m.model().summary())
    import numpy as np
    p2, p3, p4, p5, p6 = m.model().predict(np.random.rand(3, 224, 224, 3))