import tensorflow as tf



def mish(x):
    return x * tf.math.tanh(tf.math.log(1 + tf.math.exp(x)))

class DenseBlock(tf.keras.layers.Layer):
    def __init__(self,filters,padding="same"):
        super(DenseBlock, self).__init__()
        self.conv1=tf.keras.layers.Conv2D(
                                            filters,
                                            (3,3),
                                            strides=(1, 1),
                                            padding=padding,
                                            activation=None,
                                            use_bias=True,
                                            kernel_initializer='glorot_uniform',
                                            bias_initializer='zeros',
                                            kernel_regularizer='l2',
                                            
                                        )

        self.conv2=tf.keras.layers.Conv2D(
                                            filters,
                                            (3,3),
                                            strides=(1, 1),
                                            padding=padding,
                                            activation=None,
                                            use_bias=True,
                                            kernel_initializer='glorot_uniform',
                                            bias_initializer='zeros',
                                            kernel_regularizer='l2',
                                            
                                        )
        self.conv3=tf.keras.layers.Conv2D(
                                            filters,
                                            (3,3),
                                            strides=(1, 1),
                                            padding=padding,
                                            activation=None,
                                            use_bias=True,
                                            kernel_initializer='glorot_uniform',
                                            bias_initializer='zeros',
                                            kernel_regularizer='l2',
                                            
                                        )
        self.bn1=tf.keras.layers.BatchNormalization(
                                                        axis=-1,
                                                        momentum=0.99,
                                                        epsilon=0.001,
                                                        center=True,
                                                        scale=True,
                                                        
                                                    )
        self.bn2=tf.keras.layers.BatchNormalization(
                                                        axis=-1,
                                                        momentum=0.99,
                                                        epsilon=0.001,
                                                        center=True,
                                                        scale=True,
                                                        
                                                    )
        self.bn3=tf.keras.layers.BatchNormalization(
                                                        axis=-1,
                                                        momentum=0.99,
                                                        epsilon=0.001,
                                                        center=True,
                                                        scale=True,
                                                        
                                                    )

    def call(self,inputs,training=False):
        out1=self.conv1(inputs)
        out1_preact=self.bn1(out1,training)
        out1_act=mish(out1_preact)
        out2=self.conv2(out1_act)
        out2=self.bn2(out2,training=training)
        out2=out2+out1_act
        out2=mish(out2)
        out3=self.conv3(out2)
        out3=self.bn3(out3,training=training)
        out3=out1_act+out2+out3
        outfinal=mish(out3)
        return outfinal


class Encoder(tf.keras.layers.Layer):
    def __init__(self,):
        super(Encoder, self).__init__()
        self.initconv=tf.keras.layers.Conv2D(
                                            32,
                                            (3,3),
                                            strides=(1, 1),
                                            padding="same",
                                            activation=None,
                                            use_bias=True,
                                            kernel_initializer='glorot_uniform',
                                            bias_initializer='zeros',
                                            kernel_regularizer='l2',
                                            
                                        )
        self.bn1=tf.keras.layers.BatchNormalization(
                                                        axis=-1,
                                                        momentum=0.99,
                                                        epsilon=0.001,
                                                        center=True,
                                                        scale=True,
                                                        
                                                    )
        self.initpool=tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.d1=DenseBlock(filters=32)
        self.pool1=tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.d2=DenseBlock(filters=64)
        self.pool2=tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.d3=DenseBlock(filters=128)
        self.pool3=tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.d4=DenseBlock(filters=256)
        self.pool4=tf.keras.layers.MaxPooling2D(pool_size=(2, 2))

    def call(self,inputs,training=False):
        out=self.initconv(inputs)
        out=self.bn1(out,training=training)
        out=mish(out)
        outpool=self.initpool(out)
        d1out=self.d1(outpool)
        m1d1out=self.pool1(d1out)
        d2out=self.d2(m1d1out)
        m2d2out=self.pool2(d2out)
        d3out=self.d3(m2d2out)
        m3d3out=self.pool3(d3out)
        d4out=self.d4(m3d3out)
        d4out=self.pool4(d4out)
        return d1out,d2out,d3out,d4out
    
class Decoder(tf.keras.layers.Layer):
    def __init__(self,):
        super(Decoder, self).__init__()
        self.convup1=tf.keras.layers.Conv2DTranspose(
                                                512,
                                                (3,3),
                                                strides=(2, 2),
                                                padding='same',
                                                output_padding=None,
                                                data_format=None,
                                                dilation_rate=(1, 1),
                                                activation=None,
                                                use_bias=True,
                                                kernel_initializer='glorot_uniform',
                                                bias_initializer='zeros',
                                                kernel_regularizer="l2",
                                                
                                            )
        self.convup2=tf.keras.layers.Conv2DTranspose(
                                                128,
                                                (3,3),
                                                strides=(2, 2),
                                                padding='same',
                                                output_padding=None,
                                                data_format=None,
                                                dilation_rate=(1, 1),
                                                activation=None,
                                                use_bias=True,
                                                kernel_initializer='glorot_uniform',
                                                bias_initializer='zeros',
                                                kernel_regularizer="l2",
                                                
                                            )
        self.convup3=tf.keras.layers.Conv2DTranspose(
                                                64,
                                                (3,3),
                                                strides=(2, 2),
                                                padding='same',
                                                output_padding=None,
                                                data_format=None,
                                                dilation_rate=(1, 1),
                                                activation=None,
                                                use_bias=True,
                                                kernel_initializer='glorot_uniform',
                                                bias_initializer='zeros',
                                                kernel_regularizer="l2",
                                                
                                            )
        self.convup4=tf.keras.layers.Conv2DTranspose(
                                                    32,
                                                    (3,3),
                                                    strides=(2, 2),
                                                    padding='same',
                                                    output_padding=None,
                                                    data_format=None,
                                                    dilation_rate=(1, 1),
                                                    activation=None,
                                                    use_bias=True,
                                                    kernel_initializer='glorot_uniform',
                                                    bias_initializer='zeros',
                                                    kernel_regularizer="l2",
                                                    
                                                )
        self.bn1=tf.keras.layers.BatchNormalization(
                                                        axis=-1,
                                                        momentum=0.99,
                                                        epsilon=0.001,
                                                        center=True,
                                                        scale=True,
                                                        
                                                    )
        self.bn2=tf.keras.layers.BatchNormalization(
                                                        axis=-1,
                                                        momentum=0.99,
                                                        epsilon=0.001,
                                                        center=True,
                                                        scale=True,
                                                        
                                                    )
        self.bn3=tf.keras.layers.BatchNormalization(
                                                        axis=-1,
                                                        momentum=0.99,
                                                        epsilon=0.001,
                                                        center=True,
                                                        scale=True,
                                                        
                                                    )
        self.bn4=tf.keras.layers.BatchNormalization(
                                                        axis=-1,
                                                        momentum=0.99,
                                                        epsilon=0.001,
                                                        center=True,
                                                        scale=True,
                                                        
                                                    )

    def call(self,inputs,training=False):
        d1out,d2out,d3out,d4out=inputs
        out1=self.convup1(d4out)
        out1=self.bn1(out1,training=training)
        out1=mish(out1)
        out2=self.convup2(out1)
        out2=self.bn2(out2,training=training)
        out2=out2+d3out
        out2=mish(out2)
        out3=self.convup3(out2)
        out3=self.bn3(out3,training=training)
        out3=out3+d2out
        out3=mish(out3)
        out4=self.convup4(out3)
        out4=self.bn4(out4,training=training)
        out4=out4+d1out
        out4=mish(out4)
        return out4



        



