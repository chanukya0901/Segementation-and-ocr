import tensorflow as tf
from model.layer_helpers import Encoder,Decoder
from model.layer_helpers import mish

class Segmodel(tf.keras.Model):
  def __init__(self,):
    super(Segmodel, self).__init__(name='')
    self.encoder=Encoder()
    self.decoder=Decoder()
    self.finalconv=tf.keras.layers.Conv2D(
                                            8,
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
    self.convupfinal=tf.keras.layers.Conv2DTranspose(
                                                1,
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
    self.bn2=tf.keras.layers.BatchNormalization(
                                                        axis=-1,
                                                        momentum=0.99,
                                                        epsilon=0.001,
                                                        center=True,
                                                        scale=True,
                                                        
                                                    )

    
  def call(self, inputs, training=False):
    eouts=self.encoder(inputs,training=training)
    decoderout=self.decoder(eouts,training=training)
    outprefinal=self.finalconv(decoderout)
    outprefinal=self.bn1(outprefinal,training=training)
    out=mish(outprefinal)
    finalout=self.convupfinal(out)
    finalout=self.bn2(finalout,training=training)
    finallogits=tf.math.sigmoid(finalout)


    
    return finallogits
