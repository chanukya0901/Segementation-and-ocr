import tensorflow as tf
import pandas as pd
from utils.dataset import Dataset
import albumentations as A
from model.seg_model import Segmodel
import math
import numpy as np
data=pd.read_csv("train.csv")

images_paths=list(data["paths"].values)
labels=list(data["labels"].values)

def split_data(images, labels, train_size=0.9, shuffle=True):
    # 1. Get the total size of the dataset
    size = len(images)
    # 2. Make an indices array and shuffle it, if required
    indices = np.arange(size)
    if shuffle:
        np.random.shuffle(indices)
    # 3. Get the size of training samples
    train_samples = int(size * train_size)
    # 4. Split data into training and validation sets
    x_train, y_train = images[indices[:train_samples]], labels[indices[:train_samples]]
    x_valid, y_valid = images[indices[train_samples:]], labels[indices[train_samples:]]
    return x_train, x_valid, y_train, y_valid


x_train, x_valid, y_train, y_valid = split_data(np.array(images_paths), np.array(labels))

transform=A.Compose([
            A.Resize(height=416, width=416),
            A.RandomBrightnessContrast(p=0.2),
            A.CenterCrop (height=416, width=416, always_apply=False, p=0.3),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=45, p=0.2)

])
valid_transform=A.Compose([
            A.Resize(height=416, width=416),

])

train_dataset=Dataset(paths=images_paths,labels=labels,aug=transform)

valid_dataset=Dataset(paths=x_valid,labels=y_valid,aug=valid_transform)




def dice_coefficient(y_pred, y_true):
    intersection = tf.reduce_sum(y_pred * y_true)
    union = tf.reduce_sum(y_pred) + tf.reduce_sum(y_true)
    dice = (2.0 * intersection + 1e-5) / (union + 1e-5)
    return dice

def dice_loss(y_pred, y_true):
    return 1.0 - dice_coefficient(y_pred, y_true)

model=Segmodel()
sample_input=tf.random.normal(shape=(4,416,416,3))
model(sample_input)




optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)


@tf.function
def train_step(x,y):
    with tf.GradientTape() as tape:
            # Run the forward pass of the layer.
            # The operations that the layer applies
            # to its inputs are going to be recorded
            # on the GradientTape.
            logits = model(x, training=True)  # Logits for this minibatch

            # Compute the loss value for this minibatch.
            loss_value = dice_loss(y, logits)

    # Use the gradient tape to automatically retrieve
    # the gradients of the trainable variables with respect to the loss.
    
    grads = tape.gradient(loss_value, model.trainable_weights)
   

    # Run one step of gradient descent by updating
    # the value of the variables to minimize the loss.

    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    if tf.math.is_nan(loss_value):
         print("logits :",logits)
    return loss_value
@tf.function
def test_step(x,y):
     logits = model(x, training=False)  # Logits for this minibatch

            # Compute the loss value for this minibatch.
     loss_value = dice_loss(y, logits)
     return loss_value

epochs=10    

for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))

    # Iterate over the batches of the dataset.
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        

        loss_value=train_step(x_batch_train,y_batch_train)
        
        
        
        print(f"train loss value at {step} and {epoch}:",loss_value)

    for step, (x_batch_valid, y_batch_valid) in enumerate(valid_dataset):
       

        loss_value_valid=train_step(x_batch_valid,y_batch_valid)
        
        
        
        print(f"valid loss value at {step} and {epoch}:",loss_value_valid)



    if (epoch % 10):
             model.save_weights(f"wts/wts_{loss_value_valid}.h5")

        
        

