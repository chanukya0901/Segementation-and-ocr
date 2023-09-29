# Segementation-and-ocr


OCR of Odometer  reading

Two models were used to extract odometer reading from given image.


1. custom segmentaion model
2. custom ocr model

key points considered for modelling:

 -> from training data as there is no greater variation in images and there is only one class to segement.
 -> objcet (odeometer reading) is very tiny so no deeper encoder for segmentaion model is required, as local features up to 4 blocks is enough to 
    capture the context of object.so no deeper heavy pretrained model is required.
 -> for text extraction also we are having vocab are numbers and max length is 8.so no deeper heavy pretrained model is required.




1. custom segmentaion model :

  -> Model architecture:       
    
    -> followed unet style architecture for model building.   
    -> model consists of two components encoder and decoder.
    -> encoder contains 4 dense blocks .
    -> each dense block has 3 conv layers (kernel size 3 and padding same) and 3 batch norm layers.mish activation is used for non linear activation.
    -> bacth norm is added to make layer activation 0 mean and 1 std to make loss contours round such that model converges fast.
    -> 3 conv layers are connect in dense fashion resembling dense net architecture to enable efficent flow of gradients through back propogation.
    -> decoder consists of covn2d transpose layers to up sample convmaps .
    -> each layer output of decoder is added with corresponding similliars size activations from encoder blocks to group both local and high level features together.
    -> final layer is conv layer with sigmoid with 1 filter as we are having only one object to predict.

  -> Data preparation:
     
    -> albumentations are used for data transfroms. 
    -> each image is resized to 416,416 .
    -> for augmentation random brightness,shiftscale rotate, center crop is used.
    -> image is normalized by dividing pixel values with 255.

  -> Training:
    
    -> data is split into train and valid with 0.1 percent valid data.
    -> learning rate of 0.00001 is used as we are initializing weights randomly.having high learning rate may put model into higher loss and recovery may take time.


2. custom ocr model.

   -> As data is very simple to train .followed the training of ocr captcha model of keras.

       https://keras.io/examples/vision/captcha_ocr/

Inference Pipeline:
    
    1. unzip foldername.zip 
       unzip product_overlay_sappa_chanukya.zip 

    2. create virtual environment.
       python3 -m venv odometer
    
    3. activate virtual environment
       source odometer/bin/activate

    4. install necessary requirements for code to run.
       pip install -r setup.txt

    5. export pythonpath to current directory
        export PYTHONPATH=$PWD

    6. run test_predict.py with your test folder path.
        python test_predict.py --folder_path ./train/62a4ff862be4ea4a151632a9

        instead of ./train/62a4ff862be4ea4a151632a9 you need to provide your test directory path 

    7. result will be stored with name test_results.csv in the present working directory.

  Note: both model is trained on ubuntu22.04 with 16gb ram and 6 gb gpu ram.
        all the above details provided are steps to run on linux machine.  


  



