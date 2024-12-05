import matplotlib.pyplot as plt 
import tensorflow as tf 
import pandas as pd 
import numpy as np
import sys
  
import warnings 
warnings.filterwarnings('ignore') 
  
from tensorflow import keras 
from keras import layers 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense 
from tensorflow.keras.layers import Conv2D, MaxPooling2D,BatchNormalization
from tensorflow.keras.utils import image_dataset_from_directory,plot_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img 
from tensorflow.keras.preprocessing import image_dataset_from_directory 
from tensorflow.keras.preprocessing import image
import os 
import matplotlib.image as mpimg
from zipfile import ZipFile 

data_path = 'C:/Users/sh/Desktop/archive.zip'

with ZipFile(data_path, 'r') as zip: 
	zip.extractall() 
	print('The data set has been extracted.')


path = 'images/images/'
classes = os.listdir(path) 
print(classes)


fig = plt.gcf() 
fig.set_size_inches(16, 16) 

c_dir = os.path.join('images/images/crease')
cg_dir = os.path.join('images/images/crescent_gap')
i_dir = os.path.join('images/images/inclusion')
os_dir = os.path.join('images/images/oil_spot')
ph_dir = os.path.join('images/images/punching_hole')
rp_dir = os.path.join('images/images/rolled_pit')
s2_dir = os.path.join('images/images/silk_spot')
wf_dir = os.path.join('images/images/waist folding')
ws_dir = os.path.join('images/images/water_spot')
wl_dir = os.path.join('images/images/welding_line')
 
c_names = os.listdir(c_dir)
cg_names = os.listdir(cg_dir)
i_names = os.listdir(i_dir)
os_names = os.listdir(os_dir)
ph_names = os.listdir(ph_dir)
rp_names = os.listdir(rp_dir)
s2_names = os.listdir(s2_dir)
wf_names = os.listdir(wf_dir)
ws_names = os.listdir(ws_dir)
wl_names = os.listdir(wl_dir)
 

pic_index = 210

c_images = [os.path.join(c_dir, fname) 
			for fname in c_names[pic_index-8:pic_index]] 

cg_images = [os.path.join(cg_dir, fname) 
			for fname in cg_names[pic_index-8:pic_index]]

i_images = [os.path.join(i_dir, fname) 
			for fname in i_names[pic_index-8:pic_index]]

os_images = [os.path.join(os_dir, fname) 
			for fname in os_names[pic_index-8:pic_index]]

ph_images = [os.path.join(ph_dir, fname) 
			for fname in ph_names[pic_index-8:pic_index]]

rp_images = [os.path.join(rp_dir, fname) 
			for fname in rp_names[pic_index-8:pic_index]]

s2_images = [os.path.join(s2_dir, fname) 
			for fname in s2_names[pic_index-8:pic_index]]

wf_images = [os.path.join(wf_dir, fname) 
			for fname in wf_names[pic_index-8:pic_index]]
ws_images = [os.path.join(ws_dir, fname) 
			for fname in ws_names[pic_index-8:pic_index]]

wl_images = [os.path.join(wl_dir, fname) 
			for fname in wl_names[pic_index-8:pic_index]]
 

all_images = c_images + cg_images + i_images + os_images + ph_images + rp_images + s2_images + wf_images + wl_images + ws_images

# Limit the loop to iterate over the first 8 images
for i, img_path in enumerate(all_images[:8]):
    sp = plt.subplot(4, 2, i+1)
    sp.axis('Off')

    img = mpimg.imread(img_path)
    plt.imshow(img)

plt.show()


base_dir = 'images/images/'

# Create datasets 
train_datagen = image_dataset_from_directory(base_dir,image_size=(200,200),subset='training',seed = 1,validation_split=0.1,batch_size= 32) 
test_datagen = image_dataset_from_directory(base_dir,image_size=(200,200),subset='validation',seed = 1,validation_split=0.1,batch_size= 32)

# Define the model architecture
model = Sequential([ 
    Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)), 
    MaxPooling2D(2, 2), 
    Conv2D(64, (3, 3), activation='relu'), 
    MaxPooling2D(2, 2), 
    Conv2D(64, (3, 3), activation='relu'), 
    MaxPooling2D(2, 2), 
    Conv2D(64, (3, 3), activation='relu'), 
    MaxPooling2D(2, 2), 
    Flatten(), 
    Dense(512, activation='relu'), 
    BatchNormalization(), 
    Dense(512, activation='relu'), 
    Dropout(0.1), 
    BatchNormalization(), 
    Dense(512, activation='relu'), 
    Dropout(0.2), 
    BatchNormalization(), 
    Dense(1, activation='sigmoid') 
])

# Compile the model
model.compile( 
    loss='binary_crossentropy', 
    optimizer='adam', 
    metrics=['accuracy'] 
) 

# Train the model
history = model.fit(train_datagen, 
                    epochs=10, 
                    validation_data=test_datagen)

# Save the model
model.save('fabric_defect_detection_model1.h5')

# Plot training history
history_df = pd.DataFrame(history.history) 
history_df.loc[:, ['loss', 'val_loss']].plot() 
history_df.loc[:, ['accuracy', 'val_accuracy']].plot() 
plt.show()


# Input image for prediction
test_image = image.load_img('images/images/welding_line/img_08_4406424900_00001.jpg', target_size=(200, 200)) 

# Display the input image
plt.imshow(test_image) 
test_image = image.img_to_array(test_image) 
test_image = np.expand_dims(test_image, axis=0) 

# Perform prediction
result = model.predict(test_image) 

# Print the predicted class
if result >= 0.5:
    print("crease")
elif 0.3 <= result < 0.5:
    print("crescent_gap")
elif 0.2 <= result < 0.3:
    print("inclusion")
elif 0.1 <= result < 0.2:
    print("oil_spot")
elif 0.08 <= result < 0.1:
    print("punching_hole")
elif 0.06 <= result < 0.08:
    print("rolled_pit")
elif 0.04 <= result < 0.06:
    print("silk_spot")
elif 0.02 <= result < 0.04:
    print("waist_folding")
elif 0.01 <= result < 0.02:
    print("water_spot")
else:
    print("welding_line")
