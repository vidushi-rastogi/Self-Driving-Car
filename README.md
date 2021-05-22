# SELF - DRIVING - CAR 
#
#
Self driving car using applied Deep Learning.
Self generated Data set for training the model - https://github.com/Ackermann99/Training-track.git
Simulator used to generate training data set and testing the car - https://github.com/udacity/self-driving-car-sim.git

## Obtaining more uniformed data set
As the initial generated data was more centered biased, meaning the steering angle for most of the data is 0. So the data is balanced towards both left and right steering angles

```
print("Initial total data length: ", len(data))
removed_list = []
for j in range(num_bins):
  bin_list = []
  for i in range(len(data["steering"])):
    if data["steering"][i] >= bins[j] and data["steering"][i] <= bins[j+1]:
      bin_list.append(i)
  bin_list = shuffle(bin_list)
  bin_list = bin_list[samples_per_bin:]
  removed_list.extend(bin_list)

print("Length of removed data: ", len(removed_list))
data.drop(data.index[removed_list], inplace=True)
print("Length after removal: ", len(data))
```

#
#
## Applying various data augmentations method
This is done using `imgaug` python library. 
The augmentation methods applied in this project are -
1. Zoom-In
2. Panning
3. Adjusting Brightness Randomly
4. Flipping the image horizontly on random
#
#
## Creating a Batch Gengerator
What a batch generator does is it apply the augmentation on the run over the data on random basis, and it is applied for almost 50% of the time so as to save computational power.
With this the augementation methods are only applied on training set which is checked using *istraining* bool value
```
def batch_generator(image_paths, steering_ang, batch_size, istraining):
  
  while True:
    batch_img = []
    batch_steering = []
    
    for i in range(batch_size):
      random_index = random.randint(0, len(image_paths) - 1)
      
      #checks if the data is from training data
      if istraining:
        im, steering = random_augment(image_paths[random_index], steering_ang[random_index])
     
      else:
        im = mpimg.imread(image_paths[random_index])
        steering = steering_ang[random_index]
      
      im = img_preprocessing(im)
      batch_img.append(im)
      batch_steering.append(steering)
    yield (np.asarray(batch_img), np.asarray(batch_steering))
```

#
#
Then lastly training the model using batch generator and running it for validation set as well, obtaining loss value.

#
#
## Saving the model locally
```
model.save("model.h5")

from google.colab import files
files.download("model.h5")
```

##### Drive.py
This file is used to connect with the Udacity simulator, `Flask` python framework is used for the Client-Server connection.
1. Simply run it.
2. Start the simulator in Autonomous mode.
3. A *Connected* message will on the console, meaning you're now connected with the simulator

##### model.h5
File for trained model, which the simulator use to drive the car.
>Note: Save model.h5 file in the same folder as drive.py file

#
#
**Code and learning reference -**
https://www.udemy.com/course/applied-deep-learningtm-the-complete-self-driving-car-course/


