# ECS-171-Project (Face Mask Detection)

## Introduction
We chose to do this project because of the recent pandemic which made it essential to wear masks to protect one's health. Although COVID-19 has been put on the back burner, there can be resurgences or even the possibility of a future pandemic which will require mask mandates. Even in the fields of healthcare, masking is required among doctors to ensure a sterile and protected environment. Having underwent the COVID-19 pandemic, masks have certainly influenced our lives in one way or another. 

To enforce these future mask mandates, various locations like grocery stores, concerts, etc. can utilize face mask recognition technology to swiftly and accurately deny entry from maskless individuals to ensure public safety. As a result, more lives can be saved from airborne diseases and can prevent the spread of such diseases like COVID-19

## Figures
For our figure models, we had a set of images that contained masks on and another set of images without the masks on. There were around 12,000 total images. With more images there will be a higher average or data training accuracy in determining whether the image has a mask on or not.

Images were sized to be 100 x 100 to prevent long run time

Evaluation Graph plotted at the bottom depicts comparison of Model 1’s accuracy v.s. val_accuracy 

## Methods
### Data Exploration
The initial dataset consisted of almost 12,000 PNG images of various sizes so we explored the data by doing the following:

1. Counting the precise number of images from all data folders and listing all possibles sizes of images from all data folders

````
# check how many images and their sizes

# obtain paths
test_mask_path = './Face Mask Dataset/Test/WithMask'
test_nomask_path = './Face Mask Dataset/Test/WithoutMask'
train_mask_path = './Face Mask Dataset/Train/WithMask'
train_nomask_path = './Face Mask Dataset/Train/WithoutMask'
val_mask_path ='./Face Mask Dataset/Validation/WithMask'
val_nomask_path = './Face Mask Dataset/Validation/WithoutMask'

# os.listdir gets all files in a given path
image_count = 0
image_sizes = set()
paths_list = [test_mask_path, test_nomask_path, train_mask_path, train_nomask_path, val_mask_path, val_nomask_path]

#function to add width and height to image_sizes set
#and returns the number of images
def checkImages(path, image_sizes):
  count = 0
  for i in os.listdir(path):
    count += 1
    im = Image.open(os.path.abspath(path + "/" + i))
    width, height = im.size
    image_sizes.add((width, height))
  return count

for path in paths_list:
  image_count += checkImages(path, image_sizes)

#image_count = len(list(data_dir.glob('*/*.jpg')))
print("Total images:", image_count)
print("Image sizes (Width, Height):", image_sizes)
````

2. Examining/plotting the images from the data folders to observe the way mask/maskless people are presented (in our case, single headshots of people facing the camera head on with little variation)

````
# view some example images
mask_im = Image.open(os.path.abspath(test_mask_path + "/1174.png"))
nomask_im = Image.open(os.path.abspath(test_nomask_path + "/3006.png"))
display(mask_im)
display(nomask_im)
````

3. Assigning target class values for masked/maskless images (1 for masked and 0 for maskless)

### Preprocessing
After initial data exploration, we preprocessed the data by doing the following:
- Extracting images to use from the testing/training folders that are of size  100x100 or above and resizing each extracted image to 100x100
- Converting the extracted PNG images into a numpy array 
- Creating our own dataframes for training and testing data with one column being the image data and the other being the target

````
from tensorflow.keras.preprocessing.image import img_to_array

def SetUpData(mask_path):
  mask_list = []
  mask_labels = []
  for i in os.listdir(mask_path):
    im = Image.open(os.path.abspath(mask_path + "/" + i))
    width, height = im.size
    if width >= 100 and height >= 100:
      im = im.resize((100, 100))
      # convert to numpy array
      im = img_to_array(im)
      mask_list.append(im)

      append_label_value = -1
      if 'WithoutMask' in mask_path:
        append_label_value = 0
      else:
        append_label_value = 1
      # -1 means error
      #1 for mask
      #0 for no mask
      mask_labels.append(append_label_value)
  return mask_list, mask_labels


# set up data for training mask data
# establish dataframe containing masked images and their targets
train_mask_df = pd.DataFrame()
train_mask_df['image'], train_mask_df['target'] = SetUpData(train_mask_path)


# set up data for training nonmask data
# establish dataframe containing nonmasked images and their targets
train_nomask_df = pd.DataFrame()
train_nomask_df['image'], train_nomask_df['target'] = SetUpData(train_nomask_path)


# combine separate training dataframes and shuffle
train_df = pd.concat([train_mask_df, train_nomask_df])
train_df = shuffle(train_df)


# set up data for testing mask data
# establish dataframe containing masked images and their targets
test_mask_df = pd.DataFrame()
test_mask_df['image'], test_mask_df['target'] = SetUpData(test_mask_path)


# set up data for testing nomask data
# establish dataframe containing nonmasked images and their targets
test_nomask_df = pd.DataFrame()
test_nomask_df['image'], test_nomask_df['target'] = SetUpData(test_nomask_path)


# combine separate testing dataframes and shuffle
test_df = pd.concat([test_mask_df, test_nomask_df])
test_df = shuffle(test_df)
````

### Model 1
With preprocessing completed, we created our initial model by doing the following:

- Establishing a sequential model with rescaling as the first step
- Establishing three Conv2D layers with relu activation following the initial rescaling layer
- Adding MaxPooling2D layers after each Conv2D layer
- Adding a Flatten layer after the Conv2D layers to input the data into Dense layers
- Wrapping up the model with one Dense layer of relu activation followed by another Dense layer of sigmoid activation

````
# creating the model
train_model = Sequential([
  layers.Rescaling(1./255, input_shape=(100, 100, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(1, activation='sigmoid'),
])
````

For the compilation of the model, we used the adam optimizer and binary cross entropy as our loss function:
````
# compiling the model
train_model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
````

Before fitting the model, the data had to be reshaped to fit the initial input dimensions:
````
train_image_array = np.stack(np.asarray(train_df["image"]))
test_image_array = np.stack(np.asarray(test_df["image"]))
train_target_array = np.asarray(train_df["target"])
test_target_array = np.asarray(test_df["target"])
````

For fitting the model, we used a validation split of 0.15 and set the amount of epochs/batch size to 10:
````
fitted_Model = train_model.fit(train_image_array, train_target_array, validation_split=0.15, batch_size=10, epochs=10)
````

And to finally analyze the accuracy of our model for both training and testing data, we thresholded our predicted yhat values and ran a classification report:
````
## evaluate model to compare training vs. test error
from sklearn.metrics import classification_report

yhat_train = train_model.predict(train_image_array)
yhat_train = [1 if y>=0.5 else 0 for y in yhat_train]
yhat_test = train_model.predict(test_image_array)
yhat_test = [1 if y>=0.5 else 0 for y in yhat_test]

print("Training performance:\n %s" % classification_report(train_target_array, yhat_train))

print("Testing performance:\n %s" % classification_report(test_target_array, yhat_test))
````

## Results
Model 1 had an exceptional accuracy of 99% for training data and 98% for testing data, with minimal loss at 0.0448

[PLACE GRAPH HERE]

Model 2 on the other hand had a training accuracy of 55% and testing accuracy of 53% and was not predicting the maskless class at all. The loss for this model ended up being 0.0072.

[PLACE GRAPH HERE]

## Discussion
For the construction of the first model, we used a neural network:
- We first rescaled the units from smaller to bigger due to it being easier to break smaller pieces into big pieces
- 3 convolutional layers were used since they are the best to use to process images
- Each convolutional layer was followed by max pooling. By down-sampling the input representation, dimensionality is reduced, resulting in a faster running model.
- The ReLU function was used as our activation function because it is better at computing and solving vanishing gradient problems. It is generally more computationally efficient compared to sigmoid and tanh function.
- Flatten was used to flatten the multi-dimensional inputs into a single dimension.
- After flattening, two dense layers were used. The first used ReLU activation function and the second had one node to give a single output used to determine whether an image was “masked” or “maskless.” The sigmoid activation function was used in this layer as it ensures an output value between 0 and 1 which can be thresholded.

The second model is a neural network similar to the first model:
- Instead of 3 convolutional layers, 2 were used instead.
- The second dense layer uses the softmax activation function instead of the sigmoid activation function. This activation function creates a vector of values that sum to 1.

Results for Model 1 were extremely high where at first we thought that we might have overtrained the model, but since the testing data also showed high accuracy it is likely due to the lack of variation in the data.

Results for Model 2 had a drastically lower accuracy than that of model 1 which can be attributed to the following factors:
- There is one less layer and its first layer starts with twice the amount of units that the first model has in its first layer, making it too many units
- Model 2 seems to only predict one of the two classes
- The usage of softmax returning a value of 1

Data results of high accuracy could be due to large image sets that we had and not the lack of variation in data which lead to better training with images we already had and better recognition.

We're not sure if each of 12,000 pictures had a controlled environment, but we believe that we could have some outlier data that might caused wrong training of the image data such as weather conditions that might have caused a wrong prediction of whether a person was masked or not.
- Further confounding variables could be if someone has a beard, face paintings (religious), or fake masks such as bandanas that could be misconstrued for a mask, etc.
- Additionally, some of the images were not images of real people, but rather drawings or cartoon images of faces with masks. Whether or not these images positively or negatively affected the the accuracy of the model is unknown, although it probably did not negatively affect the model given the model’s high accuracy.

## Conclusion
The project could benefit from data with higher variation in data and utilizing a webcam to see if it can work with live video which can be potentially converted in to a working product.

So far, the model only detects whether or not a singular person is wearking a mask or not. As such, the model could be upgraded and used to detect multiple people in a scene and determine if each person is masked or maskless. 

The model only detects whether a person is wearing a mask or not. For a mask to be effective in stopping the spread of disease, it must be worn properly and as such, we could improve our project to detect if a person is wearing a mask improperly. 

## Collaboration
### Tracey Ngo - Leader and Coder
- Organized meetings to discuss project ideas
- Created models 1 and 2 and compiled/fitted the models
- Wrote rough outline for write-up

### Steven To - Coder and Writer
- Coded the steps involved in the data exploration/preprocessing parts of the project
- Typed out the contents for the methods section of the write-up
- Transfered contents of write-up from Google Doc to Readme.md

### Benjamin Hoang - Code Cleaner and Writer
- Rewrote some code to avoid repetition and for clarity (‘for’ loops part)
- Added some information to the introduction

### Awen Li - Collaborator, Writer, and Code Cleaner
- Worked with Benjamin on ideas to rewrite code for it to be more efficient (training and testing parts)
- Fitted results model for better visualization
- Wrote up on Figure with description and its representation, Discussion on any problems with confounding variables, and added on to the Results graph

### Jordan - Collaborator, Writer, and Coder
- Helped with testing models
- Created evaluation graph
- Helped with the discussion and conclusion parts of the write-up












