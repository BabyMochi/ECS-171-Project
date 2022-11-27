# ECS-171-Project

Preprocessing:
Based on the information from Kaggle, the images don't seem to be standardized so they will have to undergo some sort of scaling. Standardization requires normally distributed data so the pictures will have to be normalized first if they are to be standardized. Cropping the images does not seem necessary as virtually all of the image data are concise headshots of people with or without masks, but we will be resizing the images so that they are all the same size since we plan on using convolutional layers. We filtered out the images with a size of lower than 100x100 or below, and resized the rest to be 100x100. 

#Model Training:
We created the first model and the validation and testing performance is 99%. Both the evaluation graph and fitted graph are highly correlated to the training and testing data, showing no obvious split where the testing performance would go down from overtraining. 
