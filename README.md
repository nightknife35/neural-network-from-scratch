since this is for learning porpuses i decided to it would make more sense (for me), if the dataset that i got from kaggle (https://www.kaggle.com/datasets/hojjatk/mnist-dataset) would be of different format

as it stands its 2 big binary files with all the images in a big file (archive\train-images-idx3-ubyte) and all the labels in another big file (archive\train-labels-idx3-ubyte). 
what i did is broke down the big image file into smaller files (784 bytes each), where each file is an image. 
i had a similar aproach with the labels for consistency
although this might not be the most efficient way, it makes sense in my head
