# Computer Vision - Project 2

Deep fakes have been gaining more popularity every year. Thanks to new advanced
machine learning techniques, the quality of deep fakes has increased so much that many
have become difficult to detect fake faces even for humans.

Due to this growth in popularity, research in the area has increased considerably, 
resulting recently in deep fake detection challenges on Kaggle, with thousands of 
participants. Considering the race between deep fakes generators and discriminators, 
this work aims to 

1. collect relevant deep fake image data sets for experimentation, and 
2. compare different techniques for detecting deep fake generated faces on the collected data.


## prerequisites
* python > 3.6
* OpenCV > 3
* numpy
* sklearn
* matplotlib
* tensorflow

All available on requirements.txt file:

```console
pip install -r requirements.txt
```

## Extracting face photos from movie datasets

Dataset for this project is lightweight on the disk and is already included in src/data.
OpenCV was used to extract face photos from movie files in the kaggle deep fake detection challenge preview dataset, containing 400 videos.
The full Video dataset is available in https://www.kaggle.com/c/deepfake-detection-challenge/data?select=train_sample_videos .

Face photos may be extracted from any movie files, as long as there is a metadata.json file in the same folder than the videos, with fake / real labels for each video. For this, create a new folder called "deepfake-detection-challenge-data" and place the video files inside, then run:

```console
python make-dataset.py
```

face files will be created in the 'data/' folder.

## Model Training

Run train_dft_svm.py to directly train the DFT + SVM model, using features extracted from the generated face files.

```console
python train_dft_svm.py
```

For re-extraction of the features from the image files, or exploration of the the training process please check the following jupyter notebooks:
* DFT-train-test.ipynb - DFT feature extraction pipeline + SVM and LR models training and testing;
* meso-net-train-test.ipynb - MesoNet pretraining model retraining and testing on generated face images.


## Testing pre-trained DFT + SVM model on video files

Place your movie file in data/ folder and run fake-face-detection.py passing the file path as a param:

```console
python fake-face-detection.py data/fake-1.mp4
```

## Main sources:
* https://github.com/DariusAf/MesoNet
* https://github.com/cc-hpc-itwm/DeepFakeDetection