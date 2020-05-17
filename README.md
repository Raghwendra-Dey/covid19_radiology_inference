# covid19_radiology_inference
This repo contains the inference code and the saved model for the covid19 detection using cxr images

## Dependencies:

* pytorch
* torchvision
* PIL
* numpy
* argparse
* opencv2 (if using opencv2 readable image like, `.jpg`, `.jfif`, `.png`, etc)
* pydicom (if using `.dcm` image)

## Running for inferencing:
1. Clone the repo
2. Open up the terminal and type the following:<br>
   `python3 inference.py --model MODEL_NAME_FOR_INFERENCING --image_path PATH_TO_TESTING_IMAGE` <br><br>
<b>Note:</b><br>
Saved Model weights are provided with this repo, in the `saved_models` folder so if using resnet18 no need to provide the `--model` parameter. Also if you don't have image to test you can still try out inferencing on the example image provided in the `images` folder. Also the `Ground Truth` value of the test image is given in the csv file `label.csv` in the `images` folder.<br><br>
`PATH_TO_TESTING_IMAGE`: Must be with the full extension of the image i.e. with `.jpg` or `.dcm` or `.jfif`, etc.<br><br>
`MODEL_NAME_FOR_INFERENCING` choices:<br>
   1. resnet18 -- accuracy achieved: 66%<br>
   2. LeNet -- accuracy acheived: 64.9%
