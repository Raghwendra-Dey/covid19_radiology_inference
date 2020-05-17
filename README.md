# covid19_radiology_inference
This repo contains the inference code and the saved model for the covid19 detection using cxr images

## Dependencies:

* pytorch
* torchvision
* PIL
* opencv2
* argparse

## Running for inferencing:
1. Clone the repo
2. open up the terminal and type the following:<br>
   `python3 inference.py --weights_path PATH_TO_SAVED_WEIGHTS --image_path PATH_TO_TESTING_IMAGE`<br>
<br>
<b>Note:</b>
Saved Model weights are provided with this repo in the `saved_models folder` so if using the same wrights no need to provide the `--weights_path` parameter. Also if you just want don't have image to test you can still try out inferencing on the example image provided in the `images folder`. Also the `Ground Truth` value of the test image provided in given in the csv file in the `images folder`. 
