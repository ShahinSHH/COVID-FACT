# COVID-FACT
<h4>A Fully-Automated Capsule Network-based Framework for Identification of COVID-19 Cases from Chest CT scans</h4>

COVID-FACT proposes a two-stage fully-automated CT-scan based framework for identification of COVID-19 positive cases. COVID-FACT utilizes Capsule Networks, as its main building blocks and is, therefore, capable of capturing spatial information. COVID-FACT detects slices with infection in the fisrt stage, and passes the detected slices of each patient to the second stage to perform the patient-level classification.

COVID-FACT is fed with the segmented lung area as the input. In other words, instead of using an original chest CT image, first a <a href="https://github.com/JoHof/lungmask"> U-Net based segmentation model</a> is applied to extract the lung region, which is then provided as input to the COVID-FACT. Besides segmenting the lung regions, all images are normalized between 0 and 1, and resized from the original size of [512,512] to [256,256].

<b>The detailed COVID-Facts's structure and methodology is explained in detail at</b> https://arxiv.org/abs/2010.16041 .
<img src="https://github.com/ShahinSHH/COVID-FACT/blob/main/Figures/method.jpg"/>

<h3>Note : Please don’t use COVID-FACT as the self-diagnostic model without performing a clinical study and consulting with a medical specialist.</h3>
COVID-FACT is not a replacement for clinical diagnostic tests and should not be used as a self-diagnosis tool to look for COVID-19 features without a clinical study at this stage. Our team is working on enhancing the performance and generalizing the model on multi-center datasets upon receiving more data from medical collaborators and scientific community. You can track new results and versions as they will be updated on this page.<br>

A sample of lung areas contibuting the most to the final output in COVID-FACT are shown usign the <a href="https://arxiv.org/abs/1610.02391">GRAD-CAM</a> algorithm in the following image.

<img src="https://github.com/ShahinSHH/COVID-FACT/blob/main/Figures/heatmap1.jpg" width="500" height="350"/>

## Dataset
The publically available <a href="https://github.com/ShahinSHH/COVID-CT-MD">COVID-CT-MD dataset</a> is used to train/test the model.
This dataset contains volumetric chest CT scans of 171 patients positive for COVID-19 infection, 60 patients with CAP (Community Acquired Pneumonia), and 76 normal patients. Slice-Level labels (slices with the evidence of infection) are provided in this dataset.

For the detail description of the COVID-CT-MD dataset, please refer to the <a href="https://arxiv.org/abs/2009.14623">https://arxiv.org/abs/2009.14623</a>.

## Lung Segmentation
The lungmask module for the lung segmentation is adopted from <a href="https://github.com/JoHof/lungmask">here</a> and can be installed using the following line of code:
```
pip install git+https://github.com/JoHof/lungmask
```
Make sure to have torch installed in your system. Otherwise you can't use the lungmask module.
<a href = "https://pytorch.org">https://pytorch.org</a>

## Code
The code for the Capsule Network implementation is adopted from <a href="https://keras.io/examples/cifar10_cnn_capsule/">here.</a>
The brief desciption of the code's functionality is as follows:
* The code aims to test a CT scan using the COVID-FACT model.
* The training code is not provided. However, the code to implement each stage of the model is available in the code
* The outcome is in the binary (COVID-19, non-COVID) format.
* The test data should be in the DICOM format, and all DICOM files corresponding to a CT scan should be located in one folder.

Codes are available as the following list:

* COVID-FACT_binary_test.py : Codes for testing a CT scan series <b>(Set the "data_path" at the very begining of the code based on your data directoy.)</b>
* weights-stage1-final-91.h5 : Best model's weight for the first stage
* weights-stage2-final-99.h5 : Best model's weight for the second stage

## Requirements
* Tested with (tensorflow-gpu 2 and keras-gpu 2.2.4) , and (tensorflow 1.14.0 and keras 2.2.4)<br>
-- Try tensorflow.keras instead of keras if it doesn't work in your system.
* Python 3.6
* PyTorch 1.4.0
* Torch 1.5.1
* PyDicom 1.4.2 (<a href="https://pydicom.github.io/pydicom/stable/tutorials/installation.html">Installation<a/>)
* SimpleITK (<a href="https://simpleitk.readthedocs.io/en/v1.1.0/Documentation/docs/source/installation.html">Installation</a>)
* lungmask (<a href="https://github.com/JoHof/lungmask">Installation</a>)
* OpenCV
* OS
* Numpy
* Matplotlib


## Citation
If you found this code and the related paper useful in your research, please consider citing:

```
@article{Heidarian2020,
archivePrefix = {arXiv},
arxivId = {2010.16041},
author = {Heidarian, Shahin and Afshar, Parnian and Enshaei, Nastaran and Naderkhani, Farnoosh and Oikonomou, Anastasia and Atashzar, S. Farokh and Fard, Faranak Babaki and Samimi, Kaveh and Plataniotis, Konstantinos N. and Mohammadi, Arash and Rafiee, Moezedin Javad},
eprint = {2010.16041},
month = {oct},
title = {{COVID-FACT: A Fully-Automated Capsule Network-based Framework for Identification of COVID-19 Cases from Chest CT scans}},
url = {http://arxiv.org/abs/2010.16041},
year = {2020}
}

```


