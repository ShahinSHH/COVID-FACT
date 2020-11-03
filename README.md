# COVID-FACT
<h4>A Fully-Automated Capsule Network-based Framework for Identification of COVID-19 Cases from Chest CT scans</h4>

COVID-FACT proposes a two-stage fully-automated CT-scan based framework for identification of COVID-19 positive cases. COVID-FACT utilizes Capsule Networks, as its main building blocks and is, therefore, capable of capturing spatial information. In particular, to make the proposed COVID-FACT independent from sophisticated segmentations of the area of infection, slices demonstrating infection are detected
at the first stage and the second stage is responsible for classifying patients into COVID and non-COVID (Normal and Community Acquired Pneumonia) cases. COVID-FACT detects slices with infection in the fisrt stage, and pass the detected slices of each patient to the second stage to perform the patient-level classification.

COVID-FACT is fed with the segmented lung area as the input. In other words, instead of using an original chest CT image, first a <a href="https://github.com/JoHof/lungmask"> U-Net based segmentation model</a> is applied to extract the lung region, which is then provided as input to the COVID-FACT. Besides segmenting the lung regions, all images are normalized between 0 and 1, and resized from the original size of [512,512] to [256,256].

The detailed COVID-Facts's structure and methodology is explained in detail at https://arxiv.org/abs/2010.16041 .

<h3>Note : Please don’t use COVID-FACT as the self-diagnostic model without performing a clinical study and consulting with a medical specialist.</h3>
COVID-FACT is not a replacement for clinical diagnostic tests and should not be used as a self-diagnosis tool to look for COVID-19 features without a clinical study at this stage. Our team is working on enhancing the performance and generalizing the model on multi-center datasets upon receiving more data from medical collaborators and scientific community. You can track new results and versions as they will be updated on this page.<br>

<img src="https://github.com/ShahinSHH/COVID-FACT/blob/main/Figures/method.jpg"/>

## Dataset
In order to train and test the COVID-FACT model, we used the publically available <a href="https://github.com/ShahinSHH/COVID-CT-MD">COVID-CT-MD dataset</a>.
This dataset contains volumetric chest CT scans of 171 patients positive for COVID-19 infection, 60 patients with CAP (Community Acquired Pneumonia), and 76 normal patients. Slice-Level and Patient-Level labels are included in this dataset.

As the main goal of this study is to identify positive COVID-19 cases, we binarized the labels as either positive or negative. In other words the two labels of normal, and CAP together form the negative class.

For the detail description of the dataset, please refer to the <a href="https://arxiv.org/abs/2009.14623">https://arxiv.org/abs/2009.14623</a>.

## Requirements
* Tested with (tensorflow-gpu 2 and keras-gpu 2.2.4) - and (tensorflow 1.14.0 and keras 2.2.4)<br>
-- Try tensorflow.keras instead of keras if it doesn't work in your system.
* Python 3.6
* OpenCV
* Matplotlib



