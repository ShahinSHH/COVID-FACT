# COVID-FACT
<h4>A Fully-Automated Capsule Network-based Framework for Identification of COVID-19 Cases from Chest CT scans</h4>

COVID-FACT proposes a two-stage fully-automated CT-scan based framework for identification of COVID-19 positive cases. COVID-FACT utilizes Capsule Networks, as its main building blocks and is, therefore, capable of capturing spatial information. In particular, to make the proposed COVID-FACT independent from sophisticated segmentations of the area of infection, slices demonstrating infection are detected
at the first stage and the second stage is responsible for classifying patients into COVID and non-COVID cases. COVID-FACT detects slices with infection, and identifies positive COVID-19 cases using an in-house CT scan dataset, containing COVID-19, community acquired pneumonia, and normal cases.

<img src="https://github.com/ShahinSHH/COVID-FACT/blob/main/Figures/method.jpg"/>

COVID-FACT is fed with the segmented lung area as the input. In other words, instead of using an original chest CT image, first a <a href="https://github.com/JoHof/lungmask"> U-Net based segmentation model</a> is applied to extract the lung region, which is then provided as input to the COVID-FACT. Besides segmenting the lung area, all images are normalized between 0 and 1, and resized from the original size of [512,512] to [256,256].
