<div id="top"></div>

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <h3 align="center">Patient-Activity-Recognition-System</h3>
  <h4 align="center">Patient Activity Recognition System trained with CNN LSTM on Synthetic Data Generated using Unity</h4>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#dataset-and-feature-extraction">Dataset and Feature Extraction</a></li>
    <li><a href="#contributions">Contributions</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

<img src="/Assets/act.png" width="800" height="600">

Hospitals and other medical centres have facilities of keeping patients under medical attention to monitor their progress. The nurse carefully monitors the patients and takes necessary action in case of any unusual activity or sense of emergency. However, sometimes patients cannot afford this service as it’s expensive. If we have an automated system in place which tracks patient’s activity and informs concerned authorities to take immediate actions in case of emergency, then many people would be benefitted. This will incur camera installation and setting up of other logistics but would be still cheaper than human monitoring. Also, there are many instances of complications or even deaths in rehab centres because of no or delayed action in case of events like collapse, heart attack or seizures. This can be mitigated by having a 24*7 surveillance in place that recognizes an abnormality in patient’s behaviour and alerts respective authorities. We introduce a novel model in this paper to monitor patient activity using the CNN-LSTM model. <br/>
For more details, please see [Report](/Report/)

<p align="right">(<a href="#top">back to top</a>)</p>



### Built With

* [TensorFlow+Keras](https://www.tensorflow.org/)
* [NumPy](https://numpy.org/)
* [OpenCV](https://opencv.org/)
* [MediaPipe](https://google.github.io/mediapipe/solutions/pose.html)
* [Mixamo](https://www.mixamo.com/#/)
* [Unity3D](https://unity.com/)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

* TensorFlow
  ```sh
  pip install tensorflow
  ```
* NumPy
  ```sh
  pip install numpy
  ```
* OpenCV
  ```sh
  pip install opencv-python
  ```
* Mediapipe
  ```sh
  pip install mediapipe
  ```
* Unity
  ```sh
  Follow instructions: https://unity.com/download
  ```

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/parthgoe1/Patient-Activity-Recognition-System.git
   ```

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- DATASET AND FEATURE EXTRACTION -->
## Dataset and Feature Extraction
<img src="/Assets/Activities.png" width="800" height="400">
Since it was difficult to get real data of a rehab center and then train the model for some specific events, we generated synthetic data using Unity3D. We divided the training data into two broad categories; Activities that are considered normal and those which are abnormal
<b>Normal activities:</b>Sitting, Standing, Walking ,Sleeping </br>
<b>Abnormal activities:</b>Falling, pain/coughing </br>
<img src="/Assets/keypoints.png" width="600" height="400">
We divide the simulations into frames for keypoint detection model. Further, the output from keypoint models are normal- ized to pass it on to activity detection model.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- CONTRIBUTIONS -->
## Contributors

1. [Parth Goel](https://github.com/parthgoe1)
2. [Prashant Kanth](https://github.com/kanthprashant)
3. [Aditya Bhat](https://github.com/adityacbhat)
3. [Rishika Bhanushali](https://github.com/rb-rishika)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* S. N. Gowda, M. Rohrbach, and L. Sevilla-Lara, “Smart frame selection for action recognition,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 35, pp. 1451–1459, 2021.
* Z. Cao, T. Simon, S.-E. Wei, and Y. Sheikh, “Realtime multi-person 2d pose estimation using part affinity fields,” in Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 7291–7299, 2017.
* V. Bazarevsky, I. Grishchenko, K. Raveendran, T. Zhu, F. Zhang, and M. Grundmann, “Blazepose: On-device real-time body pose tracking,” arXiv preprint arXiv:2006.10204, 2020.
* T. N. Sainath, O. Vinyals, A. Senior, and H. Sak, “Convolutional, long short-term memory, fully connected deep neural networks,” in 2015 IEEE international conference on acoustics, speech and signal processing (ICASSP), pp. 4580–4584, IEEE, 2015.

<p align="right">(<a href="#top">back to top</a>)</p>

