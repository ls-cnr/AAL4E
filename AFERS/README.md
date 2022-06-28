# AFERS - Autonomous Face and Emotion Recognition System

This project wants to create a system usable on Structures for Elderly Care in order to track the assisted emotions in a span of time. It has been developed by [Pierfrancesco Martinello](htttps:github.com/pierfrancescomartinello) as a stage project.

---

## Index

- [The Idea](https://github.com/icar-aose/AAL4E/blob/main/AFERS/README.md#the-idea)
- [Background Analysis](https://github.com/icar-aose/AAL4E/blob/main/AFERS/README.md#background-analysis)
	- [Selection of Libraries](https://github.com/icar-aose/AAL4E/blob/main/AFERS/README.md#selection-of-libraries)
- [The System](https://github.com/icar-aose/AAL4E/blob/main/AFERS/README.md#the-system)
	- [Program States](https://github.com/icar-aose/AAL4E/blob/main/AFERS/README.md#program-states)
	- [State Index](https://github.com/icar-aose/AAL4E/blob/main/AFERS/README.md#state-index)
- [RoadMap](https://github.com/icar-aose/AAL4E/blob/main/AFERS/README.md#roadmap)
- [Installation Guide](https://github.com/icar-aose/AAL4E/blob/main/AFERS/README.md#installation-guide)
- [How to run the program](https://github.com/icar-aose/AAL4E/blob/main/AFERS/README.md#how-to-run-the-program)

---

## The Idea

The idea is to use the script ot create some system for the recognition and the analysis of the emotions of elderly users, both in a private environment and in Structures for Elderly Care. Hence the final users will be self-sufficient people and the the product might help them to interact with technology in a simple, interactive and transparent wat

---

## Background Analysis

### The instruments

Instruments for the realization of these project were:
- A minimalist computer, to be easily hidden to the final users. For these purpose there are a lot of possibilities such as NUC systems, Arduino boards or Raspberry Pis. For our implementation the latter has been used. Technical details of the product are here listed:
	- Product: Raspberry Pi 4 Model B Rev 1.5
	- Os: Raspbian GNU/Linux 10 (buster)
	- Architecture: armv7l
	- Kernel: 5.10.103-v7l+
	- CPU: BCM2711 (4 cores, clock speed: 1.50 Ghz)
	- RAM: 2 GB

- Raspberry Pi camera module (version 2.1)

### Selection of libraries

Some developement time was taken to search for a good State-Of-the-Art library for face analysis and recognition. here are listed the libraries take in consideration:
- [Deepface](https://github.com/serengil/deepface):<br> Deepface it's a library that allows to recognize faces and attrubutes such as age, gender and emotions by using models stored in the library.The library is easily install usding the Python Package Index (PyPI)  and it's distrubuted with MIT License.

- [Face Recognition](https://github.com/ageitgey/face_recognition):<br> Face Recognition allows faces recognition and their manipulation, such as the application of filters and masks. It allows the implementation using Deep Learning from dlib library. It's distributed with MIT License

- [FaceNet](https://github.com/davidsandberg/facenet): <br> FaceNet is an implementation of a particular face recognitor described in literature. It also have bundled with it the weights to train the model and it's distributed with MIT License.

- [InsightFace](https://github.com/deepinsight/insightface): <br> InsightFace  allows face recognition and face alingment efficiently, shared with MIT License.

- [OpenCV](https://github.com/opencv/opencv/tree/4.5.5): OpenCV is a library that covers the wide worlf of video and image analysis. Using the thousands of implemented algorithms, it is possible to create a system for face recognition. It is distributed with Apache License 2.0 .

- [OpenFace](https://github.com/cmusatyalab/openface):<br> OpenFace is a library that gives a different implementation fo the aforementioned papaer, by not using the TensorFlow library. This is also distributed with Apache License 2.0.


|Library |Face Recognition|Face Recognition|Emotion Recogntion|Licence|
| :---: | :---: | :---: | :---: | :---: |
| DeepFace | ✅ | ✅ | ✅ | MIT License |
| Face Recognition | ✅ | ✅|❌ | MIT License|
| FaceNet | ✅ |❌ | ❌| MIT License|
|InsightFace | ✅ | ✅ | ❌| MIT License |
| OpenCv | ✅ | ✅|❌ | Apache License 2.0|
| OpenFace | ✅ |❌ | ❌|Apache License 2.0|

The final choice was DeepFace, and it was the main core of the project.

---
## The System
### Program states

The code of the program itself and its running file is divived into states, remarking the idea of a [Finite State Automaton](https://en.wikipedia.org/wiki/Finite-state_machine), with a beginning state and links to various modules

![Image](https://github.com/icar-aose/AAL4E/blob/main/AFERS/Doc/FSA.png "The System")
### State Index
- Initialization: <br> Initialization operation such as obtaining a reference to all the devices used (camera, microphone and speakers), loading the models for the analysis of the emotion and preprocessing of the images stored. The next state will be "Idle" 
- Idle: <br> The system waits until a motion is recognised
- Person Recognition: <br> The system analyse the face and check if it has record of the person. In case there is a record, the sistem goesto "Known Person" state, otherwise, goes to "Registration"
- Known Person: <br> The system analyse the emotion of a person it recognise. If the emotion is positive or neutral, it gets stored in an inner database and then the system will go to the "Pre-Idle" state, otherwise it goes to the state "PER"
- Registration: <br> The system takes name and surname of the person and stores them alongside a picture in the inner database, then goes to "Registration Pre-Idle Operations"
- PER: <br> A series of images is shown (now randomly from a folder), in order to catch change of emotions in the user. The measurements are then processed in order to obtain a mean measurement, that will be later stored. The following state will be "Pre-Idle"
- Registration Pre-Idle Operations: <br> In this state, the system repreprocesses all the images stored in the database in order to include the new one as well; then the state shifts to "Pre-Idle"
- Pre-Idle: <br> Wait until there is no motion, then go to "Idle"
### RoadMap

In this deadline there were a list of ideas and features that were supposed to work at the end of the 200 hours period of the stage, in which this work was made. Here, the list of the major ideas and their current state <br>
|Feature| State |Comment|
| :---: | :---: | :---: |
|Face Recognition|🟢 | |
|Emotion Recognition| 🟢| |
|Emotion Storing|🟢||
|Input Relative Tags|🟡|Problems with database storing made impossible to retrieve the infos about the tags. The developer is confident to fix it in a future release,|
|Requests to an Online Image Library | 🟡|The requests work, but there are problems on the visualization. The developer is confident to fix it in a future release.|
|Requests to an Online Video Library|🟡|The requests work, but there are problems on the visualization. The developer is confident to fix it in a future release.|
|Requests to an Online Audio Library|🔴|An adapt library is yet to be found. The developer is confident to found it in a future release.|
---

## Installation Guide
In order to obtain the code you need to run the following script:

```bash
$git clone https://github.com/icar-aose/AAL4E
```

Make sure to also have the following requirements:

### Debian-Like systems
In order to install pip for pyhton3 (in case you just installed your system):
```bash
$sudo apt-get install python3-pip -y
```

Use the following command to install the library necessary to play MPEG from console:
```bash
$sudo apt-get install mpg321 -y
```
### Fedora
In order to install pip for pyhton3 (in case you just installed your system):
```bash
$yum install epel-release
$yum install python-pip
```
Use the following command to install the library necessary to play MPEG from console:
```bash
$yum install mpg321
```
### Arch 
In order to install pip for pyhton3 (in case you just installed your system):
```bash
$pacman -S python-pip
```
Use the following command to install the library necessary to play MPEG from console:
```bash
$yum install mpg321
```
### Pip requirements

```bash
$pip3 install opencv-python pandas gTTS SpeechRecognition deepface
```
---

## How to run the program

Once you are sure the requirements have been fulfilled, you can run this code


```bash
$cd AAL4E/
```
to enter the folder and then run it with

```bash
$python3 AFERS/main_build.py
```
