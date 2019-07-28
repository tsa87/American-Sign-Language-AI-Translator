## American Sign Language Translator

Trained on the Sign Language MNIST [dataset](https://www.kaggle.com/datamunge/sign-language-mnist) using the VGG architecture.

### Result
#### Weighted Precision: 95% 
#### Weighted Accuracy: 94%

![Imgur](https://i.imgur.com/Eq9GPCC.png?1)
![Imgur](https://i.imgur.com/X839lcF.png?1)
##### The topleft corner (PREDICTION/GROUND TRUTH)

### Getting Started
1. Clone the project 
```
git clone https://github.com/tsa87/Intuitive-Gesture-Drone-Control.git
```
2. Install the prerequisites
```
pip install pillow
pip install pandas
pip install keras
pip install sklearn
pip install opencv-contrib-python
pip install matplotlib
```
3. Train the VGG network on traing dataset
```
python train.py
```
4. Run the demo on testing dataset
```
python test.py
```
### Future Work
- Combine this network with the [hand tracker](https://github.com/tsa87/Intuitive-Gesture-Drone-Control) to allow real time translation. \
- Integrating language modeling and generating Recurrent Neural Network.
### Author

* **Tony Shen** - *Initial work*
