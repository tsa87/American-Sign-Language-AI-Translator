import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from PIL import Image
import cv2

test = pd.read_csv("./input/sign_mnist_test.csv")
target_names = ["a","b","c","d","e","f","g","h","i","k","l","m","n","o","p",
    "q","r","s","t","u","v","w","x","y"]

labels = test['label'].values
test.drop('label', axis = 1, inplace = True)

images = test.values

labels = LabelBinarizer().fit_transform(labels)
images = images.astype("float")/255.0
images = images.reshape(images.shape[0], 28, 28, 1)

model = load_model("./VGG_weights.hdf5")
preds = model.predict(images, batch_size=32).argmax(axis=1)

report = classification_report(labels.argmax(axis=1),preds,
    target_names = target_names)
print(report)

for i, image in enumerate(images):
    if i is 40:
        break
    prediction = preds[i]
    label = labels[i]
    #print(label.argmax(axis = 0))
    src = image.reshape(28, 28)
    img = cv2.resize(src, (400, 400), interpolation = cv2.INTER_CUBIC)
    cv2.putText(img, "{}/{}".format(target_names[prediction],target_names[label.argmax()]),
        (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Image", img)
    cv2.waitKey(0)
