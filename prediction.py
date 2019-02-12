
'''loding the model that was saved as mentioned on the other file.'''





from keras.models import load_model

model = load_model('drive/My Drive/models/exception_customised_64layers.h5', custom_objects = {'iou': iou})

import numpy as np
import pandas as pd

test_img = pd.read_csv('drive/My Drive/test.csv')
label = test_img["image_name"]

test_paths = ["drive/My Drive/test images/" + i for i in label]

predictions = []

for i in range(233):
    print("epoch", i)
    t = test_paths[i * (233):(i + 1) * 233]
    tem = []
    for j in t:
        img = np.array(cv2.imread(j).astype("float32")
        img = cv2.resize(img, (320, 240))
        img = img / 255.0
        tem.append(img)

        prediction = model.predict(tem)
        predictions.append(prediction)

        pred = np.array(predictions, dtype="float32")
        print(pred.shape)

        pred = pred.reshape(-1, 4)

        y_pred = 2 * y_pred

        arr = np.c_[np.array(label), y_pred]

        result = pd.DataFrame(arr, columns=["image_name", "x1", "x2", "y1", "y2"])

        result.to_csv("result_xception.csv", index=False)

        from google.colab import files

        files.download('result_xception.csv')
