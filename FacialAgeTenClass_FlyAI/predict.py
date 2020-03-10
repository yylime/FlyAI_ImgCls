from flyai.dataset import Dataset

from model import Model
from path import MODEL_PATH

data = Dataset()
model = Model(data)
p = model.predict(MODEL_PATH, image_path="img/bright_KA.AN3.41.tiff")
print(p)