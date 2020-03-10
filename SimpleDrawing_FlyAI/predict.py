from flyai.dataset import Dataset
from model import Model

class_mapping = {0: "airplane", 1:  "apple", 2: "basketball", 3: "bear", 4: "bed", 5: "bicycle",
                 6: "bridge", 7: "camera", 8: "car", 9: "cat", 10: "computer", 11: "cow",
                 12: "cup", 13: "dog", 14: "door", 15: "eye", 16: "fish", 17: "flower",
                 18: "frog", 19: "guitar", 20: "hat", 21: "horse", 22: "hospital", 23: "hourglass",
                 24: "ice cream", 25: "key", 26: "knife", 27: "lantern", 28: "lion", 29: "mailbox",
                 30: "matches", 31: "microphone", 32: "monkey", 33: "mosquito", 34: "mountain", 35: "mushroom",
                 36: "ocean", 37: "panda", 38: "piano", 39: "pig"}

data = Dataset()
model = Model(data)
x_test = [{'json_path': 'draws/draw_239207.json'}, {'json_path': 'draws/draw_239207.json'}]
p = model.predict_all(x_test)
print(p)