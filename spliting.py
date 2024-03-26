import pickle
from sklearn.model_selection import train_test_split
path= 'C:\\Users\\mitra\\PycharmProjects\\ONIEX\\datasets\\openie4.pkl'
with open(path, 'rb') as f:
  data_dict = pickle.load(f)
train_data_dict = {}
test_data_dict = {}
for key, value in data_dict.items():
    train_data_dict[key], test_data_dict[key] = train_test_split(value, test_size=0.3, random_state=42)
with open('C:\\Users\\mitra\\PycharmProjects\\ONIEX\\datasets\\train_data.pkl', 'wb') as f:
    pickle.dump(train_data_dict, f)
with open('C:\\Users\\mitra\\PycharmProjects\\ONIEX\\datasets\\test_data.pkl', 'wb') as f:
    pickle.dump(test_data_dict, f)
path= 'C:\\Users\\mitra\\PycharmProjects\\ONIEX\\datasets\\test_data.pkl'
with open(path, 'rb') as f:
  data = pickle.load(f)
print(len(data))
print(type(data))