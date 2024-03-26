import pickle
path= 'C:\\Users\\mitra\\PycharmProjects\\ONIEX\\datasets\\openie4.pkl'
with open(path, 'rb') as f:
  data = pickle.load(f)
a = data['single_arg_labels']
count_6 = 0
count_4 = 0
count_2 = 0
count_0 = 0
for sublist in a:
    if 6 in sublist:
        count_6 += 1
    if 4 in sublist:
        count_4 += 1
    if 2 in sublist:
        count_2 += 1
    if 0 in sublist:
        count_0 += 1
print("relation with 1 ent: ",(count_0-count_2)," ", (count_0-count_2)/count_0)
print("relation with 2 ent: ", (count_2- count_4)," ", (count_2- count_4)/count_0)
print("more than 2 ent: ", count_6," ", count_6/count_0)