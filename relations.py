import pickle
path= 'C:\\Users\\mitra\\PycharmProjects\\ONIEX\\datasets\\openie4.pkl'
with open(path, 'rb') as f:
  data = pickle.load(f)
a = data['tokens']
sentence_counts = {}
for sentence in a:
    sentence_key = " ".join(sentence)
    if sentence_key in sentence_counts:
        sentence_counts[sentence_key] += 1
    else:
        sentence_counts[sentence_key] = 1
repeated_twice = 0
repeated_more_than_three = 0
no_repeat = 0
for count in sentence_counts.values():
    if count == 2:
        repeated_twice += 1
    elif count >= 3:
        repeated_more_than_three += 1
    elif count == 1:
        no_repeat += 1
print("Number of sentences repeated twice:", repeated_twice)
print("Number of sentences repeated more than three times:", repeated_more_than_three)
print("Number of sentences with no repeats:", no_repeat)