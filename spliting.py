import pickle

path = 'datasets/openie4.pkl'
with open(path, 'rb') as f:
    data = pickle.load(f)

sentences = data['tokens']
P_pos = data['single_pred_labels']
ent = data['single_arg_labels']
label = data['all_pred_labels']

# تقسیم داده‌ها به سه دسته بر اساس تعداد موجودیت‌ها
count_3, count_2, count_1 = 0, 0, 0
sentences_3, P_pos_3, ent_3, label_3 = [], [], [], []
sentences_2, P_pos_2, ent_2, label_2 = [], [], [], []
sentences_1, P_pos_1, ent_1, label_1 = [], [], [], []

for i in range(len(ent)):
    if 6 in ent[i] or 4 in ent[i]:
        count_3 += 1
        sentences_3.append(sentences[i])
        P_pos_3.append(P_pos[i])
        ent_3.append(ent[i])
        label_3.append(label[i])
    elif 2 in ent[i]:
        count_2 += 1
        sentences_2.append(sentences[i])
        P_pos_2.append(P_pos[i])
        ent_2.append(ent[i])
        label_2.append(label[i])
    elif 0 in ent[i]:
        count_1 += 1
        sentences_1.append(sentences[i])
        P_pos_1.append(P_pos[i])
        ent_1.append(ent[i])
        label_1.append(label[i])

print("relation with 1 ent: ", count_1)
print("relation with 2 ent: ", count_2)
print("relation with more than 3 ent: ", count_3)

def split_data(sentences, P_pos, ent, label):
    split_index = int(len(sentences) * 0.7)
    train_data = {
        'tokens': sentences[:split_index],
        'single_pred_labels': P_pos[:split_index],
        'single_arg_labels': ent[:split_index],
        'all_pred_labels': label[:split_index]
    }
    test_data = {
        'tokens': sentences[split_index:],
        'single_pred_labels': P_pos[split_index:],
        'single_arg_labels': ent[split_index:],
        'all_pred_labels': label[split_index:]
    }
    return train_data, test_data

train_data_1, test_data_1 = split_data(sentences_1, P_pos_1, ent_1, label_1)
train_data_2, test_data_2 = split_data(sentences_2, P_pos_2, ent_2, label_2)
train_data_3, test_data_3 = split_data(sentences_3, P_pos_3, ent_3, label_3)

# ترکیب داده‌های train و test از هر دسته
train_data = {
    'tokens': train_data_1['tokens'] + train_data_2['tokens'] + train_data_3['tokens'],
    'single_pred_labels': train_data_1['single_pred_labels'] + train_data_2['single_pred_labels'] + train_data_3['single_pred_labels'],
    'single_arg_labels': train_data_1['single_arg_labels'] + train_data_2['single_arg_labels'] + train_data_3['single_arg_labels'],
    'all_pred_labels': train_data_1['all_pred_labels'] + train_data_2['all_pred_labels'] + train_data_3['all_pred_labels']
}

test_data = {
    'tokens': test_data_1['tokens'] + test_data_2['tokens'] + test_data_3['tokens'],
    'single_pred_labels': test_data_1['single_pred_labels'] + test_data_2['single_pred_labels'] + test_data_3['single_pred_labels'],
    'single_arg_labels': test_data_1['single_arg_labels'] + test_data_2['single_arg_labels'] + test_data_3['single_arg_labels'],
    'all_pred_labels': test_data_1['all_pred_labels'] + test_data_2['all_pred_labels'] + test_data_3['all_pred_labels']
}

# ذخیره داده‌های train به فایل pkl
with open('datasets/train_data.pkl', 'wb') as f:
    pickle.dump(train_data, f)

# ذخیره داده‌های test به فایل pkl
with open('datasets/test_data.pkl', 'wb') as f:
    pickle.dump(test_data, f)

print("Train and test datasets saved successfully.")