import numpy as np
import pickle
import math
import torch
from tqdm import trange
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from torch.nn import functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from models import SpanBERTForRE, TransformerEncoder, TransformerEncoderLayer, ONIEXNetwork
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
print(device)
path = 'C:\\Users\\mitra\\PycharmProjects\\ONIEX\\datasets\\openie4.pkl\\openie4.pkl'
with open(path, 'rb') as f:
  data = pickle.load(f)
  sentences = data['tokens']
  P_pos = data['single_pred_labels']
  ent = data['single_arg_labels']
  label = data['all_pred_labels']
print("1")
print(sentences[1])
ent = [[0 if x == 1 else x for x in en] for en in ent]
ent = [[1 if x == 2 else x for x in en] for en in ent]
ent = [[1 if x == 3 else x for x in en] for en in ent]
ent = [[2 if x == 4 else x for x in en] for en in ent]
ent = [[2 if x == 5 else x for x in en] for en in ent]
ent = [[3 if x == 6 else x for x in en] for en in ent]
ent = [[3 if x == 7 else x for x in en] for en in ent]
ent = [[4 if x == 8 else x for x in en] for en in ent]
P_pos = [[1 if x == 0 else x for x in P] for P in P_pos]
P_pos = [[0 if x == 2 else x for x in P] for P in P_pos]
max_len = 512
tag2idx = {
 'R-B' : 0,
 'R-I' : 1,
 'O' : 2,
 '[CLS]' : 3,
 '[SEP]' : 4 }
tag2name = { tag2idx[key] : key for key in tag2idx.keys() }
ent2idx = {
  'E0' : 0,
  'E1' : 1,
  'E2' : 2,
  'E3' : 3,
  'O' : 4,
  '[CLS]' : 5,
  '[SEP]' : 6
 }
ent2name = { ent2idx[key] : key for key in ent2idx.keys() }
model_checkpoint = "bert-base-cased"
tokenizer = BertTokenizer.from_pretrained(model_checkpoint)
tokenized_texts = []
word_piece_labels = []
ent_tags = []
p_poss = []
i_inc = 0
for word_list,lab,enttag,pos in (zip(sentences,label,ent,P_pos)):
    temp_lable = []
    temp_token = []
    temp_enttag = []
    temp_pos = []
    temp_lable.append(3)
    temp_token.append('[CLS]')
    temp_enttag.append(5)
    temp_pos.append(0)
    for word, la, en, p in zip(word_list, lab, enttag, pos):
        temp_token.append(word)
        temp_lable.append(la)
        temp_enttag.append(en)
        temp_pos.append(p)
    temp_lable.append(4)
    temp_token.append('[SEP]')
    temp_enttag.append(6)
    temp_pos.append(0)
    tokenized_texts.append(temp_token)
    word_piece_labels.append(temp_lable)
    ent_tags.append(temp_enttag)
    p_poss.append(temp_pos)
    if 10 > i_inc:
        print("No.%d,len:%d"%(i_inc,len(temp_token)))
        print("texts", temp_token)
        print("No.%d,len:%d"%(i_inc,len(temp_lable)))
        print("lables ",temp_lable)
        print("No.%d,len:%d"%(i_inc,len(temp_enttag)))
        print("ents ", temp_enttag)
        print("No.%d,len:%d"%(i_inc,len(temp_pos)))
        print("p_pos: ", temp_pos)
    i_inc += 1
input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                          maxlen = max_len, dtype = "long", truncating = "post", padding = "post")
tags = pad_sequences([[l for l in lab] for lab in word_piece_labels],
                     maxlen = max_len, value = tag2idx["O"], padding = "post",
                     dtype = "long", truncating = "post")
entss = pad_sequences([[l for l in lab] for lab in ent_tags],
                     maxlen = max_len, value = ent2idx["O"], padding = "post",
                     dtype = "long", truncating = "post")
p_pos= pad_sequences([[p for p in pos] for pos in p_poss],
                     maxlen = max_len, value = 0, padding = "post",
                     dtype = "long", truncating = "post")
attention_masks = [[int(i > 0) for i in ii] for ii in input_ids]
tr_inputs, ts_inputs, tr_tags, ts_tags, tr_masks, ts_masks, tr_ent, ts_ent, tr_ppos, ts_ppos = train_test_split(input_ids,
                                                                                                                            tags,
                                                                                                                            attention_masks,
                                                                                                                            entss,
                                                                                                                            p_pos,
                                                                                                                            random_state=4,
                                                                                                                            test_size=0.3)


print(tr_inputs.shape)
print(ts_inputs.shape)
print(type(tr_inputs))
tr_inputs = torch.tensor(tr_inputs)
ts_inputs = torch.tensor(ts_inputs)
tr_tags = torch.tensor(tr_tags)
ts_tags = torch.tensor(ts_tags)
tr_masks = torch.tensor(tr_masks)
ts_masks = torch.tensor(ts_masks)
tr_ent = torch.tensor(tr_ent)
ts_ent = torch.tensor(ts_ent)
tr_ppos = torch.tensor(tr_ppos)
ts_ppos = torch.tensor(ts_ppos)
print("2")
batch_num = 32
num_labels = len(tag2idx)
train_data = TensorDataset(tr_inputs, tr_masks, tr_ent, tr_tags, tr_ppos)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_num,drop_last=True)
ts_data = TensorDataset(ts_inputs, ts_masks, ts_ent, ts_tags, ts_ppos)
ts_sampler = SequentialSampler(ts_data)
ts_dataloader = DataLoader(ts_data, sampler=ts_sampler, batch_size=batch_num)
modelR = SpanBERTForRE(num_labels)
encoder_layer = TransformerEncoderLayer(d_model = 768, nhead = 12)
modelE = TransformerEncoder(encoder_layer, num_layers = 12,num_entities = len(ent2idx), device = device, batch_num = batch_num, max_len = max_len)
model= ONIEXNetwork(modelR, modelE, device, tag2idx)
model.to(device)
epochs = 5
num_train_optimization_steps = int( math.ceil(len(tr_inputs) / batch_num) / 1) * epochs
max_grad_norm = 0.1
optimizer = torch.optim.Adam(model.parameters(), lr = 3e-5)
def evaluateR(logits, label_ids, input_mask):
  logits = torch.argmax(F.log_softmax(logits,dim=2),dim=2)
  logits = logits.detach().cpu().numpy()
  label_ids = label_ids.to('cpu').numpy()
  input_mask = input_mask.to('cpu').numpy()
  temp_1 = []
  temp_2 = []
  for i,mask in enumerate(input_mask):
      for j, m in enumerate(mask):
          if m:
              temp_1.append(tag2name[label_ids[i][j]])
              temp_2.append(tag2name[logits[i][j]])
          else:
            break
  return temp_1, temp_2

def evaluateE(logits, ent_ids, input_mask):
  logits = torch.argmax(F.log_softmax(logits,dim=2),dim=2)
  logits = logits.detach().cpu().numpy()
  ent_ids = ent_ids.to('cpu').numpy()
  input_mask = input_mask.to('cpu').numpy()
  temp_1 = []
  temp_2 = []
  for i,mask in enumerate(input_mask):
      for j, m in enumerate(mask):
          if m == 0:
              temp_1.append(ent2name[ent_ids[i][j]])
              temp_2.append(ent2name[logits[i][j]])
  return temp_1, temp_2
def train():
    tr_loss = 0
    nb_tr_examples1, nb_tr_steps1 = 0, 0
    y_true1 = []
    y_pred1 = []
    y_true2 = []
    y_pred2 = []
    y_true3 = []
    y_pred3 = []
    model.train()
    counter = 0
    for step, batch in enumerate(train_dataloader,0):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_ent, b_labels, b_ppos= batch
        outputs = model(input_ids = b_input_ids.to(device),entities = b_ent.to(device), attention_mask = b_input_mask.to(device), labels = b_labels.to(device),b_ppos = b_ppos.to(device) )
        loss1, loss2, scores1, scores2 = outputs[:4]
        loss = loss1 + loss2
        temp_1,temp_2 = evaluateR(scores1, b_labels, b_input_mask)
        y_true1.extend(temp_1)
        y_pred1.extend(temp_2)
        y_true3.extend(temp_1)
        y_pred3.extend(temp_2)
        mask_a = torch.logical_or(torch.eq(b_ppos, tag2idx['R-B']), torch.eq(b_ppos, tag2idx['R-I']))
        mask_c = torch.eq(b_ppos, 1)
        mask_combined = torch.logical_and(mask_a, mask_c)
        mask = (torch.logical_or(torch.logical_not(b_input_mask), mask_combined)).int()
        temp_1,temp_2= evaluateE(scores2, b_ent, mask)
        y_true2.extend(temp_1)
        y_pred2.extend(temp_2)
        y_true3.extend(temp_1)
        y_pred3.extend(temp_2)
        if n_gpu > 1:
            loss = loss.mean()
        optimizer.zero_grad()
        loss.backward()
        tr_loss += loss.item()
        nb_tr_examples1 += b_input_ids.size(0)
        nb_tr_steps1 += 1
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
        optimizer.step()
        if(step >= (50 * counter)):
          print(step)
          print("Relation: ",step," ", classification_report(y_true1, y_pred1,digits=4, zero_division=1))
          print("Entity: ",step," ", classification_report(y_true2, y_pred2,digits=4, zero_division=1))
          print("ONIEX: ",step," ", classification_report(y_true3, y_pred3,digits=4, zero_division=1))
          counter = counter + 1
    return tr_loss / nb_tr_steps1, f1_score(y_true2, y_pred2, average='micro'), f1_score(y_true1, y_pred1, average='micro')
def evaluateE(logits, ent_ids, input_mask):
  logits = torch.argmax(F.log_softmax(logits,dim = 2),dim = 2)
  logits = logits.detach().cpu().numpy()
  ent_ids = ent_ids.to('cpu').numpy()
  input_mask = input_mask.to('cpu').numpy()
  temp_1 = []
  temp_2 = []
  for i,mask in enumerate(input_mask):
      for j, m in enumerate(mask):
          if m == 0:
              temp_1.append(ent2name[ent_ids[i][j]])
              temp_2.append(ent2name[logits[i][j]])
  return temp_1, temp_2
def test():
    y_true1 = []
    y_pred1 = []
    y_true2 = []
    y_pred2 = []
    y_true3 = []
    y_pred3 = []
    model.eval()
    counter = 0
    with torch.no_grad():
        for step, batch in enumerate(ts_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask,b_ent, labels, b_ppos = batch
            outputs = model(input_ids.to(device), attention_mask=input_mask.to(device))
            loss1, loss2, scores1, scores2= outputs[:4]
            temp_1,temp_2 = evaluateR(scores1, labels, input_mask)
            y_true1.extend(temp_1)
            y_pred1.extend(temp_2)
            y_true3.extend(temp_1)
            y_pred3.extend(temp_2)
            mask_a = torch.logical_or(torch.eq(b_ppos, tag2idx['R-B']), torch.eq(b_ppos, tag2idx['R-I']))
            mask_c = torch.eq(b_ppos, 1)
            mask_combined = torch.logical_and(mask_a, mask_c)
            mask = (torch.logical_or(torch.logical_not(input_mask), mask_combined)).int()
            temp_1,temp_2 = evaluateE(scores2, b_ent, mask)
            y_true2.extend(temp_1)
            y_pred2.extend(temp_2)
            y_true3.extend(temp_1)
            y_pred3.extend(temp_2)
            if(step >= (50 * counter)):
              print(step)
              print("Relation: ", step, " ", classification_report(y_true1, y_pred1, digits=4, zero_division=1))
              print("Entity: ", step, " ", classification_report(y_true2, y_pred2, digits=4, zero_division=1))
              print("ONIEX: ", step, " ", classification_report(y_true3, y_pred3, digits=4, zero_division=1))
              counter = counter + 1

    return f1_score(y_true2, y_pred2, average='micro'), f1_score(y_true1, y_pred1, average='micro')
for _ in trange(epochs,desc="Epoch"):
    train2_loss, train2_f1 , train1_loss, train1_f1 = train()
    print("Train lossR: {}".format(train1_loss), " train f1R: {}".format(train1_f1))
    print("Train lossE: {}".format(train2_loss), " train f1E: {}".format(train2_f1))
    print("\n")
test2_f1, test1_f1 = test()
print("Test f1R: {}".format(test1_f1))
print("Test f1E: {}".format(test2_f1))
print("\n")









