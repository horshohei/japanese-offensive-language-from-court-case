import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, SequentialSampler, TensorDataset
import os
from collections import defaultdict
from torch.utils.data import DataLoader
import gc
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt

#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def add_special_token(tokenizer):
    tokenizer.add_special_tokens({'additional_special_tokens': ['[IDinfo]']})
    return tokenizer

def train(model, dataloader, optimizer, device, scheduler=None):
    model = model.train()
    losses = []
    for d in dataloader:
        input_ids = d[0].to(device)
        attention_mask = d[1].to(device)
        labels = d[2].to(device)
        #optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
    return np.mean(losses)

def eval(model, dataloader, device, numlabels=1):
    model = model.eval()
    losses = []
    preds = []
    labels = []
    with torch.no_grad():
        for d in dataloader:
            input_ids = d[0].to(device)
            attention_mask = d[1].to(device)
            label = d[2].to(device)
            output = model(input_ids, attention_mask=attention_mask, labels=label)
            loss = output.loss
            logits = output.logits
            losses.append(loss.item())
            if numlabels > 2:
                preds.extend(torch.sigmoid(logits).round().detach().cpu().numpy())
                labels.extend(label.detach().cpu().numpy())
            else:
                preds.extend(torch.argmax(logits, dim=1).detach().cpu().numpy())
                labels.extend(label.detach().cpu().numpy())
    return np.mean(losses), preds, labels


def create_dataset(data, tokenizer, max_len, numlabel=2, context_mode=False):
    if numlabel > 2:
        labels = torch.tensor([d for d in data['labels']], dtype=torch.float)
    else:
        labels = torch.tensor(data['labels'])
    if context_mode:
        input_data = ['本文:'+ t + tokenizer.sep_token+'文脈:'+ c for t, c in zip(data['text'], data['context'])]
        inputs = tokenizer(input_data, padding=True, truncation=True, max_length=max_len, return_tensors='pt')
    else:
        input_data = data['text']
        inputs = tokenizer(input_data, padding=True, truncation=True, max_length=max_len, return_tensors='pt')
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    dataset = TensorDataset(input_ids, attention_mask, labels)
    return dataset

def run_experiment(train_data, test_data, experiment_settings, experiment_save_path, counter, numlabels=2):
    model_name = experiment_settings['model_name']
    history = defaultdict(list)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    rights = ['名誉権', '名誉感情', '私生活の平穏', '人格権・人格的利益', '営業権', 'プライバシー']
    if numlabels > 2:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=numlabels, problem_type='multi_label_classification')
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=numlabels, problem_type='single_label_classification')
    tokenizer = add_special_token(tokenizer)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    train_dataset = create_dataset(train_data, tokenizer, experiment['parameters']['MAX_LEN'], numlabel=num_labels,
                             context_mode=experiment['context'])
    train_dataloader = DataLoader(train_dataset, batch_size=experiment['parameters']['BATCH_SIZE'],
                            sampler=SequentialSampler(train_dataset))
    eval_dataset = create_dataset(test_data, tokenizer, experiment['parameters']['MAX_LEN'], numlabel=num_labels, context_mode=experiment['context'])
    eval_dataloader = DataLoader(eval_dataset, batch_size=experiment['parameters']['BATCH_SIZE'],
                           sampler=SequentialSampler(eval_dataset))

    early_stopping = 0
    previous = "None"
    optimizer = torch.optim.AdamW(model.parameters(), lr=experiment_settings['parameters']['LR'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, experiment_settings['parameters']['EPOCHS'])
    for epoch in tqdm(range(experiment_settings['parameters']['EPOCHS'])):
        train_loss = train(model, train_dataloader, optimizer, device, scheduler)
        print(f'Epoch {epoch} Train Loss: {train_loss}')
        eval_loss, preds, labels = eval(model, eval_dataloader, device, numlabels=numlabels)
        if numlabels > 2:
            preds = np.array(preds)
            targets = np.array(labels)
            print(f'Epoch {epoch} Eval Loss: {eval_loss}')
            print(f'Macro F1 Score: {f1_score(targets, preds, average="macro")}')
            print(f'Micro F1 Score: {f1_score(targets, preds, average="micro")}')
            print(f'Accuracy: {accuracy_score(targets, preds)}')
            print(f'Precision: {precision_score(targets, preds, average="macro")}')
            print(f'Recall: {recall_score(targets, preds, average="macro")}')
            print('')
            history['train_loss'].append(train_loss)
            history['eval_loss'].append(eval_loss)
            history['f1_score_Macro'].append(f1_score(targets, preds, average="macro"))
            history['f1_score_Micro'].append(f1_score(targets, preds, average="micro"))
            history['accuracy'].append(accuracy_score(targets, preds))
            history['precision'].append(precision_score(targets, preds, average="macro"))
            history['recall'].append(recall_score(targets, preds, average="macro"))
            for r in rights:
                print(f'{r} F1 Score: {f1_score(targets[:, rights.index(r)], preds[:, rights.index(r)])}')
                print(f'{r} Accuracy: {accuracy_score(targets[:, rights.index(r)], preds[:, rights.index(r)])}')
                print('')
                history[f'f1_score_{r}'].append(f1_score(targets[:, rights.index(r)], preds[:, rights.index(r)]))
                history[f'accuracy_{r}'].append(accuracy_score(targets[:, rights.index(r)], preds[:, rights.index(r)]))


        else:
            print(f'Epoch {epoch} Eval Loss: {eval_loss}')
            print(f'F1 Score: {f1_score(labels, preds)}')
            print(f'Accuracy: {accuracy_score(labels, preds)}')
            print(f'Precision: {precision_score(labels, preds)}')
            print(f'Recall: {recall_score(labels, preds)}')
            print('')
            history['train_loss'].append(train_loss)
            history['eval_loss'].append(eval_loss)
            history['f1_score'].append(f1_score(labels, preds))
            history['accuracy'].append(accuracy_score(labels, preds))
            history['precision'].append(precision_score(labels, preds))
            history['recall'].append(recall_score(labels, preds))
        if previous == "None" or previous > eval_loss:
            previous = eval_loss
            early_stopping = 0
            torch.save(model.state_dict(), experiment_save_path + 'best_model_state' + str(counter) + '.bin')

        elif eval_loss > previous:
            early_stopping += 1

            if early_stopping > 8:
                break
    plt.style.use('seaborn')
    plt.plot(history['accuracy'], label='val accuracy')
    plt.title('Training history')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.ylim([0, 1])
    plt.savefig(f'{experiment_save_path}accuracy_{counter}.png')
    plt.clf()
    plt.close()

    plt.plot(history['train_loss'], label='train loss')
    plt.plot(history['eval_loss'], label='val loss')
    plt.title('Training history')
    plt.ylabel('loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(f'{experiment_save_path}loss_{counter}.png')
    plt.clf()
    plt.close()
    if numlabels > 2:
        plt.plot(history['f1_score_Macro'], label='f1_score_macro')
        plt.plot(history['f1_score_Micro'], label='f1_score_micro')
        plt.title('Training history')
        plt.ylabel('f1_score')
        plt.xlabel('Epoch')
        plt.legend()
        plt.ylim([0, 1])
        plt.savefig(f'{experiment_save_path}f1_score_{counter}.png')
        plt.clf()
        plt.close()
    else:
        plt.plot(history['f1_score'], label='f1_score')
        plt.title('Training history')
        plt.ylabel('f1_score')
        plt.xlabel('Epoch')
        plt.legend()
        plt.ylim([0, 1])
        plt.savefig(f'{experiment_save_path}f1_score_{counter}.png')
        plt.clf()
        plt.close()
    torch.save(model.state_dict(), experiment_save_path + 'model_laststate' + str(counter) + '.bin')
    gc.collect()
    torch.cuda.empty_cache()

    return history



#%%
'''
savepath = './nlppaper/'
parameters = {'BATCH_SIZE': 8, 'MAX_LEN': 510, 'LR': 5e-6, 'EPOCHS': 5}
experiment = {'model_name': 'rinna/japanese-roberta-base', 'context':True, 'parameters': parameters}
savepath = f'{savepath}/{experiment["model_name"].replace("/", "-")}_{str(experiment["context"])}/'
os.makedirs(savepath, exist_ok=True)
i = 0
t = 'result'
with open(f'./experimentdata/train_{str(i)}.json') as f:
    train_data = json.load(f)
with open(f'./experimentdata/test_{str(i)}.json') as f:
    test_data = json.load(f)


def shorten(data):
    for d in data:
        data[d] = data[d][:10]
    return data


#train_data = shorten(train_data)
#test_data = shorten(test_data)
if t == 'result':
    train_labels = []
    test_labels = []
    for d in train_data['result']:
        if d:
            train_labels.append(1)
        else:
            train_labels.append(0)
    for d in test_data['result']:
        if d:
            test_labels.append(1)
        else:
            test_labels.append(0)
    num_labels = 2
    train_data['labels'] = train_labels
    test_data['labels'] = test_labels
else:
    num_labels = 6

history = defaultdict(list)
tokenizer = AutoTokenizer.from_pretrained(experiment['model_name'], trust_remote_code=True)
rights = ['名誉権', '名誉感情', '私生活の平穏', '人格権・人格的利益', '営業権', 'プライバシー']
if num_labels > 2:
    model = AutoModelForSequenceClassification.from_pretrained(experiment['model_name'], num_labels=num_labels, problem_type='multi_label_classification')
else:
    model = AutoModelForSequenceClassification.from_pretrained(experiment['model_name'], num_labels=num_labels, problem_type='single_label_classification')
tokenizer = add_special_token(tokenizer)
model.resize_token_embeddings(len(tokenizer))
model.to(device)
train_dataset = create_dataset(train_data, tokenizer, experiment['parameters']['MAX_LEN'], numlabel=num_labels,
                         context_mode=experiment['context'])
train_dataloader = DataLoader(train_dataset, batch_size=experiment['parameters']['BATCH_SIZE'],
                        sampler=SequentialSampler(train_dataset))
eval_dataset = create_dataset(test_data, tokenizer, experiment['parameters']['MAX_LEN'], numlabel=num_labels, context_mode=experiment['context'])
eval_dataloader = DataLoader(eval_dataset, batch_size=experiment['parameters']['BATCH_SIZE'],
                       sampler=SequentialSampler(eval_dataset))

early_stopping = 0
previous = "None"
optimizer = torch.optim.AdamW(model.parameters(), lr=experiment['parameters']['LR'])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, experiment['parameters']['EPOCHS'])
model = AutoModelForSequenceClassification.from_pretrained(experiment['model_name'], num_labels=2, problem_type='single_label_classification')
tokenizer = AutoTokenizer.from_pretrained(experiment['model_name'], trust_remote_code=True)
model.to(device)
tokenizer = add_special_token(tokenizer)
model.resize_token_embeddings(len(tokenizer))
train_dataset = create_dataset(train_data, tokenizer, experiment['parameters']['MAX_LEN'], numlabel=num_labels,
                         context_mode=experiment['context'])
train_dataloader = DataLoader(train_dataset, batch_size=experiment['parameters']['BATCH_SIZE'],
                        sampler=SequentialSampler(train_dataset))
eval_dataset = create_dataset(test_data, tokenizer, experiment['parameters']['MAX_LEN'], numlabel=num_labels, context_mode=experiment['context'])
eval_dataloader = DataLoader(eval_dataset, batch_size=experiment['parameters']['BATCH_SIZE'],
                       sampler=SequentialSampler(eval_dataset))
model.train()
#%%
d = iter(train_dataloader)
#%%
#history = run_experiment(train_data, test_data, experiment, savepath, i, numlabels=num_labels)
model = model.train()
losses = []
#d = iter(train_dataloader)
#for d in tqdm(train_dataloader):
input_ids = next(d)[0].to(device)
attention_mask = next(d)[1].to(device)
labels = next(d)[2].to(device)
# optimizer.zero_grad()
outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
loss = outputs.loss
loss.backward()
optimizer.step()
scheduler.step()
losses.append(loss.item())
print(np.mean(losses))

#%%
print(input_ids)

#%%
'''
if __name__ == '__main__':
    filepath = './nlppaper/'
    parameters = {'BATCH_SIZE': 64, 'MAX_LEN': 510, 'LR': 5e-6, 'EPOCHS': 50}
    print(parameters)
    task = ['label']
    experiment_settings = [
        {'model_name': 'cl-tohoku/bert-base-japanese-v3', 'context':True, 'parameters': parameters},
        {'model_name': 'cl-tohoku/bert-base-japanese-v3', 'context':False, 'parameters': parameters},
        {'model_name': 'rinna/japanese-roberta-base', 'context':True, 'parameters': parameters},
        {'model_name': 'rinna/japanese-roberta-base', 'context':False, 'parameters': parameters},
        #{'model_name': 'bert-base-multilingual-cased', 'context':True, 'parameters': parameters},
        #{'model_name': 'bert-base-multilingual-cased', 'context':False, 'parameters': parameters},
        {'model_name': 'line-corporation/line-distilbert-base-japanese', 'context':True, 'parameters': parameters},
        {'model_name': 'line-corporation/line-distilbert-base-japanese', 'context':False, 'parameters': parameters},
        {'model_name': 'intfloat/multilingual-e5-base', 'context':True, 'parameters': parameters},
        {'model_name': 'intfloat/multilingual-e5-base', 'context':False, 'parameters': parameters},
    ]
    for t in task:
        for experiment in experiment_settings:
            savepath = f'{filepath}/{t}/{experiment["model_name"].replace("/","-")}_{str(experiment["context"])}/'
            os.makedirs(savepath, exist_ok=True)
            for i in range(5):
                with open(f'./experimentdata/train_{str(i)}.json') as f:
                    train_data = json.load(f)
                with open(f'./experimentdata/test_{str(i)}.json') as f:
                    test_data = json.load(f)
                def shorten(data):
                    for d in data:
                        data[d] = data[d][:10]
                    return data
                #train_data = shorten(train_data)
                #test_data = shorten(test_data)

                if t == 'result':
                    train_labels = []
                    test_labels = []
                    for d in train_data['result']:
                        if d:
                            train_labels.append(1)
                        else:
                            train_labels.append(0)
                    for d in test_data['result']:
                        if d:
                            test_labels.append(1)
                        else:
                            test_labels.append(0)
                    num_labels = 2
                    train_data['labels'] = train_labels
                    test_data['labels'] = test_labels
                else:
                    num_labels = 6

                numlabel = 2 if t == 'result' else 6

                history = run_experiment(train_data, test_data, experiment, savepath, i, numlabels=numlabel)
                with open(savepath + 'history' + str(i) + '.json', 'w') as f:
                    json.dump(history, f)
                del history
                gc.collect()
                torch.cuda.empty_cache()

