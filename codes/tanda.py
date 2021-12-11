import os
os.chdir('/scratch/sagarsj42')
os.environ['TORCH_HOME'] = '/scratch/sagarsj42/torch-cache'
os.environ['TRANSFORMERS_CACHE'] = '/scratch/sagarsj42/transformers'
os.environ['HF_DATASETS_CACHE'] = '/scratch/sagarsj42/hf-datasets'

import random
import string
import copy

import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

import transformers
from transformers import RobertaTokenizer, RobertaModel

transformers.logging.set_verbosity_error()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', DEVICE)


class AS2Dataset(Dataset):
    def __init__(self, as2, name, tokenizer, max_length):
        super(AS2Dataset, self).__init__()
        self.as2 = as2
        self.name = name
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        if name == 'wikiqa':
            self.sentence_key = 'answer'
        else:
            self.sentence_key = 'sentence'
        
    def __len__(self):
        return len(self.as2)
    
    def __getitem__(self, idx):
        sample = self.as2[idx]
        question = sample['question'].translate(
            str.maketrans('', '', string.punctuation)).lower().strip()
        sentence = sample[self.sentence_key].translate(
            str.maketrans('', '', string.punctuation)).lower().strip()
        label = torch.zeros(2)
        label[sample['label']] = 1.0
        
        input_enc = tokenizer(text=question, text_pair=sentence, 
                              add_special_tokens=True, truncation=True, padding='max_length', 
                              max_length=self.max_length, 
                              return_tensors='pt', return_attention_mask=True, return_token_type_ids=True)
        
        if self.name == 'wikiqa':
            return (sample['question_id'], question, sentence, 
                    input_enc['input_ids'].flatten(), input_enc['attention_mask'].flatten(), 
                    input_enc['token_type_ids'].flatten(), 
                    label)
        else:
            return (question, sentence, input_enc['input_ids'].flatten(), 
                    input_enc['attention_mask'].flatten(), input_enc['token_type_ids'].flatten(), 
                    label)


class TandaTransfer(nn.Module):
    def __init__(self, encoder):
        super(TandaTransfer, self).__init__()
        self.encoder = encoder
        self.layers = nn.Sequential(
            nn.Linear(768, 2),
            nn.Dropout(p=0.25)
        )
        
    def forward(self, x):
        a = x[1]
        x = self.encoder(input_ids=x[0], attention_mask=x[1], token_type_ids=x[2]).last_hidden_state
        x = (x * a.unsqueeze(-1) / a.sum(1).view(-1, 1, 1)).sum(1)
        x = self.layers(x)
        
        return x
    
    
class BertQA(nn.Module):
    def __init__(self, encoder):
        super(BertQA, self).__init__()
        self.encoder = encoder
        self.layers = nn.Sequential(
            nn.Linear(768, 512),
            nn.Dropout(p=0.25),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(512, 2),
            nn.Dropout(p=0.15),
        )
        
    def forward(self, x):
        a = x[1]
        x = self.encoder(input_ids=x[0], attention_mask=x[1], token_type_ids=x[2]).last_hidden_state
        x = (x * a.unsqueeze(-1) / a.sum(1).view(-1, 1, 1)).sum(1)
        x = self.layers(x)
        
        return x
    
    
def get_valid_questions(wikiqa):
    question_status = dict()

    for split in wikiqa:
        split_dataset = wikiqa[split]
        n_samples = len(split_dataset)

        for i in range(n_samples):
            qid = split_dataset[i]['question_id']
            label = split_dataset[i]['label']
            if qid not in question_status:
                question_status[qid] = label
            else:
                question_status[qid] = max(question_status[qid], label)

    valid_questions = set([qid for qid in question_status if question_status[qid] > 0])
    
    return valid_questions


def train_epochs_asnq(n_epochs, dataloader, model, optimizer, criterion, 
                      dev_dataloader=None, eval_steps=10000, save_path='./best.pth', device='cpu'):
    model.train()
    n_batches = len(dataloader)
    best_loss = float('inf')
    
    for epoch in range(n_epochs):
        total_loss = 0.0
        for step, sample in enumerate(dataloader):
            sample = sample[2:6]
            sample = [s.to(device) for s in sample]
            optimizer.zero_grad()
            output = model(sample[:-1])
            loss = criterion(output, sample[-1])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if step > 0 and step % eval_steps == 0:
                dev_loss = validate_asnq(dev_dataloader, model, criterion, device=device)
                if dev_loss < best_loss:
                    best_loss = dev_loss
                    save_dict = {
                        'epoch': epoch+1,
                        'step': step,
                        'dev_loss': dev_loss,
                        'model_params': model.state_dict()
                    }
                    torch.save(save_dict, save_path)
                    print(f'Best checkpoint saved at epoch {epoch+1}, step {step}, dev loss: {dev_loss:.4f}')
                step_loss = 0.0
        
        save_dict = {
                    'epoch': epoch,
                    'step': step,
                    'dev_loss': dev_loss,
                    'model_params': model.state_dict()
                    }
        torch.save(save_dict, 'final-transfer-roberta.pth')
        total_loss /= n_batches
        print(f'Epoch {epoch+1} complete. Train loss: {total_loss:.4f}')
    
    return total_loss


def validate_asnq(dataloader, model, criterion, device='cpu'):
    model.eval()
    n_batches = len(dataloader)
    total_loss = 0.0
    
    for sample in dataloader:
        sample = sample[2:6]
        sample = [s.to(device) for s in sample]
        output = model(sample[:-1])
        loss = criterion(output, sample[-1])
        total_loss += loss.item()
        
    return total_loss / n_batches


def train_epoch(dataloader, model, optimizer, criterion, device='cpu'):
    model.train()
    n_batches = len(dataloader)
    total_loss = 0.0
    
    for sample in dataloader:
        sample = sample[3:7]
        sample = [s.to(device) for s in sample]
        optimizer.zero_grad()
        output = model(sample[:-1])
        loss = criterion(output, sample[-1])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / n_batches


def validate(dataloader, model, criterion, device='cpu'):
    model.eval()
    n_batches = len(dataloader)
    total_loss = 0.0
    
    for sample in dataloader:
        sample = sample[3:7]
        sample = [s.to(device) for s in sample]
        output = model(sample[:-1])
        loss = criterion(output, sample[-1])
        total_loss += loss.item()
        
    return total_loss / n_batches


def get_scores(dataloader, model, device='cpu'):
    model.eval()
    eval_results = list()
    
    for batch in dataloader:
        batch = copy.deepcopy(batch)
        batch_d = [b.to(device) for b in batch[3:6]]
        output = model(batch_d).detach()[:, 1]
        scores = nn.Sigmoid()(output)
        
        batch[-1] = batch[-1][:, 1].tolist()
        batch.append(scores.tolist())
        size = len(batch[0])
        eval_results.extend([[b[i] for b in batch] for i in range(size)])
        
    return eval_results


def get_question_label_scores(results):
    q_dict = dict()
    
    for result in results:
        qid = result[0]
        if qid in q_dict:
            q_dict[qid][1].append(result[2])
            q_dict[qid][2].append(result[6])
            q_dict[qid][3].append(result[7])
        else:
            q_dict[qid] = [result[1], [result[2]], [result[6]], [result[7]]]
            
    return q_dict


def calculate_metrics(question_scores):
    total = len(question_scores) * 1.0
    accuracy = 0
    mrr = 0.0
    all_labels = list()
    all_scores = list()
    
    for qid, values in question_scores.items():
        values = copy.deepcopy(values)
        labels = values[2]
        scores = values[3]
        all_labels.extend(labels)
        all_scores.extend(scores)
        actual = np.array(labels).argmax()
        predicted = np.array(scores).argmax()
        expected_max = scores[actual]
        scores.sort(reverse=True)
        given_rank = scores.index(expected_max) + 1
        
        if actual == predicted:
            accuracy += 1
        mrr += (1.0/given_rank)
        
    accuracy /= total
    mean_ap = average_precision_score(all_labels, all_scores)
    mrr /= total
    
    return accuracy, mean_ap, mrr


asnq_f = datasets.load_from_disk('asnq-2-3')
print('Loaded filtered ASNQ:', asnq_f)

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
encoder = RobertaModel.from_pretrained('roberta-base')

batch_size = 16 * 3
transfer_epochs = 2
transfer_learning_rate = 1e-6

pos_weight=torch.tensor([1.0, 16.0], dtype=torch.float).to(DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

asnq_train_ds = AS2Dataset(asnq_f['train'], 'asnq', tokenizer, max_length=128)
asnq_dev_ds = AS2Dataset(asnq_f['validation'], 'asnq', tokenizer, max_length=128)

asnq_train_dl = DataLoader(asnq_train_ds, batch_size=batch_size, shuffle=True)
asnq_dev_dl = DataLoader(asnq_dev_ds, batch_size=batch_size, shuffle=False)

model = TandaTransfer(encoder)
model.to(DEVICE)
model = nn.DataParallel(model)

optimizer = Adam(model.parameters(), lr=transfer_learning_rate)

train_loss = train_epochs_asnq(transfer_epochs, asnq_train_dl, model, optimizer, criterion, 
                               dev_dataloader=asnq_dev_dl, eval_steps=10000, save_path='best-transfer-roberta.pth', 
                               device=DEVICE)
print('Transfer complete. Train loss:', train_loss)

model = TandaTransfer(encoder)
save_dict = torch.load('best-transfer-roberta.pth')
model.load_state_dict(save_dict['model_params'], strict=False)

wikiqa = datasets.load_dataset('wiki_qa')
valid_questions = get_valid_questions(wikiqa)
wikiqa_f = wikiqa.filter(lambda sample: sample['question_id'] in valid_questions)
print('Filtered wikiqa:', wikiqa_f)

adapt_learning_rate = 5e-6
adapt_epochs = 10

pos_weight=torch.tensor([1.0, 8.0], dtype=torch.float).to(DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

model = BertQA(model.encoder)
model = nn.DataParallel(model)
model.to(DEVICE)

optimizer = Adam(model.parameters(), lr=adapt_learning_rate, betas=(0.9, 0.999), eps=1e-7)

wikiqa_train_ds = AS2Dataset(wikiqa_f['train'], 'wikiqa', tokenizer, max_length=64)
wikiqa_dev_ds = AS2Dataset(wikiqa_f['validation'], 'wikiqa', tokenizer, max_length=64)
wikiqa_test_ds = AS2Dataset(wikiqa_f['test'], 'wikiqa', tokenizer, max_length=64)

wikiqa_train_dl = DataLoader(wikiqa_train_ds, batch_size=batch_size, shuffle=True)
wikiqa_dev_dl = DataLoader(wikiqa_dev_ds, batch_size=batch_size, shuffle=False)
wikiqa_test_dl = DataLoader(wikiqa_test_ds, batch_size=batch_size, shuffle=False)

best_mrr = float('-inf')
best_model = None
train_stats = list()
dev_stats = list()

for epoch in range(adapt_epochs):
    train_loss = train_epoch(wikiqa_train_dl, model, optimizer, criterion, device=DEVICE)
    dev_loss = validate(wikiqa_dev_dl, model, criterion, device=DEVICE)
    
    train_results = get_scores(wikiqa_train_dl, model, device=DEVICE)
    train_qscores = get_question_label_scores(train_results)
    train_acc, train_map, train_mrr = calculate_metrics(train_qscores)
    train_stats.append((train_loss, train_acc, train_map, train_mrr))
    
    dev_results = get_scores(wikiqa_dev_dl, model, device=DEVICE)
    dev_qscores = get_question_label_scores(dev_results)
    dev_acc, dev_map, dev_mrr = calculate_metrics(dev_qscores)
    dev_stats.append((dev_loss, dev_acc, dev_map, dev_mrr))
    
    if dev_mrr > best_mrr:
        best_mrr = dev_mrr
        best_model = copy.deepcopy(model)
    
    print(f'Epoch {epoch+1} complete. Train loss: {train_loss:.4f}, Dev loss: {dev_loss:.4f}')
    print(f'Accuracy: train = {train_acc}, dev = {dev_acc}')
    print(f'MAP: train = {train_map}, dev = {dev_map}')
    print(f'MRR: train = {train_mrr}, dev = {dev_mrr}')
    print('-'*80)
    
save_dict = {'model_params': best_model.state_dict(), 
             'train_stats': train_stats, 
             'dev_stats': dev_stats
            }
torch.save(save_dict, 'best-wikiqa-adapt-roberta.pth')

save_dict = torch.load('best-wikiqa-adapt-roberta.pth')
train_stats = save_dict['train_stats']
dev_stats = save_dict['dev_stats']

model = BertQA(encoder)
model.load_state_dict(save_dict['model_params'], strict=False)
model.to(DEVICE)

test_loss = validate(wikiqa_test_dl, model, criterion, device=DEVICE)
test_results = get_scores(wikiqa_test_dl, model, device=DEVICE)
test_qscores = get_question_label_scores(test_results)
test_acc, test_map, test_mrr = calculate_metrics(test_qscores)

print(f'Test data: Loss = {test_loss}, Accuracy = {test_acc}, MAP = {test_map}, MRR = {test_mrr}')
