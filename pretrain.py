#!/usr/bin/env python3
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertForPreTraining, BertConfig
from pytorch_pretrained_bert import BertForSequenceClassification, BertForMultiLabelSequenceClassification
import pickle as p
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from pytorch_pretrained_bert.optimization import BertAdam
from torch.nn import CrossEntropyLoss
import gc
import tensorflow as tf
from tensorflow import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


### Example Object
class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

### Prepare Unsupervised Data

def prepare_unsupervised_data(src, backtrad,max_seq_length=256):
    print('Preparing Unsupervised Data ...')
    unsupervised_data = []
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
#     with open(path_data+'/original/'+filename) as f:
#             text_original = f.readlines()

#     with open(path_data+'/Translated/'+filename) as f:
#         text_translated = f.readlines()

    assert len(src) == len(backtrad)
    for i in tqdm(range(len(src))):
        instance = convert_examplesUDA_to_features([src[i],backtrad[i]], max_seq_length=max_seq_length, tokenizer=tokenizer,
                                                   output_mode="UDA")
        unsupervised_data.append(instance)
    print('Unsupervised Data prepared !')
    return list(np.array(unsupervised_data).reshape(-1))


def prepare_unsupervised_data_triplet(src, backtrad,max_seq_length=256):
    print('Preparing triplet data...')
    unsupervised_data = []
    tokenizer = BertTokenizer.from_pretrained('bert-large-cased') ## TO DO STOP HARDCODING

    assert len(src) == len(backtrad)
    for i in tqdm(range(len(src))):
        j = np.random.randint(len(src))
        while j == i:
            j = np.random.randint(len(src))
        instance = convert_examplesUDA_to_features([src[i],backtrad[i],src[j]], max_seq_length=max_seq_length, tokenizer=tokenizer,
                                                   output_mode="UDA")
        unsupervised_data.append(instance)
    print('Unsupervised Data prepared !')
    return list(np.array(unsupervised_data).reshape(-1))

def convert_examplesUDA_to_features(examples, max_seq_length,
                                 tokenizer, output_mode,label_list = None):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    example = examples[0]
    example_2 = examples[1]
    triplet = False
    if len(examples)>2:
        example_3 = examples[2]
        triplet =True
    tokens_a = tokenizer.tokenize(example)

    tokens_b = tokenizer.tokenize(example_2)

    if triplet :
        tokens_c =tokenizer.tokenize(example_3)
        if len(tokens_c) > max_seq_length - 2:
            tokens_c = tokens_c[:(max_seq_length - 2)]
        tokens3 = ["[CLS]"] + tokens_c + ["[SEP]"]
        segment_ids3 = [0] * len(tokens3)
        input_ids3 = tokenizer.convert_tokens_to_ids(tokens3)
        input_mask3 = [1] * len(input_ids3)
        padding3 = [0] * (max_seq_length - len(input_ids3))
        input_ids3 += padding3
        input_mask3 += padding3
        segment_ids3 += padding3


    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[:(max_seq_length - 2)]
    if len(tokens_b) > max_seq_length - 2:
        tokens_b = tokens_b[:(max_seq_length - 2)]

    tokens1 = ["[CLS]"] + tokens_a + ["[SEP]"]
    segment_ids1 = [0] * len(tokens1)
    tokens2 = ["[CLS]"] + tokens_b + ["[SEP]"]
    segment_ids2 = [0] * len(tokens2)
    #


    input_ids1 = tokenizer.convert_tokens_to_ids(tokens1)
    input_ids2 = tokenizer.convert_tokens_to_ids(tokens2)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask1 = [1] * len(input_ids1)
    input_mask2 = [1] * len(input_ids2)

    # Zero-pad up to the sequence length.
    padding1 = [0] * (max_seq_length - len(input_ids1))
    padding2 = [0] * (max_seq_length - len(input_ids2))
    input_ids1 += padding1
    input_mask1 += padding1
    segment_ids1 += padding1

    input_ids2 += padding2
    input_mask2 += padding2
    segment_ids2 += padding2

    assert len(input_ids1) == max_seq_length
    assert len(input_mask1) == max_seq_length
    assert len(segment_ids1) == max_seq_length
    assert len(input_ids2) == max_seq_length
    assert len(input_mask2) == max_seq_length
    assert len(segment_ids2) == max_seq_length

    if output_mode == "classification":
        label_id = label_list[ex_index]
    elif output_mode == "regression":
        label_id = float(label_list[ex_index])
    elif output_mode == "UDA":
        label_id = None
    else:
        raise KeyError(output_mode)

    if triplet: 
        features.append(InputFeatures(input_ids=[input_ids1,input_ids2,input_ids3],
                        input_mask=[input_mask1,input_mask2,input_mask3],
                        segment_ids=[segment_ids1,segment_ids2,segment_ids3],
                        label_id=label_id))
    return features



### Functions

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))



def kl_for_log_probs(log_p, log_q):
    p = torch.exp(log_p)
    neg_ent = torch.sum(p * log_p, dim=-1)
    neg_cross_ent = torch.sum(p * log_q, dim=-1)
    kl = neg_ent - neg_cross_ent
    return kl

def get_tsa_threshold(global_step, num_train_step, start, end,schedule = 'linear', scale = 5):
    '''
    Schedule: Must be either linear, log or exp. Defines the type of schedule used for the annealing.
    start = 1 / K , K being the number of classes
    end = 1
    scale = exp(-scale) close to zero
    '''
    assert schedule in ['linear','log','exp']
    training_progress = global_step / num_train_step
    
    if schedule == 'linear' :
        threshold = training_progress
    elif schedule == 'exp' :
        threshold = torch.exp((training_progress-1) * scale)
    elif schedule == 'log' :
        threshold = 1 - torch.exp((-training_progress) * scale)
    return threshold * (end - start) + start


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--preprocess_data",
                        action='store_true',
                        help="to activate the preprocessing of the data if not done yet")
    parser.add_argument("--triplet_loss",
                        action='store_true',
                        help="to activate the use of triplet loss")
    parser.add_argument("--regularisation_only",
                        action='store_true',
                        help="to deactivate the kl uda loss")
    parser.add_argument("--tsa",
                        default = 3000,
                        type = int, 
                        help="Number of steps to perform tsa over")              
    parser.add_argument("--sequence_length",
                        default = 256,
                        type = int, 
                        help="Length of the sequence used in the model")
    parser.add_argument("--load_model", 
                        default = None, 
                        required = False, 
                        type = str,
                        help="Name of a save model file to load and start from")
    parser.add_argument("--unsup_input",
                        default='data',
                        type=str,
                        required=False,
                        help="The input unlabelled pickle file. If preprocess_data is activate please enter the prefix of the files.")
    parser.add_argument("--uda",
                        default = True,
                        type = bool,
                        help = "Whether or not to use uda.")
    parser.add_argument("--multi_gpu",
                        action = 'store_true',
                        help = 'to activate multi gpus')
    parser.add_argument("--batch_size",
                        default=4,
                        type=int,
                        help='Batch size of the labelled data')
    parser.add_argument("--num_labels",
                        default=2,
                        type=int,
                        help='Number of labels')
    parser.add_argument("--gradient_accumulation",
                        default = 3,
                        type = int, 
                        help = "how many gradients to accumulate before stepping down.")
    parser.add_argument("--lr_model",
                        default = 2e-5,
                        type = float,
                        help = "Learning rate applied to the whole model.")
    parser.add_argument('--verbose',
                        action= 'store_true',
                        help="to activate the printing of intermediate values")
    parser.add_argument('--tensorboard',
                        action= 'store_true',
                        help="to activate tensorboard on port")
    parser.add_argument("--epoch",
                        default = 3,
                        type = int, 
                        help = "how many epochs to perform")                  
    parser.add_argument("--temperature",
                        default = 0.85,
                        type = float,
                        help = "Set the temperature on the pre_softmax layer for unsupervisded entropy")
    parser.add_argument("--uda_threshold",
                        default = -1,
                        type = float,
                        help = "Set the minimal acceptable max probability for unsupervised data")
    parser.add_argument("--tsa_method",
                        default = 'linear',
                        type = str,
                        help = "Set the method to perform threshold annealing on supervised data")
    parser.add_argument("--regularisation",
                        default = -1,
                        type = float, 
                        help='Regularize the last layer instead of the output with a gamma parameter')
    parser.add_argument("--clip_grad",
                        default = 1.,
                        type = float,
                        help='Clip gradient after accumulution')
    parser.add_argument("--mse",
                        action='store_true',
                        help="to opt for the mse loss if combined with regularisation, else nothing.")

    args = parser.parse_args()
    train_log_dir = 'logs/'
    train_summary_writer = summary.create_file_writer(train_log_dir)

    if args.preprocess_data :
        with open(args.unsup_input +'/original.txt') as original:
            src = original.readlines()
        with open(args.unsup_input +'/paraphrase.txt') as paraphrase:
            tgt = paraphrase.readlines()
            unsupervised_data = prepare_unsupervised_data_triplet(src, tgt,max_seq_length=args.sequence_length)
            p.dump(unsupervised_data, open('data/unsupervised_triplet.p', 'wb'))
        


    unsupervised_data = p.load(open('data/unsupervised_triplet.p', 'rb'))


    
    ### Recuperation sous tensors des données non supervisées
    original_input_ids = torch.tensor([f.input_ids[0] for f in unsupervised_data], dtype=torch.long)
    original_input_mask = torch.tensor([f.input_mask[0] for f in unsupervised_data], dtype=torch.long)
    original_segment_ids = torch.tensor([f.segment_ids[0] for f in unsupervised_data], dtype=torch.long)

    augmented_input_ids = torch.tensor([f.input_ids[1] for f in unsupervised_data], dtype=torch.long)
    augmented_input_mask = torch.tensor([f.input_mask[1] for f in unsupervised_data], dtype=torch.long)
    augmented_segment_ids = torch.tensor([f.segment_ids[1] for f in unsupervised_data], dtype=torch.long)

    triplet_input_ids = torch.tensor([f.input_ids[2] for f in unsupervised_data], dtype=torch.long)
    triplet_input_mask = torch.tensor([f.input_mask[2] for f in unsupervised_data], dtype=torch.long)
    triplet_segment_ids = torch.tensor([f.segment_ids[2] for f in unsupervised_data], dtype=torch.long)


    ### Creation des datasets

    unsupervised_dataset = TensorDataset(original_input_ids,\
                                    augmented_input_ids,\
                                    triplet_input_ids)



    ### Variables
    unsup_train_batch_size = args.batch_size 
    unsup_train_sampler = RandomSampler(unsupervised_dataset)
    unsup_train_dataloader = DataLoader(unsupervised_dataset, sampler=unsup_train_sampler, batch_size=unsup_train_batch_size)


    num_labels = args.num_labels

    if args.load_model is not None:
        model = torch.load(args.load_model)
    else:
        model = BertForSequenceClassification.from_pretrained('bert-large-uncased', num_labels = num_labels).to(device)

    if args.multi_gpu:
        model = nn.DataParallel(model)


    ### Parameters
    param_classifier_optimizer = list(model.module.classifier.named_parameters())
    param_body_optimizer = list(model.module.bert.parameters())
    lr_bert = args.lr_model
    epochs = args.epoch
    accumulation_steps = args.gradient_accumulation
    uda_threshold = args.uda_threshold
    temperature = args.temperature

    tsa = args.tsa
    verbose = False




    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay': 0., 'lr' : lr_bert,'max_grad_norm':1},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.01, 'lr' : lr_bert, 'max_grad_norm':1},
        {'params' :param_body_optimizer , 'weight_decay' : 0.01, 'lr' : lr_bert,'max_grad_norm':1}
        ]
    #optimizer = BertAdam(optimizer_grouped_parameters)
    optimizer = torch.optim.Adam(optimizer_grouped_parameters)
    # Locally used variables
    global_step = 0
    accuracy = 0
    # counter = 1
    test_counter = 0
    loss_function = CrossEntropyLoss(reduction = 'none')
    optimizer.zero_grad()
    best = 0  
    MSE = nn.MSELoss(reduction = 'mean')
    ### TRAINING
    for epoch in range(epochs):                      
        for step, batch in tqdm(enumerate(unsup_train_dataloader)):
            model.train()       
        ### Unsupervised Loss
            batch = tuple(t.to(device) for t in batch)
            if args.triplet_loss:
                original_input,augmented_input,triplet_input =batch
                triplet = True
            else:
                original_input,augmented_input = batch
            
            ### REGULARISATION LAST LAYER 
            if args.regularisation>0:
                with torch.no_grad():
                    last_layer_original = model.module.bert(original_input)[1]
                    log_probas = F.log_softmax(model.module.classifier(last_layer_original)/temperature,dim=-1)
                    entropy = -torch.exp(log_probas)*log_probas
                    
                    with train_summary_writer.as_default():
                        tf.summary.scalar('entropy', entropy.sum(-1).mean(0).item(), step=global_step)
                        tf.summary.histogram('proba',torch.exp(log_probas).cpu().data.numpy(), step = global_step)

                ## CLEANING MEMORY
                del entropy
                gc.collect()
                torch.cuda.ipc_collect()
                torch.cuda.empty_cache()
                ## END OF CLEANING

                last_layer_augmented = model.module.bert(augmented_input)[1]
                
                last_layer_triplet = model.module.bert(triplet_input)[1]
                loss_triplet = -MSE(last_layer_triplet, last_layer_original) 
                with train_summary_writer.as_default():
                    tf.summary.scalar('loss_triplet', loss_triplet.item(), step=global_step)

                loss_unsup_regu = MSE(logits_augmented, logits_original) * args.regularisation
                
                if args.regularisation_only:
                    loss_unsup_uda = torch.tensor([0.]).to(device)
                else:
                    log_probas_augmented = F.log_softmax(model.module.classifier(logits_augmented), dim=-1)
                    loss_unsup_uda = kl_for_log_probs(log_probas,log_probas_augmented)

                if uda_threshold > 0:
                    max_logits = torch.max(torch.exp(log_probas), dim=-1)[0]
                    loss_unsup_mask = torch.where(max_logits.cpu() < np.log(uda_threshold),
                                                  torch.tensor([1], dtype=torch.uint8),
                                                  torch.tensor([0], dtype=torch.uint8)).view(-1)
                    loss_unsup_mask.to(device)
                    loss_unsup_uda[loss_unsup_mask] = 0
                    loss_unsup_uda = loss_unsup_uda[loss_unsup_uda > 0.]
                if loss_unsup_uda.size(0) > 0 :
                    loss_unsup_mean = loss_unsup_uda.mean(-1) + loss_unsup_regu + loss_triplet
                else:
                    loss_unsup_mean = loss_unsup_regu + loss_triplet

                
                with train_summary_writer.as_default():

                    tf.summary.scalar('Number of elements unsup', loss_unsup_uda.size(0),global_step)
                    tf.summary.scalar('Loss_Unsup_uda', loss_unsup_uda.mean(-1).item(), step=global_step)
                    tf.summary.scalar('Loss_Unsup_regu', loss_unsup_regu.item(), step=global_step)
                loss_unsup_mean.backward()


            else:
                with torch.no_grad():
                    originals = model(original_input) / temperature
                    logits_original = F.log_softmax(originals, dim = -1)
                    entropy = -torch.exp(logits_original)*logits_original
                    with train_summary_writer.as_default():
                        tf.summary.scalar('entropy', entropy.sum(-1).mean(0).item(), step=global_step)
                        tf.summary.histogram('proba',torch.exp(logits_original).cpu().data.numpy(), step=global_step)
                    max_logits = torch.max(logits_original, dim =-1)[0]

                logits_augmented = F.log_softmax(model(augmented_input), dim = -1)
                loss_unsup = kl_for_log_probs(logits_augmented,logits_original)
                if uda_threshold > 0:
                    loss_unsup_mask = torch.where(max_logits.cpu() < np.log(uda_threshold),
                                                  torch.tensor([1], dtype=torch.uint8),
                                                  torch.tensor([0], dtype=torch.uint8)).view(-1)
                    loss_unsup_mask.to(device)
                    loss_unsup[loss_unsup_mask] = 0
                    loss_unsup = loss_unsup[loss_unsup > 0.]

                if loss_unsup.size(0) > 0:
                    loss_unsup_mean = loss_unsup.mean(-1)
                    with train_summary_writer.as_default():
                        tf.summary.scalar('Number of elements unsup', loss_unsup.size(0),global_step)
                        tf.summary.scalar('Loss_Unsup', loss_unsup_mean.item(), step=global_step)
                    loss_unsup_mean.backward()

        ### Cleaning
            #del entropy
            try:
                del loss_unsup_uda
            except:
                pass
            try:
                del loss_unsup
            except:
                pass
            del loss_unsup_mean
            del logits_original 
            del logits_augmented         
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

            ### Accumulation Steps and Gradient steps
            if (step+1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_value_(model.parameters(),args.clip_grad)
                optimizer.step()
                optimizer.zero_grad()
            
            if (step+1) % 200  == 0:
                model_to_save = model.module if hasattr(model, 'module') else model
                torch.save(model_to_save, "model_pretrained_"+str(step)+".pt")

 ### Increase the global step tracker
            global_step += 1

if __name__ == '__main__':
    main()
