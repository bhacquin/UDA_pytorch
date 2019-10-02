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
    return unsupervised_data
### Translate each sentence


def convert_examplesUDA_to_features(examples, max_seq_length,
                                 tokenizer, output_mode,label_list = None):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    example = examples[0]
    example_2 = examples[1]

    tokens_a = tokenizer.tokenize(example)

    tokens_b = tokenizer.tokenize(example_2)

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


    features.append(InputFeatures(input_ids=[input_ids1,input_ids2],
                        input_mask=[input_mask1,input_mask2],
                        segment_ids=[segment_ids1,segment_ids2],
                        label_id=label_id))
    return features


### Prepare Labelled Data

def prepare_supervised_data(src,max_seq_length=256):
    print('Preparing supervised data...')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#     with open(path_data+'/original/'+filename) as f:
#             text_original = f.readlines()

#     with open(path_data+'/Translated/'+filename) as f:
#         text_translated = f.readlines()

    
    
    supervised_data = convertLABEL_examples_to_features(list(src['content']),list(src['label']), max_seq_length=max_seq_length, tokenizer=tokenizer,
                                                   output_mode="classification")
    print('Supervised Data prepared !')
    return np.array(supervised_data)



def convertLABEL_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode):
    """Loads a data file into a list of `InputBatch`s."""

    #   label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in tqdm(enumerate(examples)):


        
        tokens_a = tokenizer.tokenize(example)
        
        
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)


        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_list[ex_index]
        elif output_mode == "regression":
            label_id = float(label_list[ex_index])
        else:
            raise KeyError(output_mode)


        features.append(InputFeatures(input_ids=input_ids,
                            input_mask=input_mask,
                            segment_ids=segment_ids,
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
    parser.add_argument("--sup_input",
                        default='data',
                        type=str,
                        required=False,
                        help="The input labelled pickle file")
    parser.add_argument("--pickle_input_sup",
                        default="supervised.p",
                        required = False, 
                        help="The preprocessed supervised data to unpickle from")
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
    parser.add_argument("--pretraining",
                        action = 'store_true',
                        help = "To pretrain network with unsupervised loss")
    parser.add_argument("--multi_gpu",
                        action = 'store_true',
                        help = 'to activate multi gpus')
    parser.add_argument("--batch_size",
                        default=4,
                        type=int,
                        help='Batch size of the labelled data')
    parser.add_argument('--unsup_ratio',
                        default = 2,
                        type = int, 
                        help = 'To define the batch_size of unlabelled data, unsup_ratio * batch_size.')
    parser.add_argument("--gradient_accumulation",
                        default = 3,
                        type = int, 
                        help = "how many gradients to accumulate before stepping down.")
    parser.add_argument("--lr_classifier",
                        default = 2e-5,
                        type = float,
                        help = " Learning rate applied to the last layer - classifier layer - .")
    parser.add_argument("--lr_model",
                        default = 2e-5,
                        type = float,
                        help = "Learning rate applied to the whole model bar the classifier layer.")
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
    parser.add_argument("--labelled_examples",
                        default = 20,
                        type = int, 
                        help = "how many labelled examples to learn from")
    parser.add_argument("--temperature",
                        default = 0.85,
                        type = float,
                        help = "Set the temperature on the pre_softmax layer for unsupervisded entropy")
    parser.add_argument("--uda_threshold",
                        default = -1,
                        type = float,
                        help = "Set the minimal acceptable max probability for unsupervised data")
    parser.add_argument("--tsa",
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
        unsupervised_data = prepare_unsupervised_data(src, tgt,max_seq_length=args.sequence_length)
        df_train = p.load(open(args.sup_input+'/train_label.p','rb'))   
        df_test = p.load(open(args.sup_input+'/test_label.p','rb')) 
        supervised_data = prepare_supervised_data(df_train,max_seq_length=args.sequence_length)
        test_data = prepare_supervised_data(df_test,max_seq_length=args.sequence_length)
        p.dump(unsupervised_data, open('data/unsupervised.p', 'wb'))
        p.dump(supervised_data, open(args.pickle_input_sup, 'wb'))
        p.dump(test_data, open('data/test.p', 'wb'))

    unsupervised_data = p.load(open('data/unsupervised.p', 'rb'))
    unsupervised_data = list(np.array(unsupervised_data).reshape(-1))
    supervised_data = p.load(open('data/'+args.pickle_input_sup,'rb'))   
    test_data = p.load(open('data/test.p','rb')) 

    
    ### Recuperation sous tensors des données non supervisées
    original_input_ids = torch.tensor([f.input_ids[0] for f in unsupervised_data], dtype=torch.long)
    original_input_mask = torch.tensor([f.input_mask[0] for f in unsupervised_data], dtype=torch.long)
    original_segment_ids = torch.tensor([f.segment_ids[0] for f in unsupervised_data], dtype=torch.long)

    augmented_input_ids = torch.tensor([f.input_ids[1] for f in unsupervised_data], dtype=torch.long)
    augmented_input_mask = torch.tensor([f.input_mask[1] for f in unsupervised_data], dtype=torch.long)
    augmented_segment_ids = torch.tensor([f.segment_ids[1] for f in unsupervised_data], dtype=torch.long)

    ### Recuperation sous tensors des données supervisées
    supervised_input_ids = torch.tensor([f.input_ids for f in supervised_data], dtype=torch.long)
    supervised_input_mask = torch.tensor([f.input_mask for f in supervised_data], dtype=torch.long)
    supervised_segment_ids = torch.tensor([f.segment_ids for f in supervised_data], dtype=torch.long)
    supervised_label_ids = torch.tensor([f.label_id for f in supervised_data], dtype=torch.long)

    test_input_ids = torch.tensor([f.input_ids for f in test_data], dtype=torch.long)
    test_input_mask = torch.tensor([f.input_mask for f in test_data], dtype=torch.long)
    test_segment_ids = torch.tensor([f.segment_ids for f in test_data], dtype=torch.long)
    test_label_ids = torch.tensor([f.label_id for f in test_data], dtype=torch.long)

    ### Creation des datasets
    unsupervised_dataset = TensorDataset(original_input_ids, original_input_mask, original_segment_ids,\
                                    augmented_input_ids,augmented_input_mask,augmented_segment_ids)

    supervised_dataset = TensorDataset(supervised_input_ids,\
                                    supervised_input_mask,supervised_segment_ids,\
                                    supervised_label_ids)

    test_dataset = TensorDataset(test_input_ids,\
                                    test_input_mask,test_segment_ids,\
                                    test_label_ids)



    ### Variables
    unsup_train_batch_size = args.batch_size * args.unsup_ratio 
    sup_train_batch_size = args.batch_size
    labelled_examples = args.labelled_examples
    unsup_train_sampler = RandomSampler(unsupervised_dataset)
    unsup_train_dataloader = DataLoader(unsupervised_dataset, sampler=unsup_train_sampler, batch_size=unsup_train_batch_size)


    # sup_train_sampler = RandomSampler(supervised_dataset)
    sup_subset_sampler = torch.utils.data.SubsetRandomSampler(\
                                            np.random.randint(supervised_input_ids.size(0), size=labelled_examples))
    sup_train_dataloader = DataLoader(supervised_dataset, sampler=sup_subset_sampler, batch_size=sup_train_batch_size)

    test_sampler = torch.utils.data.SubsetRandomSampler(\
                                            np.random.randint(test_input_ids.size(0), size=10000))
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=16)



    num_labels = 2

    if args.load_model is not None:
        model = torch.load(args.load_model)

    else:
        model = BertForSequenceClassification.from_pretrained('bert-large-uncased', num_labels = num_labels).to(device)

    if args.multi_gpu:
        model = nn.DataParallel(model)


    ### Parameters
    param_optimizer = list(model.module.classifier.named_parameters())
    lr = args.lr_classifier
    lr_bert = args.lr_model

    epochs = args.epoch
    accumulation_steps = args.gradient_accumulation
    uda_threshold = args.uda_threshold
    temperature = args.temperature
    tsa = True
    verbose = False




    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay': 0., 'lr' : lr,'max_grad_norm':1},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.01, 'lr' : lr, 'max_grad_norm':1},
        {'params' : model.module.bert.parameters(), 'weight_decay' : 0.01, 'lr' : lr_bert,'max_grad_norm':1}
        ]
    #optimizer = BertAdam(optimizer_grouped_parameters)
    optimizer = torch.optim.Adam(optimizer_grouped_parameters)
    # Locally used variables
    global_step = 0
    accuracy = 0
    counter = 1
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
            original_input, _, _, augmented_input,_,_ = batch

            if args.regularisation>0:
                with torch.no_grad():
                    logits_original = model.module.bert(original_input)[1]
                    log_probas = F.log_softmax(model.module.classifier(logits_original),dim=-1)
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

                logits_augmented = model.module.bert(augmented_input)[1]
                log_probas_augmented = F.log_softmax(model.module.classifier(logits_augmented)/temperature, dim=-1)
                loss_unsup_regu = MSE(logits_augmented, logits_original) * args.regularisation
                loss_unsup_uda = kl_for_log_probs(log_probas,log_probas_augmented)

                #

                if uda_threshold > 0:
                    max_logits = torch.max(torch.exp(log_probas), dim=-1)[0]
                    loss_unsup_mask = torch.where(max_logits.cpu() < np.log(uda_threshold),
                                                  torch.tensor([1], dtype=torch.uint8),
                                                  torch.tensor([0], dtype=torch.uint8)).view(-1)
                    loss_unsup_mask.to(device)
                    loss_unsup_uda[loss_unsup_mask] = 0
                    loss_unsup_uda = loss_unsup_uda[loss_unsup > 0.]
                if loss_unsup_uda.size(0) > 0:
                    loss_unsup_mean = loss_unsup_uda.mean(-1) + loss_unsup_regu
                else:
                    loss_unsup_mean = loss_unsup_regu
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
            
            if args.pretrained: ## To do reformatting
                continue
        ### Supervised Loss
            for i , batch_sup in enumerate(sup_train_dataloader):
                if counter % (i+1) == 0 :
                    batch_sup = tuple(t.to(device) for t in batch_sup)
                    input_ids, input_mask, segment_ids, label_ids = batch_sup 
                   # tf.summary.scalar('learning rate', np.max(optimizer.get_lr(), step=global_step)
                    logits = model(input_ids)
                    loss_sup = loss_function(logits.view(-1,2),label_ids.view(-1))

                    with torch.no_grad():
                        outputs = F.softmax(logits,dim=-1)
                        sentiment_corrects = torch.sum(torch.max(outputs, -1)[1] == label_ids)
                        sentiment_acc = sentiment_corrects.double() / sup_train_batch_size
                        accuracy += sentiment_acc
                    with train_summary_writer.as_default():
                        tf.summary.scalar('Batch_score', sentiment_acc.item(), step=global_step)
                    number_of_elements = outputs.size(0)
                    
            ### Threshold Annealing
                    if tsa:
                        tsa_start = 1. / num_labels
                        tsa_threshold = get_tsa_threshold(global_step = global_step,\
                                                        num_train_step  = 1500, start = tsa_start,\
                                                        end=1.,schedule = 'linear', scale = 5)
                        probas = torch.gather(outputs, dim = -1, index = label_ids.unsqueeze(1)).cpu()
                        loss_mask = torch.where(probas > tsa_threshold, torch.tensor([1], dtype=torch.uint8), torch.tensor([0], dtype=torch.uint8))
                        loss_mask.to(device)
                        loss_mask = loss_mask.view(-1)
                        with train_summary_writer.as_default():
                            tf.summary.scalar('tsa_threshold',tsa_threshold, global_step)
                            tf.summary.scalar('loss_sup', loss_sup.mean(-1).item(), step=global_step)
                        loss_sup[loss_mask] = 0.
                        number_of_elements = loss_mask.size(0)-loss_mask.sum(0)
                        if verbose:
                            print('outputs', outputs)
                            print('tsa_threshold',tsa_threshold)
                            print('label_ids', label_ids)
                            print('probas', probas)
                            print('mask', loss_mask)
                            print('post_loss', loss_sup)
                            print('number_of_elements : ',loss_mask.size(0)-loss_mask.sum(0))

                    if number_of_elements > 0:
                        loss_sup = loss_sup[loss_sup > 0.]
                        nb_elements = loss_sup.size(0)
                        loss_sup = loss_sup.mean(-1)
                        loss_sup.backward()
                    else:
                        nb_elements = 0
                        loss_sup = torch.tensor([0.])
                    with train_summary_writer.as_default():
                        tf.summary.scalar('nb_elements_sup', nb_elements, global_step)
                        tf.summary.scalar('Post_loss', loss_sup.item(), step=global_step)

            ### Cleaning
                    del loss_sup 
                    del logits
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                    counter += 1                
                    if counter > labelled_examples +1 :
                        counter = 1
                    break
                else:
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                    continue
            
            ### Accumulation Steps and Gradient steps
            if (step+1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_value_(model.parameters(),args.clip_grad)
                optimizer.step()
                optimizer.zero_grad()

            ### Test set and Evaluation  every x gradient steps         
            if (step+1) % 100  == 0:
                loss = []
                sentiment_test_acc = 0
                for test_step, test_batch in enumerate(test_dataloader):
                    test_batch = tuple(t.to(device) for t in test_batch)
                    input_ids, input_mask, segment_ids, label_ids = test_batch
                    with torch.no_grad():
                        logits = model(input_ids)
                        loss_test = loss_function(logits.view(-1,2),label_ids.view(-1)).mean(-1)
                        with train_summary_writer.as_default():
                            tf.summary.scalar('Test_loss_continuous', loss_test.item(), step=test_step+test_counter*len(test_dataloader))
                        loss.append(loss_test.item())
                        outputs = F.softmax(logits,dim=-1)
                        sentiment_corrects = torch.sum(torch.max(outputs, -1)[1] == label_ids)                        
                        sentiment_test_acc += sentiment_corrects.double()
                        accuracy += sentiment_acc / input_ids.size(0)
                sentiment_test_acc = sentiment_test_acc / len(test_dataloader)		  
                with train_summary_writer.as_default():
                    tf.summary.scalar('Test_score', sentiment_test_acc.item()/16, step=global_step)
                    tf.summary.scalar('test_loss', np.array(loss).mean(), step=global_step)
                    tf.summary.scalar('test_loss_std', np.array(loss).std(), step=global_step)
                test_counter += 1
                print('best_score',best)
                if sentiment_test_acc.item()/16 > best :
                    model_to_save = model.module if hasattr(model, 'module') else model
                    torch.save(model_to_save, "best_model_score.pt")
                    best = sentiment_test_acc.item()/16
               
            ### Increase the global step tracker
            global_step += 1

if __name__ == '__main__':
    main()
