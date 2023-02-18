import numpy as np
import os
import json
import pickle as pkl
import torch
from transformers import BertTokenizer, BertModel
import torch_scatter
import itertools as it
from itertools import accumulate, takewhile

device = ("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained(
    'bert-base-uncased', output_hidden_states=True)
model.eval()
model.to(device)

"""

This code uses the entire text to retrieve node embedding.

"""


def longest_text(type):
    data_dir = "/data/radgraph-extracting-clinical-entities-and-relations-from-radiology-reports-1.0.0"
    filename = type + ".json"
    file_path = os.path.join(data_dir, filename)
    text_token_size = []
    with open(file_path) as f:
        js_graph = json.load(f)
        for g_idx, patient in enumerate(js_graph):
            text = js_graph[patient]['text']
            marked_text = "[CLS] " + text + " [SEP]"
            text_token_size.append(len(tokenizer.tokenize(marked_text)))
    return text_token_size


def get_last_word_idx(num_tokens, max_seq_len=512):
    """forward looking

    Args:
        num_tokens ([type]): [description]

    Returns:
        [type]: word-level index
    """
    accum_f = list(accumulate(num_tokens))
    # forward
    last_word_idx = len(list(takewhile(lambda x: x < max_seq_len, accum_f))) - 1
    # space for the last word for [SEP]
    last_word_idx -= 1

    assert last_word_idx > 0

    return last_word_idx


def get_first_word_idx(num_tokens, max_seq_len=512):
    """

    Args:
        num_tokens ([type]): [description]

    Returns:
    [type]: last word index count backward

    """

    accum_b = list(accumulate(reversed(num_tokens)))
    # last word count backward
    last_word_idx_b = len(list(takewhile(lambda x: x < max_seq_len, accum_b))) - 1
    # space for the last word for [CLS]
    last_word_idx_b -= 1

    num_words_2nd_part = last_word_idx_b + 1
    num_words = len(num_tokens)
    # first word count forward
    first_word_idx_f = num_words - num_words_2nd_part

    return first_word_idx_f


def tokenize_forward(tokens, word_idx_for_token, last_word_idx=None):
    # last_word_idx inclusive
    if last_word_idx:
        end = last_word_idx + 1
    else:
        end = None
    # include both CLS and SEP
    total_num_words = len(tokens)
    word_idx_for_token_flat = list(it.chain.from_iterable(
        word_idx_for_token[:end]))
    tokenized_text = list(it.chain.from_iterable(tokens[:end]))
    if last_word_idx:
        tokenized_text = tokenized_text + [' [SEP]']
    # add last index = last_word_idx + 1 for ' [SEP]'
    # it's for scatter i.e. used to select the last row of token embedding (SEP)

    if last_word_idx:
        word_idx_for_token_tensor = torch.tensor(
            word_idx_for_token_flat +
        [last_word_idx + 1])
        # indicate the original word index exclusive [CLS] and [SEP]
        word_idx = [w[0] for w in word_idx_for_token if w[0]
                    <= last_word_idx and w[0] > 0]
    else:
        word_idx_for_token_tensor = torch.tensor(
            word_idx_for_token_flat)
        word_idx = [w[0] for w in word_idx_for_token]

    return tokenized_text, word_idx_for_token_tensor, word_idx


def tokenize_backward(tokens, word_idx_for_token, first_word_idx_f):
    """[summary]

    Args:
        tokens ([type]): [description]
        word_idx_for_token ([type]): [description] 2nd part
        last_word_idx ([type], optional): last word count backward

    Returns:
        [type]: [description]
    """

    word_idx_for_token_flat = list(it.chain.from_iterable(
        word_idx_for_token[first_word_idx_f:]))     # no correct
    tokenized_text = list(it.chain.from_iterable(
        tokens[first_word_idx_f:]))
    tokenized_text = ['[CLS] '] + tokenized_text
    word_idx_for_token_tensor = torch.tensor(
        [0] + word_idx_for_token_flat)
    shift = word_idx_for_token_flat[0]
    word_idx_scatter = word_idx_for_token_tensor - shift + 1
    word_idx_scatter[0] = 0
    # word_idx should match the embedding vector exclusive of
    word_idx = [w[0] for w in word_idx_for_token if w[0]
                >= first_word_idx_f and w[0] < word_idx_for_token[-1][0]]

    return tokenized_text, word_idx_scatter, word_idx


def get_embedding(model, tokenized_text, word_idx_for_token, device="cuda"):
    model.to(device)

    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    # Mark each of the tokens as belonging to sentence "1".
    segments_ids = [1] * len(indexed_tokens)

    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    # extracting embedding
    with torch.no_grad():
        outputs = model(tokens_tensor.to(device), segments_tensors.to(device))
        hidden_states = outputs[2]
    token_embeddings = torch.stack(hidden_states, dim=0).squeeze(1)
    token_embeddings = token_embeddings.permute(1, 0, 2)
    # sum the last 4 layers
    token_vec_sum = token_embeddings[:, -4:].sum(dim=1)

    word_embeddings = torch_scatter.scatter(
        token_vec_sum, word_idx_for_token.to(device), dim=0, reduce='sum')

    return word_embeddings


def word_embedding(model, text, max_seq_len=512):
    marked_text = "[CLS] " + text + " [SEP]"
    words = marked_text.split(' ')
    tokens = [tokenizer.tokenize(w) for w in words]
    word_idx_for_token = [[i] * len(t) for i, t in enumerate(tokens)]
    # number of tokens for each word
    num_tokens = [len(t) for t in tokens]
    total_num_tokens = sum(num_tokens)
    num_of_words = len(words)

    if total_num_tokens < max_seq_len:
        tokenized_text, word_idx_for_token, _ = tokenize_forward(
            tokens, word_idx_for_token)
        word_embedding = get_embedding(model, tokenized_text, word_idx_for_token)
    else:
        last_word_idx = get_last_word_idx(num_tokens, max_seq_len)
        tokenized_text_1, word_idx_for_token_1, word_idx_1 = tokenize_forward(
            tokens, word_idx_for_token, last_word_idx)
        word_embedding_1 = get_embedding(
            model, tokenized_text_1, word_idx_for_token_1)
        # sanity check
        assert word_embedding_1.shape[0] == word_idx_for_token_1[-1] + 1
        # rows of word_embedding_1 should match word_idx_1
        assert word_embedding_1.shape[0] == len(word_idx_1) + 2

        first_word_idx = get_first_word_idx(num_tokens, max_seq_len)

        # ensure overlapping
        assert last_word_idx > first_word_idx, "require longer max_seq_len to overlap"

        tokenized_text_2, word_idx_adjusted, word_idx_2 = tokenize_backward(
            tokens, word_idx_for_token, first_word_idx)
        word_embedding_2 = get_embedding(
            model, tokenized_text_2, word_idx_adjusted)
        assert word_embedding_2.shape[0] == word_idx_adjusted[-1] + 1
        # rows of word_embedding_2 should match word_idx_2
        assert word_embedding_2.shape[0] == len(word_idx_2) + 2

        # TODO: check
        emb_idx_2 = np.where(np.array(word_idx_2) == last_word_idx)[0].item() + 1
        word_embedding = torch.cat(
            (word_embedding_1[0:-1], word_embedding_2[emb_idx_2+1:]), dim=0)

        assert word_embedding.shape[0] == num_of_words

    return word_embedding


if __name__ == "__main__":

    text = 'WET READ : ___ ___ ___ 8 : 22 PM ET tube 4 . 1 cm from the carina . Enteric tube with the tip and side-port in the stomach . Right internal jugular central venous catheter in the mid SVC . Two drains projecting over the mediastinum and one over the left hemi thorax . Multiple epicardial pacing leads projecting over the heart . Extensive subcutaneous gas along the left chest wall . The lung apices are omitted from view . No large pneumothorax is seen on this supine view . There is likely small left pleural effusion . There is a an opacity at the left upper lobe projecting from the aorta to the left apex . This is new since the study of ___ at 14 : 29 . It is unclear whether this is expected postoperative change or hemorrhage . If there is any concern for postoperative bleeding CT of the chest should be obtained . The findings were telephoned to ___ by ___ at 21 : 30 , ___ , ___ min after discovery . The provider explain that the opacity likely reflects expected postoperative hematoma and that there was no evidence of active bleeding WET READ VERSION #___ ___ ___ 9 : 34 PM ET tube 4 . 1 cm from the carina . Enteric tube with the tip and side-port in the stomach . Right internal jugular central venous catheter in the mid SVC . Two drains projecting over the mediastinum and one over the left hemi thorax . Multiple epicardial pacing leads projecting over the heart . Extensive subcutaneous gas along the left chest wall . The lung apices are omitted from view . No large pneumothorax is seen on this supine view . There is likely small left pleural effusion . There is a an opacity at the left upper lobe projecting from the aorta to the left apex . This is new since the study of ___ at 14 : 29 . It is unclear whether this is expected postoperative change or hemorrhage . If there is any concern for postoperative bleeding CT of the chest should be obtained . The findings were telephoned to ___ by ___ at 21 : 30 , ___ , ___ min after discovery . The provider explain that the opacity likely reflects expected postoperative hematoma and that there was no evidence of active bleeding ______________________________________________________________________________ FINAL REPORT INDICATION : ___ year old man s / p takeback for mediastinal washout / / Post-op CXR COMPARISON : Comparison is made with prior studies including ___ at 16 : 30 . IMPRESSION : The lung apices are not included on this study limiting observation . There is no definite pneumothorax or pneumomediastinum . There is subcutaneous emphysema on the left as on the earlier study . Extensive postoperative changes are present . There is a left chest tube and mediastinal drains are present . Endotracheal tube tip is 4 cm above the carina . Right central line tip is in the superior vena cava . Nasogastric tube tip is in the stomach . There is better aeration of the left lung as compared to the earlier study . There is an area of density in the left apex medially which could represent atelectasis , but correlation with surgical intervention that region is recommended . This may be accentuated by slight patient rotation to the left .'

    a = word_embedding(text, 5)


