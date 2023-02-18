import numpy as np
import os
import json
import torch
from transformers import BertModel
from word_embedding import word_embedding


""" ##############################  WARNING  ################################

This code uses the entire text to retrieve node embedding (with word_embedding
function) If you do not want to use the full text, please adjust it accordingly
(e.g. change word_embedding function).

"""

RadGraph_json_dir = "/data"
save_dir = "/data/radgraph/"


model = BertModel.from_pretrained(
    'bert-base-uncased', output_hidden_states=True)
model.eval()

device = ("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def get_node_type():
    node_types = ['obs', 'anat']
    node_attr = ['dp', 'da', 'u']
    return [t + '-' + a for t in node_types for a in node_attr]


def nodetype2idx():
    a = get_node_type()
    return {t: i for i, t in enumerate(a)}


def get_edge_label():
    relations = ['located_at', 'modify', 'suggestive_of']
    return {t: i for i, t in enumerate(relations)}


def collect_subgraph_nodes(type):
    """[summary]

    Args:
        type ([type]): "train", "dev" or "test"
    """
    data_dir = os.path.join(RadGraph_json_dir, "radgraph-extracting-clinical-entities-and-relations-from-radiology-reports-1.0.0")
    filename = type + ".json"
    file_path = os.path.join(data_dir, filename)
    rel2idx = get_edge_label()
    with open(file_path) as f:
        js_graph = json.load(f)
        num_graphs = len(js_graph)
        graph_indicator = []
        # nodesidx = []
        idx2node_global = dict()
        idx2node_patient = dict()
        node_labels = []
        node_labels_text = []
        node_embeddings = []
        type2idx = nodetype2idx()
        edge_labels = []
        A = []
        n = 1
        for g_idx, patient in enumerate(js_graph):
            # save in multiple files
            patient2global = dict()
            patient_text = js_graph[patient]['text']
            if type == "test":
                patient_nodes = js_graph[patient]['labeler_1']['entities']
            else:
                patient_nodes = js_graph[patient]['entities']
            word_embeddings = word_embedding(model, patient_text)
            # node part
            for node_id, node_info in patient_nodes.items():
                tok = node_info['tokens'].lower()
                lab = node_info['label'].lower()
                # build up index for the entire dataset (e.g. train)
                idx2node_global[n] = tok
                # existing node index under a patient
                idx2node_patient[node_id] = tok
                patient2global[node_id] = n
                # nodes label starts from 0
                # print(lab)
                node_labels_text.append(lab)
                node_labels.append(type2idx[lab])
                start_ix = int(node_info['start_ix'])
                end_ix = int(node_info['end_ix'])
                if start_ix == end_ix:
                    # the first word is CLS so the correct location needs to shift 1
                    embed_vec = word_embeddings[start_ix+1]
                else:
                    embed_vec = word_embeddings[start_ix+1:end_ix+2].mean(dim=0)
                node_embeddings.append(embed_vec)
                # graph label starts from 1
                graph_indicator.append(g_idx+1)
                n += 1
            # edge part
            for node_id, node_info in patient_nodes.items():
                if node_info['relations']:
                    head_global_idx = patient2global[node_id]
                    # r[0]: relation (e.g. 'located_at')
                    # r[1]: pointed node id (e.g. "7")
                    rel_pointer = node_info['relations'][0]
                    rel = rel_pointer[0]
                    edge_labels.append(rel2idx[rel])
                    # local index of pointed (tail) node
                    pointed_node_idx = rel_pointer[1]
                    tail_global_idx = patient2global[pointed_node_idx]
                    A.append((head_global_idx, tail_global_idx))

            embed_file_name = f"node_attributes-{g_idx}.pth"
            if node_embeddings:
                embed_obj = torch.stack(node_embeddings, dim=0)
                save_files(type, embed_obj, embed_file_name)
            else:
                print(f"{patient}-{g_idx} has no entities! skip saving")
            node_embeddings = []


def save_files(data_type, object, file_name):
    """[summary]

    Args:
        data_type ([type]): "train", "dev", "test"
        object (list or dict): [description]
        file_name ([type]): e.g. "edge_labels.txt"
    """
    if data_type == "MIMIC-CXR_graphs":
        data_type = "train"
    elif data_type == "dev":
        data_type = "valid"

    out_file = os.path.join(save_dir, data_type, "raw",
                            data_type + "_" + file_name)

    if file_name.split('-')[0] == "node_attributes":
        torch.save(object, out_file)
    else:
        with open(out_file, "x") as f:
            if file_name == "idx2node_global.txt":
                for idx, node in object.items():
                    _ = f.write(str(idx) + '\t' + node + '\n')
            elif file_name == "A.txt":
                for pair in object:
                    _ = f.write(str(pair[0]) + ', ' + str(pair[1]) + '\n')
            else:
                for idx in object:
                    _ = f.write(str(idx) + '\n')


if __name__ == "__main__":

    for type in ["dev", "test", "MIMIC-CXR_graphs"]:
        collect_subgraph_nodes(type)

