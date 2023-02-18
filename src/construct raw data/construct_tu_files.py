import numpy as np
import os
import json
import pickle as pkl
import torch

"""
Construct the raw data for radgraph_dataset.py

"""

RadGraph_json_dir = "/data"


def collect_graph_patient_id(type):
    data_dir = os.path.join(RadGraph_json_dir, "radgraph-extracting-clinical-entities-and-relations-from-radiology-reports-1.0.0")
    filename = type + ".json"
    file_path = os.path.join(data_dir, filename)
    patient_id = []
    with open(file_path) as f:
        js_graph = json.load(f)
        for g_idx, patient in enumerate(js_graph):
            patient_id.append(patient)

    if type == 'dev':
        type = 'valid'
    save_files(type, patient_id, "patient_id.txt")


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
        graph_indicator = []
        # nodesidx = []
        idx2node_global = dict()
        idx2node_patient = dict()
        node_labels = []
        node_labels_text = []
        type2idx = nodetype2idx()
        edge_labels = []
        A = []
        n = 1
        for g_idx, patient in enumerate(js_graph):
            patient2global = dict()
            if type == "test":
                all_nodes = js_graph[patient]['labeler_1']['entities']
            else:
                all_nodes = js_graph[patient]['entities']
            # node part
            for node_id, v in all_nodes.items():
                tok = v['tokens'].lower()
                lab = v['label'].lower()
                # build up index for the entire dataset (e.g. train)
                idx2node_global[n] = tok
                # existing node index under a patient
                idx2node_patient[node_id] = tok
                patient2global[node_id] = n
                # nodesidx.append(n)
                # nodes label starts from 0
                # print(lab)
                node_labels_text.append(lab)
                node_labels.append(type2idx[lab])
                # graph label starts from 1
                graph_indicator.append(g_idx+1)
                n += 1
            # edge part
            for node_id, v in all_nodes.items():
                if v['relations']:
                    head_global_idx = patient2global[node_id]
                    # r[0]: relation (e.g. 'located_at')
                    # r[1]: pointed node id (e.g. "7")
                    r = v['relations'][0]
                    edge_labels.append(rel2idx[r[0]])
                    # local index of pointed (tail) node
                    pointed_node_idx = r[1]
                    tail_global_idx = patient2global[pointed_node_idx]
                    A.append((head_global_idx, tail_global_idx))


        file_names = ["graph_indicator.txt", "node_labels.txt",
                      "edge_labels.txt", "idx2node_global.txt",
                      "A.txt"]

        objects = [graph_indicator, node_labels,
                   edge_labels, idx2node_global, A
                   ]

        for f, obj in zip(file_names, objects):
            save_files(type, obj, f)


def save_files(data_type, object, file_name):
    """[summary]

    Args:
        data_type ([type]): "train", "dev", "test"
        object (list or dict): [description]
        file_name ([type]): e.g. "edge_labels.txt"
    """
    if data_type == "MIMIC-CXR_graphs":
        data_type = "train"

    save_dir = "/data/radgraph/" + data_type + "/raw"
    out_file = os.path.join(save_dir,
                            data_type + "_" + file_name)

    with open(out_file, "x") as f:
        if isinstance(object, dict):
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
        collect_graph_patient_id(type)

