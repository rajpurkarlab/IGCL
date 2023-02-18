import os

# Compile all main stats in two easy-to-read csv files (one for eval, one for test)

BASE_ROOT = "/deep/u/sameerk/ImgGraph/dracon_train"
MODEL_ROOT = "/deep/u/sameerk/ImgGraph/dracon_train/saved_models/best_downstream_models/"
EVAL_ROOT = "/deep/u/sameerk/ImgGraph/dracon_train/eval_csv"
TEST_ROOT = "/deep/u/sameerk/ImgGraph/dracon_train/test_csv"
EVAL_CSV_FILE = "/deep/u/sameerk/ImgGraph/dracon_train/complete_eval/eval.csv"
TEST_CSV_FILE = "/deep/u/sameerk/ImgGraph/dracon_train/complete_eval/test.csv"

columns = "Image Encoder,Graph Encoder,Node Attributes,Edge Attributes,Graph Transform,GNN Layers,GNN Hidden,Batch Size,Mean AUC (14),Mean AUC (6),Atelectasis,Cardiomegaly,Consolidation,Edema,No Finding,Pleural Effusion\n"

model_name = "Vit-DRACON"

def compute(out_file, in_dir):
    model_dict = {}
    # First, eval csvs!
    with open(out_file, "w") as OUT:
        OUT.write(columns)
        for dirpath,_,filenames in os.walk(in_dir):
            for f in filenames:
                abs_path = os.path.abspath(os.path.join(dirpath, f))
                if ".csv" not in abs_path:
                    continue

                # ViT-DRACON-2-None-True-128-8_FINAL.csv

                basename = os.path.basename(abs_path)
                filename = basename.split(".")[0]
                model_name = filename.split("_")[0]
                file_vals = model_name.split("-")

                num_attn_heads = int(file_vals[2])
                graph_transform = file_vals[3]
                use_pool = file_vals[4] == "True"
                batch_size = int(file_vals[5])
                epochs = int(file_vals[-1])

                model_code = f"{num_attn_heads}-{graph_transform}-{use_pool}-{batch_size}-{epochs}"

                pool_str = ""
                if use_pool:
                    pool_str = " | global pooling"

                out = f"Vit,DRACON ({epochs} epochs{pool_str} | {num_attn_heads} attention heads),yes,yes,{graph_transform},2,512,{batch_size},"

                with open(abs_path, "r") as FILE:
                    lines = [line.rstrip() for line in FILE]
                    important_line = lines[1]
                    sum_val = sum([float(val) for val in important_line.split(",")])
                    mean_val = sum_val / len(important_line.split(","))

                    out = out + f",{mean_val},{important_line}"

                    if model_code in model_dict: 
                        if mean_val > model_dict[model_code]["mean_val"]:
                            model_dict[model_code]["str"] = out
                            model_dict[model_code]["mean_val"] = mean_val
                    else:
                        model_dict[model_code] = {}
                        model_dict[model_code]["str"] = out
                        model_dict[model_code]["mean_val"] = mean_val

        for model_code in model_dict:
            out_str = model_dict[model_code]["str"]
            OUT.write(f"{out_str}\n")

compute(EVAL_CSV_FILE, EVAL_ROOT)
compute(TEST_CSV_FILE, TEST_ROOT)
