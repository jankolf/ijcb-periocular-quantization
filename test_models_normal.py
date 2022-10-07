from argparse import ArgumentParser
from pathlib import Path
import os
import sys
import traceback
from unittest import result
from tqdm import tqdm

import torch

from periocular.backbones import get_model
from periocular.datasets import PeriocularTest
from periocular.datasets import ProtocolType
from periocular.evaluation.extraction import extract_embeddings
from periocular.evaluation.ranking import evaluate_rank_accuracy
from periocular.evaluation.verification import evaluate_verification

def get_last_checkpoint(model_directory):
    
    
    best_backbone = model_directory / f"backbone_{model_directory.name}.pth"
    
    if best_backbone.exists():
        return str(best_backbone)

    sel = max([int(str(d).split("step")[-1].split(".")[0]) 
                 for d in model_directory.glob("backbone_*") if "_step" in str(d)])
    model_path = list(model_directory.glob(f"backbone_*{sel}*"))[0]
    
    return model_path

def build_string(model_name, metrics_verification, metrics_ranking):
    return f"{model_name};{metrics_ranking.rank_1_acc};{metrics_ranking.rank_5_acc};" +\
           f"{metrics_verification.auc};{metrics_verification.eer};{metrics_verification.fnmr_1e_2};" +\
           f"{metrics_verification.fnmr_1e_3};{metrics_verification.fnmr_1e_4};{metrics_verification.fnmr_1e_5};{metrics_verification.fnmr_1e_6}\n"

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--gpu", help="The GPU to use.", type=str)
    parser.add_argument("--model_folder", help="The folder of the models.", type=str)
    parser.add_argument("--protocol", help="The protocol used to train the models.", type=str, choices=["open_world", "open_world_valopen", "closed_world"])
    parser.add_argument("--test_dataset", type=str, choices=["nir", "ufpr"])
    parser.add_argument("--flip_images", help="Use this to flip the images.", action="store_true")
    args = parser.parse_args()
    

    # Reading the arguments
    protocol = {"open_world_valopen":ProtocolType.OPEN_WORLD_OPEN_VAL, 
                "closed_world":ProtocolType.CLOSED_WORLD,
                "open_world":ProtocolType.OPEN_WORLD_CLOSED_VAL}[args.protocol]

    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

    FLIP_L2R = args.flip_images
    model_path = f"/data/{args.model_folder}"
    result_path = f"{model_path}.txt"

    models_fold1 = [f for f in Path(model_path).glob("*_f1_*") if f.is_dir()]
    models_fold2 = [f for f in Path(model_path).glob("*_f2_*") if f.is_dir()]
    models_fold3 = [f for f in Path(model_path).glob("*_f3_*") if f.is_dir()]

    with open(result_path, "a") as f:
        f.write("=> ====== Testing ======\n")
        print("====== Testing ======")
        
        f.write(f"=> Base Path: {model_path}\n")
        print("Base Path:", model_path)
        
        f.write(f"=> Result File: {result_path}\n")
        print("Result File:", result_path)
        
        f.write(f"=> Protocol: {protocol.value}\n")
        print("Protocol:", protocol.value)
        
        f.write(f"=> Flipping: {FLIP_L2R}\n")
        print("Flipping:", FLIP_L2R)

    
    print("# Models Fold 1:", len(models_fold1))
    print("# Models Fold 2:", len(models_fold2))
    print("# Models Fold 3:", len(models_fold3))

    


    for fold_nr, models in [(1, models_fold1), (2, models_fold2), (3, models_fold3)]:
        
        print(f"=== Fold {fold_nr} ===")

        test_set = PeriocularTest(protocol, fold_nr, flip_L_to_R=FLIP_L2R)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        with open(result_path, "a") as f:
            f.write("Model-Name;Rank1;Rank5;AUC;EER;FNMR@1e-2;FNMR@1e-3;FNMR@1e-4;FNMR@1e-5;FNMR@1e-6\n")

        for model_path in tqdm(models):
            try:
                if "mobilenet_v2" in model_path.name:
                    model_name = "mobilenet_v2"
                elif "mobilenet_v3" in model_path.name:
                    model_name = "mobilenet_v3"
                else:
                    model_name = model_path.name.split("_")[0]

                model_embs = int(model_path.name.split("emb")[-1].split("_")[0])

                backbone = get_model(model_name, num_features=model_embs)
                checkpoint_path = get_last_checkpoint(model_path)
                checkpoint = torch.load(str(checkpoint_path))
                backbone.load_state_dict(checkpoint)

                model = torch.nn.DataParallel(backbone)
                model.eval()

                embeddings = extract_embeddings(test_set, model, device)
                metrics_verification = evaluate_verification(test_set, embeddings)
                metrics_ranking = evaluate_rank_accuracy(test_set, embeddings)

                line = build_string(model_path.name, metrics_verification, metrics_ranking)

                with open(result_path, "a") as f:
                    f.write(line)
            except KeyboardInterrupt:
                sys.exit(0)
            except Exception as e:
                traceback.print_exc()

        with open(result_path, "a") as f:
            f.write("=> Info: ")
            f.write(f"Protocol: {protocol.value} ")
            f.write(f"Base Path: {model_path}\n")