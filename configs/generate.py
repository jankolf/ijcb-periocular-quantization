from types import SimpleNamespace
import pickle
from pathlib import Path

if __name__ == "__main__":


    config = SimpleNamespace()
    config.seed         = 10
    config.batch_size   = 32
    config.epochs       = 12
    config.shuffle      = True
    config.emb_size     = 512
    config.m            = 0.5
    config.s            = 64
    config.base_lr      = 0.1
    config.quant_lr     = 0.01
    config.weight_decay = 5e-4
    config.fold         = 3
    config.model        = "resnet18"
    config.img_size     = 224
    config.data_path    = "/data/"
    config.ufpr_path    = "/data/datasets/UFPR-Periocular/"
    config.protocol     = "open_world_valclosed"
    config.log_interval = 50
    config.save_interval= 200
    config.val_interval = 2000
    config.wq           = 8
    config.aq           = 8
    config.base_model   = ""
    config.flip_images  = True

    r = 0
    for emb in [512, 256, 128]:
        for m in [1.0, 1.1, 1.2, 1.3, 1.4]:
            for fold in [1,2,3]:
                for model in ["resnet18"]:
                    if model == "resnet50" and emb == 512:
                        bs = 16
                    else:
                        bs = 32
                    
                    config.model = model
                    config.m = m
                    config.emb_size = emb
                    config.batch_size = bs
                    config.fold = fold

                    model_name = f"{config.model}_f{config.fold}_m{config.m}_s{config.s}_emb{config.emb_size}_lr{config.base_lr}_b{config.batch_size}"

                    config.base_model = f"/data/models_verification_open_world_valclosed_all_folds/{model_name}/"
                    
                    with open(f"run_{r:05d}.pkl", "wb") as f:
                        pickle.dump(config, f)

                    r += 1

                    