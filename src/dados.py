import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A


def obter_transformacoes_treino():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.1,
            rotate_limit=15,
            border_mode=0,
            p=0.5
        ),
    ])


class DatasetLCA(Dataset):
   #Dataset customizado do PyTorch para carregar pares de fatias (.npy) de RM do joelho e mascaras do LCA.

    def __init__(self, pasta_base, exames_ids, transform=None):
        self.pasta_imagens = os.path.join(pasta_base, "Imagens")
        self.pasta_mascaras = os.path.join(pasta_base, "MASCARAS")
        self.transform = transform
        self.pares = []

        # Filtra apenas as fatias pertencentes aos exames do grupo atual (treino, validacao ou teste)
        exames_set = set(exames_ids)
        for nome_img in sorted(os.listdir(self.pasta_imagens)):
            if nome_img.startswith("img_") and "_fatia_" in nome_img and nome_img.endswith(".npy"):
                # Extrai o ID do paciente (ex: "img_LCA_0001_fatia_015.npy" -> "LCA_0001")
                exame_id = nome_img.split("img_")[1].split("_fatia_")[0]
                if exame_id in exames_set:
                    # Associa cada imagem a sua mascara correspondente trocando o prefixo
                    nome_mask = nome_img.replace("img_", "mask_", 1)
                    caminho_img = os.path.join(self.pasta_imagens, nome_img)
                    caminho_mask = os.path.join(self.pasta_mascaras, nome_mask)
                    if os.path.exists(caminho_mask):
                        self.pares.append((caminho_img, caminho_mask))

    def __len__(self):
        return len(self.pares)

    def __getitem__(self, idx):
        caminho_img, caminho_mask = self.pares[idx]

        img = np.load(caminho_img).astype(np.float32)
        mask = np.load(caminho_mask).astype(np.float32)
        mask = (mask > 0).astype(np.float32)

        # Aplica a mesma transformacao na imagem e na mascara 
        if self.transform is not None:
            aumentado = self.transform(image=img, mask=mask)
            img = aumentado["image"]
            mask = aumentado["mask"]

        img_tensor = torch.from_numpy(img).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)

        return img_tensor, mask_tensor


def criar_dataloaders(
    pasta_base,
    caminho_splits="metadata/splits.csv",
    fold_val=1,
    batch_size=8,
    usar_augmentation=False,
    num_workers=2
):
    df = pd.read_csv(caminho_splits)

    # Separa os IDs por exame para evitar vazamento de dados entre grupos
    exames_treino = df[(df["split"] == "cv") & (df["fold"] != fold_val)]["exame_id"].tolist()
    exames_val = df[(df["split"] == "cv") & (df["fold"] == fold_val)]["exame_id"].tolist()
    exames_teste = df[df["split"] == "teste"]["exame_id"].tolist()

    # Data Augmentation e aplicado exclusivamente no conjunto de treino
    transform_treino = obter_transformacoes_treino() if usar_augmentation else None

    ds_treino = DatasetLCA(pasta_base, exames_treino, transform=transform_treino)
    ds_val = DatasetLCA(pasta_base, exames_val, transform=None)
    ds_teste = DatasetLCA(pasta_base, exames_teste, transform=None)

    # Embaralha apenas o treino a cada epoca; validacao e teste mantem ordem fixa
    loader_treino = DataLoader(ds_treino, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    loader_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    loader_teste = DataLoader(ds_teste, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return loader_treino, loader_val, loader_teste