
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


def criar_unet_pretrained():
    """
    Cria uma U-Net com encoder ResNet18 pré-treinado no ImageNet.

    Entrada:
        Imagem de ressonância magnética com 1 canal.

    Saída:
        Máscara de segmentação binária com 1 canal.
    """

    modelo = smp.Unet( #Cria a arquitetura U-Net.
        encoder_name="resnet18", #Escolhe a ResNet18 como encoder.
        encoder_weights="imagenet",#Aproveita os pesos que já foram aprendidos com o ImageNet
        in_channels=1, #Recebe imagens em escala de cinza
        classes=1, #Produz uma máscara binária.
        activation=None #Produz uma máscara binária.
    )

    return modelo
