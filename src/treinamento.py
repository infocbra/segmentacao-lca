
import os
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from .avaliacao import avaliar_modelo
except ImportError:
    from avaliacao import avaliar_modelo


def calcular_dice_loss(logits, mascaras, smooth=1e-7):
    predicoes = torch.sigmoid(logits)

    intersecao = (predicoes * mascaras).sum(dim=(1, 2, 3))
    total = predicoes.sum(dim=(1, 2, 3)) + mascaras.sum(dim=(1, 2, 3))

    dice = (2 * intersecao + smooth) / (total + smooth)

    return 1 - dice.mean()


def treinar_modelo(
    modelo,
    loader_treino,
    loader_val,
    dispositivo,
    epocas=30,
    taxa_aprendizado=0.0001,
    caminho_modelo="melhor_modelo.pth"
):
    modelo = modelo.to(dispositivo)

    criterio_bce = nn.BCEWithLogitsLoss()
    otimizador = optim.Adam(
        modelo.parameters(),
        lr=taxa_aprendizado
    )

    melhor_dice = -1
    historico = []

    for epoca in range(epocas):
        modelo.train()

        perda_total = 0

        for imagens, mascaras in loader_treino:
            imagens = imagens.to(dispositivo)
            mascaras = mascaras.to(dispositivo)

            otimizador.zero_grad()

            logits = modelo(imagens)

            perda_bce = criterio_bce(logits, mascaras)
            perda_dice = calcular_dice_loss(logits, mascaras)

            perda = perda_bce + perda_dice

            perda.backward()
            otimizador.step()

            perda_total += perda.item()

        perda_media = perda_total / len(loader_treino)

        metricas = avaliar_modelo(
            modelo,
            loader_val,
            dispositivo
        )

        dice_val = metricas["dice"]
        iou_val = metricas["iou"]

        historico.append({
            "epoca": epoca + 1,
            "perda": perda_media,
            "dice": dice_val,
            "iou": iou_val
        })

        print(
            f"Época {epoca + 1}/{epocas} | "
            f"Perda: {perda_media:.4f} | "
            f"Dice: {dice_val:.4f} | "
            f"IoU: {iou_val:.4f}"
        )

        if dice_val > melhor_dice:
            melhor_dice = dice_val

            pasta = os.path.dirname(caminho_modelo)
            if pasta:
                os.makedirs(pasta, exist_ok=True)

            torch.save(modelo.state_dict(), caminho_modelo)

            print("Melhor modelo salvo!")

  # Recuperar os melhores pesos encontrados durante a validação
  modelo.load_state_dict(
    torch.load(caminho_modelo, map_location=dispositivo)
)

return modelo, historico
