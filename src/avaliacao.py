#Dice: mede a sobreposição entre a máscara prevista e a máscara correta.
#IoU: mede a interseção sobre a união entre as duas máscaras.


import torch


def calcular_dice(predicao, mascara, smooth=1e-7):
    predicao = (predicao > 0.5).float()
    mascara = (mascara > 0.5).float()

    intersecao = (predicao * mascara).sum()
    total = predicao.sum() + mascara.sum()

    dice = (2 * intersecao + smooth) / (total + smooth)
    return dice.item()


def calcular_iou(predicao, mascara, smooth=1e-7):
    predicao = (predicao > 0.5).float()
    mascara = (mascara > 0.5).float()

    intersecao = (predicao * mascara).sum()
    uniao = predicao.sum() + mascara.sum() - intersecao

    iou = (intersecao + smooth) / (uniao + smooth)
    return iou.item()


def avaliar_modelo(modelo, dataloader, dispositivo):
    modelo.eval()

    resultados_dice = []
    resultados_iou = []

    with torch.no_grad():
        for imagens, mascaras in dataloader:
            imagens = imagens.to(dispositivo)
            mascaras = mascaras.to(dispositivo)

            logits = modelo(imagens)
            predicoes = torch.sigmoid(logits)

            for i in range(imagens.size(0)):
                dice = calcular_dice(predicoes[i], mascaras[i])
                iou = calcular_iou(predicoes[i], mascaras[i])

                resultados_dice.append(dice)
                resultados_iou.append(iou)

    if not resultados_dice:
        raise ValueError("O conjunto de dados está vazio.")

    return {
        "dice": sum(resultados_dice) / len(resultados_dice),
        "iou": sum(resultados_iou) / len(resultados_iou)
    }
