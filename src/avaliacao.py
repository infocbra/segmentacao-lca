#Dice: mede a sobreposição entre a máscara prevista e a máscara correta.
#IoU: mede a interseção sobre a união entre as duas máscaras.



import torch

def calcular_dice(predicao, mascara, smooth=1e-7):
    # Transforma as previsões em valores binários:
    # acima de 0.5 = 1 (LCA)
    # abaixo ou igual a 0.5 = 0 (fundo)
    predicao = (predicao > 0.5).float()
    mascara = (mascara > 0.5).float()

    # Conta quantos pixels estão presentes nas duas máscaras
    intersecao = (predicao * mascara).sum()

    # Soma a quantidade de pixels positivos das duas máscaras
    total = predicao.sum() + mascara.sum()

    # Aplica a fórmula do Dice Score
    dice = (2 * intersecao + smooth) / (total + smooth)

    # Retorna o resultado como número decimal
    return dice.item()


def calcular_iou(predicao, mascara, smooth=1e-7):
    # Converte as máscaras em valores binários
    predicao = (predicao > 0.5).float()
    mascara = (mascara > 0.5).float()

    # Calcula os pixels que aparecem nas duas máscaras
    intersecao = (predicao * mascara).sum()

    # Calcula a união das duas máscaras
    uniao = predicao.sum() + mascara.sum() - intersecao

    # Aplica a fórmula do IoU
    iou = (intersecao + smooth) / (uniao + smooth)

    return iou.item()


def avaliar_modelo(modelo, dataloader, dispositivo):
    """
    Avalia o modelo utilizando um conjunto de dados.

    Percorre as imagens, gera as previsões e calcula
    as médias das métricas Dice e IoU por imagem.
    """

    # Coloca o modelo em modo de avaliação
    modelo.eval()

    # Listas para armazenar os resultados individuais
    resultados_dice = []
    resultados_iou = []

    # Desativa o cálculo de gradientes, pois não estamos treinando
    with torch.no_grad():

        # Percorre todos os lotes de imagens
        for imagens, mascaras in dataloader:

            # Envia imagens e máscaras para a GPU ou CPU
            imagens = imagens.to(dispositivo)
            mascaras = mascaras.to(dispositivo)

            # O modelo gera as previsões
            logits = modelo(imagens)

            # Converte os logits em probabilidades entre 0 e 1
            predicoes = torch.sigmoid(logits)

            # Calcula as métricas individualmente para cada imagem
            for i in range(imagens.size(0)):

                dice = calcular_dice(
                    predicoes[i],
                    mascaras[i]
                )

                iou = calcular_iou(
                    predicoes[i],
                    mascaras[i]
                )

                # Guarda os resultados calculados
                resultados_dice.append(dice)
                resultados_iou.append(iou)

    # Verifica se existem imagens para avaliação
    if not resultados_dice:
        raise ValueError("O conjunto de dados está vazio.")

    # Calcula a média das métricas de todas as imagens
    media_dice = sum(resultados_dice) / len(resultados_dice)
    media_iou = sum(resultados_iou) / len(resultados_iou)

    # Retorna as duas métricas em um dicionário
    return {
        "dice": media_dice,
        "iou": media_iou
    }

