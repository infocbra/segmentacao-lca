

import os
import torch
import torch.nn as nn
import torch.optim as optim

# Importa a função responsável por avaliar o modelo
try:
    from .avaliacao import avaliar_modelo
except ImportError:
    from avaliacao import avaliar_modelo


def calcular_dice_loss(logits, mascaras, smooth=1e-7):
    """
    Calcula a Dice Loss.

    Diferentemente do Dice Score, que mede o desempenho,
    a Dice Loss mede o erro da segmentação.

    Quanto menor a perda, melhor.
    """

    # Transforma os logits em probabilidades entre 0 e 1
    predicoes = torch.sigmoid(logits)

    # Calcula a interseção entre previsão e máscara correta
    intersecao = (predicoes * mascaras).sum(dim=(1, 2, 3))

    # Soma os pixels positivos das duas máscaras
    total = (
        predicoes.sum(dim=(1, 2, 3))
        + mascaras.sum(dim=(1, 2, 3))
    )

    # Calcula o Dice Score de cada imagem
    dice = (2 * intersecao + smooth) / (total + smooth)

    # Transforma o Dice Score em perda
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
    """
    Executa o treinamento da U-Net.

    Utiliza:
    - BCEWithLogitsLoss
    - Dice Loss
    - Otimizador Adam
    - Avaliação por Dice e IoU

    Salva os melhores pesos encontrados durante o treinamento.
    """

    # Envia o modelo para a GPU ou CPU
    modelo = modelo.to(dispositivo)

    # Calcula a Binary Cross Entropy Loss
    criterio_bce = nn.BCEWithLogitsLoss()

    # Define o otimizador Adam, responsável por atualizar os pesos
    otimizador = optim.Adam(
        modelo.parameters(),
        lr=taxa_aprendizado
    )

    # Guarda o melhor Dice encontrado na validação
    melhor_dice = -1

    # Armazena os resultados de cada época
    historico = []

    # Inicia o treinamento pelo número de épocas definido
    for epoca in range(epocas):

        # Coloca o modelo em modo de treinamento
        modelo.train()

        # Acumula as perdas dos lotes da época
        perda_total = 0

        # Percorre todos os lotes de treinamento
        for imagens, mascaras in loader_treino:

            # Envia os dados para a GPU ou CPU
            imagens = imagens.to(dispositivo)
            mascaras = mascaras.to(dispositivo)

            # Zera os gradientes da etapa anterior
            otimizador.zero_grad()

            # Faz a previsão da máscara do LCA
            logits = modelo(imagens)

            # Calcula a perda usando Binary Cross Entropy
            perda_bce = criterio_bce(logits, mascaras)

            # Calcula a perda usando Dice
            perda_dice = calcular_dice_loss(logits, mascaras)

            # Combina as duas perdas
            perda = perda_bce + perda_dice

            # Calcula os gradientes para corrigir os pesos
            perda.backward()

            # Atualiza os pesos da rede neural
            otimizador.step()

            # Acumula a perda do lote atual
            perda_total += perda.item()

        # Calcula a perda média da época
        perda_media = perda_total / len(loader_treino)

        # Avalia o modelo utilizando os dados de validação
        metricas = avaliar_modelo(
            modelo,
            loader_val,
            dispositivo
        )

        # Recupera os valores de Dice e IoU
        dice_val = metricas["dice"]
        iou_val = metricas["iou"]

        # Guarda os resultados da época no histórico
        historico.append({
            "epoca": epoca + 1,
            "perda": perda_media,
            "dice": dice_val,
            "iou": iou_val
        })

        # Mostra os resultados da época no Colab
        print(
            f"Época {epoca + 1}/{epocas} | "
            f"Perda: {perda_media:.4f} | "
            f"Dice: {dice_val:.4f} | "
            f"IoU: {iou_val:.4f}"
        )

        # Verifica se o modelo melhorou em relação às épocas anteriores
        if dice_val > melhor_dice:

            # Atualiza o melhor Dice encontrado
            melhor_dice = dice_val

            # Cria a pasta de destino, caso ela não exista
            pasta = os.path.dirname(caminho_modelo)

            if pasta:
                os.makedirs(pasta, exist_ok=True)

            # Salva os pesos do melhor modelo
            torch.save(modelo.state_dict(), caminho_modelo)

            print("Melhor modelo salvo!")

    # Recupera os melhores pesos encontrados na validação
    modelo.load_state_dict(
        torch.load(
            caminho_modelo,
            map_location=dispositivo,
            weights_only=True
        )
    )

    # Retorna o modelo com os melhores pesos e o histórico
    return modelo, historico
