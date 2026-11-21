import os
import torch
import numpy as np
import matplotlib.pyplot as plt


# Calcula o Dice de uma unica fatia para mostrar no titulo da imagem
def calcular_dice_fatia(mask_real, mask_pred, eps=1e-7):
    intersecao = np.sum(mask_real * mask_pred)
    soma = np.sum(mask_real) + np.sum(mask_pred)
    return (2.0 * intersecao + eps) / (soma + eps)

# Carrega os pesos salvos (.pth), roda no conjunto de teste e salva figuras comparativas em PNG
def gerar_predicoes_visuais(
    modelo,
    caminho_pesos,
    loader_teste,
    pasta_saida,
    dispositivo="cuda",
    limiar=0.5,
    max_figuras=None,  
):
    os.makedirs(pasta_saida, exist_ok=True)

    # Carrega os pesos treinados e coloca o modelo em modo de avaliacao
    modelo.load_state_dict(
        torch.load(caminho_pesos, map_location=dispositivo, weights_only=True)
    )
    modelo.to(dispositivo)
    modelo.eval()

    salvas = 0

    # Desativa o calculo de gradientes
    with torch.no_grad():
        for imagens, mascaras in loader_teste:
            imagens = imagens.to(dispositivo)

            # Gera logits, aplica Sigmoid (0 a 1) e transforma em mascara binaria (0 ou 1)
            logits = modelo(imagens)
            probs = torch.sigmoid(logits)
            preds = (probs > limiar).float()

            # Move para CPU/NumPy para plotar com o Matplotlib
            imgs_np = imagens.cpu().numpy()
            masks_np = mascaras.cpu().numpy()
            preds_np = preds.cpu().numpy()

            for i in range(imgs_np.shape[0]):
                if max_figuras is not None and salvas >= max_figuras:
                    print(f"{salvas} figuras salvas em: {pasta_saida}")
                    return

                img = imgs_np[i, 0]
                mask_real = masks_np[i, 0]
                mask_pred = preds_np[i, 0]

                dice = calcular_dice_fatia(mask_real, mask_pred)

                fig, axes = plt.subplots(1, 4, figsize=(16, 4))

                axes[0].imshow(img, cmap="gray")
                axes[0].set_title("RM Original")

                axes[1].imshow(mask_real, cmap="gray")
                axes[1].set_title("Mascara Manual")

                axes[2].imshow(mask_pred, cmap="gray")
                axes[2].set_title("Mascara Predita")

                # Sobreposicao na RM: verde = manual, vermelho = predita
                axes[3].imshow(img, cmap="gray")
                axes[3].imshow(
                    np.ma.masked_where(mask_real == 0, mask_real),
                    cmap="Greens",
                    alpha=0.5,
                    vmin=0,
                    vmax=1,
                )
                axes[3].imshow(
                    np.ma.masked_where(mask_pred == 0, mask_pred),
                    cmap="Reds",
                    alpha=0.5,
                    vmin=0,
                    vmax=1,
                )
                axes[3].set_title(f"Sobreposicao (Dice: {dice:.3f})")

                for ax in axes:
                    ax.axis("off")

                plt.tight_layout()
                caminho_png = os.path.join(pasta_saida, f"pred_teste_{salvas +