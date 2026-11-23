# Resultados dos Experimentos

Esta seção apresenta a consolidação dos resultados quantitativos obtidos na avaliação dos 4 cenários experimentais da arquitetura U-Net para a segmentação automática do Ligamento Cruzado Anterior (LCA).

## Tabela Comparativa Final
Os modelos treinados nos 5 folds da validação cruzada foram avaliados sobre o mesmo conjunto de teste fixo (composto por 4 exames independentes). Os valores abaixo representam a média e o desvio padrão das métricas Dice Score e IoU obtidas pelos 5 modelos de cada cenário:

| Cenário Experimental | Dice (Média ± DP) | IoU (Média ± DP) |
| :--- | :---: | :---: |
| **U-Net Zero (Original)** | 0.6775 ± 0.0413 | 0.5187 ± 0.0452 |
| **U-Net Zero (Com Augmentation)** | 0.5726 ± 0.0715 | 0.4309 ± 0.0652 |
| **U-Net Pré-treinada (Original)** | 0.5710 ± 0.0916 | 0.4357 ± 0.0898 |
| **U-Net Pré-treinada (Com Augmentation)** | 0.6124 ± 0.0683 | 0.4680 ± 0.0577 |

## Visualizações Gráficas
Os gráficos gerados durante a análise comparativa encontram-se salvos na pasta `Graficos`.
