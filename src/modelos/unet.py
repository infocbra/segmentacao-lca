import torch
import torch.nn as nn

class BlocoDuplaConv(nn.Module):
    def __init__(self, canais_entrada, canais_saida):
        super().__init__()
        self.bloco = nn.Sequential(
            nn.Conv2d(canais_entrada, canais_saida, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(canais_saida),
            nn.ReLU(inplace=True),
            nn.Conv2d(canais_saida, canais_saida, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(canais_saida),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.bloco(x)

class UNet(nn.Module):
    def __init__(self, canais_entrada=1, canais_saida=1, filtros_base=64):
        super().__init__()
        f = filtros_base

        # Encoder (caminho de contracao: extrai caracteristicas e reduz resolucao)
        self.enc1 = BlocoDuplaConv(canais_entrada, f)
        self.enc2 = BlocoDuplaConv(f, f * 2)
        self.enc3 = BlocoDuplaConv(f * 2, f * 4)
        self.enc4 = BlocoDuplaConv(f * 4, f * 8)

        # Reduz o tamanho da imagem pela metade
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck (base do "U", onde a representacao e mais compacta e profunda)
        self.bottleneck = BlocoDuplaConv(f * 8, f * 16)

        # Decoder 
        self.up4 = nn.ConvTranspose2d(f * 16, f * 8, kernel_size=2, stride=2)
        self.dec4 = BlocoDuplaConv(f * 16, f * 8)

        self.up3 = nn.ConvTranspose2d(f * 8, f * 4, kernel_size=2, stride=2)
        self.dec3 = BlocoDuplaConv(f * 8, f * 4)

        self.up2 = nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2)
        self.dec2 = BlocoDuplaConv(f * 4, f * 2)

        self.up1 = nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
        self.dec1 = BlocoDuplaConv(f * 2, f)

        self.saida = nn.Conv2d(f, canais_saida, kernel_size=1)

    def forward(self, x):
        # Descida
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))

        # Base 
        b = self.bottleneck(self.pool(s4))

        # Subida
        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, s4], dim=1))
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, s3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, s2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, s1], dim=1))

        return self.saida(d1)


# Teste de dimensoes 
if __name__ == "__main__":
    modelo = UNet(canais_entrada=1, canais_saida=1)
    entrada = torch.randn((2, 1, 256, 256))
    saida = modelo(entrada)
    print(f"Entrada: {entrada.shape} -> Saida: {saida.shape}")