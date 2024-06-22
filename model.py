import timm
import torch.nn as nn

class ExampleModel(nn.Module):
    """
    timm içerisinde restnet18, efficientnet, restnext gibi önceden kurgulanmış
    modeller var.
    inherit
    """
    def __init__(self, num_classes):
        super(ExampleModel, self).__init__()
        self.base_model = timm.create_model("efficientnet_b0", pretrained=True)
        self.features = nn.Sequential(*list(self.base_model.childiren())[:-1])
        # Hidden statelerin hepsini alıp, son layerı değiştiyorum. Burada bir transfer learning
        # yapıyoruz çünkü. clasification modelinieldeki veri fine tune etmiş olacağız.

        network_out_size = 1280

        self.classifier == nn.Sequential(nn.Flatten(), nn.linear(network_out_size, num_classes))

        def forward(self, x):
            x = self.features(x)
            output = self.classifier(x)
            return output



