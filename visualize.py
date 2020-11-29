import matplotlib.pyplot as plt
from torchvision.utils import make_grid



def show_tensor_images(image_tensor, num_images=25, size=(3, 32, 32), nrow=5, show=True):
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=nrow)
    plt.imshow(image_grid.permute(1,2,0).squeeze())
    if show:
        plt.show()