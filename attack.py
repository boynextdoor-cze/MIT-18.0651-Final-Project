import torch
from tqdm import tqdm_notebook as tqdm
import torchvision.models as models
import torch.nn.functional as F

cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device)


def fgsm(x, class_id, epsilon):
    """
    :param x: input tensor.
    :param class_id: the id of the target class.
    :return: a tensor for the adversarial example
    """
    ground_truth = torch.zeros(logit.shape).to(device)
    ground_truth[:, class_id] = 1
    model.eval()
    for _ in tqdm(range(200)):
        logit = model(x)
        loss = F.cross_entropy(logit, ground_truth)
        gradient, = torch.autograd.grad(loss, x)
        sign_grad = gradient.sign()
        x += epsilon*sign_grad
        x = torch.clamp(x, 0, 1)
    return x
