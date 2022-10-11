import numpy as np
import torch
from torchvision.datasets import mnist
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import time
batch_size=256
test_dataset = mnist.MNIST(root='../test', train=False, transform=ToTensor())
test_loader = DataLoader(test_dataset, batch_size=batch_size)
def calc():
    all_correct_num = 0
    all_sample_num = 0
    model=torch.load("mnist_0.97720.pkl")
    model = model.to("cpu")
    model.eval()
    t_ls=[]
    fps_ls=[]
    for idx, (test_x, test_label) in enumerate(test_loader):
        test_x = test_x.to('cpu')
        start = time.time()
        predict_y = model(test_x.float()).cpu().detach()
        end = time.time()
        total_time = (end - start)
        if total_time!=0:
            fps=1/(total_time)
            fps_ls.append(fps)
        t_ls.append(total_time)
        predict_y = np.argmax(predict_y, axis=-1)
        current_correct_num = predict_y == test_label
        all_correct_num += np.sum(current_correct_num.numpy(), axis=-1)
        all_sample_num += current_correct_num.shape[0]
    t=sum(t_ls)
    fps=sum(fps_ls)/len(fps_ls)
    avg_time=t/len(t_ls)
    acc = all_correct_num / all_sample_num
    return t,avg_time,fps,acc
a=calc()
print("FPS :",a[2]," Acc :",a[3])