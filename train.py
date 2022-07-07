import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets1 import Datasets
from Model.MiniIR import IRModel
from utils import Visualizer, compute_loss,ssim
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# h, w
IMG_SIZE = 256, 256
K = [0.01, 0.03]
L = 255
window_size = 7

vis = Visualizer(env="IR")




def train(model, epoch, dataloader, optimizer, criterion, scheduler):
    model.train()
    for itr, (image, hm) in enumerate(dataloader):
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            hm = hm.to(device)
            image = image.to(device)

        bs = image.shape[0]

        output = model(image)

        hm = hm.float()
        #output=output.float()
        #loss1=torch.sum(torch.abs(hm-output))
        loss = criterion(output, hm)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()

        if itr % 2 == 0:
            print("epoch:%2d|step:%04d|loss:%.6f" %
                  (epoch, itr, loss.item()/bs))
            vis.plot_many_stack({"train_loss": loss.item()/bs})


def test(model, epoch, dataloader, criterion):
    model.eval()
    sum_loss = 0
    n_sample = 0
    for itr, (image, hm) in enumerate(dataloader):
        if torch.cuda.is_available():
            hm = hm.cuda()
            image = image.cuda()

        output = model(image)
        hm = hm.float()

        loss1 = criterion(output, hm)

        loss=loss1

        sum_loss += loss.item()
        n_sample += image.shape[0]

    print("TEST: epoch:%02d-->loss:%.6f" % (epoch, sum_loss/n_sample))
    if epoch > 1:
        vis.plot_many_stack({"test_loss": sum_loss/n_sample})
    return sum_loss / n_sample


if __name__ == "__main__":
    Loss_list = []



    total_epoch = 200

    bs =2
    ########################################
    transforms_all = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4372, 0.4372, 0.4373],
                             std=[0.2479, 0.2475, 0.2485])
    ])

    datasets = Datasets(root_dir="./datanew", transforms=transforms_all)

    data_loader = DataLoader(datasets, shuffle=True,
                             batch_size=bs, collate_fn=datasets.collect_fn,drop_last=True)

    model = IRModel()


    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        model=model.to(device)
        #model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    criterion = torch.nn.MSELoss()  # compute_loss
    #criterion=torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=30,
                                                gamma=0.1)

    for epoch in range(total_epoch):
        train(model, epoch, data_loader, optimizer, criterion, scheduler)
        loss = test(model, epoch, data_loader, criterion)

        Loss_list.append(loss)

        # 我这里迭代了200次，所以x的取值范围为(0，200)，然后再将每次相对应的准确率以及损失率附在x上


        if epoch % 1 == 0:
            torch.save(model.state_dict(),
                       "weightstest4/epoch_%d_%.3f.pt" % (epoch, loss*10000))
    # x2 = range(0, 20)
    # y2 = Loss_list
    # plt.figure()
    # plt.plot(x2, y2, 'b.-')
    # plt.xlabel(' loss vs. epoches')
    # plt.ylabel('loss')
    # plt.show()
    # plt.savefig("loss.jpg")




