import argparse
import MyData
import torch
import model
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.models as models
import cv2
import numpy as np
import time
from torch.autograd import Variable
import torch.nn.functional as F
parser = argparse.ArgumentParser(description='DenseNet')
parser.add_argument('--save', type=str, default='./DenseNet.pt',
                    help='path to save the final model')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--unuse-cuda', action='store_true',
                    help='unuse cuda')

parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--epochs', type=int, default=5,
                    help='number of epochs for train')
parser.add_argument('--batch-size', type=int, default=1,
                    help='batch size for training')
parser.add_argument('--weight-decay', type=float, default=1e-4)
parser.add_argument('--momentum', type=float, default=.9)
parser.add_argument('--augmentation', action='store_true')
parser.add_argument('--dropout', type=float, default=0.)

parser.add_argument('--num-class', type=int, default=2)
parser.add_argument('--growth-rate', type=int, default=12)
parser.add_argument('--channels', type=int, default=1)
parser.add_argument('--rawpicroot', type=str, default='C:\\Users\\lyming\\Desktop\\SFCdata\\raw\\')
parser.add_argument('--maskpicroot', type=str, default='C:\\Users\\lyming\\Desktop\\SFCdata\\mask\\maskdata\\')
parser.add_argument('--Train', type=bool, default=False)
args = parser.parse_args()

transform=transforms.Compose([
    transforms.ToTensor(), #将图片转换为Tensor,归一化至[0,1]
    transforms.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5]) #标准化至[-1,1]
])


use_cuda = torch.cuda.is_available() and not args.unuse_cuda
args.dropout = args.dropout if args.augmentation else 0.

train_data=MyData.MyDataset(maskroot=args.maskpicroot,rawroot=args.rawpicroot,datatxt='list.txt', transform=transform)
data_loader = DataLoader(train_data, batch_size=1,shuffle=True)
print(len(data_loader))

# ##############################################################################
# Build model
# ##############################################################################


vggnet = models.vgg11(pretrained=True)
fc_features=vggnet.fc.in_features
vggnet.fc=torch.nn.Linear(fc_features)
SFCNET = model.SFCNET(args)
if args.Train==True:
    SFCNET._initialize_weights()
else:
    SFCNET.load_state_dict(torch.load(args.save)['model'])
if use_cuda:
    SFCNET = SFCNET.cuda()
optimizer = torch.optim.Adadelta(SFCNET.parameters(), lr=args.lr)
criterion = torch.nn.CrossEntropyLoss()
L1loss = torch.nn.L1Loss()
testtransform =transforms.ToTensor()
# ##############################################################################
# Training
# ##############################################################################



def train():
    corrects = total_loss = 0
    for i, (rawdata, maskdata, label) in enumerate(data_loader):
        maskdata, label = Variable(maskdata), Variable(label)
        rawdata = Variable(rawdata)
        if use_cuda:
            rawdata, maskdata, label = rawdata.cuda(),maskdata.cuda(), label.cuda()

        QI = vggnet(rawdata)
        QIm = vggnet(maskdata)
        QI = QI.view(1,2).detach()
        QIm = QIm.view(1, 2)
        #label = label.view(2)
        losscla = criterion(QIm, label)
        lossdis = L1loss(QIm, QI)
        #loss = losscla+0.5*lossdis
        loss = losscla
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.data
        #corrects += (torch.max(target, 1)[1].view(label.size()).data == label.data).sum()

    return total_loss[0]

def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in range(112, image.shape[0]-112, stepSize):
		for x in range(112, image.shape[1]-112, stepSize):
		    yield (x, y, image[y - 112:y + 112, x- 112:x + 112])

def test():
    testimg = cv2.imread("C:\\Users\lyming\Desktop\SFCdata\\raw\\1.jpg")
    testimgshape = np.shape(testimg)
    result = np.zeros((testimgshape))
    for [x,y,testdata] in sliding_window(testimg,2,(224,224)):
        print(x, y, testdata.shape)
        testdata = Variable(testtransform(testdata))
        testdata = testdata.cuda()
        testdata = testdata.unsqueeze(0)
        testdata = F.softmax(SFCNET(testdata).view(2,),dim=0)
        result[y,x,0:2] = testdata.detach().cpu().numpy()
    return result[:,:,0:2]
# ##############################################################################
# Save Model
# ##############################################################################
best_acc = None
total_start_time = time.time()

try:
    print('-' * 90)
    if args.Train==True:
        for epoch in range(1, args.epochs+1):
            epoch_start_time = time.time()
            loss= train()
            print(epoch,loss)
            #optimizer.update_learning_rate()
    else:
        result = test()
        Pmap = result[:, :, 0]
        Nmap = result[:, :, 1]
        Pmap = cv2.GaussianBlur(Pmap, (10, 10), 15)*255
        Nmap = cv2.GaussianBlur(Nmap, (10, 10), 15)*255
        cv2.imwrite("Nmap.jpg",Nmap)
        cv2.imwrite("Pmap.jpg", Pmap)
    model_state_dict = SFCNET.state_dict()
    model_source = {
        "settings": args,
        "model": model_state_dict
    }
    torch.save(model_source, args.save)
except KeyboardInterrupt:
    print("-"*90)
    print("Exiting from training early | cost time: {:5.2f}min".format((time.time() - total_start_time)/60.0))

