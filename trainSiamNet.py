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
import Siamesevgg
parser = argparse.ArgumentParser(description='SiameseNet')
parser.add_argument('--save', type=str, default='./SiameseNet_80.pt',
                    help='path to save the final model')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--unuse-cuda', action='store_true',
                    help='unuse cuda')

parser.add_argument('--lr', type=float, default=0.00001)
parser.add_argument('--epochs', type=int, default=200,
                    help='number of epochs for train')
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch size for training')
parser.add_argument('--weight-decay', type=float, default=1e-4)
parser.add_argument('--momentum', type=float, default=.9)
parser.add_argument('--augmentation', action='store_true')
parser.add_argument('--dropout', type=float, default=0.)

parser.add_argument('--num-class', type=int, default=2)
parser.add_argument('--growth-rate', type=int, default=12)
parser.add_argument('--channels', type=int, default=1)
parser.add_argument('--root', type=str, default='/home/liuyuming/SiameseNet/数据/武汉180605/')
parser.add_argument('--rawpicroot', type=str, default='/home/liuyuming/SiameseNet/数据/武汉180605/L_regionraw/')
parser.add_argument('--turepicroot', type=str, default='/home/liuyuming/SiameseNet/数据/武汉180605/L_regionture/')
parser.add_argument('--falsepicroot', type=str, default='/home/liuyuming/SiameseNet/数据/武汉180605/L_regionfalse/')
parser.add_argument('--GPU', type=int, default=0)
parser.add_argument('--Train', type=bool, default=True)
args = parser.parse_args()

transform=transforms.Compose([
    transforms.ToTensor(), #将图片转换为Tensor,归一化至[0,1]
    transforms.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5]) #标准化至[-1,1]
])


use_cuda = torch.cuda.is_available() and not args.unuse_cuda
torch.cuda.set_device(1)

args.dropout = args.dropout if args.augmentation else 0.

if args.Train==True:
    train_data=MyData.MyDataset(rawroot=args.rawpicroot,tureroot=args.turepicroot,falseroot=args.falsepicroot,datatxt='/home/liuyuming/SiameseNet/数据/武汉180605/regionlisttrain.txt', transform=transform)
    data_loader = DataLoader(train_data, batch_size=args.batch_size,shuffle=True)
    print(len(data_loader))

    val_data=MyData.MyDataset(rawroot=args.rawpicroot,tureroot=args.turepicroot,falseroot=args.falsepicroot,datatxt='/home/liuyuming/SiameseNet/数据/武汉180605/regionlistval.txt', transform=transform)
    val_loader = DataLoader(val_data, batch_size=1,shuffle=False)
    print(len(val_loader))
else:
    test_data=MyData.TestDataset(rawroot=args.rawpicroot,tureroot=args.turepicroot,datatxt='/home/liuyuming/SiameseNet/数据/武汉180605/regionlistval.txt', transform=transform)
    test_loader = DataLoader(test_data, batch_size=16,shuffle=False)

# ##############################################################################
# Build model
# ##############################################################################



Siamesenet = Siamesevgg.vgg16()

if args.Train==True:
    model_dict = Siamesenet.state_dict()
    vgg = models.vgg16(pretrained=True)
    pretrained_dict = vgg.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    model_dict.update(pretrained_dict)
    Siamesenet.load_state_dict(model_dict)
    #Siamesenet._initialize_weights()
else:
    Siamesenet.load_state_dict(torch.load(args.save)['model'])
if use_cuda:
    Siamesenet = Siamesenet.cuda()
optimizer = torch.optim.Adam(Siamesenet.parameters(), lr=args.lr)
criterion = torch.nn.CrossEntropyLoss()
L1loss = torch.nn.L1Loss()
Siamloss = Siamesevgg.ContrastiveLoss()
testtransform =transforms.ToTensor()
# ##############################################################################
# Training
# ##############################################################################



def train():
    corrects = total_loss = 0
    for i, (rawdata, paridata, label) in enumerate(data_loader):
        paridata, label = Variable(paridata), Variable(label)
        rawdata = Variable(rawdata)
        if use_cuda:
            rawdata, paridata, label = rawdata.cuda(),paridata.cuda(), label.cuda().float()

        rawout,pariout = Siamesenet(rawdata,paridata)
        #print(out[0])
        #out = out.view(1,2).detach()
        #pariout = pariout.view(1, 2)
        #label = label.view(2)
        #losscla = criterion(out, label)
        #loss = losscla+0.5*lossdis
        loss = Siamloss(rawout,pariout,label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.data
        #print(loss.data)
        #corrects += (torch.max(target, 1)[1].view(label.size()).data == label.data).sum()

    return total_loss


def val():
    corrects = total_loss = 0
    for i, (rawdata, paridata, label) in enumerate(val_loader):
        paridata, label = Variable(paridata), Variable(label)
        rawdata = Variable(rawdata)
        if use_cuda:
            rawdata, paridata, label = rawdata.cuda(),paridata.cuda(), label.cuda().float()

        rawout,pariout = Siamesenet(rawdata,paridata)
        loss = Siamloss(rawout,pariout,label)

        total_loss += loss.data
        #corrects += (torch.max(target, 1)[1].view(label.size()).data == label.data).sum()

    return total_loss

def test(f):
    corrects = total_loss = 0
    for i, (rawdata, paridata, label,fn) in enumerate(test_loader):
        paridata, label = Variable(paridata), Variable(label)
        rawdata = Variable(rawdata)
        if use_cuda:
            rawdata, paridata, label = rawdata.cuda(),paridata.cuda(), label.cuda().float()

        rawout,pariout = Siamesenet(rawdata,paridata)
        euclidean_distance = F.pairwise_distance(rawout, pariout).cpu()
        scorewrite = euclidean_distance.detach().numpy().copy()
        euclidean_distance[np.where(euclidean_distance<=3)] = 0
        euclidean_distance[np.where(euclidean_distance > 3)] = 1
        corrects += (euclidean_distance== label.cpu().data).sum()
        temp = np.where(euclidean_distance >=1)[0]
        for a in range(len(temp)):
            print(fn[temp[a]] + '_' + str(scorewrite[temp[a]]) + '\n')
            f.write(fn[temp[a]]+'_'+str(scorewrite[temp[a]])+'_'+'\n')
    return i*16,corrects

def readedgexml():
    edgeboxregionroot = args.root + 'L_edgeboxlabel/'
    testimglist = args.root + 'testlist.txt'
    edgelabeldict={}
    with open(testimglist , 'r') as ftl:
        for line in ftl.readlines():
            if not line:
                break
            labellist=[]
            try:
                with open(edgeboxregionroot + line[0:-5]+'.txt', 'r') as fxml:
                    for labelline in fxml.readlines():
                        if not labelline:
                            break
                        labellist.append(labelline)
                edgelabeldict[line]=labellist
            except:
                continue

    return edgelabeldict

def predict():
    edgelabeldict = readedgexml()
    predictlist = args.root + 'predictabnormal.txt'
    testimgroot = args.root + 'L_test/'
    resroot = args.root + 'resroot/'
    img=[]
    lastfilename=[]
    with open(predictlist , 'r') as fpredict:
        for predictline in fpredict.readlines():
            if not predictline:
                break
            word = predictline.split('_')
            filename, number,score = word[0], int(word[1][0:-4]),float(word[2])
            [x, y, w, h] = edgelabeldict[filename+'.JPG'+'\n'][number].split()
            x = int(x)
            y = int(y)
            w = int(w)
            h = int(h)
            if img==[] or filename != lastfilename:
                #cv2.imshow('a', img)
                if(lastfilename):
                    cv2.imwrite(resroot + lastfilename + '.jpg', img)
                img = cv2.imread(testimgroot + filename + '.JPG')
                cv2.rectangle(img, (x, y), (x + w, y + h), 255, 2)
                cv2.putText(img,str(number)+'_'+str(score),(x,y+20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            else:
                cv2.rectangle(img, (x, y), (x + w, y + h), 255, 2)
                cv2.putText(img,str(number)+'_'+str(score),(x,y+20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            lastfilename = filename
        if (lastfilename):
            cv2.imwrite(resroot + lastfilename + '.jpg', img)

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
            print("*******************")
            print(epoch,loss)
            if(epoch%1==0):
                val_loss = val()
                print("val_loss: {:5.2f}".format(val_loss))
                model_state_dict = Siamesenet.state_dict()
                model_source = {
                    "settings": args,
                    "model": model_state_dict
                }
                model_name = 'SiameseNet' + '_' + str(epoch) + '.pt'
                torch.save(model_source, model_name)
            print("*******************")

        model_state_dict = Siamesenet.state_dict()
        model_source = {
            "settings": args,
            "model": model_state_dict
        }
        torch.save(model_source, args.save)
    else:
        #f = open('/home/liuyuming/SWINT/数据/武汉180605/predictabnormal.txt','w')
        #count,result = test(f)
        #f.close()
        predict()


except KeyboardInterrupt:
    print("-"*90)
    print("Exiting from training early | cost time: {:5.2f}min".format((time.time() - total_start_time)/60.0))

