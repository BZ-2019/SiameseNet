import argparse
import MyData_triple
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
import Siamesevggtriple
import os
parser = argparse.ArgumentParser(description='SiameseNet')
parser.add_argument('--save', type=str, default='./SiameseNet.pt',
                    help='path to save the final model')


parser.add_argument('--unuse-cuda', action='store_true',
                    help='unuse cuda')

parser.add_argument('--lr', type=float, default=0.0001)   #20190401 0.000001
parser.add_argument('--epochs', type=int, default=1,
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
parser.add_argument('--root', type=str, default='E:\\liuyuming\\SiameseNet\\DATA\\WH180605\\L\\')
parser.add_argument('--GPU', type=int, default=1)
parser.add_argument('--Train', type=bool, default=False)
args = parser.parse_args()
if args.Train:
    dataset = 'TRAIN'
else:
    dataset = 'VAL2'

rawpicroot = args.root+'regionraw/'
turepicroot = args.root+'regionture/'
falsepicroot = args.root+'regionfalse/'
RegionListTrain = args.root+'regionlist/'+dataset+'.txt'
RegionListVal = args.root+'regionlist/'+dataset+'.txt'
RegionListTest = args.root+'regionlist/'+dataset+'.txt'

edgeboxregionroot = args.root + 'edgeboxlabel/'
testimglist = args.root + 'testlist/'+dataset+'.txt'
predictlist = args.root + 'predict/'+dataset+'.txt'
if dataset == 'VAL' or dataset == 'TEST' or dataset == 'ALL':
    testimgroot = args.root + 'DATASET/'+'MergeImg'+dataset+'/'
else:
    testimgroot = args.root + 'DATASET/' + dataset + '/'
resroot = args.root + 'resroot/'+dataset+'/'
predtxtroot = args.root + 'restxt/'+dataset+'/'
use_cuda = torch.cuda.is_available() and not args.unuse_cuda
if args.Train:
    torch.cuda.set_device(args.GPU)
else:
    torch.cuda.set_device(args.GPU+1)
transform = transforms.Compose([
    transforms.ToTensor(), #将图片转换为Tensor,归一化至[0,1]
    transforms.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5]) #标准化至[-1,1]
])
print("迭代%d次，小规模数据，adam lr:%f, batch:%d"%(args.epochs, args.lr, args.batch_size))



args.dropout = args.dropout if args.augmentation else 0.

if args.Train==True:
    train_data=MyData_triple.MyDataset(rawroot=rawpicroot,tureroot=turepicroot,falseroot=falsepicroot,datatxt=RegionListTrain, transform=transform)
    data_loader = DataLoader(train_data, batch_size=args.batch_size,shuffle=True)
    print(len(data_loader))

    # val_data=MyData_triple.MyDataset(rawroot=rawpicroot,tureroot=turepicroot,falseroot=falsepicroot,datatxt=RegionListVal, transform=transform)
    # val_loader = DataLoader(val_data, batch_size=128,shuffle=True)
    # print(len(val_loader))
else:
    test_data=MyData_triple.TestDataset(rawroot=rawpicroot,tureroot=turepicroot,falseroot=falsepicroot,datatxt=RegionListTest, transform=transform)
    test_loader = DataLoader(test_data, batch_size=1,shuffle=False)
    print(len(test_loader))

# ##############################################################################
# Build model
# ##############################################################################



Siamesenet = Siamesevggtriple.vgg16()

if args.Train==True:
    model_dict = Siamesenet.state_dict()
    vgg = models.vgg16(pretrained=False)
    #pretrained_dict = vgg.state_dict()
    #pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#
    #model_dict.update(pretrained_dict)
    #Siamesenet.load_state_dict(model_dict)
    Siamesenet._initialize_weights()
    #Siamesenet = torch.nn.DataParallel(Siamesenet, device_ids=[1, 2, 3])
else:
    Siamesenet.load_state_dict(torch.load(args.save)['model'])
if use_cuda:
    Siamesenet = Siamesenet.cuda()


optimizer = torch.optim.Adam(Siamesenet.parameters(), lr=args.lr)
criterion = torch.nn.TripletMarginLoss()
L1loss = torch.nn.L1Loss()
# Siamloss = Siamesevggtriple.ContrastiveLoss()
testtransform =transforms.ToTensor()
# ##############################################################################
# Training
# ##############################################################################



def train():
    corrects = total_loss = 0
    for i, (rawdata, turedata, falsedata) in enumerate(data_loader):
        rawdata = Variable(rawdata)
        turedata = Variable(turedata)
        falsedata = Variable(falsedata)

        if use_cuda:
            rawdata, turedata, falsedata = rawdata.cuda(),turedata.cuda(), falsedata.cuda()

        out1,out2,out3 = Siamesenet(rawdata,turedata,falsedata)
        loss = criterion(out1,out2,out3)
        #loss = Siamloss(rawout,pariout,label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.data
        print(loss.data)
    return total_loss/len(data_loader)

# def val():
#     corrects = total_loss = 0
#     for i, (rawdata, turedata, falsedata) in enumerate(val_loader):
#         paridata, label = Variable(paridata), Variable(label)
#         rawdata = Variable(rawdata)
#         if use_cuda:
#             rawdata, paridata, label = rawdata.cuda(),paridata.cuda(), label.cuda().float()
#         #rawout,pariout = Siamesenet(rawdata,paridata)
#         out = Siamesenet(rawdata, paridata)
#
#         loss = criterion(out, label)
#         total_loss += loss.data
#         #corrects += (torch.max(target, 1)[1].view(label.size()).data == label.data).sum()
#         if i>10:
#             break
#     return total_loss//10

def test(f):
    corrects = total_loss = 0
    count =0
    Siamesenet.eval()
    for i, (rawdata, turedata, falsedata,label,fn) in enumerate(test_loader):
        rawdata = Variable(rawdata)
        turedata = Variable(turedata)
        falsedata = Variable(falsedata)

        if use_cuda:
            rawdata, turedata, falsedata = rawdata.cuda(),turedata.cuda(), falsedata.cuda()

        out1,out2,out3 = Siamesenet(rawdata,turedata,falsedata)
        euclidean_distance = F.pairwise_distance(out1, out2)
        scorewrite=euclidean_distance.clone()
        L2_threshold=1.1
        euclidean_distance[euclidean_distance <=L2_threshold] = 0
        euclidean_distance[euclidean_distance > L2_threshold] = 1
        predicttemp=euclidean_distance
        #rawout,pariout = Siamesenet(rawdata,paridata)
        #euclidean_distance = F.pairwise_distance(rawout, pariout).cpu()
        #scorewrite = euclidean_distance.detach().numpy().copy()
        #euclidean_distance[np.where(euclidean_distance<=1.5)] = 0
        #euclidean_distance[np.where(euclidean_distance > 1.5)] = 1
        #scorewrite = out.detach().cpu().numpy().copy()
        scorewrite = scorewrite.detach().cpu().numpy()
        corrects +=(predicttemp== 0).cpu().sum().numpy()
        count +=rawdata.shape[0]

        temp = np.where(predicttemp.cpu() == 1)[0]
        for a in range(len(temp)):
            print(fn[temp[a]] + '_' + str(scorewrite[temp[a]]) + '\n')
            f.write(fn[temp[a]]+'_'+str(scorewrite[temp[a]])+'_'+'\n')
    return count, corrects

def readedgexml():

    edgelabeldict={}
    with open(testimglist , 'r') as ftl:
        for line in ftl.readlines():
            if not line:
                break
            labellist=[]
            try:
                with open(edgeboxregionroot + line[0:5]+'.txt', 'r') as fxml:
                    for labelline in fxml.readlines():
                        if not labelline:
                            break
                        labellist.append(labelline)
                if line[-1]=='\n':
                    edgelabeldict[line] = labellist
                else:
                    edgelabeldict[line + '\n'] = labellist
            except:
                continue

    return edgelabeldict

def predict():
    edgelabeldict = readedgexml()

    img=[]
    lastfilename=[]
    # os.remove(predtxtroot)
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
            x2 = x+w
            y2 = y+h

            if img==[] or filename != lastfilename:

                #cv2.imshow('a', img)
                if(lastfilename):
                    cv2.imwrite(resroot + lastfilename + '.jpg', img)
                    predtxtfile.close()
                predtxtfile = open(predtxtroot + filename + '.txt', 'a+')
                img = cv2.imread(testimgroot + filename + '.JPG')
                cv2.rectangle(img, (x, y), (x + w, y + h), 255, 2)
                cv2.putText(img,str(number)+'_'+str(score),(x,y+20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
                predtxtfile.write(str(x) + '\000' + str(y) + '\000' + str(x2) + '\000' + str(y2) + '\n')
            else:
                cv2.rectangle(img, (x, y), (x + w, y + h), 255, 2)
                cv2.putText(img,str(number)+'_'+str(score),(x,y+20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
                predtxtfile.write(str(x) + '\000' + str(y) + '\000' + str(x2) + '\000' + str(y2) + '\n')

            lastfilename = filename
        if (lastfilename):
            cv2.imwrite(resroot + lastfilename + '.jpg', img)
            predtxtfile.close()

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
            print("begin")
            loss = train()
            print("*****************************************************&*********")
            print(epoch,loss)
            if (epoch%5==0):               #print("val_loss: {:5.2f}".format(val_loss))
                model_state_dict = Siamesenet.state_dict()
                model_source = {
                    "settings": args,
                    "model": model_state_dict
                }
                model_name = 'SiameseNet' + '_' + str(epoch) + '.pt'
                torch.save(model_source, model_name)
            print("***************************************************************")

        model_state_dict = Siamesenet.state_dict()
        model_source = {
            "settings": args,
            "model": model_state_dict
        }
        torch.save(model_source, args.save)
    else:
        f = open(predictlist,'w')
        count,result = test(f)
        f.close()
        predict()
        #print("test done.regioncount:%d,predict:%d, acc = %.4f"%(count,count - result,result/count))


except KeyboardInterrupt:
    print("-"*90)
    print("Exiting from training early | cost time: {:5.2f}min".format((time.time() - total_start_time)/60.0))

