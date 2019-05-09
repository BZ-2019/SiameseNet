import glob
import os
root = 'E:\\liuyuming\\SiameseNet\\DATA\\WH180605\\L\\'
dataset = 'VAL1'
preddir=root+'restxt\\'+dataset
gtdir=root+'gttxt\\'+dataset
IOU_value=0.5
TurePositive = 0
FalsePositive = 0
dectecount = 0
def compute_iou(rec1, rec2):
    """
    computing IoU
    :param rec1: (y0, x0, y1, x1), which reflects
            (top, left, bottom, right)
    :param rec2: (y0, x0, y1, x1)
    :return: scala value of IoU
    """
    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

    # computing the sum_area
    sum_area = S_rec1 + S_rec2

    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])

    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return intersect / (sum_area - intersect)


txtlist=glob.glob(r'%s\*.txt' %gtdir)
print(txtlist)
gt_rec_count={}
gt_dec_count={}
sum_gt_rec =0
sum_gt_dec =0
for path in os.listdir(gtdir):
    fgt = open(gtdir+'\\'+path)
    tempgt_rec_count=0

    for predictline in fgt.readlines():
        tempgt_rec_count+=1
        sum_gt_rec+=1
    gt_rec_count.update({path:tempgt_rec_count})
for path in os.listdir(preddir):
    if gtdir+'\\'+path in txtlist:
        fpre = open(preddir+'\\'+path)
        fgt = open(gtdir+'\\'+path)
        gtrec=[]
        for predictline in fgt.readlines():
            x1 = int(predictline.split(' ')[0])
            y1 = int(predictline.split(' ')[1])
            x2 = int(predictline.split(' ')[2])
            y2 = int(predictline.split(' ')[3])
            gtrec.append(tuple((y1,x1,y2,x2)))
        dectec_gt_indexlist = []
        for predictline in fpre.readlines():
            x1 = int(predictline.split('\000')[0])
            y1 = int(predictline.split('\000')[1])
            x2 = int(predictline.split('\000')[2])
            y2 = int(predictline.split('\000')[3])
            prerec = tuple((y1, x1, y2, x2))


            for index in range(len(gtrec)):
                IOU = compute_iou(gtrec[index],prerec)
                if IOU>IOU_value:
                    dectecount += 1
                    TurePositive+=1
                    if index not in dectec_gt_indexlist:
                        dectec_gt_indexlist.append(index)
                else:
                    dectecount += 1
                    FalsePositive+=1
        gt_dec_count.update({path:len(dectec_gt_indexlist)})
        sum_gt_dec+=len(dectec_gt_indexlist)
    else:
        fpre = open(preddir + '\\' + path)
        for predictline in fpre.readlines():
            dectecount += 1
            FalsePositive += 1

print("deteccount:{},TurePositive:{},FalsePositive:{}".format(dectecount,TurePositive,FalsePositive))
print("abnormalsum:{},abnormaldec:{}".format(sum_gt_rec,sum_gt_dec))



