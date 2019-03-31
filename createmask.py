import cv2
import numpy
import os
import xml.etree.ElementTree as ET
fileroot = "C:\\Users\\lyming\\Desktop\\SFCdata\\mask\\"
rawfile = os.listdir(fileroot)
for fileindex in rawfile:
    filename = fileroot+fileindex
    rawpic = cv2.imread(filename)
    if filename[-3:]=='jpg':
        #cv2.imshow('a',rawpic)
        #cv2.waitKey(0)
        xmlfilename = filename[:-3]+'xml'
        tree = ET.parse(xmlfilename)
        rect = {}
        line = ""
        root = tree.getroot()
        for name in root.iter('path'):
            rect['path'] = name.text
        proposalnum = 0
        for ob in root.iter('object'):
            proposalnum = proposalnum+1;
        for ob in root.iter('object'):
            for bndbox in ob.iter('bndbox'):
                # for l in bndbox:
                #     print(l.text)
                for xmin in bndbox.iter('xmin'):
                    rect['xmin'] = xmin.text
                for ymin in bndbox.iter('ymin'):
                    rect['ymin'] = ymin.text
                for xmax in bndbox.iter('xmax'):
                    rect['xmax'] = xmax.text
                for ymax in bndbox.iter('ymax'):
                    rect['ymax'] = ymax.text
                rawpic[int(ymin.text):int(ymax.text),int(xmin.text):int(xmax.text)] = 0
        #cv2.imshow('a',rawpic)
        #cv2.waitKey(0)
        savename = 'C:\\Users\lyming\Desktop\SFCdata\mask\maskdata\\'+fileindex
        cv2.imwrite(savename,rawpic)


