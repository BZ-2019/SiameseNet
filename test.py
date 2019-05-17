# -*- coding: UTF-8 -*-
import struct
# import matplotlib.pyplot as plt
with open('E:\\liuyuming\\无载高低\\列采\\沈阳局轨廓波形\\20190403140041京沈线上行.txt','rb') as file:
    while(1):
        line = file.read(96034)
        if len(line)!=96034:
            break
        s =struct.Struct('6f2i12000d2c')
        unpacked_data = s.unpack(line)
        print(unpacked_data)
        a = list(unpacked_data)
        # plt.scatter(a[10:100],a[3010:3100])
        with open('111.txt', 'w+') as f:
            for i in range(a[6]):
                f.write(str(a[i + 8])+'\t'+str(a[i+3000+8])+'\n')
            f.close()
file.close()
