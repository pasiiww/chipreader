import os
import xlwt
import chipreader as cr

pic_path = './chips_cut/'
chiptypes = ['精度','装填','杀伤','破甲']

book = xlwt.Workbook(encoding='utf-8', style_compression=0)
sheet = book.add_sheet('chips', cell_overwrite_ok=True)
sheet.write(0,0,'文件')
sheet.write(0,1,'序号')
sheet.write(0,2,'形状')
sheet.write(0,3,'颜色')
sheet.write(0,4,'强化等级')
sheet.write(0,5,'杀伤')
sheet.write(0,6,'破防')
sheet.write(0,7,'精度')
sheet.write(0,8,'装填')

num = 1
for name in os.listdir(pic_path):
    regions = cr.read_full_pic(pic_path+name)
    print(regions)
    for i in range(len(regions)):
        color, chiptype, equip, attr, inter = regions[i]

        sheet.write(num,0,name)
        sheet.write(num,1,str(i))
        sheet.write(num,2,chiptype)
        sheet.write(num,3,color)
        sheet.write(num,4,str(inter))
        sheet.write(num,5,str(attr[2]))
        sheet.write(num,6,str(attr[3]))
        sheet.write(num,7,str(attr[0]))
        sheet.write(num,8,str(attr[1]))

        #sheet.write(num,4,str(t))
        num +=1

save_path = './out.xls'
book.save(save_path)
