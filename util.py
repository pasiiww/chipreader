import numpy as np
import cv2
import matplotlib as plt

def mysort(x,y):
    for i in range(len(x)-1):
        for j in range(len(x)-i-1):
            if abs(y[j] - y[j+1]) < 10 and x[j] > x[j+1]:
                x[j], x[j+1] = [x[j+1], x[j]]


def recoverlink(img, length=25):
    x_len = img.shape[0]
    y_len = img.shape[1]

    if length > x_len/2:
        raise 'error : ??????'

    img_c = img.copy()
    img_c[0, :] = np.sum(img[0:length, :], axis=0)
    for i in range(1, length):
        img_c[i, :] = img_c[i-1, :] + img[i+length, :]

    for i in range(length, x_len-length):
        img_c[i, :] = img_c[i-1, :] - img[i-length, :] + img[i+length, :]

    for i in range(x_len-length, x_len):
        img_c[i, :] = img_c[i-1, :] - img[i-length, :]

    img_d = img.copy()
    img_d[:, 0] = np.sum(img[:, 0:length], axis=1)
    for i in range(1, length):
        img_d[:, i] = img_d[:, i-1] + img[:, i+length]

    for i in range(length, y_len-length):
        img_d[:, i] = img_d[:, i-1] - img[:, i-length] + img[:, i+length]

    for i in range(y_len-length, y_len):
        img_d[:, i] = img_d[:, i-1] - img[:, i-length]

    img = img_c + img_d
    img = img > length
    return img


ver2order = {
    "1":((0,3,0,3),
         (0,0,2,2)),
    "2":((0,3,2,3,1,2,0,1),
         (0,0,1,1,2,2,3,3)),
    "3":((0,4,0,1,3,4,1,3),
         (0,0,1,1,1,1,2,2)),
    "4a":((0,3,0,1,3,4,1,4),
          (0,0,1,1,1,1,2,2)),
    "4b":((1,4,0,1,3,4,0,3),
          (0,0,1,1,1,1,2,2)),
    "5":((0,2,0,1,2,3,0,1,2,3,0,2),
         (0,0,1,1,1,1,2,2,2,2,3,3)),
    "6":((2,3,0,2,3,4,0,2,3,4,2,3),
         (0,0,1,1,1,1,2,2,2,2,3,3)),
    "7":((0,6,0,6),
         (0,0,1,1)),
    "8":((0,3.6,1,2.6,0,1,2.6,3.6),
         (0,0,1,1,2,2,2,2)),
    "9":((0,2,2,3,0,1,2,3,1,2),
         (0,0,1,1,2,2,2,2,3,3)),
    "Fa":((1,3,0,1,2,3,0,1,1,2),
          (0,0,1,1,1,1,2,2,3,3)),
    "Fb":((0,2,0,1,2,3,2,3,1,2),
          (0,0,1,1,1,1,2,2,3,3)),
    "Na":((0,3,0,2,3,4,2,4),
          (0,0,1,1,1,1,2,2)),
    "Nb":((1,4,0,1,2,4,0,2),
          (0,0,1,1,1,1,2,2)),
    "T":((0,1,1,3,1,3,0,1),
         (0,0,1,1,2,2,3,3)),
    "W":((2,3,1,2,1,2,3,0,2),
         (0,0,1,1,2,2,2,3,3)),
    "X":((1,2,0,1,2,3,0,1,2,3,1,2),
         (0,0,1,1,1,1,2,2,2,2,3,3)),
    "Ya":((2,3,0,2,3,4,0,4),
          (0,0,1,1,1,1,2,2)),
    "Yb":((1,2,0,1,2,4,0,4),
          (0,0,1,1,1,1,2,2)),
    "52-1a":((0,3,2,3,0,2), #以下的芯片都是非最优类型 开头的数字为格子数
           (0,0,1,1,2,2)),
    "52-1b":((0,2,2,3,0,3),
           (0,0,1,1,2,2)),
    "52-2":((0,5,0,5),
           (0,0,1,1)),
    "52-3":((0,2,0,1,0,1,0,2),
           (0,0,1,1,2,2,3,3)),
    "52-4a":((0,2,0,1,2,3,1,3),
           (0,0,1,1,2,2,3,3)),
    "52-4b":((1,3,2,3,0,1,0,2),
            (0,0,1,1,2,2,3,3)),
    "52-5":((0,3,2,3,0,1),
           (0,0,1,1,3,3)),
    "52-6a":((0,4,1,4,0,1),
           (0,0,1,1,2,2)),
    "52-6b":((0,1,1,4,0,4),
            (0,0,1,1,2,2)),
    "4-1":((0,4,0,4),
          (0,0,1,1)),
    "4-2":((0,2,0,2),
          (0,0,2,2)),
    "4-3a":((0,1,1,3,0,3),
          (0,0,1,1,2,2)),
    "4-3b":((0,1,1,2,0,2),
           (0,0,2,2,3,3)),
    "4-4a":((1,3,0,1,2,3,0,2),
           (0,0,1,1,1,1,2,2)),
    "4-4b":((0,2,0,1,2,3,1,3),
           (0,0,1,1,1,1,2,2)),
    "4-5":((1,2,0,1,2,3,0,3),
          (0,0,1,1,1,1,2,2)),
    "3-1":((0,1,1,2,0,2),
          (0,0,1,1,2,2)),
    "3-2":((0,1,0,1),(0,0,3,3))
}

vec_dict = [dict() for i in range(4)]


def turn90(x,y):
    new_x = y.max() - y
    new_y = x
    return new_x,new_y


for k,v in ver2order.items():
    gt_width = max(v[0])-min(v[0])
    gt_x = np.array(v[0])/gt_width
    gt_y = np.array(v[1])/gt_width
    for i in range(4):
        vec_dict[i][k] = (gt_x,gt_y)
        gt_x, gt_y = turn90(gt_x,gt_y)
        gt_width = gt_x.max() - gt_x.min()
        gt_x = gt_x/gt_width
        gt_y = gt_y/gt_width



def get_type_from_ver(x,y):
    mysort(x, y)
    ver_x = x[:1]
    ver_y = y[:1]
    for i in range(1, len(x)):
        if -10 < x[i] - x[i - 1] < 10 and -10 < y[i] - y[i - 1] < 10:
            continue
        else:
            ver_y = np.insert(ver_y, len(ver_y), y[i], axis=0)
            ver_x = np.insert(ver_x, len(ver_x), x[i], axis=0)

    width = ver_x.max() - ver_x.min()
    ver_x = (ver_x - ver_x.min()) / width
    ver_y = (ver_y - ver_y.min()) / width

    L1D = 1000
    name = ""
    t = 0
    calu_piont = len(ver_x)
    for i in range(4):
        #print(i)
        for k, v in vec_dict[i].items():
            gt_x , gt_y = v
            if calu_piont < len(gt_x) - 1:
                # 一般来说被计算的点都会比实际的点多。。。留个1当误差吧
                continue
            distance = 0
            for j in range(len(ver_x)):
                d = np.min(np.abs(gt_x-ver_x[j]) + np.abs(gt_y-ver_y[j]))
                distance += d
            if distance < L1D :
                L1D = distance
                name = k
                t = i *90
    return name, t, L1D/calu_piont


def process_icon(icon,is_number=False,esp=240):
    #cv2.imshow('xx',icon)
    #cv2.waitKey(1000)
    icon = np.sum(icon,axis=2) > esp
    if not is_number:
        icon = 1-icon
    x_sum = np.sum(icon,axis=0)
    y_sum = np.sum(icon,axis=1)
    if not is_number:
        x_w = np.where((x_sum > 2)*(x_sum < 49))
        y_w = np.where((y_sum > 2)*(y_sum < 49))
        if not len(x_w[0]) or not len(y_w[0]):
            return icon
    else:
        x_w = np.where((x_sum > 0))
        y_w = np.where((y_sum > 0))
        if not len(x_w[0]) or not len(y_w[0]):
            return icon

    ys = np.min(y_w)
    ye = np.max(y_w)
    xs = np.min(x_w)
    xe = np.max(x_w)
    return icon[ys:ye+1, xs:xe+1]


#icons = np.load('icons.npy')
'''
ord type shape white scale
0:  精度 37X29 297   0.2768
1:  装填 32X29 401   0.4321
2:  火力 33X39 664   0.5159
3:  穿甲 34X38 459   0.3553
'''
icons_para = [
    [37,29,297,0.2768],
    [32,29,401,0.4321],
    [33,39,664,0.5159],
    [34,38,459,0.3553]
    ]

def get_icon_type(icon):
    #TODO: .....
    icon_types = []
    for i in icon:
        #cv2.imshow('xxx',i)
        #cv2.waitKey(1000)
        type_id = -1
        i = process_icon(i,False,300)
        #cv2.imshow('xx0',i.astype('float32'))
        #cv2.waitKey(1999)

        sum_white = np.sum(i)
        #print(sum_white)
        scale = sum_white / i.shape[0] / i.shape[1]
        if i.shape == (50,50) or scale > 0.9 or i.shape[0] + i.shape[1] < 40:
            type_id = -1
        else:
            min_score = 30
            min_id = 0
            id = 0
            for para in icons_para:
                score = abs(i.shape[0] - para[0]) + \
                    abs(i.shape[1] - para[1]) + \
                    0.1 * abs(sum_white - para[2]) + \
                    100 * abs(scale - para[3])
                #print(score)
                if score < min_score:
                    min_score = score
                    min_id = id
                id +=1
            if min_score < 30:
                type_id = min_id

        icon_types.append(type_id)
    #print(icon_types)
    return icon_types

numbers = np.load('numbers.npy')
def get_single_num(num):
    if num.shape[1] == 0:
        return -1
    value1 = (num.shape[0]/num.shape[1]) > 2.7
    #print(num.shape,value1)
    num = cv2.resize(num.astype('uint8'),(18,36)).astype('bool')
    mindiff = 800
    id = -1
    for i in range(10):
        if i == 1 and not value1:
            continue
        diff = num ^ numbers[i]
        diff = np.sum(diff)
        if diff< mindiff:
            mindiff=diff
            id = i
    if id != -1 and value1:
        id =1
    #print(id)
    return id

def get_number(num):
    #cv2.imshow('xxx',num)
    #cv2.waitKey(1999)
    num = process_icon(num,True)
    num_white = np.sum(num,axis=0)
    idxs = np.where(num_white == 0)[0]
    width = num.shape[1]/num.shape[0]
    if len(idxs) > 0:
        idx = np.min(idxs)
        idxe = np.max(idxs)
        num1 = num[:,0:idx]
        num2 = num[:,idxe+1:]
        num1 = get_single_num(num1)
        num2 = get_single_num(num2)
        if num1 == -1 or num2 ==-1 or num1 == 0:
            raise Exception('error when processing number')

        return num1 * 10 + num2
    else:
        #print(width)
        if width > 0.63:
            idx = 12
            num1 = num[:,0:idx]
            num2 = num[:,idx+1:]
            num1 = get_single_num(num1)
            num2 = get_single_num(num2)
            if num1 == -1 or num2 ==-1 or num1 == 0:
                raise Exception('error when processing number')

            return num1 * 10 + num2
        num = get_single_num(num)
        if num <= 0 :
            raise Exception('error when processing number')
        return num

def get_inter_num(num):
    num = process_icon(num,True,esp=600)
    num_white = np.sum(num,axis=0)
    idx = 14
    #idx = np.min(np.where(num_white == 0))
    num1_start = 0
    num1_end = 0
    num2_start = 0
    for i in range(idx,len(num_white)):
        if not num1_start and num_white[i] > 0 :
            num1_start = i
        if num1_start and not num1_end and num_white[i] ==0:
            num1_end = i
        if num1_end and num_white[i]>0 and not num2_start:
            num2_start = i

    if num2_start != 0:
        num1 = num[:,num1_start:num1_end]
        num1 = get_single_num(num1)

        num2  = num[:,num2_start:]
        num2 = get_single_num(num2)
        if num1 == -1 or num2 ==-1 or num1 == 0:
            raise Exception('error processing number')
        return num1 * 10 + num2
    else:
        num1 = num[:,num1_start:]
        num1 = get_single_num(num1)
        if num1 <= 0 :
            raise Exception('error processing number')
        return num1


