import cv2
import numpy as np
# from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import util


def get_section(x):
    kernel = np.array([1, 0, 1])
    x = np.convolve(x, kernel, 'same')

    max_ = np.max(x)
    segement = max_ / 3 * 2  # magic

    x = x > segement  # or not
    kernel = np.array([-1, 1])
    x = np.abs (np.convolve(x, kernel, 'same'))

    seg_index = np.where(x == 1)[0]
    kernel = np.array([1, -1])
    seg_distance = np.convolve(seg_index,kernel,'valid')
    max_distance= np.max(seg_distance)
    distance = max_distance / 3 * 2  # magic~
    start_index = np.where(seg_distance > distance)

    '''# maybe not 3 centers so kmeans not used
    max_idx = np.where(seg_distance == max_distance)[0]
    init_center = np.array([[min_distance],[min_distance+max_distance/10],[max_distance]])
    kmeans = KMeans(n_clusters=3, init=init_center,n_init=1).fit(seg_distance.reshape(-1,1)) 
    start_index = np.where ( kmeans.labels_ == kmeans.labels_[max_idx])
    '''

    return seg_index[start_index], max_distance - 1


# color unequip           equip
# white 243 243 243     153 153 153
# red   239 107 67      146 65 41
# blue  111 136 217      69  85 135
# inter 24,205,168     15,127,104
# size 264 428

# !!!! BGR IN OPENCV ????
red = (67,107,239)
blue = (217,136,111)
inter_color = (168,205,24)

esp = 31
MIN_white = 233 # 210
scale = 1.5882353


def recover_from_equip(equip_chip):
    equip_chip = np.multiply(equip_chip,scale)
    equip_chip = np.clip(equip_chip,0,243)
    return equip_chip.astype('uint8')


def get_type(gray):

    gray = np.uint8(gray)
    # how to once...
    wlinknum, labels = cv2.connectedComponents(gray,connectivity=4)
    blinknum, labels = cv2.connectedComponents(1-gray,connectivity=4)
    #print(wlinknum,blinknum)
    if wlinknum != 2 or blinknum != 3:
        # print('uncomplete') # or with noise
        # recover link masked (not work well)  and remove noise
        gray = np.int32(gray)
        gray = util.recoverlink(gray)
        gray = np.uint8(gray)

        '''maybe not need ..'''
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        gray = cv2.dilate(gray, kernel)
        gray = cv2.erode(gray, kernel)

    dst = cv2.cornerHarris(gray, 6, 5, 0.04)
    y,x = np.where(dst>0.1*dst.max())
    #gray[dst > 0.1 * dst.max()] = 255

    chip_type, t, diff = util.get_type_from_ver(x,y)
    #print(chip_type,t,diff)
    #cv2.imshow('iii',gray)
    #cv2.waitKey(1111)
    return chip_type, t, diff


boxs = ((10, 225, 60, 275),
        (138, 225, 188, 275),
        (10, 285, 60, 335),
        (138, 285, 188, 335))

wzboxs = ((70, 235, 120, 280),
         (200, 235, 250, 280),
         (70, 295, 120, 340),
         (200, 295, 250, 340))

inter_box = (220,245,260,285)
inter_box2 = (220,185,260,225)

def read_single_chip(chip):
    #cv2.imshow('xxx',chip)
    #cv2.waitKey(1000)
    equiped = False
    color_type = 'red'
    chip = cv2.resize(chip,(264,428))
    chip_whilt = np.sum(chip > MIN_white,axis=-1) == 3
    y_while = np.sum(chip_whilt,axis=1)
    if np.sum(y_while) < 3900 : # ?
        equiped = True
        chip = recover_from_equip(chip)
        chip_whilt = np.sum(chip > MIN_white,axis=-1) == 3
        y_while = np.sum(chip_whilt,axis=1)

    # ....... basechip 6,280
    y_bias = 295
    while y_while[y_bias] > 40:
        y_bias-=1
    y_bias = y_bias - 280
    x_while = np.sum(chip_whilt,axis=0)
    x_bias = 5
    while x_while[x_bias] > 40 and x_bias > 0:
        x_bias -= 1
    x_bias = x_bias - 5

    chip_core = chip[40+y_bias:270+y_bias,:]
    chip_core_line = np.sum(np.abs(chip_core -red),axis=-1) < esp
    if np.sum(chip_core_line) < 15:
        color_type = 'blue'
        chip_core_line = np.sum(np.abs(chip_core -blue),axis=-1) < esp

    core_type, t, type_diff = get_type(chip_core_line)

    icons = [ chip[box[1]+ y_bias:box[3]+y_bias, box[0]+x_bias:box[2]+x_bias, :] for box in boxs]
    nums = [ chip[box[1]+ y_bias:box[3]+y_bias, box[0]+x_bias:box[2]+x_bias, :] for box in wzboxs]

    icons_nums = 0
    attr_nums = [0 for i in range(4)]
    icons_types = util.get_icon_type(icons)
    for i in range(len(icons_types)):
        if icons_types[i] != -1:
            icons_nums += 1
            num = util.get_number(nums[i])
            attr_nums[icons_types[i]] = num

    if icons_nums <=2 :
        inter_class = chip[inter_box[1]+ y_bias:inter_box[3]+y_bias, inter_box[0]+x_bias:, :]
        inter_chip = np.sum( np.sum(np.abs(inter_class-inter_color),axis=2) < esp)
    else:
        inter_class = chip[inter_box2[1]+ y_bias:inter_box2[3]+y_bias, inter_box2[0]+x_bias:, :]
        inter_chip = np.sum( np.sum(np.abs(inter_class-inter_color),axis=2) < esp)

    inter_num = 0
    if inter_chip > 2:
        inter_num = util.get_inter_num(inter_class)
    #cv2.imshow('xx',chip)
    #cv2.waitKey(1000)

    return color_type, core_type, equiped, attr_nums, inter_num

# 178,178,178 for chips segment
MIN = 120
MAX = 190


def read_full_pic(path):
    img = cv2.imread(path)
    cut_ = ( img > MIN ) * (img < MAX)
    cut_ = cut_[:,:,0] * cut_[:,:,1] * cut_[:,:,2]
    y = np.sum(cut_,axis=1)
    x = np.sum(cut_,axis=0)

    x_index, x_len = get_section(x)
    y_index, y_len = get_section(y)
    if not ( 0.6 < x_len/y_len < 0.64 ):
        raise 'error for pic ' + path

    if 0:
        xs = x_index[3] # x start
        ys = y_index[1]
        x = read_single_chip(img[ys + 5:ys+y_len-5,xs+5:xs+x_len-5,:])
        print(x)
        exit()
    chip_list = []
    for j in range(len(y_index)):
        for i in range(len(x_index)):
            xs = x_index[i] # x start
            ys = y_index[j]
            try:
                single_chip = read_single_chip(img[ys + 5:ys+y_len-5,xs+5:xs+x_len-5,:])
                chip_list.append(single_chip)
            except:
                print('error when procrss chip ' + path + ' id:' + str(j * len(x_index) + i))

    return chip_list

if __name__ == '__main__':
    path = './chips_cut/img_1202.png'

    #path = './dd/test.jpg'
    a = read_full_pic(path)
    print(a)
