# -*- coding: UTF-8 -*-
"""
time: 2022-07-16
author: Liu Yong-Ce
"""
import os
import requests
import jieba
import cv2
import base64
import re
import collections
import numpy as np
import moviepy.editor as mp
from bs4 import BeautifulSoup
from aip import AipBodyAnalysis
from PIL import Image
from wordcloud import WordCloud


def download_video(video_url, save_path, video_name):
    '''
    youget 下载视频
    :param video_url:视频链接
    :param save_path: 保存路径
    :param video_name: 视频命名
    :return:
    '''
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    if len(os.listdir(save_path)) > 0:
        return None

    cmd = 'you-get -o {} -O {} {}'.format(save_path, video_name, video_url)
    print("开始下载视频！")
    res = os.popen(cmd)
    tmp = res.buffer.read().decode('utf-8')
    print(tmp)# 打印输出
    print("下载视频结束，存于{}".format(save_path))


def extract_audio(videos_file_path, mp3_save_path, name, output_video_file):
    if not os.path.exists(mp3_save_path):
        os.makedirs(mp3_save_path)
    path = os.path.join(mp3_save_path, name)
    print("开始提取背景音乐！")
    my_clip = mp.VideoFileClip(videos_file_path)
    my_clip.audio.write_audiofile(path)
    print("提取结束，存于{}".format(path))
    print("写入输出视频！")
    video = mp.VideoFileClip(output_video_file)
    videos = video.set_audio(mp.AudioFileClip(path))
    videos.write_videofile("./output_with_audio.mp4")
    print("Enjoy Yourself!")
    

def download_danmu(cid, save_path):
    '''弹幕下载并存储'''
    url = 'http://comment.bilibili.com/{}.xml'.format(cid)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    name = 'danmu.txt'
    path = os.path.join(save_path, name)
    
    if len(os.listdir(save_path)) > 0:
        return None

    f = open(path,'w+',encoding='utf-8') #打开 txt 文件
    res = requests.get(url)
    res.encoding = 'utf-8'
    soup = BeautifulSoup(res.text,'lxml')
    items = soup.find_all('d')# 找到 d 标签
    print("分割弹幕！")

    for item in items:
        text = item.text
        # print('---------------------------------'*10)
        # print(text)

        seg_list = jieba.cut(text,cut_all =True)# 对字符串进行分词处理，方便后面制作词云图
        for j in seg_list:
            # print(j)
            f.write(j)
            f.write('\n')
    f.close()
    print("分割结束，存于{}".format(save_path))


def video2pic(video_path, pic_path):
    # 切割视频
    if not os.path.exists(pic_path):
        os.makedirs(pic_path)
    vc = cv2.VideoCapture(video_path)
    c =0
    if vc.isOpened():
        rval,frame = vc.read()# 读取视频帧
    else:
        rval=False
        print("Error!")

    print("开始切割视频，获得图片！")
    while rval:
        cv2.imwrite(os.path.join(pic_path,'{}.jpg'.format(c)),frame)
        rval,frame = vc.read()# 读取每一视频帧，并保存至图片中
        c += 1
        # print('第 {} 张图片存放成功！'.format(c))
    print("切割完毕，共计{}张图片，存于{}".format(c, pic_path))


def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        content = fp.read()
        fp.close()
        return content


def seg_people(jpg_path, save_path, crop_path=None):
    # 找自己百度智能云id，注册获取
    APP_ID = ""
    API_KEY = ''
    SECRET_KEY = ''

    client = AipBodyAnalysis(APP_ID, API_KEY, SECRET_KEY)
    # 文件夹
    jpg_file = os.listdir(jpg_path)  
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # 要保存的文件夹
    print("开始人形分割图片！")
    for i in jpg_file:
        open_file = os.path.join(jpg_path, i)
        save_file = os.path.join(save_path, i)
        if not os.path.exists(save_file):#文件不存在时，进行下步操作
            img = cv2.imread(open_file)  # 获取图像尺寸
            height, width, _ = img.shape
            if crop_path:# 若Crop_path 不为 None,则不进行裁剪
                crop_file = os.path.join(crop_path,i)
                img = img[100:-1,300:-400] #图片太大，对图像进行裁剪里面参数根据自己情况设定
                cv2.imwrite(crop_file,img)
                image= get_file_content(crop_file)
            else:
                image = get_file_content(open_file)

            res = client.bodySeg(image)#调用百度API 对人像进行分割
            labelmap = base64.b64decode(res['labelmap'])
            labelimg = np.frombuffer(labelmap,np.uint8)# 转化为np数组 0-255
            labelimg = cv2.imdecode(labelimg,1)
            labelimg = cv2.resize(labelimg,(width,height),interpolation=cv2.INTER_NEAREST)
            img_new = np.where(labelimg==1,255,labelimg)# 将 1 转化为 255
            cv2.imwrite(save_file,img_new)
            # print(save_file,'save successfully')
    print("提取人形图片结束，共计{}张，存于{}".format(i, save_path))


def plot_wordcloud(danmu_path, mask_path, cloud_path):
    if not os.path.exists(cloud_path):
        os.makedirs(cloud_path)

    word_list = []
    with open(danmu_path,encoding='utf-8') as f:
        con = f.read().split('\n')# 读取txt文本词云文本
        for i in con:
            if re.findall('[\u4e00-\u9fa5]+', str(i), re.S): #去除无中文的词频
                word_list.append(i)
    print("开始绘制词云图片！")
    
    for i in os.listdir(mask_path):
        open_file = os.path.join(mask_path,i)
        save_file = os.path.join(cloud_path,i)
        try:
            if not os.path.exists(save_file):
                # 随机索引前 start 频率词
                start = np.random.randint(0, 15)
                word_counts = collections.Counter(word_list)
                word_counts = dict(word_counts.most_common()[start:])
                background = 255- np.array(Image.open(open_file))

                wc =WordCloud(
                    background_color='black',
                    max_words=500,
                    mask=background,
                    mode = 'RGB',
                    font_path ="C:\Windows\Fonts\simhei.ttf",# 设置字体路径，用于设置中文,
                ).generate_from_frequencies(word_counts)
                wc.to_file(save_file)
                # print(save_file,'Save Sucessfully!')
        except:
            print("error")
            continue
    print("词云绘制完毕，共计{}张，存于{}".format(i, cloud_path))


def jpg2video(video_path, origin_path, wordart_path):
    num_list = [int(str(i).split('.')[0]) for i in os.listdir(origin_path)]
    fps = 24# 视频帧率，越大越流畅
    height,width,_=cv2.imread(os.path.join(origin_path,'{}.jpg'.format(num_list[0]))).shape # 视频高度和宽度
    width = width*2
    # 创建一个写入操作;
    video_writer = cv2.VideoWriter(video_path,cv2.VideoWriter_fourcc(*'mp4v'),fps,(width,height))

    for i in sorted(num_list):
        i = '{}.jpg'.format(i)
        ori_jpg = os.path.join(origin_path,str(i))
        word_jpg = os.path.join(wordart_path,str(i))
        # com_jpg = os.path.join(Composite_path,str(i))
        ori_arr = cv2.imread(ori_jpg)
        word_arr = cv2.imread(word_jpg)
        try:
            # 利用 Numpy 进行拼接
            com_arr = np.hstack((ori_arr,word_arr))
            cv2.imwrite("./tem.jpg",com_arr)# 合成图保存
        except:
            print("pass")
            continue
        video_writer.write(com_arr) # 将每一帧画面写入视频流中
        print("{} Save Sucessfully---------".format(ori_jpg))
        


if __name__ == "__main__":
    # 下载视频
    video_url = "https://www.bilibili.com/video/BV1sJ411P7CF?p=2&vd_source=1feeca73967f668257ea49ca540e34d5"
    video_path = "./data/video"
    video_name = "XiaoMeng"
    # download_video(video_url=video_url, save_path=video_path, video_name=video_name)

    # 下载、切割弹幕
    my_cid =  116963870
    danmu_path = "./data/danmu"
    # download_danmu(cid=my_cid, save_path=danmu_path)

    # # 切割图片并保存
    pic_path = "./data/pic"
    # video2pic(video_path="./data/video/XiaoMeng.flv", pic_path=pic_path)

    seg_save_path = "./data/seg_pic"
    # seg_people(jpg_path=pic_path, save_path=seg_save_path)

    # plot_wordcloud(danmu_path="./data/danmu/danmu.txt", mask_path="./data/seg_pic", cloud_path="./data/word_cloud_pic")

    # jpg2video(video_path="./output.mp4v", origin_path="./data/pic", wordart_path="./data/word_cloud_pic")

    # 提取背景音乐, 写入
    extract_audio(videos_file_path="./data/video/XiaoMeng.flv", mp3_save_path="./data/music", name="XiaoMeng.mp3", output_video_file='./output.mp4v')

