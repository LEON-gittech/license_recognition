import sys
import cv2 as cv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random

import argparse
import torch.backends.cudnn as cudnn

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QFileDialog, QMainWindow, QInputDialog, QMessageBox, QApplication

from demo1 import Ui_MainWindow
from models.experimental import *
from utils.datasets import *
from utils.utils import *
from models.LPRNet import *
from models.STN import *

matplotlib.use('TkAgg')


def detect(opt, save_img=False):
    classify, out, source, det_weights, rec_weights, rec_weights_b, view_img, save_txt, imgsz = \
        opt.classify, opt.output, opt.source, opt.det_weights, opt.rec_weights, opt.rec_weights_b, opt.view_img, opt.save_txt, opt.img_size
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = torch_utils.select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete rec_result folder
    os.makedirs(out)  # make new rec_result folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load yolov5 model
    model = attempt_load(det_weights, map_location=device)  # load FP32 model
    print("load det pretrained model successful!")
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier  也就是rec 字符识别
    if classify:
        #############绿牌
        modelc = LPRNet(lpr_max_len=8, phase=False, class_num=len(CHARS), dropout_rate=0).to(device)
        modelc.load_state_dict(torch.load(rec_weights, map_location=torch.device('cpu'))['net_state_dict'])
        print("load rec pretrained model successful!")
        modelc.to(device).eval()

        ############蓝牌
        modelb = LPRNet(lpr_max_len=8, phase=False, class_num=len(CHARS), dropout_rate=0).to(device)
        modelb.load_state_dict(torch.load(rec_weights_b, map_location=torch.device('cpu')))
        print("load rec pretrained model successful!")
        modelb.to(device).eval()
    ###STN
    # STN = STNet()
    # STN.to(device)
    # STN.load_state_dict(torch.load('D:\cv\YOLOv5-LPRNet-Licence-Recognition-master\weights\Final_STN_model.pth', map_location=lambda storage, loc: storage))
    # STN.eval()
    ###

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size demo
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run demo
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        # if(len(pred)>1):
        #     pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        # Apply Classifier
        if classify:
            # plt.imshow(im0s)
            # plt.show()
            pred, plat_num = apply_classifier(pred, modelc, modelb, img, im0s)
            # labels, pred_labels = apply_classifier(pred, modelc, img, im0s)
        t2 = torch_utils.time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for de, lic_plat in zip(det, plat_num):
                    # xyxy,conf,cls,lic_plat=de[:4],de[4],de[5],de[6:]
                    *xyxy, conf, cls = de

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 5 + '\n') % (cls, xywh))  # label format

                    if save_img or view_img:  # Add bbox to image
                        # label = '%s %.2f' % (names[int(cls)], conf)
                        lb = ""
                        for a, i in enumerate(lic_plat):
                            # if a ==0:
                            #     continue
                            lb += CHARS[int(i)]
                        # label = '%s %.2f' % (lb, conf)
                        label = '%s' % (lb)
                        im0 = plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # Print time (demo + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images' or dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # rec_result video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


class MyWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        self.setupUi(self)

        self.captured = None  # 第一页打开的图片
        self.captured1 = None  # 第二页打开的第一张图片
        self.captured2 = None  # 第二页打开的第二张图片
        self.captured3 = None  # 第三页打开的图片

        self.is_read = False  # 第一页有没有打开图片
        self.is_read1 = False  # 第二页有没有打开第一张图片
        self.is_read2 = False  # 第二页有没有打开第二张图片
        self.is_read3 = False  # 第三页有没有打开图片

        self.output = None  # 第一页图片输出结果
        self.output1 = None  # 第二页图片输出结果
        self.output2 = None  # 第三页图片输出结果
        self.filename = None  # 第三页图片的地址

        self.kernel = None  # 当前的核

    def btnReadImage_Clicked(self):
        """
        从本地读取图片
        """
        # 打开文件选取对话框
        filename,  _ = QFileDialog.getOpenFileName(self, '打开图片')
        if filename:
            self.is_read = True
            self.captured = cv.imread(str(filename))
            # OpenCV图像以BGR通道存储，显示时需要从BGR转到RGB
            img = cv.cvtColor(self.captured, cv.COLOR_BGR2RGB)
            # 获取图像的高，宽以及深度
            rows, cols, channels = img.shape
            bytesPerLine = channels * cols
            # 创建QImage格式的图像，并读入图像信息
            QImg = QImage(img.data, cols, rows, bytesPerLine, QImage.Format_RGB888)
            self.output = None
            self.labelOriginal.clear()
            self.labelResult.clear()
            self.labelOriginal.setPixmap(QPixmap.fromImage(QImg).scaled(
                self.labelOriginal.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def btnReadImage2_Clicked(self):
        """
        从本地读取图片1
        """
        # 打开文件选取对话框
        filename,  _ = QFileDialog.getOpenFileName(self, '打开图片')
        if filename:
            self.is_read1 = True
            self.captured1 = cv.imread(str(filename))
            # OpenCV图像以BGR通道存储，显示时需要从BGR转到RGB
            img = cv.cvtColor(self.captured1, cv.COLOR_BGR2RGB)
            # 获取图像的高，宽以及深度
            rows, cols, channels = img.shape
            bytesPerLine = channels * cols
            # 创建QImage格式的图像，并读入图像信息
            QImg = QImage(img.data, cols, rows, bytesPerLine, QImage.Format_RGB888)
            self.labelOriginal_2.clear()
            self.labelResult_2.clear()
            self.labelOriginal_2.setPixmap(QPixmap.fromImage(QImg).scaled(
                self.labelOriginal_2.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def btnReadImage3_Clicked(self):
        """
        从本地读取图片2
        """
        # 打开文件选取对话框
        filename,  _ = QFileDialog.getOpenFileName(self, '打开图片')
        if filename:
            self.is_read2 = True
            self.captured2 = cv.imread(str(filename))
            # OpenCV图像以BGR通道存储，显示时需要从BGR转到RGB
            img = cv.cvtColor(self.captured2, cv.COLOR_BGR2RGB)
            # 获取图像的高，宽以及深度
            rows, cols, channels = img.shape
            bytesPerLine = channels * cols
            # 创建QImage格式的图像，并读入图像信息
            QImg = QImage(img.data, cols, rows, bytesPerLine, QImage.Format_RGB888)
            self.labelOriginal_3.clear()
            self.labelResult_2.clear()
            self.labelOriginal_3.setPixmap(QPixmap.fromImage(QImg).scaled(
                self.labelOriginal_3.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def btnReadImage4_Clicked(self):
        """
        从本地读取图片
        """
        # 打开文件选取对话框
        filename,  _ = QFileDialog.getOpenFileName(self, '打开图片')
        self.filename = filename
        if filename:
            self.is_read3 = True
            self.captured3 = cv.imread(str(filename))
            # OpenCV图像以BGR通道存储，显示时需要从BGR转到RGB
            img = cv.cvtColor(self.captured3, cv.COLOR_BGR2RGB)
            # 获取图像的高，宽以及深度
            rows, cols, channels = img.shape
            bytesPerLine = channels * cols
            # 创建QImage格式的图像，并读入图像信息
            QImg = QImage(img.data, cols, rows, bytesPerLine, QImage.Format_RGB888)
            self.labelOriginal_4.clear()
            self.labelResult_3.clear()
            self.labelOriginal_4.setPixmap(QPixmap.fromImage(QImg).scaled(
                self.labelOriginal_4.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def btnSaveImage_Clicked(self):
        """
        保存第一页图片
        """
        if self.output is None:
            return
        cv.imwrite('output_page1.png', self.output, [int(cv.IMWRITE_PNG_COMPRESSION), 3])

    def btnSaveImage2_Clicked(self):
        """
        保存第二页图片
        """
        if self.output1 is None:
            return
        cv.imwrite('output_page2.png', self.output1, [int(cv.IMWRITE_PNG_COMPRESSION), 3])

    def btnHorizontalFlip_Clicked(self):
        """
        水平翻转
        """
        # 如果没有打开图片，则不执行操作
        if not self.is_read:
            return
        # 水平翻转
        horizon_img = cv.flip(self.captured, 1)
        self.output = horizon_img
        horizon_img = cv.cvtColor(horizon_img, cv.COLOR_BGR2RGB)
        rows, columns, channels = horizon_img.shape
        bytesPerLine = columns * channels
        QImg = QImage(horizon_img.data, columns, rows, bytesPerLine, QImage.Format_RGB888)
        self.labelResult.clear()
        self.labelResult.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.labelResult.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def btnVerticalFlip_Clicked(self):
        """
        垂直翻转
        """
        # 如果没有打开图片，则不执行操作
        if not self.is_read:
            return
        # 垂直翻转
        vertical_img = cv.flip(self.captured, 0)
        self.output = vertical_img
        vertical_img = cv.cvtColor(vertical_img, cv.COLOR_BGR2RGB)
        rows, columns, channels = vertical_img.shape
        bytesPerLine = columns * channels
        QImg = QImage(vertical_img.data, columns, rows, bytesPerLine, QImage.Format_RGB888)
        self.labelResult.clear()
        self.labelResult.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.labelResult.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def btnDiagonalFlip_Clicked(self):
        """
        对角翻转
        """
        # 如果没有打开图片，则不执行操作
        if not self.is_read:
            return
        # 对角翻转
        diagonal_img = cv.flip(self.captured, -1)
        self.output = diagonal_img
        diagonal_img = cv.cvtColor(diagonal_img, cv.COLOR_BGR2RGB)
        rows, columns, channels = diagonal_img.shape
        bytesPerLine = columns * channels
        QImg = QImage(diagonal_img.data, columns, rows, bytesPerLine, QImage.Format_RGB888)
        self.labelResult.clear()
        self.labelResult.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.labelResult.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def btnAffineTransform_Clicked(self):
        """
        仿射变化
        """
        # 如果没有打开图片，则不执行操作
        if not self.is_read:
            return
        img = self.captured
        rows, cols = img.shape[: 2]
        # 设置图像仿射变化矩阵
        post1 = np.float32([[50, 50], [200, 50], [50, 200]])
        post2 = np.float32([[10, 100], [200, 50], [100, 250]])
        M = cv.getAffineTransform(post1, post2)
        img = cv.warpAffine(img, M, (rows, cols))
        self.output = img
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        rows, columns, channels = img.shape
        bytesPerLine = columns * channels
        QImg = QImage(img.data, columns, rows, bytesPerLine, QImage.Format_RGB888)
        self.labelResult.clear()
        self.labelResult.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.labelResult.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def btnGray_Clicked(self):
        """
        灰度图
        """
        # 如果没有打开图片，则不执行操作
        if not self.is_read:
            return
        img = cv.cvtColor(self.captured, cv.COLOR_RGB2GRAY)
        self.output = img
        rows, columns = img.shape
        bytesPerLine = columns
        # 灰度图是单通道，所以需要用Format_Indexed8
        QImg = QImage(img.data, columns, rows, bytesPerLine, QImage.Format_Indexed8)
        self.labelResult.clear()
        self.labelResult.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.labelResult.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def btnGrayHist_Clicked(self):
        """
        灰度直方图
        """
        # 如果没有打开图片，则不执行操作
        if not self.is_read:
            return
        img = cv.cvtColor(self.captured, cv.COLOR_RGB2GRAY)
        hist = cv.calcHist([img], [0], None, [256], [0, 256])
        # 自定义画布大小
        plt.figure()
        plt.title("grayhist")
        # 设置X轴标签
        plt.xlabel("bins")
        # 设置Y轴标签
        plt.ylabel("fixels")
        plt.plot(hist)
        plt.show()

    def btnColorHist_Clicked(self):
        """
        彩色直方图
        """
        # 如果没有打开图片，则不执行操作
        if not self.is_read:
            return
        img = cv.cvtColor(self.captured, cv.COLOR_BGR2RGB)
        color = ["r", "g", "b"]
        plt.figure()
        plt.title("colorhist")
        # 设置X轴标签
        plt.xlabel("bins")
        # 设置Y轴标签
        plt.ylabel("fixels")
        for i, col in enumerate(color):
            hist = cv.calcHist([img], [i], None, [256], [0, 256])
            plt.plot(hist, color=col)
            plt.xlim([0, 256])
        plt.show()

    def btnPiecewiseLinearVary_Clicked(self):
        """
        分段线性转换
        """
        if not self.is_read:
            return
        img = cv.cvtColor(self.captured, cv.COLOR_RGB2GRAY)
        h, w = img.shape[:2]
        out = np.zeros(img.shape, np.uint8)
        # 通过遍历对不同像素范围内进行分段线性变化
        num1 = 70
        num2 = 180
        num3 = 0.8
        num4 = 0
        num5 = 2.6
        num6 = -210
        num7 = 0.138
        num8 = 195
        for i in range(h):
            for j in range(w):
                pix = img[i][j]
                if (pix < num1).all():
                    out[i][j] = num3 * pix + num4
                elif (pix < num2).all():
                    out[i][j] = num5 * pix + num6
                else:
                    out[i][j] = num7 * pix + num8
        out = np.around(out)
        out = out.astype(np.uint8)
        self.output = out
        rows, columns = out.shape
        bytesPerLine = columns
        # 灰度图是单通道，所以需要用Format_Indexed8
        QImg = QImage(out.data, columns, rows, bytesPerLine, QImage.Format_Indexed8)
        self.labelResult.clear()
        self.labelResult.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.labelResult.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def btnHoughLines_Clicked(self):
        """
        Hough变换线条变化检测
        """
        # 如果没有打开图片，则不执行操作
        if not self.is_read:
            return
        img = self.captured
        img = cv.GaussianBlur(img, (3, 3), 0)
        edges = cv.Canny(img, 50, 150, apertureSize=3)
        lines = cv.HoughLines(edges, 1, np.pi / 2, 118)
        result = img.copy()
        for i_line in lines:
            for line in i_line:
                rho = line[0]  # 第一个元素是距离rho
                theta = line[1]  # 第二个元素是角度theta
                if (theta < (np.pi / 4.)) or (theta > (3. * np.pi / 4.0)):  # 垂直直线
                    pt1 = (int(rho / np.cos(theta)), 0)
                    pt2 = (int((rho - result.shape[0] * np.sin(theta)) / np.cos(theta)), result.shape[0])
                    cv.line(result, pt1, pt2, (0, 0, 255))
                else:
                    pt1 = (0, int(rho / np.sin(theta)))
                    pt2 = (result.shape[1], int((rho - result.shape[1] * np.cos(theta)) / np.sin(theta)))
                    cv.line(result, pt1, pt2, (0, 0, 255), 1)
        minLineLength = 200
        maxLineGap = 15
        linesP = cv.HoughLinesP(edges, 1, np.pi / 180, 80, minLineLength, maxLineGap)
        result_P = img.copy()
        for i_P in linesP:
            for x1, y1, x2, y2 in i_P:
                cv.line(result_P, (x1, y1), (x2, y2), (0, 255, 0), 3)
        result_P = cv.cvtColor(result_P, cv.COLOR_BGR2RGB)
        rows, columns, channels = result_P.shape
        bytesPerLine = columns * channels
        QImg = QImage(result_P.data, columns, rows, bytesPerLine, QImage.Format_RGB888)
        self.labelResult.clear()
        self.labelResult.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.labelResult.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def btnB_Clicked(self):
        """
        B通道图片
        """
        # 如果没有打开图片，则不执行操作
        if not self.is_read:
            return
        img = self.captured
        img_b = img[:, :, 0]
        self.output = img_b
        img_b = cv.cvtColor(img_b, cv.COLOR_BGR2RGB)
        rows, columns, channels = img_b.shape
        bytesPerLine = columns * channels
        QImg = QImage(img_b.data, columns, rows, bytesPerLine, QImage.Format_RGB888)
        self.labelResult.clear()
        self.labelResult.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.labelResult.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def btnG_Clicked(self):
        """
        G通道图片
        """
        # 如果没有打开图片，则不执行操作
        if not self.is_read:
            return
        img = self.captured
        img_g = img[:, :, 1]
        self.output = img_g
        img_g = cv.cvtColor(img_g, cv.COLOR_BGR2RGB)
        rows, columns, channels = img_g.shape
        bytesPerLine = columns * channels
        QImg = QImage(img_g.data, columns, rows, bytesPerLine, QImage.Format_RGB888)
        self.labelResult.clear()
        self.labelResult.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.labelResult.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def btnR_Clicked(self):
        """
        R通道图片
        """
        # 如果没有打开图片，则不执行操作
        if not self.is_read:
            return
        img = self.captured
        img_r = img[:, :, 2]
        self.output = img_r
        img_r = cv.cvtColor(img_r, cv.COLOR_BGR2RGB)
        rows, columns, channels = img_r.shape
        bytesPerLine = columns * channels
        QImg = QImage(img_r.data, columns, rows, bytesPerLine, QImage.Format_RGB888)
        self.labelResult.clear()
        self.labelResult.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.labelResult.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def btnH_Clicked(self):
        """
        H通道图片
        """
        # 如果没有打开图片，则不执行操作
        if not self.is_read:
            return
        img = cv.cvtColor(self.captured, cv.COLOR_BGR2HSV)
        img_h = img[:, :, 0]
        self.output = img_h
        img_h = cv.cvtColor(img_h, cv.COLOR_BGR2RGB)
        rows, columns, channels = img_h.shape
        bytesPerLine = columns * channels
        QImg = QImage(img_h.data, columns, rows, bytesPerLine, QImage.Format_RGB888)
        self.labelResult.clear()
        self.labelResult.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.labelResult.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def btnS_Clicked(self):
        """
        S通道图片
        """
        # 如果没有打开图片，则不执行操作
        if not self.is_read:
            return
        img = cv.cvtColor(self.captured, cv.COLOR_BGR2HSV)
        img_s = img[:, :, 1]
        self.output = img_s
        img_s = cv.cvtColor(img_s, cv.COLOR_BGR2RGB)
        rows, columns, channels = img_s.shape
        bytesPerLine = columns * channels
        QImg = QImage(img_s.data, columns, rows, bytesPerLine, QImage.Format_RGB888)
        self.labelResult.clear()
        self.labelResult.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.labelResult.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def btnV_Clicked(self):
        """
        V通道图片
        """
        # 如果没有打开图片，则不执行操作
        if not self.is_read:
            return
        img = cv.cvtColor(self.captured, cv.COLOR_BGR2HSV)
        img_v = img[:, :, 2]
        self.output = img_v
        img_v = cv.cvtColor(img_v, cv.COLOR_BGR2RGB)
        rows, columns, channels = img_v.shape
        bytesPerLine = columns * channels
        QImg = QImage(img_v.data, columns, rows, bytesPerLine, QImage.Format_RGB888)
        self.labelResult.clear()
        self.labelResult.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.labelResult.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def btnAnd_Clicked(self):
        """
        与运算
        """
        # 如果没有打开图片，则不执行操作
        if not self.is_read1:
            return
        if not self.is_read2:
            return
        img1 = cv.cvtColor(self.captured1, cv.COLOR_RGB2GRAY)
        img2 = cv.cvtColor(self.captured2, cv.COLOR_RGB2GRAY)
        img1 = cv.resize(img1, (256, 256))
        img2 = cv.resize(img2, (256, 256))
        img = img1 & img2
        self.output1 = img
        rows, columns = img.shape
        bytesPerLine = columns
        # 灰度图是单通道，所以需要用Format_Indexed8
        QImg = QImage(img.data, columns, rows, bytesPerLine, QImage.Format_Indexed8)
        self.labelResult_2.clear()
        self.labelResult_2.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.labelResult_2.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def btnOr_Clicked(self):
        """
        或运算
        """
        # 如果没有打开图片，则不执行操作
        if not self.is_read1:
            return
        if not self.is_read2:
            return
        img1 = cv.cvtColor(self.captured1, cv.COLOR_RGB2GRAY)
        img2 = cv.cvtColor(self.captured2, cv.COLOR_RGB2GRAY)
        img1 = cv.resize(img1, (256, 256))
        img2 = cv.resize(img2, (256, 256))
        img = img1 | img2
        self.output1 = img
        rows, columns = img.shape
        bytesPerLine = columns
        # 灰度图是单通道，所以需要用Format_Indexed8
        QImg = QImage(img.data, columns, rows, bytesPerLine, QImage.Format_Indexed8)
        self.labelResult_2.clear()
        self.labelResult_2.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.labelResult_2.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def btnNot_Clicked(self):
        """
        非运算
        """
        # 如果没有打开图片，则不执行操作
        if not self.is_read1:
            return
        self.labelOriginal_3.clear()
        img1 = cv.cvtColor(self.captured1, cv.COLOR_RGB2GRAY)
        img1 = cv.resize(img1, (256, 256))
        img = ~img1
        self.output1 = img
        rows, columns = img.shape
        bytesPerLine = columns
        # 灰度图是单通道，所以需要用Format_Indexed8
        QImg = QImage(img.data, columns, rows, bytesPerLine, QImage.Format_Indexed8)
        self.labelResult_2.clear()
        self.is_read2 = False
        self.labelResult_2.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.labelResult_2.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def btnPlus_Clicked(self):
        """
        加法运算
        """
        # 如果没有打开图片，则不执行操作
        if not self.is_read1:
            return
        if not self.is_read2:
            return
        img1 = cv.cvtColor(self.captured1, cv.COLOR_BGR2RGB)
        img2 = cv.cvtColor(self.captured2, cv.COLOR_BGR2RGB)
        img1 = cv.resize(img1, (256, 256))
        img2 = cv.resize(img2, (256, 256))
        img = cv.add(img1, img2)
        self.output1 = img
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        rows, columns, channels = img.shape
        bytesPerLine = columns * channels
        QImg = QImage(img.data, columns, rows, bytesPerLine, QImage.Format_RGB888)
        self.labelResult_2.clear()
        self.labelResult_2.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.labelResult_2.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def btnSub_Clicked(self):
        """
        减法运算
        """
        # 如果没有打开图片，则不执行操作
        if not self.is_read1:
            return
        if not self.is_read2:
            return
        img1 = cv.cvtColor(self.captured1, cv.COLOR_BGR2RGB)
        img2 = cv.cvtColor(self.captured2, cv.COLOR_BGR2RGB)
        img1 = cv.resize(img1, (256, 256))
        img2 = cv.resize(img2, (256, 256))
        img = cv.subtract(img1, img2)
        self.output1 = img
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        rows, columns, channels = img.shape
        bytesPerLine = columns * channels
        QImg = QImage(img.data, columns, rows, bytesPerLine, QImage.Format_RGB888)
        self.labelResult_2.clear()
        self.labelResult_2.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.labelResult_2.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def btnMulti_Clicked(self):
        """
        乘法运算
        """
        # 如果没有打开图片，则不执行操作
        if not self.is_read1:
            return
        if not self.is_read2:
            return
        img1 = cv.cvtColor(self.captured1, cv.COLOR_BGR2RGB)
        img2 = cv.cvtColor(self.captured2, cv.COLOR_BGR2RGB)
        img1 = cv.resize(img1, (256, 256))
        img2 = cv.resize(img2, (256, 256))
        img = cv.multiply(img1, img2)
        self.output1 = img
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        rows, columns, channels = img.shape
        bytesPerLine = columns * channels
        QImg = QImage(img.data, columns, rows, bytesPerLine, QImage.Format_RGB888)
        self.labelResult_2.clear()
        self.labelResult_2.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.labelResult_2.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def btnDivi_Clicked(self):
        """
        除法运算
        """
        # 如果没有打开图片，则不执行操作
        if not self.is_read1:
            return
        if not self.is_read2:
            return
        img1 = cv.cvtColor(self.captured1, cv.COLOR_BGR2RGB)
        img2 = cv.cvtColor(self.captured2, cv.COLOR_BGR2RGB)
        img1 = cv.resize(img1, (256, 256))
        img2 = cv.resize(img2, (256, 256))
        img = cv.divide(img1, img2)
        self.output1 = img
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        rows, columns, channels = img.shape
        bytesPerLine = columns * channels
        QImg = QImage(img.data, columns, rows, bytesPerLine, QImage.Format_RGB888)
        self.labelResult_2.clear()
        self.labelResult_2.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.labelResult_2.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def btnChangeSize_Clicked(self):
        """
        改变图片大小
        """
        # 如果没有打开图片，则不执行操作
        if not self.is_read:
            return
        num1, ok1 = QInputDialog.getInt(self, '输入长度', '输入新的长度，单位为像素')
        if ok1 and num1:
            num2, ok2 = QInputDialog.getInt(self, '输入宽度', '输入新的宽度，单位为像素')
            if ok2 and num2:
                img = self.captured
                resize_img = cv.resize(img, (num1, num2), interpolation=cv.INTER_LINEAR)
                self.output = resize_img
                resize_img = cv.cvtColor(resize_img, cv.COLOR_BGR2RGB)
                rows, columns, channels = resize_img.shape
                bytesPerLine = columns * channels
                QImg = QImage(resize_img.data, columns, rows, bytesPerLine, QImage.Format_RGB888)
                self.labelResult.clear()
                self.labelResult.setPixmap(QPixmap.fromImage(QImg).scaled(
                    self.labelResult.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def btnMove_Clicked(self):
        """
        平移
        """
        # 如果没有打开图片，则不执行操作
        if not self.is_read:
            return
        num1, ok1 = QInputDialog.getInt(self, '向右水平平移', '输入向右水平平移的长度，单位为像素')
        if ok1 and num1:
            num2, ok2 = QInputDialog.getInt(self, '向下垂直平移', '输入向下垂直平移的长度，单位为像素')
            if ok2 and num2:
                img = self.captured
                height, width, channel = img.shape
                M = np.float32([[1, 0, num1], [0, 1, num2]])
                shifted_img = cv.warpAffine(img, M, (width, height))
                self.output = shifted_img
                shifted_img = cv.cvtColor(shifted_img, cv.COLOR_BGR2RGB)
                rows, columns, channels = shifted_img.shape
                bytesPerLine = columns * channels
                QImg = QImage(shifted_img.data, columns, rows, bytesPerLine, QImage.Format_RGB888)
                self.labelResult.clear()
                self.labelResult.setPixmap(QPixmap.fromImage(QImg).scaled(
                    self.labelResult.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def btnRotate_Clicked(self):
        """
        旋转
        """
        # 如果没有打开图片，则不执行操作
        if not self.is_read:
            return
        num, ok = QInputDialog.getInt(self, '输入旋转角度', '输入旋转角度，单位为度')
        if ok and num:
            img = self.captured
            rows, cols, depth = img.shape
            M = cv.getRotationMatrix2D((cols / 2, rows / 2), num, 1)  # 旋转中心，旋转角度，缩放因子
            rotate_img = cv.warpAffine(img, M, (cols, rows))  # M放射变换矩阵
            self.output = rotate_img
            rotate_img = cv.cvtColor(rotate_img, cv.COLOR_BGR2RGB)
            rows, columns, channels = rotate_img.shape
            bytesPerLine = columns * channels
            QImg = QImage(rotate_img.data, columns, rows, bytesPerLine, QImage.Format_RGB888)
            self.labelResult.clear()
            self.labelResult.setPixmap(QPixmap.fromImage(QImg).scaled(
                self.labelResult.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def btnEnhance_Clicked(self):
        """
        利用梯度图像增强
        """
        # 如果没有打开图片，则不执行操作
        if not self.is_read:
            return
        img = cv.cvtColor(self.captured, cv.COLOR_RGB2GRAY)  # 灰度模式读取图像
        img = img.astype('float')
        row, column = img.shape[:2]
        gradient = np.zeros((row, column))
        for x in range(row - 1):
            for y in range(column - 1):
                gx = abs(img[x + 1, y] - img[x, y])
                gy = abs(img[x, y + 1] - img[x, y])
                gradient[x, y] = gx + gy
        sharp = img + gradient
        sharp = np.where(sharp > 255, 255, sharp)
        sharp = np.where(sharp < 0, 0, sharp)
        sharp = sharp.astype('uint8')
        self.output = sharp
        rows, columns = sharp.shape
        bytesPerLine = columns
        # 灰度图是单通道，所以需要用Format_Indexed8
        QImg = QImage(sharp.data, columns, rows, bytesPerLine, QImage.Format_Indexed8)
        self.labelResult.clear()
        self.labelResult.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.labelResult.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def btnRoberts_Clicked(self):
        """
        Roberts
        """
        # 如果没有打开图片，则不执行操作
        if not self.is_read:
            return
        img = cv.cvtColor(self.captured, cv.COLOR_RGB2GRAY)  # 灰度模式读取图像
        # Roberts算子
        kernelx = np.array([[-1, 0], [0, 1]], dtype=int)
        kernely = np.array([[0, -1], [1, 0]], dtype=int)
        # 卷积操作
        x = cv.filter2D(img, cv.CV_16S, kernelx)
        y = cv.filter2D(img, cv.CV_16S, kernely)
        # 数据格式转换
        absX = cv.convertScaleAbs(x)
        absY = cv.convertScaleAbs(y)
        Roberts = cv.addWeighted(absX, 0.5, absY, 0.5, 0)
        self.output = Roberts
        rows, columns = Roberts.shape
        bytesPerLine = columns
        # 灰度图是单通道，所以需要用Format_Indexed8
        QImg = QImage(Roberts.data, columns, rows, bytesPerLine, QImage.Format_Indexed8)
        self.labelResult.clear()
        self.labelResult.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.labelResult.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def btnSobel_Clicked(self):
        """
        Sobel
        """
        # 如果没有打开图片，则不执行操作
        if not self.is_read:
            return
        img = cv.cvtColor(self.captured, cv.COLOR_RGB2GRAY)  # 灰度模式读取图像
        # Sobel算子
        x = cv.Sobel(img, cv.CV_16S, 1, 0)
        y = cv.Sobel(img, cv.CV_16S, 0, 1)
        # 数据格式转换
        absX = cv.convertScaleAbs(x)
        absY = cv.convertScaleAbs(y)
        Sobel = cv.addWeighted(absX, 0.5, absY, 0.5, 0)
        self.output = Sobel
        rows, columns = Sobel.shape
        bytesPerLine = columns
        # 灰度图是单通道，所以需要用Format_Indexed8
        QImg = QImage(Sobel.data, columns, rows, bytesPerLine, QImage.Format_Indexed8)
        self.labelResult.clear()
        self.labelResult.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.labelResult.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def btnLaplacian_Clicked(self):
        """
        Laplacian
        """
        # 如果没有打开图片，则不执行操作
        if not self.is_read:
            return
        img = cv.cvtColor(self.captured, cv.COLOR_RGB2GRAY)  # 灰度模式读取图像
        # 高斯滤波
        img1 = cv.GaussianBlur(img, (5, 5), 0, 0)
        # 拉普拉斯算法
        dst = cv.Laplacian(img1, cv.CV_16S, ksize=3)
        # 数据格式转换
        Laplacian = cv.convertScaleAbs(dst)
        self.output = Laplacian
        rows, columns = Laplacian.shape
        bytesPerLine = columns
        # 灰度图是单通道，所以需要用Format_Indexed8
        QImg = QImage(Laplacian.data, columns, rows, bytesPerLine, QImage.Format_Indexed8)
        self.labelResult.clear()
        self.labelResult.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.labelResult.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def btnLoG_Clicked(self):
        """
        LoG
        """
        # 如果没有打开图片，则不执行操作
        if not self.is_read:
            return
        img = cv.cvtColor(self.captured, cv.COLOR_RGB2GRAY)  # 灰度模式读取图像
        # 边缘扩充处理图像并使用高斯滤波处理该图像
        # 使用`BORDER_REPLICATE`参数扩展边缘，分别在四周扩展2个像素单位
        image0 = cv.copyMakeBorder(img, 2, 2, 2, 2, borderType=cv.BORDER_REPLICATE)
        # 使用高斯滤波平滑处理，核大小为（3，3），x、y的偏差为0
        image = cv.GaussianBlur(image0, (3, 3), 0, 0)
        # 使用Numpy定义LoG算子
        m1 = np.array(
            [[0, 0, -1, 0, 0], [0, -1, -2, -1, 0], [-1, -2, 16, -2, -1], [0, -1, -2, -1, 0], [0, 0, -1, 0, 0]])
        rows, cols = image.shape[:2]
        image1 = np.zeros((rows, cols))
        #  卷积运算
        # 为了使卷积对每个像素都进行运算，原图像的边缘像素要对准模板的中心。
        # 由于图像边缘扩大了2像素，因此要从位置2到行(列)-2
        for i in range(2, rows - 2):
            for j in range(2, cols - 2):
                # (i, j) 的位置表示卷积核的中心位置
                # 把卷积核与对应位置的原像素相乘求和，完成卷积操作
                # 替换卷积核中心位置的原像素值
                image1[i, j] = np.sum(m1 * image[i - 2:i + 3, j - 2:j + 3])
        # 数据格式转换
        image1 = cv.convertScaleAbs(image1)
        self.output = image1
        rows, columns = image1.shape
        bytesPerLine = columns
        # 灰度图是单通道，所以需要用Format_Indexed8
        QImg = QImage(image1.data, columns, rows, bytesPerLine, QImage.Format_Indexed8)
        self.labelResult.clear()
        self.labelResult.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.labelResult.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def btnCanny_Clicked(self):
        """
        Canny
        """
        # 如果没有打开图片，则不执行操作
        if not self.is_read:
            return
        # 高斯滤波
        blur = cv.GaussianBlur(self.captured, (3, 3), 0)
        # 灰度转换
        grayImage = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)
        # 求x，y方向的Sobel算子
        gradx = cv.Sobel(grayImage, cv.CV_16SC1, 1, 0)
        grady = cv.Sobel(grayImage, cv.CV_16SC1, 0, 1)
        # 使用Canny函数处理图像，低阈值50，高阈值150
        edge_output = cv.Canny(gradx, grady, 50, 150)
        self.output = edge_output
        rows, columns = edge_output.shape
        bytesPerLine = columns
        # 灰度图是单通道，所以需要用Format_Indexed8
        QImg = QImage(edge_output.data, columns, rows, bytesPerLine, QImage.Format_Indexed8)
        self.labelResult.clear()
        self.labelResult.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.labelResult.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def btnOpen_Clicked(self):
        """
        开运算
        """
        # 如果没有打开图片，则不执行操作
        if not self.is_read:
            return
        if self.kernel is None:
            self.labelResult.clear()
            QMessageBox.information(self, '请先设置结构元', '请先设置结构元')
            return
        img = cv.cvtColor(self.captured, cv.COLOR_RGB2GRAY)
        ret, img = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
        im_op = cv.morphologyEx(img, cv.MORPH_OPEN, self.kernel)
        self.output = im_op
        rows, columns = im_op.shape
        bytesPerLine = columns
        # 灰度图是单通道，所以需要用Format_Indexed8
        QImg = QImage(im_op.data, columns, rows, bytesPerLine, QImage.Format_Indexed8)
        self.labelResult.clear()
        self.labelResult.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.labelResult.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def btnClose_Clicked(self):
        """
        闭运算
        """
        # 如果没有打开图片，则不执行操作
        if not self.is_read:
            return
        if self.kernel is None:
            self.labelResult.clear()
            QMessageBox.information(self, '请先设置结构元', '请先设置结构元')
            return
        img = cv.cvtColor(self.captured, cv.COLOR_RGB2GRAY)
        ret, img = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
        im_cl = cv.morphologyEx(img, cv.MORPH_CLOSE, self.kernel)
        self.output = im_cl
        rows, columns = im_cl.shape
        bytesPerLine = columns
        # 灰度图是单通道，所以需要用Format_Indexed8
        QImg = QImage(im_cl.data, columns, rows, bytesPerLine, QImage.Format_Indexed8)
        self.labelResult.clear()
        self.labelResult.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.labelResult.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def btnBinarization_Clicked(self):
        """
        二值化
        """
        # 如果没有打开图片，则不执行操作
        if not self.is_read:
            return
        img = cv.cvtColor(self.captured, cv.COLOR_RGB2GRAY)
        ret, img = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
        self.output = img
        rows, columns = img.shape
        bytesPerLine = columns
        # 灰度图是单通道，所以需要用Format_Indexed8
        QImg = QImage(img.data, columns, rows, bytesPerLine, QImage.Format_Indexed8)
        self.labelResult.clear()
        self.labelResult.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.labelResult.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def btnDilate_Clicked(self):
        """
        膨胀
        """
        # 如果没有打开图片，则不执行操作
        if not self.is_read:
            return
        if self.kernel is None:
            self.labelResult.clear()
            QMessageBox.information(self, '请先设置结构元', '请先设置结构元')
            return
        img = cv.cvtColor(self.captured, cv.COLOR_RGB2GRAY)
        ret, img = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
        dilation = cv.dilate(img, self.kernel)
        self.output = dilation
        rows, columns = dilation.shape
        bytesPerLine = columns
        # 灰度图是单通道，所以需要用Format_Indexed8
        QImg = QImage(dilation.data, columns, rows, bytesPerLine, QImage.Format_Indexed8)
        self.labelResult.clear()
        self.labelResult.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.labelResult.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def btnErode_Clicked(self):
        """
        腐蚀
        """
        # 如果没有打开图片，则不执行操作
        if not self.is_read:
            return
        if self.kernel is None:
            self.labelResult.clear()
            QMessageBox.information(self, '请先设置结构元', '请先设置结构元')
            return
        img = cv.cvtColor(self.captured, cv.COLOR_RGB2GRAY)
        ret, img = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
        erosion = cv.erode(img, self.kernel)
        self.output = erosion
        rows, columns = erosion.shape
        bytesPerLine = columns
        # 灰度图是单通道，所以需要用Format_Indexed8
        QImg = QImage(erosion.data, columns, rows, bytesPerLine, QImage.Format_Indexed8)
        self.labelResult.clear()
        self.labelResult.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.labelResult.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def btnSet_Clicked(self):
        """
        设置结构元种类
        """
        num, ok = QInputDialog.getInt(self, '输入结构元大小', '输入结构元大小')
        if ok and num:
            cur_type = self.comboBox.currentText()
            if cur_type == "矩形结构元":
                self.kernel = cv.getStructuringElement(cv.MORPH_RECT, (num, num), (-1, -1))
            if cur_type == "交叉形结构元":
                self.kernel = cv.getStructuringElement(cv.MORPH_CROSS, (num, num), (-1, -1))
            if cur_type == "椭圆形结构元":
                self.kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (num, num), (-1, -1))
        return

    def btnAverage_Clicked(self):
        """
        均值类滤波
        """
        # 如果没有打开图片，则不执行操作
        if not self.is_read:
            return
        image = cv.cvtColor(self.captured, cv.COLOR_RGB2GRAY)
        cur_type = self.comboBox_2.currentText()
        out = np.zeros(image.shape, np.uint8)
        if cur_type == "算数均值滤波器":
            # 遍历图像，进行均值滤波
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    # 滤波器内像素值的和
                    my_sum = 0
                    # 遍历滤波器内的像素值
                    for m in range(-1, 2):
                        for n in range(-1, 2):
                            # 防止越界
                            if 0 <= i + m < image.shape[0] and 0 <= j + n < image.shape[1]:
                                # 像素值求和
                                my_sum += image[i + m][j + n]
                    # 求均值，作为最终的像素值
                    out[i][j] = int(my_sum / 9)
        if cur_type == "几何均值滤波器":
            # 遍历图像，进行均值滤波
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    # 计算均值,完成对图片src的几何均值滤波
                    ji = 1.0
                    # 遍历滤波器内的像素值
                    for m in range(0, 1):
                        for n in range(-1, 2):
                            # 防止越界
                            if 0 <= i + m < image.shape[0] and 0 <= j + n < image.shape[1]:
                                # 像素值求乘积
                                ji = ji * image[i + m][j + n]
                    # 取一个指数值
                    out[i][j] = (pow(ji, 1.0 / 3.0))
        if cur_type == "谐波均值滤波器":
            img_h = image.shape[0]
            img_w = image.shape[1]
            m, n = 3, 3
            order = m * n
            kernalMean = np.ones((m, n), np.float32)  # 生成盒式核
            hPad = int((m - 1) / 2)
            wPad = int((n - 1) / 2)
            imgPad = np.pad(image.copy(), ((hPad, m - hPad - 1), (wPad, n - wPad - 1)), mode="edge")
            epsilon = 1e-8
            out = image.copy()
            for i in range(hPad, img_h + hPad):
                for j in range(wPad, img_w + wPad):
                    sumTemp = np.sum(1.0 / (imgPad[i - hPad:i + hPad + 1, j - wPad:j + wPad + 1] + epsilon))
                    out[i - hPad][j - wPad] = order / sumTemp
        self.output = out
        rows, columns = out.shape
        bytesPerLine = columns
        # 灰度图是单通道，所以需要用Format_Indexed8
        QImg = QImage(out.data, columns, rows, bytesPerLine, QImage.Format_Indexed8)
        self.labelResult.clear()
        self.labelResult.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.labelResult.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def get_middle(self, array):
        """
        获取列表的中间值的函数
        """
        # 列表的长度
        length = len(array)
        # 对列表进行选择排序，获得有序的列表
        for i in range(length):
            for j in range(i + 1, length):
                # 选择最大的值
                if array[j] > array[i]:
                    # 交换位置
                    temp = array[j]
                    array[j] = array[i]
                    array[i] = temp
        return array[int(length / 2)]

    def btnSort_Clicked(self):
        """
        统计排序类滤波
        """
        # 如果没有打开图片，则不执行操作
        if not self.is_read:
            return
        image = cv.cvtColor(self.captured, cv.COLOR_RGB2GRAY)
        cur_type = self.comboBox_3.currentText()
        out = np.zeros(image.shape, np.uint8)
        if cur_type == "中值滤波器":
            # 存储滤波器范围内的像素值
            array = []
            # 遍历图像，进行中值滤波
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    # 清空滤波器内的像素值
                    array.clear()
                    # 遍历滤波器内的像素
                    for m in range(-1, 2):
                        for n in range(-1, 2):
                            # 防止越界
                            if 0 <= i + m < image.shape[0] and 0 <= j + n < image.shape[1]:
                                # 像素值加到列表中
                                array.append(image[i + m][j + n])
                                # 求中值，作为最终的像素值
                    out[i][j] = self.get_middle(array)
        if cur_type == "最大值滤波器":
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    # 最大值滤波器
                    my_max = -999999
                    for m in range(-1, 2):
                        for n in range(-1, 2):
                            # 防止越界
                            if 0 <= i + m < image.shape[0] and 0 <= j + n < image.shape[1]:
                                if image[i + m][j + n] > my_max:
                                    my_max = image[i + m][j + n]
                    out[i][j] = my_max
        if cur_type == "最小值滤波器":
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    # 最大值滤波器
                    my_min = 999999
                    for m in range(-1, 2):
                        for n in range(-1, 2):
                            # 防止越界
                            if 0 <= i + m < image.shape[0] and 0 <= j + n < image.shape[1]:
                                if image[i + m][j + n] < my_min:
                                    my_min = image[i + m][j + n]
                    out[i][j] = my_min
        self.output = out
        rows, columns = out.shape
        bytesPerLine = columns
        # 灰度图是单通道，所以需要用Format_Indexed8
        QImg = QImage(out.data, columns, rows, bytesPerLine, QImage.Format_Indexed8)
        self.labelResult.clear()
        self.labelResult.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.labelResult.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def btnSelect_Clicked(self):
        """
        选择性滤波
        """
        # 如果没有打开图片，则不执行操作
        if not self.is_read:
            return
        image = cv.cvtColor(self.captured, cv.COLOR_RGB2GRAY)
        cur_type = self.comboBox_4.currentText()
        out = np.zeros(image.shape, np.uint8)
        if cur_type == "低通滤波器":
            my_max = 150
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    if my_max > image[i][j]:
                        out[i][j] = image[i][j]
                    else:
                        out[i][j] = 255
        if cur_type == "高通滤波器":
            my_min = 160
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    if my_min < image[i][j]:
                        out[i][j] = image[i][j]
                    else:
                        out[i][j] = 0
        if cur_type == "带通滤波器":
            # 带通的范围
            my_min = 20
            my_max = 220
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    if my_min >= image[i][j]:
                        out[i][j] = 0
                    elif my_min < image[i][j] < my_max:
                        out[i][j] = image[i][j]
                    else:
                        out[i][j] = 255
        if cur_type == "带阻滤波器":
            my_min = 160
            my_max = 220
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    if my_min >= image[i][j]:
                        out[i][j] = image[i][j]
                    elif my_min < image[i][j] < my_max:
                        out[i][j] = 190
                    else:
                        out[i][j] = image[i][j]
        self.output = out
        rows, columns = out.shape
        bytesPerLine = columns
        # 灰度图是单通道，所以需要用Format_Indexed8
        QImg = QImage(out.data, columns, rows, bytesPerLine, QImage.Format_Indexed8)
        self.labelResult.clear()
        self.labelResult.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.labelResult.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def btnGaussian_Clicked(self):
        """
        高斯噪声
        """
        # 如果没有打开图片，则不执行操作
        if not self.is_read:
            return
        image = self.captured
        # 将图片的像素值归一化，存入矩阵中
        image = np.array(image / 255, dtype=float)
        # 生成正态分布的噪声，其中0表示均值，0.1表示方差
        noise = np.random.normal(0, 0.1, image.shape)
        # 将噪声叠加到图片上
        out = image + noise
        # 将图像的归一化像素值控制在0和1之间，防止噪声越界
        out = np.clip(out, 0.0, 1.0)
        # 将图像的像素值恢复到0到255之间
        out = np.uint8(out * 255)
        self.output = out
        out = cv.cvtColor(out, cv.COLOR_BGR2RGB)
        rows, columns, channels = out.shape
        bytesPerLine = columns * channels
        QImg = QImage(out.data, columns, rows, bytesPerLine, QImage.Format_RGB888)
        self.labelResult.clear()
        self.labelResult.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.labelResult.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def btnSaltAndPepper_Clicked(self):
        """
        椒盐噪声
        """
        # 如果没有打开图片，则不执行操作
        if not self.is_read:
            return
        image = self.captured
        # 待输出的图片
        out = np.zeros(image.shape, np.uint8)
        # 椒盐噪声的阈值
        prob = 0.2
        thres = 1 - prob
        # 遍历图像，获取叠加噪声后的图像
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                rdn = random.random()
                if rdn < prob:
                    # 添加胡椒噪声
                    out[i][j] = 0
                elif rdn > thres:
                    # 添加食盐噪声
                    out[i][j] = 255
                else:
                    # 不添加噪声
                    out[i][j] = image[i][j]
        self.output = out
        out = cv.cvtColor(out, cv.COLOR_BGR2RGB)
        rows, columns, channels = out.shape
        bytesPerLine = columns * channels
        QImg = QImage(out.data, columns, rows, bytesPerLine, QImage.Format_RGB888)
        self.labelResult.clear()
        self.labelResult.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.labelResult.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def btnDetect_Clicked(self):
        """
        车牌检测
        """
        if self.filename is None:
            return
        parser = argparse.ArgumentParser()
        parser.add_argument('--classify', nargs='+', type=str, default=True, help='True rec')
        parser.add_argument('--det-weights', nargs='+', type=str, default=r'weights/last.pt',
                            help='model.pt path(s)')
        parser.add_argument('--rec-weights', nargs='+', type=str,
                            default='weights/lprnet_green.pth',
                            help='model.pt path(s)')
        parser.add_argument('--rec-weights-b', nargs='+', type=str,
                            default='weights/lprnet-pretrain.pth',
                            help='model.pt path(s)')
        parser.add_argument('--source', type=str, default=str(self.filename), help='source')  # file/folder, 0 for webcam
        parser.add_argument('--output', type=str, default=r'runs/result', help='rec_result folder')  # rec_result folder
        parser.add_argument('--img-size', type=int, default=640, help='demo size (pixels)')
        parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
        parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--view-img', action='store_true', help='display results')
        parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented demo')
        parser.add_argument('--update', action='store_true', help='update all models')
        parser.add_argument("--scale", dest='scale', help="scale the iamge", default=1, type=int)
        parser.add_argument('--mini_lp', dest='mini_lp', help="Minimum face to be detected", default=(50, 15), type=int)
        opt = parser.parse_args()
        print(opt)
        with torch.no_grad():
            if opt.update:  # update all models (to fix SourceChangeWarning)
                for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt', 'yolov3-spp.pt']:
                    detect(opt)
                    create_pretrained(opt.weights, opt.weights)
            else:
                detect(opt)
        file_str = self.filename
        i = len(file_str) - 1
        str_list = []
        while file_str[i] != '/':
            str_list.append(file_str[i])
            i = i - 1
        str_list.reverse()
        output_img = cv.imread('runs/result/' + ''.join(str_list))
        output_img = cv.cvtColor(output_img, cv.COLOR_BGR2RGB)
        rows, columns, channels = output_img.shape
        bytesPerLine = columns * channels
        QImg = QImage(output_img.data, columns, rows, bytesPerLine, QImage.Format_RGB888)
        self.labelResult_3.clear()
        self.labelResult_3.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.labelResult_3.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def btnDetectVideo_Clicked(self):
        """
        视频检测
        """
        Videofilename, _ = QFileDialog.getOpenFileName(self, '打开视频')
        if Videofilename:
            parser = argparse.ArgumentParser()
            parser.add_argument('--classify', nargs='+', type=str, default=True, help='True rec')
            parser.add_argument('--det-weights', nargs='+', type=str, default=r'weights/last.pt',
                                help='model.pt path(s)')
            parser.add_argument('--rec-weights', nargs='+', type=str,
                                default='weights/lprnet_green.pth',
                                help='model.pt path(s)')
            parser.add_argument('--rec-weights-b', nargs='+', type=str,
                                default='weights/lprnet-pretrain.pth',
                                help='model.pt path(s)')
            parser.add_argument('--source', type=str, default=str(Videofilename), help='source')  # file/folder, 0 for webcam
            parser.add_argument('--output', type=str, default=r'runs/result', help='rec_result folder')  # rec_result folder
            parser.add_argument('--img-size', type=int, default=640, help='demo size (pixels)')
            parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
            parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
            parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
            parser.add_argument('--view-img', action='store_true', help='display results')
            parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
            parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
            parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
            parser.add_argument('--augment', action='store_true', help='augmented demo')
            parser.add_argument('--update', action='store_true', help='update all models')
            parser.add_argument("--scale", dest='scale', help="scale the iamge", default=1, type=int)
            parser.add_argument('--mini_lp', dest='mini_lp', help="Minimum face to be detected", default=(50, 15), type=int)
            opt = parser.parse_args()
            print(opt)
            with torch.no_grad():
                if opt.update:  # update all models (to fix SourceChangeWarning)
                    for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt', 'yolov3-spp.pt']:
                        detect(opt)
                        create_pretrained(opt.weights, opt.weights)
                else:
                    detect(opt)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())
