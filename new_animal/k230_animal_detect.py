"""
K230 动物识别推理代码 (YOLOv5 Nano)
=====================================
基于 K230 SDK 示例 object_detect_yolov8n.py 的调用方式

使用方法：
    将 best_nano_320.kmodel 放入 /sdcard/examples/kmodel/
    python k230_animal_detect.py

注意：模型文件为 YOLOv5 Nano (320x320)，比原始 640x640 省电省内存
"""

from libs.PipeLine import PipeLine
from libs.AIBase import AIBase
from libs.AI2D import Ai2d
from libs.Utils import *
import os,sys,ujson,gc,math
from media.media import *
import nncase_runtime as nn
import ulab.numpy as np
import image
import aidemo


class AnimalDetectApp(AIBase):
    def __init__(self,
                 kmodel_path,
                 labels,
                 model_input_size,
                 max_boxes_num,
                 confidence_threshold=0.3,
                 nms_threshold=0.4,
                 rgb888p_size=[640,480],
                 display_size=[1920,1080],
                 debug_mode=0):
        super().__init__(kmodel_path,model_input_size,rgb888p_size,debug_mode)
        self.kmodel_path=kmodel_path
        self.labels=labels
        self.model_input_size=model_input_size
        self.confidence_threshold=confidence_threshold
        self.nms_threshold=nms_threshold
        self.max_boxes_num=max_boxes_num
        self.rgb888p_size=[ALIGN_UP(rgb888p_size[0],16),rgb888p_size[1]]
        self.display_size=[ALIGN_UP(display_size[0],16),display_size[1]]
        self.debug_mode=debug_mode
        self.color_four=get_colors(len(self.labels))
        self.x_factor=float(self.rgb888p_size[0])/self.model_input_size[0]
        self.y_factor=float(self.rgb888p_size[1])/self.model_input_size[1]
        self.ai2d=Ai2d(debug_mode)
        self.ai2d.set_ai2d_dtype(nn.ai2d_format.NCHW_FMT,nn.ai2d_format.NCHW_FMT,np.uint8,np.uint8)

    def config_preprocess(self,input_image_size=None):
        with ScopedTiming("set preprocess config",self.debug_mode > 0):
            ai2d_input_size=input_image_size if input_image_size else self.rgb888p_size
            top,bottom,left,right,self.scale=letterbox_pad_param(self.rgb888p_size,self.model_input_size)
            self.ai2d.pad([0,0,0,0,top,bottom,left,right],0,[128,128,128])
            self.ai2d.resize(nn.interp_method.tf_bilinear,nn.interp_mode.half_pixel)
            self.ai2d.build([1,3,ai2d_input_size[1],ai2d_input_size[0]],[1,3,self.model_input_size[1],self.model_input_size[0]])

    def preprocess(self,input_np):
        with ScopedTiming("preprocess",self.debug_mode > 0):
            return [nn.from_numpy(input_np)]

    def postprocess(self,results):
        with ScopedTiming("postprocess",self.debug_mode > 0):
            # YOLOv5 Nano 输出 shape: [1, 6300, 10] (5类 + 4坐标 + 1置信度)
            # 转置后送入 aidemo 通用后处理
            new_result = results[0][0].transpose()
            det_res = aidemo.yolov8_det_postprocess(
                new_result.copy(),
                [self.rgb888p_size[1],self.rgb888p_size[0]],
                [self.model_input_size[1],self.model_input_size[0]],
                [self.display_size[1],self.display_size[0]],
                len(self.labels),
                self.confidence_threshold,
                self.nms_threshold,
                self.max_boxes_num
            )
            return det_res

    def draw_result(self,pl,dets):
        with ScopedTiming("display_draw",self.debug_mode > 0):
            if dets:
                pl.osd_img.clear()
                for i in range(len(dets[0])):
                    x,y,w,h = map(lambda v: int(round(v,0)),dets[0][i])
                    label_id=dets[1][i]
                    score=dets[2][i]
                    pl.osd_img.draw_rectangle(x,y,w,h,color=self.color_four[label_id],thickness=4)
                    label_text=" " + self.labels[label_id] + " " + str(round(score,2))
                    pl.osd_img.draw_string_advanced(x,y-50,32,label_text,color=self.color_four[label_id])
            else:
                pl.osd_img.clear()


if __name__=="__main__":
    # 显示模式，默认hdmi，可选hdmi/lcd
    display_mode="hdmi"
    # AI输入分辨率（摄像头画面分辨率）
    rgb888p_size=[640,480]
    # 模型路径（使用 YOLOv5 Nano 320×320 版）
    kmodel_path="/sdcard/examples/kmodel/best_nano_320.kmodel"
    # 动物类别标签
    labels=["monkey","elephant","peacock","tiger","wolf"]
    # 检测参数
    confidence_threshold=0.3
    nms_threshold=0.4
    max_boxes_num=30

    # 初始化 PipeLine（摄像头+显示）
    pl=PipeLine(rgb888p_size=rgb888p_size,display_mode=display_mode)
    pl.create()
    display_size=pl.get_display_size()

    # 初始化动物检测器
    animal_det=AnimalDetectApp(
        kmodel_path=kmodel_path,
        labels=labels,
        model_input_size=[320,320],
        max_boxes_num=max_boxes_num,
        confidence_threshold=confidence_threshold,
        nms_threshold=nms_threshold,
        rgb888p_size=rgb888p_size,
        display_size=display_size,
        debug_mode=0
    )
    animal_det.config_preprocess()

    print("[INFO] K230 Animal Detection Started (YOLOv5 Nano 320)")
    print("[INFO] Labels:",labels)
    print("[INFO] Press Ctrl+C to stop.")

    try:
        while True:
            with ScopedTiming("total",1):
                # 获取当前帧
                img=pl.get_frame()
                # 推理当前帧
                res=animal_det.run(img)
                # 绘制结果
                animal_det.draw_result(pl,res)
                # 显示
                pl.show_image()
                gc.collect()
    except KeyboardInterrupt:
        print("[INFO] Stopped by user.")
    except Exception as e:
        print("[ERROR]",e)
        import sys
        sys.print_exception(e)
    finally:
        animal_det.deinit()
        pl.destroy()
        gc.collect()
        print("[INFO] Resources released.")
