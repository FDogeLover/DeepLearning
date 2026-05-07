"""
K230 动物识别推理代码 (YOLOv8n, 320×320)
==========================================
基于 K230 官方 yolov8n_obb.py 示例格式
对齐官方示例：320×320 输入、不使用 ALIGN_UP、不覆写 preprocess()

模型：YOLOv8n — 320×320 输入，5 类动物检测
类别：tiger / wolf / monkey / peacock / elephant

使用方法：
    将 animal_yolov8n_320.kmodel 放入 /sdcard/examples/kmodel/
    通过 CanMV IDE 运行本脚本
"""

from libs.PipeLine import PipeLine
from libs.AIBase import AIBase
from libs.AI2D import Ai2d
from libs.Utils import *
import os, sys, ujson, gc, math
from media.media import *
import nncase_runtime as nn
import ulab.numpy as np
import image
import aidemo


class AnimalDetectApp(AIBase):
    def __init__(self, kmodel_path, labels, model_input_size, max_boxes_num,
                 confidence_threshold=0.5, nms_threshold=0.2,
                 rgb888p_size=[640, 480], display_size=[800, 480], debug_mode=0):
        super().__init__(kmodel_path, model_input_size, rgb888p_size, debug_mode)
        self.kmodel_path = kmodel_path
        self.labels = labels
        self.d = {i: 0 for i in range(len(self.labels))}
        # 模型输入分辨率
        self.model_input_size = model_input_size
        # 阈值设置
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.max_boxes_num = max_boxes_num
        # sensor 给到 AI 的图像分辨率（不对齐，对齐官方示例）
        self.rgb888p_size = [rgb888p_size[0], rgb888p_size[1]]
        # 显示分辨率
        self.display_size = [display_size[0], display_size[1]]
        self.debug_mode = debug_mode
        # 检测框预置颜色值
        self.color_four = get_colors(len(self.labels))
        self.scale = 1.0
        # Ai2d 实例，用于实现模型预处理
        self.ai2d = Ai2d(debug_mode)
        # 设置 Ai2d 的输入输出格式和类型
        self.ai2d.set_ai2d_dtype(nn.ai2d_format.NCHW_FMT, nn.ai2d_format.NCHW_FMT, np.uint8, np.uint8)

    # 配置预处理操作：letterbox pad + resize
    def config_preprocess(self, input_image_size=None):
        with ScopedTiming("set preprocess config", self.debug_mode > 0):
            ai2d_input_size = input_image_size if input_image_size else self.rgb888p_size
            top, bottom, left, right, self.scale = letterbox_pad_param(self.rgb888p_size, self.model_input_size)
            self.ai2d.pad([0, 0, 0, 0, top, bottom, left, right], 0, [128, 128, 128])
            self.ai2d.resize(nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel)
            self.ai2d.build([1, 3, ai2d_input_size[1], ai2d_input_size[0]],
                            [1, 3, self.model_input_size[1], self.model_input_size[0]])

    # 不覆写 preprocess()，使用 AIBase 基类默认实现

    # YOLOv8 后处理
    def postprocess(self, results):
        with ScopedTiming("postprocess", self.debug_mode > 0):
            new_result = results[0][0].transpose()
            det_res = aidemo.yolov8_det_postprocess(
                new_result.copy(),
                [self.rgb888p_size[1], self.rgb888p_size[0]],
                [self.model_input_size[1], self.model_input_size[0]],
                [self.display_size[1], self.display_size[0]],
                len(self.labels),
                self.confidence_threshold,
                self.nms_threshold,
                self.max_boxes_num
            )
            return det_res

    # 绘制检测框和标签
    def draw_result(self, pl, dets):
        with ScopedTiming("display_draw", self.debug_mode > 0):
            pl.osd_img.clear()
            if dets:
                for i in range(len(dets[0])):
                    x, y, w, h = map(lambda v: int(round(v, 0)), dets[0][i])
                    label_id = dets[1][i]
                    score = dets[2][i]
                    pl.osd_img.draw_rectangle(x, y, w, h, color=self.color_four[label_id], thickness=4)
                    label_text = " " + self.labels[label_id] + " " + str(round(score, 2))
                    pl.osd_img.draw_string_advanced(x, y - 50, 24, label_text, color=self.color_four[label_id])
                    self.d[label_id] += 1
                # 顶部显示各类别计数
                text = ""
                for j in range(len(self.labels)):
                    if self.d[j] != 0:
                        text += self.labels[j] + ": " + str(self.d[j]) + ";  "
                        self.d[j] = 0
                pl.osd_img.draw_string_advanced(50, 50, 24, text, color=[0, 255, 0])


if __name__ == "__main__":
    # ========== 显示模式 ==========
    display_mode = "hdmi"

    # ========== 摄像头 AI 输入分辨率（对齐官方示例，不 ALIGN_UP） ==========
    rgb888p_size = [640, 480]

    # ========== 模型路径 ==========
    kmodel_path = "/sdcard/examples/kmodel/animal_yolov8n_320.kmodel"

    # ========== 动物类别标签（与训练一致） ==========
    labels = ["tiger", "wolf", "monkey", "peacock", "elephant"]

    # ========== 检测参数（对齐官方示例，confidence 偏低） ==========
    confidence_threshold = 0.1
    nms_threshold = 0.6
    max_boxes_num = 30

    # ========== 模型输入尺寸 ==========
    model_input_size = [320, 320]

    # ========== 初始化 PipeLine ==========
    pl = PipeLine(rgb888p_size=rgb888p_size, display_mode=display_mode)
    pl.create()
    display_size = pl.get_display_size()

    # ========== 初始化动物检测器 ==========
    animal_det = AnimalDetectApp(
        kmodel_path=kmodel_path,
        labels=labels,
        model_input_size=model_input_size,
        max_boxes_num=max_boxes_num,
        confidence_threshold=confidence_threshold,
        nms_threshold=nms_threshold,
        rgb888p_size=rgb888p_size,
        display_size=display_size,
        debug_mode=0
    )
    animal_det.config_preprocess()

    print("=" * 50)
    print("  K230 Animal Detection (YOLOv8n 320x320)")
    print("  Model: animal_yolov8n_320.kmodel")
    print("  Classes:", labels)
    print("  Confidence:", confidence_threshold, " NMS:", nms_threshold)
    print("  Press Ctrl+C to stop")
    print("=" * 50)

    try:
        while True:
            with ScopedTiming("total", 1):
                img = pl.get_frame()
                res = animal_det.run(img)
                animal_det.draw_result(pl, res)
                pl.show_image()
                gc.collect()
    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
    except Exception as e:
        print("[ERROR]", e)
        import sys
        sys.print_exception(e)
    finally:
        animal_det.deinit()
        pl.destroy()
        gc.collect()
        print("[INFO] Resources released.")
