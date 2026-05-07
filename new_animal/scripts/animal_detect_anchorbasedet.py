"""
K230 动物识别推理代码 (AnchorBaseDet 256×256)
==============================================
使用 best_AnchorBaseDet_can2_5_n_20260427180224.kmodel，
标注文件 deploy_config.json 中 5 类动物（tiger/wolf/monkey/peacock/elephant）。

使用方法：
    将 kmodel 放入 /sdcard/examples/mycode/
    python animal_detect_anchorbasedet.py

参数来源：
    推理参数（阈值、anchors、mean/std）均取自 deploy_config.json，
    请确保配套的 deploy_config.json 在同一目录下。
"""

from libs.PipeLine import PipeLine
from libs.AIBase import AIBase
from libs.AI2D import Ai2d
from libs.Utils import *
import os, sys, ujson, gc, math
from media.media import *
from media.sensor import *
import nncase_runtime as nn
import ulab.numpy as np
import image
import aidemo


class AnchorBaseDetApp(AIBase):
    def __init__(self,
                 kmodel_path,
                 labels,
                 model_input_size,
                 max_boxes_num,
                 confidence_threshold=0.5,
                 nms_threshold=0.5,
                 rgb888p_size=[256, 256],
                 display_size=[1920, 1080],
                 debug_mode=0):
        super().__init__(kmodel_path, model_input_size, rgb888p_size, debug_mode)
        self.kmodel_path = kmodel_path
        self.labels = labels
        self.model_input_size = model_input_size
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.max_boxes_num = max_boxes_num
        self.rgb888p_size = [ALIGN_UP(rgb888p_size[0], 16), rgb888p_size[1]]
        self.display_size = [ALIGN_UP(display_size[0], 16), display_size[1]]
        self.debug_mode = debug_mode
        self.color_four = get_colors(len(self.labels))
        self.x_factor = float(self.rgb888p_size[0]) / self.model_input_size[0]
        self.y_factor = float(self.rgb888p_size[1]) / self.model_input_size[1]
        self.ai2d = Ai2d(debug_mode)
        self.ai2d.set_ai2d_dtype(nn.ai2d_format.NCHW_FMT, nn.ai2d_format.NCHW_FMT, np.uint8, np.uint8)

    def config_preprocess(self, input_image_size=None):
        with ScopedTiming("set preprocess config", self.debug_mode > 0):
            ai2d_input_size = input_image_size if input_image_size else self.rgb888p_size
            top, bottom, left, right, self.scale = letterbox_pad_param(self.rgb888p_size, self.model_input_size)
            self.ai2d.pad([0, 0, 0, 0, top, bottom, left, right], 0, [128, 128, 128])
            self.ai2d.resize(nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel)
            self.ai2d.build([1, 3, ai2d_input_size[1], ai2d_input_size[0]],
                            [1, 3, self.model_input_size[1], self.model_input_size[0]])

    def preprocess(self, input_np):
        with ScopedTiming("preprocess", self.debug_mode > 0):
            return [nn.from_numpy(input_np)]

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

    def draw_result(self, pl, dets):
        with ScopedTiming("display_draw", self.debug_mode > 0):
            if dets:
                pl.osd_img.clear()
                for i in range(len(dets[0])):
                    x, y, w, h = map(lambda v: int(round(v, 0)), dets[0][i])
                    label_id = dets[1][i]
                    score = dets[2][i]
                    pl.osd_img.draw_rectangle(x, y, w, h, color=self.color_four[label_id], thickness=4)
                    label_text = " " + self.labels[label_id] + " " + str(round(score, 2))
                    pl.osd_img.draw_string_advanced(x, y - 50, 32, label_text, color=self.color_four[label_id])
            else:
                pl.osd_img.clear()


if __name__ == "__main__":
    # 显示模式，默认 hdmi，可选 hdmi/lcd/lt9611/st7701/hx8399
    display_mode = "hdmi"

    # 模型路径
    kmodel_path = "/sdcard/examples/mycode/best_AnchorBaseDet_can2_5_n_20260427180224.kmodel"

    # ============================================================
    # 以下参数来源于配套的 deploy_config.json
    # 如调整，请确保与模型训练时的配置一致
    # ============================================================
    # 动物类别标签（5 类，与训练数据一致）
    labels = ["tiger", "wolf", "monkey", "peacock", "elephant"]
    # 模型输入分辨率（AnchorBaseDet 256×256）
    model_input_size = [256, 256]
    # 摄像头分辨率（与模型输入保持一致，减少预处理开销）
    rgb888p_size = [256, 256]
    # 检测参数（来自 deploy_config.json）
    confidence_threshold = 0.5   # 置信度阈值
    nms_threshold = 0.5          # NMS 阈值
    max_boxes_num = 30           # 最大检测框数

    # 初始化 PipeLine（摄像头 + 显示）
    pl = PipeLine(rgb888p_size=rgb888p_size, display_mode=display_mode)
    pl.create(sensor = Sensor(id=1,fps=30))
    display_size = pl.get_display_size()

    # 初始化动物检测器
    animal_det = AnchorBaseDetApp(
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

    print("[INFO] K230 Animal Detection Started (AnchorBaseDet 256)")
    print("[INFO] Labels:", labels)
    print("[INFO] Model:", kmodel_path.split("/")[-1])
    print("[INFO] Press Ctrl+C to stop.")

    frame_count = 0
    try:
        while True:
            with ScopedTiming("total", 1):
                # 获取当前帧
                img = pl.get_frame()
                # 推理当前帧
                res = animal_det.run(img)
                # 绘制结果到 PipeLine 的 osd 图像
                animal_det.draw_result(pl, res)
                # 显示当前的绘制结果
                pl.show_image()
                # 每帧回收内存，防止累积
                gc.collect()
                frame_count += 1
                if frame_count % 300 == 0:
                    print("[INFO] Running... frames:", frame_count)
    except KeyboardInterrupt:
        print("[INFO] Stopped by user.")
    except Exception as e:
        print("[ERROR]", e)
        import sys
        sys.print_exception(e)
    finally:
        animal_det.deinit()
        pl.destroy()
        gc.collect()
        print("[INFO] Resources released.")
