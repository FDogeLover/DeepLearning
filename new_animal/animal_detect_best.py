"""
K230 动物识别推理代码 (使用 best.kmodel)
=========================================
基于 K230 SDK 示例 object_detect_yolov8n.py 的调用方式

模型文件: best.kmodel（约 11.6 MB，比 best_nano_320.kmodel 更大，精度更高）

使用方法：
    1. 将 best.kmodel 放入 /sdcard/examples/kmodel/
    2. python animal_detect_best.py

标签：monkey, elephant, peacock, tiger, wolf（共 5 类）

注意事项：
    - best.kmodel 较大（11.6MB），对 K230 内存有一定压力
    - 默认 rgb888p_size=[320,320] 与 model_input_size=[320,320] 保持一致（参考原版写法）
    - 如需更高精度，可尝试调高 rgb888p_size，但需确保 K230 内存足够
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


class AnimalDetectBestApp(AIBase):
    def __init__(self,
                 kmodel_path,
                 labels,
                 model_input_size,
                 max_boxes_num,
                 confidence_threshold=0.3,
                 nms_threshold=0.4,
                 rgb888p_size=[320, 320],
                 display_size=[1920, 1080],
                 debug_mode=0):
        super().__init__(kmodel_path, model_input_size, rgb888p_size, debug_mode)
        self.kmodel_path = kmodel_path
        self.labels = labels
        # 模型输入分辨率
        self.model_input_size = model_input_size
        # 阈值设置
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.max_boxes_num = max_boxes_num
        # sensor 给到 AI 的图像分辨率
        self.rgb888p_size = [ALIGN_UP(rgb888p_size[0], 16), rgb888p_size[1]]
        # 显示分辨率
        self.display_size = [ALIGN_UP(display_size[0], 16), display_size[1]]
        self.debug_mode = debug_mode
        # 检测框预置颜色值
        self.color_four = get_colors(len(self.labels))
        # 宽高缩放比例
        self.x_factor = float(self.rgb888p_size[0]) / self.model_input_size[0]
        self.y_factor = float(self.rgb888p_size[1]) / self.model_input_size[1]
        # Ai2d 实例，用于实现模型预处理
        self.ai2d = Ai2d(debug_mode)
        # 设置 Ai2d 的输入输出格式和类型
        self.ai2d.set_ai2d_dtype(nn.ai2d_format.NCHW_FMT, nn.ai2d_format.NCHW_FMT, np.uint8, np.uint8)

    # 配置预处理操作
    def config_preprocess(self, input_image_size=None):
        with ScopedTiming("set preprocess config", self.debug_mode > 0):
            ai2d_input_size = input_image_size if input_image_size else self.rgb888p_size
            top, bottom, left, right, self.scale = letterbox_pad_param(self.rgb888p_size, self.model_input_size)
            # 配置 padding 和 resize 预处理
            self.ai2d.pad([0, 0, 0, 0, top, bottom, left, right], 0, [128, 128, 128])
            self.ai2d.resize(nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel)
            self.ai2d.build([1, 3, ai2d_input_size[1], ai2d_input_size[0]],
                            [1, 3, self.model_input_size[1], self.model_input_size[0]])

    def preprocess(self, input_np):
        with ScopedTiming("preprocess", self.debug_mode > 0):
            return [nn.from_numpy(input_np)]

    # 自定义当前任务的后处理
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

    # 绘制结果
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
    # ============================================================
    # [重要] 断连原因与修复说明
    #
    # 对比原版 object_detect_yolov8n.py 发现：
    #   原版: rgb888p_size=[224,224] = model_input_size=[224,224]  ✅ 一致
    #   旧版: rgb888p_size=[640,480] ≠ model_input_size=[320,320]  ❌ 多了缩放开销
    #
    # 核心修复：让 rgb888p_size 和 model_input_size 保持一致，
    #           就像原版 SDK 示例那样，避免额外的内存开销。
    #
    # 此外，best.kmodel 比 yolov8n_224.kmodel 大 5.7MB，
    # 所以 rgb888p_size 也应适当降低以补偿模型多占用的内存。
    # ============================================================
    # 摄像头分辨率 == 模型输入分辨率（参考原版写法，保持一致）
    rgb888p_size = [320, 320]
    # 模型路径（使用 best.kmodel）
    kmodel_path = "/sdcard/examples/mycode/best.kmodel"
    # 动物类别标签（与训练数据一致，共 5 类）
    labels = ["monkey", "elephant", "peacock", "tiger", "wolf"]
    # 模型输入分辨率（与 best.kmodel 导出时一致，与 rgb888p_size 保持一致）
    model_input_size = [320, 320]
    # 检测参数
    confidence_threshold = 0.3       # 置信度阈值
    nms_threshold = 0.4              # NMS 阈值
    max_boxes_num = 30               # 最大检测框数

    # 初始化 PipeLine（摄像头 + 显示）
    pl = PipeLine(rgb888p_size=rgb888p_size, display_mode=display_mode)
    pl.create()
    display_size = pl.get_display_size()

    # 初始化动物检测器
    animal_det = AnimalDetectBestApp(
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

    print("[INFO] K230 Animal Detection Started (best.kmodel)")
    print("[INFO] Labels:", labels)
    print("[INFO] rgb888p_size:", rgb888p_size)
    print("[INFO] max_boxes_num:", max_boxes_num)
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
                # 每一帧都强制回收内存，防止累积
                gc.collect()
                frame_count += 1
                # 每 300 帧（约 10 秒）打印一次运行状态
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
