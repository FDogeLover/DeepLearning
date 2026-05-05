import sys
from PyQt5.QtWidgets import QApplication
from ui.drawing_board import DrawingBoard
from model.similarity import CLIPSimilarity

def run_drawing_board():
    """启动绘图板界面"""
    app = QApplication(sys.argv)
    window = DrawingBoard(prompt_file="assets/prompts.txt")
    window.show()
    sys.exit(app.exec_())

def test_similarity():
    """测试图像相似度"""
    checker = CLIPSimilarity()

    # 测试：图像-图像相似度
    image1_path = "assets/参考图/心形.png"
    image2_path = "assets/参考图/心形.png"
    sim_img = checker.image_to_image_similarity(image1_path, image2_path)
    print(f"图像相似度分数 ({image1_path} vs {image2_path}) ：{sim_img:.4f}")

    # 测试：图像-文本相似度
    prompt = "心形"
    sim_text = checker.image_to_prompt_similarity(image1_path, prompt)
    print(f"图像与文本 ('{prompt}') 匹配度：{sim_text:.4f}")

if __name__ == "__main__":
    # ====== 选择模式 ======
    mode = "test"  # 改成 "test" 可以切换成测试相似度

    if mode == "gui":
        run_drawing_board()
    elif mode == "test":
        test_similarity()
    else:
        print("未知模式，请设置 mode = 'gui' 或 'test'")
