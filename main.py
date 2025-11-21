import os
import sys
import cv2
import numpy as np
from skeleton_detector import skeleton_longest_endpoints
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel,
                             QPushButton, QVBoxLayout, QFileDialog, QTextEdit,
                             QScrollArea, QSizePolicy, QDialog, QLineEdit,
                             QFormLayout, QMessageBox)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap


class ScrollableImageLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(640, 480)
        self.scale_factor = 1.0
        self._pixmap = None  # 保存原始像素图

    def setPixmap(self, pixmap):
        self._pixmap = pixmap
        self.update_scaled_pixmap()

    def update_scaled_pixmap(self):
        if self._pixmap:
            scaled_pixmap = self._pixmap.scaled(
                self._pixmap.size() * self.scale_factor,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            super().setPixmap(scaled_pixmap)
            self.adjustSize()

    def wheelEvent(self, event):
        # 按住Ctrl时使用滚轮缩放
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            delta = event.angleDelta().y()
            if delta > 0:
                self.zoom(1.25)
            else:
                self.zoom(0.8)
            event.accept()
        else:
            super().wheelEvent(event)

    def zoom(self, factor):
        self.scale_factor *= factor
        self.scale_factor = max(0.1, min(self.scale_factor, 10.0))  # 限制缩放范围
        self.update_scaled_pixmap()

    def mouseDoubleClickEvent(self, event):
        # 双击重置缩放
        self.scale_factor = 1.0
        self.update_scaled_pixmap()
        event.accept()


class ParamsDialog(QDialog):
    def __init__(self, parent=None, params=None):
        super().__init__(parent)
        self.setWindowTitle("调整图像处理参数")
        self.params = params.copy() if params else {}
        self.init_ui()

    def init_ui(self):
        layout = QFormLayout()

        # 高斯模糊参数
        self.blur_size_edit = QLineEdit(str(self.params.get('blur_size', 5)))
        # Canny参数
        self.canny_low_edit = QLineEdit(str(self.params.get('canny_low', 50)))
        self.canny_high_edit = QLineEdit(str(self.params.get('canny_high', 150)))
        # 形态学操作参数
        self.morph_size_edit = QLineEdit(str(self.params.get('morph_size', 5)))
        # 轮廓筛选参数
        self.min_contour_length_edit = QLineEdit(str(self.params.get('min_contour_length', 50)))

        layout.addRow("高斯模糊核大小 (奇数):", self.blur_size_edit)
        layout.addRow("Canny低阈值:", self.canny_low_edit)
        layout.addRow("Canny高阈值:", self.canny_high_edit)
        layout.addRow("形态学核大小:", self.morph_size_edit)
        layout.addRow("最小轮廓长度:", self.min_contour_length_edit)

        # 比例尺参数
        self.pixels_per_mm_edit = QLineEdit(str(self.params.get('pixels_per_mm', 84.07)))  # 默认值840.7像素/10mm
        layout.addRow("像素/毫米比例尺:", self.pixels_per_mm_edit)

        btn_confirm = QPushButton("确认")
        btn_confirm.clicked.connect(self.validate_and_accept)
        layout.addRow(btn_confirm)

        self.setLayout(layout)

    def validate_and_accept(self):
        try:
            params = {
                'blur_size': int(self.blur_size_edit.text()),
                'canny_low': int(self.canny_low_edit.text()),
                'canny_high': int(self.canny_high_edit.text()),
                'morph_size': int(self.morph_size_edit.text()),
                'min_contour_length': int(self.min_contour_length_edit.text()),
                'pixels_per_mm': float(self.pixels_per_mm_edit.text())
            }

            # 验证参数有效性
            if params['blur_size'] % 2 == 0 or params['blur_size'] < 1:
                raise ValueError("模糊核大小必须是正奇数")
            if params['canny_low'] >= params['canny_high']:
                raise ValueError("Canny低阈值必须小于高阈值")
            if params['morph_size'] < 1:
                raise ValueError("形态学核大小必须大于0")
            if params['min_contour_length'] < 1:
                raise ValueError("最小轮廓长度必须大于0")
            if params['pixels_per_mm'] <= 0:
                raise ValueError("比例尺必须大于0")

            self.params = params
            self.accept()
        except ValueError as e:
            QMessageBox.warning(self, "输入错误", f"参数错误: {str(e)}")


class PowderSegmentApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("粉段检测软件 v2.0")
        self.setGeometry(100, 100, 800, 600)

        # 初始化默认参数
        self.params = {
            'blur_size': 5,
            'canny_low': 50,
            'canny_high': 150,
            'morph_size': 5,
            'min_contour_length': 50,
            'pixels_per_mm': 84.07  # 默认值840.7像素对应10mm
        }

        self.current_image = None
        self.processed_image = None  # 保存最近一次处理后的结果图像
        self.init_ui()

    def init_ui(self):
        self.image_label = ScrollableImageLabel()
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.image_label)
        scroll_area.setWidgetResizable(True)

        self.btn_file = QPushButton("选择本地图片")
        self.btn_params = QPushButton("参数调整")
        self.btn_save = QPushButton("导出结果图片")
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMaximumHeight(150)

        # 按钮布局
        button_layout = QVBoxLayout()
        button_layout.addWidget(self.btn_file)
        button_layout.addWidget(self.btn_params)
        button_layout.addWidget(self.btn_save)

        main_layout = QVBoxLayout()
        main_layout.addWidget(scroll_area)
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.result_text)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # 连接信号
        self.btn_file.clicked.connect(self.open_image)
        self.btn_params.clicked.connect(self.open_params_dialog)
        self.btn_save.clicked.connect(self.save_result_image)

    def open_params_dialog(self):
        dialog = ParamsDialog(self, self.params)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.params = dialog.params
            self.result_text.append("参数已更新：")
            self.result_text.append("\n".join([f"{k}: {v}" for k, v in self.params.items()]))

            # 如果有已加载的图像，立即重新处理
            if self.current_image is not None:
                self.process_image(self.current_image)

    def open_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "Image Files (*.jpg *.jpeg *.png)")
        if file_path:
            image = cv2.imread(file_path)
            if image is not None:
                self.current_image = image.copy()
                self.process_image(image)

    def save_result_image(self):
        """
        将最近一次处理后的结果图像导出到本地文件。
        """
        if self.processed_image is None:
            QMessageBox.information(self, "提示", "当前没有可导出的结果图片，请先选择图片并完成处理。")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存结果图片",
            "processed_result.png",
            "Image Files (*.png *.jpg *.jpeg)"
        )
        if not file_path:
            return

        # 使用 OpenCV 保存图像（注意 BGR 通道顺序）
        try:
            cv2.imwrite(file_path, self.processed_image)
            QMessageBox.information(self, "保存成功", f"结果图片已保存到:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"保存结果图片时发生错误:\n{str(e)}")

    def process_image(self, image):
        """
        改进版：
        - 使用灰度 + Otsu 二值化 + 形态学闭运算
        - 用 minAreaRect 拿到旋转矩形
        - 通过 长宽比 + 面积 筛掉圆形/小块，只保留条状粉末段
        - 有效长度 = 长边 - 短边（大致扣掉两端圆头/椭圆弧）
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # 高斯模糊
            blur_size = self.params['blur_size']
            gray_blur = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)

            # Otsu 自动阈值二值化（默认认为粉末比背景更亮）
            # 如果你的图像里粉末更暗，可以把 THRESH_BINARY 改成 THRESH_BINARY_INV
            # Otsu 自动阈值二值化 + 自适应前景极性
            # 先假定粉末比背景更亮
            _, binary = cv2.threshold(
                gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

            # 根据白色像素占比，自动决定是否需要反色
            white_ratio = cv2.countNonZero(binary) / binary.size

            # 如果白色占比过大，通常说明背景被分成了白色，此时反色，
            # 让粉末变成白色、背景为黑色，便于后续轮廓检测
            if white_ratio > 0.5:
                binary = cv2.bitwise_not(binary)
                self.result_text.append("自动判断：粉末较暗，使用反色二值化")
            else:
                self.result_text.append("自动判断：粉末较亮，使用正常二值化")

            # 形态学闭运算：连通碎裂的粉末
            morph_size = self.params['morph_size']
            kernel = np.ones((morph_size, morph_size), np.uint8)
            closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            open_kernel = np.ones((3, 3), np.uint8)
            cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, open_kernel)

            # 找轮廓
            contours, _ = cv2.findContours(
                cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            result_image = image.copy()
            detected_segments = []

            pixels_per_mm = self.params['pixels_per_mm']
            min_perimeter = self.params['min_contour_length']

            # 这两个阈值是关键：长宽比 + 面积（你可以按实际图像调整）
            MIN_ASPECT_RATIO = 3.0   # 最小长宽比，>3 基本可以认为是“条状”
            MIN_AREA = 200.0         # 最小面积，过滤掉很小的噪声点

            # 颜色列表，不同粉段使用不同颜色
            colors = [
                (0, 255, 0),
                (0, 0, 255),
                (255, 0, 0),
                (255, 255, 0),
                (255, 0, 255),
                (0, 255, 255)
            ]

            h_img, w_img = cleaned.shape

            for cnt in contours:
                perimeter = cv2.arcLength(cnt, True)
                if perimeter < min_perimeter:
                    continue

                area = cv2.contourArea(cnt)
                if area < MIN_AREA:
                    continue

                rect = cv2.minAreaRect(cnt)
                (cx, cy), (w, h), angle = rect

                if w <= 0 or h <= 0:
                    continue

                long_side = max(w, h)
                short_side = min(w, h)
                aspect_ratio = long_side / short_side if short_side > 0 else np.inf

                # 只要“细长条”，丢掉接近圆的轮廓
                if aspect_ratio < MIN_ASPECT_RATIO:
                    continue

                x, y, bw, bh = cv2.boundingRect(cnt)
                pad = max(4, morph_size)
                x0 = max(x - pad, 0)
                y0 = max(y - pad, 0)
                x1 = min(x + bw + pad, w_img)
                y1 = min(y + bh + pad, h_img)

                if x1 <= x0 or y1 <= y0:
                    continue

                roi_mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
                shifted_cnt = cnt - [x0, y0]
                cv2.drawContours(roi_mask, [shifted_cnt], -1, 255, -1)

                skel_info = skeleton_longest_endpoints(roi_mask)
                if skel_info and skel_info.get("length_px", 0) > 0:
                    effective_length_px = skel_info["length_px"]
                    pt1_global = (skel_info["pt1"][0] + x0, skel_info["pt1"][1] + y0)
                    pt2_global = (skel_info["pt2"][0] + x0, skel_info["pt2"][1] + y0)
                else:
                    # 回退到几何近似长度
                    effective_length_px = max(long_side - short_side, 0)
                    pt1_global = None
                    pt2_global = None

                length_mm = effective_length_px / pixels_per_mm
                height_mm = short_side / pixels_per_mm

                box = cv2.boxPoints(rect)
                box = np.intp(box)

                detected_segments.append({
                    "center": (cx, cy),
                    "box": box,
                    "long_side_px": long_side,
                    "short_side_px": short_side,
                    "length_px": effective_length_px,
                    "height_px": short_side,
                    "length_mm": length_mm,
                    "height_mm": height_mm,
                    "area": area,
                    "aspect_ratio": aspect_ratio,
                    "endpoints": (pt1_global, pt2_global)
                })

            if not detected_segments:
                self.result_text.setText(
                    "未检测到符合条件的条状粉末段：\n"
                    "- 可以尝试降低 MIN_ASPECT_RATIO 或 MIN_AREA；\n"
                    "- 或者调整模糊、形态学核、最小轮廓长度等参数。"
                )
                self.show_image(result_image)
                return

            # 按中心 x 坐标排序，从左到右编号
            detected_segments.sort(key=lambda s: s["center"][0])

            output_lines = []
            output_lines.append(f"检测到粉末段数量: {len(detected_segments)}")
            output_lines.append(f"当前比例尺: {pixels_per_mm:.2f} 像素/毫米\n")

            for idx, seg in enumerate(detected_segments, start=1):
                color = colors[(idx - 1) % len(colors)]
                box = seg["box"]
                cx, cy = seg["center"]
                length_mm = seg["length_mm"]
                height_mm = seg["height_mm"]

                # 画旋转矩形
                cv2.drawContours(result_image, [box], 0, color, 2)

                if seg.get("endpoints") and all(seg["endpoints"]):
                    cv2.circle(result_image, seg["endpoints"][0], 4, color, -1)
                    cv2.circle(result_image, seg["endpoints"][1], 4, color, -1)

                # 文本标注：编号 + d/h
                text = f"{idx}: d={length_mm:.2f} mm, h={height_mm:.2f} mm"
                # 将文字尽量放在粉末外侧：优先框上方，不够则框下方
                min_x = int(np.min(box[:, 0]))
                min_y = int(np.min(box[:, 1]))
                max_y = int(np.max(box[:, 1]))
                text_x = max(min_x, 5)
                text_y = min_y - 8
                if text_y < 15:
                    text_y = max_y + 18
                text_org = (text_x, text_y)
                cv2.putText(
                    result_image,
                    text,
                    text_org,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                    cv2.LINE_AA
                )

                output_lines.append(
                    f"粉末段 {idx}: "
                    f"中心=({cx:.1f}, {cy:.1f}), "
                    f"有效长度 d={length_mm:.2f} mm "
                    f"(像素 {seg['length_px']:.1f}), "
                    f"高度 h={height_mm:.2f} mm "
                    f"(像素 {seg['height_px']:.1f}), "
                    f"面积≈{seg['area']:.1f} 像素^2, "
                    f"长宽比≈{seg['aspect_ratio']:.2f}"
                )

            self.result_text.setText("\n".join(output_lines))

            # 保存处理后的结果图像，便于导出
            self.processed_image = result_image.copy()
            self.show_image(result_image)

        except Exception as e:
            QMessageBox.critical(self, "处理错误", f"图像处理时发生错误: {str(e)}")


    def get_longest_line(self, box):
        # 这个函数现在暂时没用到，可以保留备用，也可以删除
        dists = [
            (np.linalg.norm(box[i] - box[(i + 1) % 4]), (box[i], box[(i + 1) % 4]))
            for i in range(4)
        ]
        return max(dists, key=lambda x: x[0])[1]

    def project_point(self, A, B, P):
        # 同上，暂时没用到
        AB = np.array(B) - np.array(A)
        AP = np.array(P) - np.array(A)
        AB_norm = AB / np.linalg.norm(AB)
        projection_length = np.dot(AP, AB_norm)
        P_prime = A + projection_length * AB_norm
        return tuple(map(int, P_prime))

    def show_image(self, image):
        h, w, ch = image.shape
        bytes_per_line = ch * w
        q_img = QImage(image.data, w, h, bytes_per_line, QImage.Format.Format_BGR888)
        pixmap = QPixmap.fromImage(q_img)

        # 获取当前滚动条位置
        scroll_area = self.centralWidget().findChild(QScrollArea)
        h_bar = scroll_area.horizontalScrollBar()
        v_bar = scroll_area.verticalScrollBar()
        old_h_pos = h_bar.value()
        old_v_pos = v_bar.value()

        # 设置新的pixmap
        self.image_label.setPixmap(pixmap)

        # 恢复滚动条位置
        QApplication.processEvents()  # 确保布局已更新
        h_bar.setValue(int(old_h_pos * self.image_label.scale_factor))
        v_bar.setValue(int(old_v_pos * self.image_label.scale_factor))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PowderSegmentApp()
    window.show()
    sys.exit(app.exec())
