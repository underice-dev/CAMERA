import sys
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from skeleton_detector import skeleton_longest_endpoints, prune_skeleton
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QFileDialog,
    QTextEdit,
    QScrollArea,
    QSizePolicy,
    QDialog,
    QLineEdit,
    QFormLayout,
    QMessageBox,
    QComboBox,
    QCheckBox,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap


def imread_any(path: str):
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def imwrite_any(path: str, image: np.ndarray):
    try:
        ext = Path(path).suffix
        if ext == "":
            ext = ".png"
        ok, buf = cv2.imencode(ext, image)
        if not ok:
            return False
        buf.tofile(path)
        return True
    except Exception:
        return False


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def save_debug(name: str, img: np.ndarray, params: Dict[str, Any]):
    if not params.get("debug_save_steps"):
        return
    out_dir = Path(params.get("debug_out_dir", "debug_out"))
    ensure_dir(out_dir)
    out_path = out_dir / f"{name}.png"
    cv2.imwrite(str(out_path), img)


def validate_canny(params: Dict[str, Any]):
    low = params.get("canny_low", 50)
    high = params.get("canny_high", 150)
    low = max(0, min(int(low), 255))
    high = max(1, min(int(high), 255))
    if low >= high:
        low = max(0, high - 1)
    params["canny_low"] = low
    params["canny_high"] = high


def detect_scratches(gray: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    validate_canny(params)
    edges = cv2.Canny(
        gray,
        params["canny_low"],
        params["canny_high"],
        apertureSize=3,
        L2gradient=True,
    )
    min_line = int(params.get("scratch_min_line_len", 60))
    threshold = int(params.get("scratch_hough_threshold", 60))
    max_gap = int(params.get("scratch_hough_max_gap", 6))
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=threshold,
        minLineLength=min_line,
        maxLineGap=max_gap,
    )
    mask = np.zeros_like(gray, dtype=np.uint8)
    erase_width = int(params.get("scratch_erase_width_px", 2))
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0, :]:
            cv2.line(mask, (x1, y1), (x2, y2), 255, erase_width)
    save_debug("scratch_mask", mask, params)
    return mask


def repair_with_inpaint(gray: np.ndarray, scratch_mask: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    if not params.get("enable_inpaint", False):
        return gray
    radius = int(params.get("inpaint_radius", 3))
    radius = max(1, radius)
    repaired = cv2.inpaint(gray, scratch_mask, radius, cv2.INPAINT_TELEA)
    save_debug("gray_inpaint", repaired, params)
    return repaired


def remove_thin_lines(gray: np.ndarray, params: Dict[str, Any], min_length: int = 80, thickness: int = 2):
    """Use Hough lines to remove very thin long scratches/lines."""
    validate_canny(params)
    edges = cv2.Canny(
        gray,
        params["canny_low"],
        params["canny_high"],
        apertureSize=3,
        L2gradient=True,
    )
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=60, minLineLength=min_length, maxLineGap=5
    )
    cleaned = gray.copy()
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0, :]:
            cv2.line(cleaned, (x1, y1), (x2, y2), 0, thickness)
    return cleaned


def suppress_scratches(mask: np.ndarray, params: Dict[str, Any], min_length: int, erase_width: int):
    """Detect long thin lines (scratches) on a binary mask and erase them."""
    validate_canny(params)
    edges = cv2.Canny(
        mask,
        params["canny_low"],
        params["canny_high"],
        apertureSize=3,
        L2gradient=True,
    )
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=int(params.get("scratch_hough_threshold", 60)),
        minLineLength=min_length,
        maxLineGap=int(params.get("scratch_hough_max_gap", 6)),
    )
    cleaned = mask.copy()
    if lines is not None:
        erase_width = max(1, int(erase_width))
        for x1, y1, x2, y2 in lines[:, 0, :]:
            cv2.line(cleaned, (x1, y1), (x2, y2), 0, erase_width)
    return cleaned


def filter_components(mask: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """
    Remove small/noisy components based on area and aspect ratio before contouring.
    """
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return mask
    min_area = params.get("min_area_px", 50)
    min_width = params.get("min_width_px", 4.0) * 0.5
    min_height = params.get("min_length_px", 35) * 0.5
    keep = np.zeros_like(mask)
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        if w < min_width or h < min_height:
            continue
        if w == 0 or h == 0:
            continue
        aspect = max(w, h) / max(1, min(w, h))
        if aspect < params.get("min_aspect_ratio", 2.2) * 0.5:
            continue
        keep[labels == i] = 255
    return keep if np.any(keep) else mask


class ScrollableImageLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(640, 480)
        self.scale_factor = 1.0
        self._pixmap = None

    def setPixmap(self, pixmap):
        self._pixmap = pixmap
        self.update_scaled_pixmap()

    def update_scaled_pixmap(self):
        if self._pixmap:
            scaled_pixmap = self._pixmap.scaled(
                self._pixmap.size() * self.scale_factor,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            super().setPixmap(scaled_pixmap)
            self.adjustSize()

    def wheelEvent(self, event):
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
        self.scale_factor = max(0.1, min(self.scale_factor, 10.0))
        self.update_scaled_pixmap()

    def mouseDoubleClickEvent(self, event):
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
        self.presets = {
            "默认稳定": {
                "blur_size": 5,
                "canny_low": 50,
                "canny_high": 150,
                "morph_size": 5,
                "min_contour_length": 50,
                "pixels_per_mm": 84.07,
                "min_width_px": 4.0,
                "min_area_px": 50,
                "min_aspect_ratio": 2.2,
                "min_length_px": 35,
                "min_texture_std": 3.8,
                "angle_tol_deg": 15,
                "branch_prune_len": 8,
                "max_tracks": 5,
                "scratch_min_line_len": 60,
                "scratch_hough_threshold": 60,
                "scratch_hough_max_gap": 6,
                "scratch_erase_width_px": 2,
                "enable_inpaint": True,
                "inpaint_radius": 3,
                "auto_rotate": False,
                "rotate_max_deg": 10,
                "debug_save_steps": False,
                "debug_out_dir": "debug_out",
            },
            "1": {
                "blur_size": 5,
                "canny_low": 40,
                "canny_high": 120,
                "morph_size": 5,
                "min_contour_length": 50,
                "pixels_per_mm": 84.07,
                "min_width_px": 5.0,
                "min_area_px": 80,
                "min_aspect_ratio": 2.2,
                "min_length_px": 40,
                "min_texture_std": 3.5,
                "angle_tol_deg": 10,
                "branch_prune_len": 8,
                "max_tracks": 5,
                "scratch_min_line_len": 80,
                "scratch_hough_threshold": 80,
                "scratch_hough_max_gap": 5,
                "scratch_erase_width_px": 3,
                "enable_inpaint": True,
                "inpaint_radius": 3,
                "auto_rotate": False,
                "rotate_max_deg": 10,
                "debug_save_steps": False,
                "debug_out_dir": "debug_out",
            },
            "2": {
                "blur_size": 3,
                "canny_low": 40,
                "canny_high": 120,
                "morph_size": 4,
                "min_contour_length": 40,
                "pixels_per_mm": 84.07,
                "min_width_px": 3.5,
                "min_area_px": 40,
                "min_aspect_ratio": 2.0,
                "min_length_px": 30,
                "min_texture_std": 3.5,
                "angle_tol_deg": 15,
                "branch_prune_len": 8,
                "max_tracks": 5,
                "scratch_min_line_len": 60,
                "scratch_hough_threshold": 60,
                "scratch_hough_max_gap": 6,
                "scratch_erase_width_px": 2,
                "enable_inpaint": True,
                "inpaint_radius": 3,
                "auto_rotate": False,
                "rotate_max_deg": 10,
                "debug_save_steps": False,
                "debug_out_dir": "debug_out",
            },
        }

        main_layout = QVBoxLayout()
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("预设"))
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(list(self.presets.keys()))
        preset_row.addWidget(self.preset_combo)
        self.btn_apply_preset = QPushButton("应用预设")
        preset_row.addWidget(self.btn_apply_preset)
        self.adv_checkbox = QCheckBox("高级选项")
        self.adv_checkbox.setChecked(False)
        preset_row.addWidget(self.adv_checkbox)
        main_layout.addLayout(preset_row)

        basic_form = QFormLayout()
        advanced_form = QFormLayout()

        self.blur_size_edit = QLineEdit(str(self.params.get("blur_size", 5)))
        self.canny_low_edit = QLineEdit(str(self.params.get("canny_low", 50)))
        self.canny_high_edit = QLineEdit(str(self.params.get("canny_high", 150)))
        self.morph_size_edit = QLineEdit(str(self.params.get("morph_size", 5)))
        self.pixels_per_mm_edit = QLineEdit(str(self.params.get("pixels_per_mm", 84.07)))
        self.max_tracks_edit = QLineEdit(str(self.params.get("max_tracks", 5)))

        basic_form.addRow("高斯模糊核大小(奇数):", self.blur_size_edit)
        basic_form.addRow("Canny低阈值:", self.canny_low_edit)
        basic_form.addRow("Canny高阈值:", self.canny_high_edit)
        basic_form.addRow("形态学操作核大小:", self.morph_size_edit)
        basic_form.addRow("像素/毫米:", self.pixels_per_mm_edit)
        basic_form.addRow("最大轨迹数:", self.max_tracks_edit)

        self.min_contour_length_edit = QLineEdit(str(self.params.get("min_contour_length", 50)))
        self.min_width_px_edit = QLineEdit(str(self.params.get("min_width_px", 4.0)))
        self.min_area_px_edit = QLineEdit(str(self.params.get("min_area_px", 50)))
        self.min_aspect_ratio_edit = QLineEdit(str(self.params.get("min_aspect_ratio", 2.2)))
        self.min_length_px_edit = QLineEdit(str(self.params.get("min_length_px", 35)))
        self.min_texture_std_edit = QLineEdit(str(self.params.get("min_texture_std", 3.8)))
        self.angle_tol_deg_edit = QLineEdit(str(self.params.get("angle_tol_deg", 15)))
        self.branch_prune_len_edit = QLineEdit(str(self.params.get("branch_prune_len", 8)))
        self.scratch_min_line_len_edit = QLineEdit(str(self.params.get("scratch_min_line_len", 60)))
        self.scratch_hough_threshold_edit = QLineEdit(str(self.params.get("scratch_hough_threshold", 60)))
        self.scratch_hough_max_gap_edit = QLineEdit(str(self.params.get("scratch_hough_max_gap", 6)))
        self.scratch_erase_width_px_edit = QLineEdit(str(self.params.get("scratch_erase_width_px", 2)))
        self.enable_inpaint_cb = QCheckBox("启用修复")
        self.enable_inpaint_cb.setChecked(bool(self.params.get("enable_inpaint", True)))
        self.inpaint_radius_edit = QLineEdit(str(self.params.get("inpaint_radius", 3)))
        self.auto_rotate_cb = QCheckBox("自动旋转")
        self.auto_rotate_cb.setChecked(bool(self.params.get("auto_rotate", False)))
        self.rotate_max_deg_edit = QLineEdit(str(self.params.get("rotate_max_deg", 10)))
        self.debug_save_steps_cb = QCheckBox("调试保存步骤")
        self.debug_save_steps_cb.setChecked(bool(self.params.get("debug_save_steps", False)))
        self.debug_out_dir_edit = QLineEdit(str(self.params.get("debug_out_dir", "debug_out")))

        advanced_form.addRow("最小轮廓长度", self.min_contour_length_edit)
        advanced_form.addRow("最小宽度(px)", self.min_width_px_edit)
        advanced_form.addRow("最小面积(px)", self.min_area_px_edit)
        advanced_form.addRow("最小长宽比", self.min_aspect_ratio_edit)
        advanced_form.addRow("最小长度(px)", self.min_length_px_edit)
        advanced_form.addRow("最小纹理标准差", self.min_texture_std_edit)
        advanced_form.addRow("角度容忍度(°)", self.angle_tol_deg_edit)
        advanced_form.addRow("分支修剪长度", self.branch_prune_len_edit)
        advanced_form.addRow("划痕最小线段长度", self.scratch_min_line_len_edit)
        advanced_form.addRow("划痕Hough阈值", self.scratch_hough_threshold_edit)
        advanced_form.addRow("划痕Hough最大间隙", self.scratch_hough_max_gap_edit)
        advanced_form.addRow("划痕擦除宽度(px)", self.scratch_erase_width_px_edit)
        advanced_form.addRow("启用修复", self.enable_inpaint_cb)
        advanced_form.addRow("修复半径(px)", self.inpaint_radius_edit)
        advanced_form.addRow("自动旋转", self.auto_rotate_cb)
        advanced_form.addRow("最大旋转角度(°)", self.rotate_max_deg_edit)
        advanced_form.addRow("调试保存步骤", self.debug_save_steps_cb)
        advanced_form.addRow("调试输出目录", self.debug_out_dir_edit)

        self.advanced_widget = QWidget()
        self.advanced_widget.setLayout(advanced_form)
        self.advanced_widget.setVisible(False)

        main_layout.addLayout(basic_form)
        main_layout.addWidget(self.advanced_widget)

        btn_confirm = QPushButton("确认")
        btn_confirm.clicked.connect(self.validate_and_accept)
        main_layout.addWidget(btn_confirm)

        self.btn_apply_preset.clicked.connect(self.apply_selected_preset)
        self.adv_checkbox.stateChanged.connect(self.toggle_advanced)
        self.setLayout(main_layout)
        self.apply_selected_preset()
    def toggle_advanced(self):
        self.advanced_widget.setVisible(self.adv_checkbox.isChecked())

    def apply_selected_preset(self):
        name = self.preset_combo.currentText()
        preset = self.presets.get(name)
        if not preset:
            return
        fields = {
            "blur_size": self.blur_size_edit,
            "canny_low": self.canny_low_edit,
            "canny_high": self.canny_high_edit,
            "morph_size": self.morph_size_edit,
            "pixels_per_mm": self.pixels_per_mm_edit,
            "min_contour_length": self.min_contour_length_edit,
            "min_width_px": self.min_width_px_edit,
            "min_area_px": self.min_area_px_edit,
            "min_aspect_ratio": self.min_aspect_ratio_edit,
            "min_length_px": self.min_length_px_edit,
            "min_texture_std": self.min_texture_std_edit,
            "angle_tol_deg": self.angle_tol_deg_edit,
            "branch_prune_len": self.branch_prune_len_edit,
            "max_tracks": self.max_tracks_edit,
            "scratch_min_line_len": self.scratch_min_line_len_edit,
            "scratch_hough_threshold": self.scratch_hough_threshold_edit,
            "scratch_hough_max_gap": self.scratch_hough_max_gap_edit,
            "scratch_erase_width_px": self.scratch_erase_width_px_edit,
            "inpaint_radius": self.inpaint_radius_edit,
            "rotate_max_deg": self.rotate_max_deg_edit,
            "debug_out_dir": self.debug_out_dir_edit,
        }
        for k, v in preset.items():
            if k in fields:
                fields[k].setText(str(v))
        self.enable_inpaint_cb.setChecked(bool(preset.get("enable_inpaint", True)))
        self.auto_rotate_cb.setChecked(bool(preset.get("auto_rotate", False)))
        self.debug_save_steps_cb.setChecked(bool(preset.get("debug_save_steps", False)))

    def _parse_bool(self, text: str) -> bool:
        return str(text).strip().lower() in ["1", "true", "yes", "y", "t"]

    def validate_and_accept(self):
        try:
            params = {
                "blur_size": int(self.blur_size_edit.text()),
                "canny_low": int(self.canny_low_edit.text()),
                "canny_high": int(self.canny_high_edit.text()),
                "morph_size": int(self.morph_size_edit.text()),
                "pixels_per_mm": float(self.pixels_per_mm_edit.text()),
                "max_tracks": int(self.max_tracks_edit.text()),
                "min_contour_length": int(self.min_contour_length_edit.text()),
                "min_width_px": float(self.min_width_px_edit.text()),
                "min_area_px": float(self.min_area_px_edit.text()),
                "min_aspect_ratio": float(self.min_aspect_ratio_edit.text()),
                "min_length_px": float(self.min_length_px_edit.text()),
                "min_texture_std": float(self.min_texture_std_edit.text()),
                "angle_tol_deg": float(self.angle_tol_deg_edit.text()),
                "branch_prune_len": int(self.branch_prune_len_edit.text()),
                "scratch_min_line_len": int(self.scratch_min_line_len_edit.text()),
                "scratch_hough_threshold": int(self.scratch_hough_threshold_edit.text()),
                "scratch_hough_max_gap": int(self.scratch_hough_max_gap_edit.text()),
                "scratch_erase_width_px": int(self.scratch_erase_width_px_edit.text()),
                "enable_inpaint": self.enable_inpaint_cb.isChecked(),
                "inpaint_radius": int(self.inpaint_radius_edit.text()),
                "auto_rotate": self.auto_rotate_cb.isChecked(),
                "rotate_max_deg": float(self.rotate_max_deg_edit.text()),
                "debug_save_steps": self.debug_save_steps_cb.isChecked(),
                "debug_out_dir": self.debug_out_dir_edit.text(),
            }
            if params["blur_size"] % 2 == 0 or params["blur_size"] < 1:
                raise ValueError("模糊核大小必须是正奇数")
            if params["canny_low"] >= params["canny_high"]:
                raise ValueError("Canny低阈值必须小于高阈值")
            if params["morph_size"] < 1:
                raise ValueError("形态学核大小必须大于0")
            if params["min_contour_length"] < 1:
                raise ValueError("形态学核大小必须大于0")
            if params["pixels_per_mm"] <= 0:
                raise ValueError("比例尺必须大于0")
            if params["min_width_px"] <= 0:
                raise ValueError("最小宽度必须大于0")
            if params["max_tracks"] < 1:
                raise ValueError("输出最大条数必须>=1")
            self.params = params
            self.accept()
        except ValueError as e:
            QMessageBox.warning(self, "输入错误", f"参数错误: {str(e)}")


class PowderSegmentApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("粉段检测软件 v2.0")
        self.setGeometry(100, 100, 800, 600)
        self.params = {
            "pixels_per_mm": 84.07,
            "blur_size": 5,
            "morph_size": 5,
            "min_contour_length": 50,
            "min_width_px": 4.0,
            "canny_low": 50,
            "canny_high": 150,
            "min_area_px": 50,
            "min_aspect_ratio": 2.2,
            "min_length_px": 35,
            "min_texture_std": 3.8,
            "angle_tol_deg": 15,
            "branch_prune_len": 8,
            "max_tracks": 5,
            "scratch_min_line_len": 60,
            "scratch_hough_threshold": 60,
            "scratch_hough_max_gap": 6,
            "scratch_erase_width_px": 2,
            "enable_inpaint": True,
            "inpaint_radius": 3,
            "auto_rotate": False,
            "rotate_max_deg": 10,
            "debug_save_steps": False,
            "debug_out_dir": "debug_out",
        }
        self.current_image = None
        self.processed_image = None
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

        self.btn_file.clicked.connect(self.open_image)
        self.btn_params.clicked.connect(self.open_params_dialog)
        self.btn_save.clicked.connect(self.save_result_image)

    def open_params_dialog(self):
        dialog = ParamsDialog(self, self.params)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.params = dialog.params
            self.result_text.append("参数已更新：")
            self.result_text.append("\n".join([f"{k}: {v}" for k, v in self.params.items()]))
            if self.current_image is not None:
                self.process_image(self.current_image)

    def open_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "Image Files (*.jpg *.jpeg *.png)"
        )
        if file_path:
            image = imread_any(file_path)
            if image is not None:
                self.current_image = image.copy()
                self.process_image(image)
            else:
                QMessageBox.warning(self, "读取失败", f"无法读取图片：{file_path}")

    def save_result_image(self):
        if self.processed_image is None:
            QMessageBox.information(self, "提示", "当前没有可导出的结果图片，请先选择图片并完成处理。")
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存结果图片", "processed_result.png", "Image Files (*.png *.jpg *.jpeg)"
        )
        if not file_path:
            return
        if imwrite_any(file_path, self.processed_image):
            QMessageBox.information(self, "保存成功", f"结果图片已保存到:\n{file_path}")
        else:
            QMessageBox.critical(self, "保存失败", "保存结果图片时发生错误：无法写入文件路径，请检查。")

    def auto_rotate_if_needed(self, gray: np.ndarray) -> Tuple[np.ndarray, float]:
        if not self.params.get("auto_rotate", False):
            return gray, 0.0
        angles = []
        edges = cv2.Canny(gray, self.params["canny_low"], self.params["canny_high"], L2gradient=True)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=80, maxLineGap=10)
        if lines is not None:
            for x1, y1, x2, y2 in lines[:, 0, :]:
                ang = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                angles.append(ang)
        if not angles:
            return gray, 0.0
        ang = float(np.median(angles))
        ang_norm = ((ang + 90) % 180) - 90
        if abs(ang_norm) > self.params.get("rotate_max_deg", 10):
            return gray, 0.0
        h, w = gray.shape
        M = cv2.getRotationMatrix2D((w / 2, h / 2), -ang_norm, 1.0)
        rotated = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return rotated, ang_norm

    def distance_width_stats(self, mask: np.ndarray, skeleton_mask: np.ndarray, pixels_per_mm: float) -> Tuple[float, float]:
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        ys, xs = np.nonzero(skeleton_mask)
        if len(xs) == 0:
            return 0.0, 0.0
        widths = dist[ys, xs] * 2.0
        med = float(np.median(widths))
        mean = float(np.mean(widths))
        return med / pixels_per_mm, mean / pixels_per_mm

    def _draw_text_rotated(self, image, text, center, angle_deg, color, scale=0.6, thickness=2):
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        pad = 4
        text_img = np.zeros((th + pad * 2, tw + pad * 2, 3), dtype=np.uint8)
        cv2.putText(text_img, text, (pad, th + pad), font, scale, color, thickness, cv2.LINE_AA)
        (h, w) = text_img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
        cos = abs(M[0, 0])
        sin = abs(M[0, 1])
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        M[0, 2] += (nW / 2) - w / 2
        M[1, 2] += (nH / 2) - h / 2
        rotated = cv2.warpAffine(text_img, M, (nW, nH), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
        mask = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
        img_h, img_w = image.shape[:2]
        x0 = int(center[0] - rotated.shape[1] / 2)
        y0 = int(center[1] - rotated.shape[0] / 2)
        x0 = max(0, min(x0, img_w - rotated.shape[1]))
        y0 = max(0, min(y0, img_h - rotated.shape[0]))
        roi = image[y0 : y0 + rotated.shape[0], x0 : x0 + rotated.shape[1]]
        mask_roi = mask > 0
        roi[mask_roi] = rotated[mask_roi]

    def process_image(self, image):
        try:
            validate_canny(self.params)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray_rot, angle_applied = self.auto_rotate_if_needed(gray)
            if angle_applied != 0:
                gray = gray_rot

            gray_med = cv2.medianBlur(gray, 3)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray_eq = clahe.apply(gray_med)
            bg_kernel = max(15, self.params["morph_size"] * 6 | 1)
            bg = cv2.GaussianBlur(gray_eq, (bg_kernel, bg_kernel), 0)
            gray_flat = cv2.addWeighted(gray_eq, 1.0, bg, -1.0, 128)
            tophat_kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (max(5, self.params["morph_size"] * 2), max(5, self.params["morph_size"] * 2)),
            )
            gray_tophat = cv2.morphologyEx(gray_med, cv2.MORPH_TOPHAT, tophat_kernel)
            gray_mix = cv2.addWeighted(gray_flat, 0.6, gray_tophat, 0.4, 0)
            gray_mix = cv2.addWeighted(gray_mix, 0.8, gray_eq, 0.2, 0)
            blur_size = self.params["blur_size"]
            gray_blur = cv2.GaussianBlur(gray_mix, (blur_size, blur_size), 0)
            save_debug("gray_flat", gray_flat, self.params)
            save_debug("gray_tophat", gray_tophat, self.params)

            scratch_mask = detect_scratches(gray_blur, self.params)
            gray_repaired = repair_with_inpaint(gray_blur, scratch_mask, self.params)
            save_debug("gray_repaired", gray_repaired, self.params)

            _, binary = cv2.threshold(gray_repaired, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            white_ratio = cv2.countNonZero(binary) / binary.size
            if white_ratio > 0.5:
                binary = cv2.bitwise_not(binary)
            morph_size = self.params["morph_size"]
            kernel = np.ones((morph_size, morph_size), np.uint8)
            closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            open_kernel = np.ones((3, 3), np.uint8)
            cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, open_kernel)
            cleaned = suppress_scratches(
                cleaned,
                self.params,
                min_length=max(60, int(self.params["min_contour_length"] * 0.8)),
                erase_width=self.params["scratch_erase_width_px"],
            )
            cleaned = filter_components(cleaned, self.params)
            save_debug("binary", binary, self.params)
            save_debug("cleaned", cleaned, self.params)

            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            result_image = image.copy()
            pixels_per_mm = self.params["pixels_per_mm"]
            min_perimeter = self.params["min_contour_length"]

            MIN_ASPECT_RATIO = self.params["min_aspect_ratio"]
            MIN_AREA = self.params["min_area_px"]
            MIN_WIDTH_PX = self.params["min_width_px"]
            MIN_LENGTH_PX = self.params["min_length_px"]
            MIN_TEXTURE_STD = self.params["min_texture_std"]
            ANGLE_TOL = self.params["angle_tol_deg"]
            MAX_TRACKS = self.params["max_tracks"]
            BRANCH_PRUNE_LEN = self.params["branch_prune_len"]

            colors = [
                (0, 255, 0),
                (0, 0, 255),
                (255, 0, 0),
                (255, 255, 0),
                (255, 0, 255),
                (0, 255, 255),
            ]

            h_img, w_img = cleaned.shape
            candidates = []

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
                if aspect_ratio < MIN_ASPECT_RATIO:
                    continue
                if short_side < MIN_WIDTH_PX:
                    continue

                angle_long = angle + 90 if w < h else angle
                if angle_long > 90:
                    angle_long -= 180
                if angle_long < -90:
                    angle_long += 180

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

                roi_gray = gray_mix[y0:y1, x0:x1]
                _, stddev = cv2.meanStdDev(roi_gray, mask=roi_mask)
                if stddev[0][0] < MIN_TEXTURE_STD:
                    continue

                skel_info = skeleton_longest_endpoints(
                    roi_mask, min_branch_len=BRANCH_PRUNE_LEN
                )
                if skel_info and skel_info.get("length_px", 0) > 0:
                    effective_length_px = skel_info["length_px"]
                    pt1_global = (skel_info["pt1"][0] + x0, skel_info["pt1"][1] + y0)
                    pt2_global = (skel_info["pt2"][0] + x0, skel_info["pt2"][1] + y0)
                    skel_u8 = np.zeros_like(roi_mask)
                    skel = skel_info["skeleton"]
                    skel_u8 = skel
                else:
                    effective_length_px = max(long_side - short_side, 0)
                    pt1_global = None
                    pt2_global = None
                    skel_u8 = (roi_mask > 0).astype(np.uint8) * 0

                if effective_length_px < MIN_LENGTH_PX:
                    continue

                length_mm = effective_length_px / pixels_per_mm
                height_mm = short_side / pixels_per_mm

                box = cv2.boxPoints(rect)
                box = np.intp(box)

                width_mm_med, width_mm_mean = self.distance_width_stats(roi_mask, skel_u8 // 255, pixels_per_mm)

                candidates.append(
                    {
                        "center": (cx, cy),
                        "box": box,
                        "long_side_px": long_side,
                        "short_side_px": short_side,
                        "length_px": effective_length_px,
                        "height_px": short_side,
                        "length_mm": length_mm,
                        "height_mm": height_mm,
                        "width_mm_med": width_mm_med,
                        "width_mm_mean": width_mm_mean,
                        "area": area,
                        "aspect_ratio": aspect_ratio,
                        "endpoints": (pt1_global, pt2_global),
                        "angle_deg": angle_long,
                    }
                )

            if not candidates:
                self.result_text.setText(
                    "未检测到符合条件的条状粉末段：\n"
                    "- 可尝试降低各类最小阈值；\n"
                    "- 或调整模糊、形态学核、最小轮廓长度等参数。"
                )
                self.show_image(result_image)
                return

            angles = np.array([c["angle_deg"] for c in candidates], dtype=np.float32)
            main_angle = float(np.median(angles))

            def angle_diff(a, b):
                d = abs(a - b) % 180
                return min(d, 180 - d)

            identified = [c for c in candidates if angle_diff(c["angle_deg"], main_angle) <= ANGLE_TOL]

            lengths = np.array([c["length_px"] for c in identified], dtype=np.float32)
            widths = np.array([c["short_side_px"] for c in identified], dtype=np.float32)
            areas = np.array([c["area"] for c in identified], dtype=np.float32)
            max_len = float(np.max(lengths))
            len_keep = max(MIN_LENGTH_PX, 0.45 * max_len)
            med_w = float(np.median(widths))
            med_area = float(np.median(areas))

            def in_band(val, med, low, high):
                return (val >= med * low) and (val <= med * high)

            robust = [
                c
                for c in identified
                if c["length_px"] >= len_keep
                and in_band(c["short_side_px"], med_w, 0.5, 2.2)
                and in_band(c["area"], med_area, 0.4, 2.5)
            ]
            if len(robust) >= 2:
                identified = robust

            identified.sort(key=lambda s: s["length_px"], reverse=True)
            detected_segments = identified[:MAX_TRACKS]

            if len(detected_segments) < MAX_TRACKS:
                extras = sorted(candidates, key=lambda s: s["length_px"], reverse=True)
                for seg in extras:
                    if seg not in detected_segments:
                        detected_segments.append(seg)
                    if len(detected_segments) >= MAX_TRACKS:
                        break

            var_x = np.var([s["center"][0] for s in detected_segments])
            var_y = np.var([s["center"][1] for s in detected_segments])
            if var_x >= var_y:
                detected_segments.sort(key=lambda s: s["center"][0])
            else:
                detected_segments.sort(key=lambda s: s["center"][1])

            output_lines = []
            output_lines.append(f"检测到粉末段数量: {len(detected_segments)}")
            output_lines.append(f"当前比例尺: {pixels_per_mm:.2f} 像素/毫米\n")

            for idx, seg in enumerate(detected_segments, start=1):
                color = colors[(idx - 1) % len(colors)]
                box = seg["box"]
                cx, cy = seg["center"]
                length_mm = seg["length_mm"]
                height_mm = seg["height_mm"]
                width_mm_med = seg.get("width_mm_med", height_mm)

                cv2.drawContours(result_image, [box], 0, color, 2)

                if seg.get("endpoints") and all(seg["endpoints"]):
                    cv2.circle(result_image, seg["endpoints"][0], 4, color, -1)
                    cv2.circle(result_image, seg["endpoints"][1], 4, color, -1)

                text = f"{idx}: L={length_mm:.2f} mm  W~{width_mm_med:.2f} mm"
                min_x = int(np.min(box[:, 0]))
                max_x = int(np.max(box[:, 0]))
                min_y = int(np.min(box[:, 1]))
                max_y = int(np.max(box[:, 1]))
                text_x = max(min_x, 5)

                angle = seg.get("angle_deg", 0)
                vertical_label = abs(angle) > 60

                if vertical_label:
                    target_x = min_x - 20
                    if target_x < 5:
                        target_x = max_x + 20
                    center = (int(target_x), int((min_y + max_y) / 2))
                    self._draw_text_rotated(result_image, text, center, -90, color, scale=0.6, thickness=2)
                else:
                    layer_offset = 16 * ((idx - 1) % 3)
                    text_y = min_y - 10 - layer_offset
                    if text_y < 15:
                        text_y = max_y + 20 + layer_offset
                    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    text_x = max(5, min(text_x, w_img - tw - 5))
                    text_y = max(th + 5, min(text_y, h_img - 5))
                    cv2.putText(
                        result_image,
                        text,
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                        cv2.LINE_AA,
                    )

                output_lines.append(
                    f"粉末段 {idx}: "
                    f"中心=({cx:.1f}, {cy:.1f}), "
                    f"长度 L={length_mm:.2f} mm (像素 {seg['length_px']:.1f}), "
                    f"宽度 W~{width_mm_med:.2f} mm, "
                    f"面积≈{seg['area']:.1f} 像素^2, "
                    f"长宽比≈{seg['aspect_ratio']:.2f}"
                )

            self.result_text.setText("\n".join(output_lines))
            self.processed_image = result_image.copy()
            self.show_image(result_image)

        except Exception as e:
            QMessageBox.critical(self, "处理错误", f"图像处理时发生错误: {str(e)}")

    def show_image(self, image):
        h, w, ch = image.shape
        bytes_per_line = ch * w
        q_img = QImage(image.data, w, h, bytes_per_line, QImage.Format.Format_BGR888)
        pixmap = QPixmap.fromImage(q_img)

        scroll_area = self.centralWidget().findChild(QScrollArea)
        h_bar = scroll_area.horizontalScrollBar()
        v_bar = scroll_area.verticalScrollBar()
        old_h_pos = h_bar.value()
        old_v_pos = v_bar.value()

        self.image_label.setPixmap(pixmap)
        QApplication.processEvents()
        h_bar.setValue(int(old_h_pos * self.image_label.scale_factor))
        v_bar.setValue(int(old_v_pos * self.image_label.scale_factor))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PowderSegmentApp()
    window.show()
    sys.exit(app.exec())