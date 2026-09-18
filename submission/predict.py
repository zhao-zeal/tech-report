# -*- coding: utf-8 -*-
"""Three-task inference: city Ridge/CNY, TA Ridge/templates, PV HYX ensemble."""
import os
import sys
import warnings

warnings.filterwarnings("ignore")

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from utils import config
from utils.pipeline import (
    MODEL_NAME as CITY_MODEL_NAME,
    model_dir as city_model_dir,
    predict_city_model,
)
from utils.hyx_pv_pipeline import (
    MANIFEST_NAME as PV_MANIFEST_NAME,
    model_dir as pv_model_dir,
    predict_pv_model,
)
from utils.hyx_ta_pipeline import (
    MODEL_NAME as TA_MODEL_NAME,
    model_dir as ta_model_dir,
    predict_ta_model,
)


class Predictor:
    """统一封装三个任务的模型检查与推理。"""

    def InitModel(self):
        """检查三个任务推理所需的主要模型文件。"""
        ret, err_message = True, "normal"
        try:
            self.city_model_path = city_model_dir() / CITY_MODEL_NAME
            if not self.city_model_path.is_file():
                raise FileNotFoundError(
                    "Task 1 model not found: " + str(self.city_model_path)
                )

            self.pv_manifest_path = pv_model_dir() / PV_MANIFEST_NAME
            if not self.pv_manifest_path.is_file():
                raise FileNotFoundError(
                    "Task 3 model not found: " + str(self.pv_manifest_path)
                )

            self.ta_model_path = ta_model_dir() / TA_MODEL_NAME
            if not self.ta_model_path.is_file():
                raise FileNotFoundError(
                    "Task 2 model not found: " + str(self.ta_model_path)
                )

            print(
                "[Info] 模型检查完成: "
                "city=Ridge/CNY, ta=HYX-Ridge-template-v1, pv=HYX-ET-ensemble"
            )
        except Exception as err:
            ret = False
            err_message = "[Error] model init failed: " + ExceptionMessage(err)
            print(err_message)
        return ret, err_message

    def Detect(self):
        """依次执行三个任务，按官方要求返回 (pv, ta, city)。"""
        try:
            df_city = self._predict_city()
            df_ta = self._predict_ta()
            df_pv = self._predict_pv()
            return df_pv, df_ta, df_city
        except Exception as err:
            message = "[Error] Detect failed: " + ExceptionMessage(err)
            print(message)
            # 不返回 Exception 对象，否则平台会误报“输出类型错误”。
            raise RuntimeError(message) from err

    def _predict_city(self):
        """任务一：保留现有地市 pipeline 的调用方式。"""
        print("[Info] 任务一：地市用电量预测...")
        result = predict_city_model()
        return result[["city_id", "date", "pred"]]

    def _predict_ta(self):
        """任务二：新台区模型只读 NPZ，不再读取 ta/data 或旧权重。"""
        print("[Info] 任务二：台区用电量预测...")
        result = predict_ta_model(
            model_path=self.ta_model_path,
            save_output=False,
        )
        print("[Info] 任务二：台区用电量预测完成")
        return result[["ta_id", "date", "pred"]]

    def _predict_pv(self):
        """任务三：保留现有 HYX 光伏 pipeline 的调用方式。"""
        print("[Info] 任务三：光伏发电量预测...")
        result = predict_pv_model(save_output=False)
        return result[["pv_id", "date", "pred"]]


def save_outputs(df_city, df_ta, df_pv):
    """将三个任务结果保存到各自 output 目录。"""
    destinations = [
        (df_city, config.LOCAL_CITY_OUTPUT_DIR, "任务一"),
        (df_ta, config.LOCAL_TA_OUTPUT_DIR, "任务二"),
        (df_pv, config.LOCAL_PV_OUTPUT_DIR, "任务三"),
    ]
    for frame, directory, task_name in destinations:
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, "submit_result.csv")
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        print("[Info] %s结果已保存: %s" % (task_name, path))


def ExceptionMessage(err):
    """格式化异常信息，附带最接近异常位置的文件与行号。"""
    trace = err.__traceback__
    if trace is None:
        return str(err)
    while trace.tb_next is not None:
        trace = trace.tb_next
    return "%s:%s行:%s" % (
        trace.tb_frame.f_globals.get("__file__", "?"),
        trace.tb_lineno,
        err,
    )


if __name__ == "__main__":
    predictor = Predictor()
    ret, err = predictor.InitModel()
    if not ret:
        raise RuntimeError(err)

    df_pv_result, df_ta_result, df_city_result = predictor.Detect()
    save_outputs(df_city_result, df_ta_result, df_pv_result)

    print("\n========== 结果汇总 ==========")
    print("光伏(HYX ExtraTrees):", df_pv_result.shape)
    print("台区(HYX Ridge/template v1):", df_ta_result.shape)
    print("地市(Ridge/CNY):", df_city_result.shape)

