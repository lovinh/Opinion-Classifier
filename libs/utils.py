import pandas as pd 
import numpy as np
import sys, os
import matplotlib.pyplot as plt
import pickle
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score
import logging
import traceback

# Variable
IMAGES_PATH : str = os.path.join(os.getcwd(), "image") 

LOGGING_FORMAT_STRING = """%(asctime)s %(name)s %(levelname)s: %(message)s\n\t\tat Line %(lineno)d [%(funcName)s() in %(filename)s, %(module)s]"""

LOGGING_FORMAT_TIME_FORMAT = "%H:%M:%S"

LOGGING_LEVEL = logging.INFO

DATASET_PATH = ""


# Logging
logging.basicConfig(format=LOGGING_FORMAT_STRING, datefmt=LOGGING_FORMAT_TIME_FORMAT, level=LOGGING_LEVEL)

# Functions

def save_fig(fig_id : str, image_path : str | None = None, tight_layout : bool =True, fig_extension : str ="png", resolution : str | int ='figure'):
    """
        Lưu một figure thành một ảnh dưới định dạng png (mặc định), hoặc chỉ định định dạng khác.

        Ảnh được lưu mặc định trong đường dẫn: {Your Project}/image/ . Để thay đổi đường dẫn, config lại biến IMAGE_PATH.

        Parameters:
        ------
        fig_id: str. Tên của figure cần lưu. Tên này được sử dụng để đặt tên cho file ảnh.

        image_path: str | None. Đường dẫn tới ảnh cần lưu. Mặc định image_path = `None`. Nếu là `None`, ảnh sẽ được lưu vào folder `image` trong project (Nếu project chưa tồn tại folder `image`, sẽ khởi tạo folder này). Ngược lại là đường dẫn đến folder lưu ảnh (Chỉ folder) 

        tight_layout: bool. Chưa biết vì sao có nó. Mặc định là `True`. (Cập nhật cách SD sau)

        fig_extention: str. Phần mở rộng của file ảnh. Mặc định là `png`.

        resolution: str | int. Độ phân giải ảnh. Nếu resolution = 'figure', lưu toàn bộ điểm ảnh.

        Return:
        ------

            None

    """
    try:
        if not image_path:
            path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
        else:
            path = os.path.join(image_path, fig_id + '.' + fig_extension)

        logging.info(f"Saving figure '{fig_id}' ...")
        if tight_layout:
            plt.tight_layout()
        plt.savefig(path, format=fig_extension, dpi=resolution)
        logging.info(f"Saved figure '{fig_id}'! File location: {path}")
    except Exception as err:
        logging.error(f"Có lỗi xảy ra trong quá trình đọc dữ liệu. Chi tiết lỗi: {err.__class__} " + str(err))
        traceback.print_exc()

def save_model(file_name : str, estimator):
    """
    Lưu model thành file .pkl. File này có thể load lại thành model bằng cách sử dụng hàm load_model().

    Parameters
    ---

        file_name: str. Tên file. Mặc định lưu file tại thư mục cùng thư mục chứa file đang thực thi. Nếu muốn lưu tại thư mục khác cần chỉ định đường dẫn.

        estimator: Mô hình.

    Return
    ---

        None
    """
    try:
        logging.info(f"Đang lưu lại mô hình...")
        with open(file_name, "wb") as f:
            pickle.dump(estimator, f)
        
        logging.info(f"Lưu thành công!")
    except Exception as err:
        logging.error(f"Có lỗi xảy ra trong quá trình đọc dữ liệu. Chi tiết lỗi: {err.__class__} " + str(err))
        traceback.print_exc()

def load_model(file_name : str):
    '''
    Mở một model đã lưu ở dạng file .pkl. Mô hình được lưu sử dụng hàm save_model()
    
    Parameters
    ---
        file_name: str. Đường dẫn tới file model đã lưu. Mặc định tìm kiếm tại thư mục chứa file đang thực thi.
    
    
    Return
    ---
        estimator

            Mô hình được load từ file. 
    '''
    try:
        return pickle.load(open(file_name, 'rb'))
    except Exception as err:
        logging.error(f"Có lỗi xảy ra trong quá trình đọc dữ liệu. Chi tiết lỗi: {err.__class__} " + str(err))
        traceback.print_exc()

def encode_label(array : list[str] | pd.Series | np.ndarray | pd.DataFrame, label_encode_pattern : dict[str, int]) -> list[int]:
    '''
    Mã hóa các nhãn (lớp) thành các số tương ứng được định nghĩa trong bảng `label_encode_pattern`
    
    Parameters
    ---
        array: list[str] | pd.Series | np.ndarray | pd.DataFrame. Danh sách lớp đầu vào.

        label_encode_pattern : dict[str, int]. Bảng quy đổi các lớp thành số
    
    Return
    ---
        list[int]
            Một danh sách các mã hóa theo thứ tự các lớp đầu vào

    Examples
    ---

    >>> y = ['A', 'B', 'A', 'B', 'C', 'A', 'A', 'C']
    >>> label_encode_pattern = {
        'A' : 0,
        'B' : 1,
        'C' : 2,
    }
    >>> encode_label(y, label_encode_pattern)
    [0, 1, 0, 1, 2, 0, 0, 2]
    '''


    encoded_label_list : list[int] = []
    for label in array:
        encoded_label_list.append(label_encode_pattern[label])
    return encoded_label_list


def get_reported_score(y_true, y_pred, average: str = None) -> tuple[float] | None:

    '''
    Báo cáo điểm số Precision, Recall và F1-Score từ hai tập y_true và y_pred đã cho.

    Điểm số Precision biểu thị độ chính xác của dự đoán của mô hình. Công thức tính Precision:

        `Precision` = `TP / (TP + FP)` , với `TP` là tần suất số đối tượng lớp mục tiêu được mô hình đoán đúng, `FP` là tần suất số đối tượng lớp khác bị mô hình dự đoán nhầm thành lớp mục tiêu.

    Điểm số Recall biểu thị độ chính xác của dự đoán mô hình trên thực tế. Công thức tính Recall:

        `Recall` = `TP / (TP + FN)`, với `FN` là tần suất số đối tượng thực tế là lớp khác bị mô hình đoán là lớp mục tiêu.
    
    Điểm số F1 biểu thị trung bình trọng số của Precision và Recall. Công thức thường gặp của F1:

        `F1` = `2 / ((1 / Precision) + (1 / Recall))`, trong công thức này coi như trọng số của các lớp cân bằng và bằng 1. Tùy vào giá trị tham số `average`, cách tính điểm sẽ khác.

    Parameters
    ---
        y_true: Tập nhãn (lớp) đúng. 

        y_pred: Tập nhãn (lớp) dự đoán.

        average: str. Giá trị biểu thị cách tính điểm. average có 5 giá trị chính là None, "binary", "micro", "macro" và "weighted":

            `average = None`. Tính điểm Precision, Recall và F1 theo từng lớp riêng biệt.

            `average = "binary"`. Tính điểm Precision, Recall và F1 theo lớp dương. Chỉ áp dụng khi tập lớp chỉ chứa 2 lớp.

            `average = "micro"`. Tính điểm Precision, Recall và F1 tổng quát theo tổng số lượng đối tượng `TP`, `FP` và `FN`.

            `average = "macro"`. Tính điểm Precision, Recall và F1 theo từng lớp riêng biệt, sau đó tính trung bình không trọng số (trung bình cộng) của các điểm lại và cho ra điểm cuối cùng.

            `average = "weight"`. Tính điểm Precision, Recall và F1 theo từng lớp riêng biệt, sau đó tính trung bình có trọng số của các điểm lại và cho ra điểm cuối cùng.
    
    Return
    ---
        tuple[float]
            Một danh sách 3 giá trị là precision, recall và f1.

    Examples:
    ---
    >>> y_true = [0,1,0,1,0,1]
    >>> y_pred = [1,1,0,1,1,1]
    >>> # Try with average = None.
    >>> get_reported_score(y_true, y_pred, None)
    Average: None
    Precision:      [1.  0.6]
    Recall:         [0.33333333 1.        ]
    --> F1-score:   [0.5  0.75]
    (array([1. , 0.6]), array([0.33333333, 1.        ]), array([0.5 , 0.75]))

    >>> # Try with average = "binary".
    >>> get_reported_score(y_true, y_pred, "binary")
    Average: binary
    Precision:      0.6
    Recall:         1.0
    --> F1-score:   0.7499999999999999
    (0.6, 1.0, 0.7499999999999999)
    
    >>> # Try with average = "micro".
    >>> get_reported_score(y_true, y_pred, "micro")
    Average: micro
    Precision:      0.6666666666666666
    Recall:         0.6666666666666666
    --> F1-score:   0.6666666666666666
    (0.6666666666666666, 0.6666666666666666, 0.6666666666666666)

    >>> # Try with average = "macro".
    >>> get_reported_score(y_true, y_pred, "macro")
    Average: macro
    Precision:      0.8
    Recall:         0.6666666666666666
    --> F1-score:   0.625
    (0.8, 0.6666666666666666, 0.625)

    >>> # Try with average = "weighted".
    >>> get_reported_score(y_true, y_pred, "weighted")
    Average: weighted
    Precision:      0.7999999999999999
    Recall:         0.6666666666666666
    --> F1-score:   0.6249999999999999
    (0.7999999999999999, 0.6666666666666666, 0.6249999999999999)    
    '''
    try:
        precision = precision_score(y_true, y_pred, average=average)
        recall = recall_score(y_true, y_pred, average=average)
        f1 = f1_score(y_true, y_pred, average=average)
        logging.info(f"\nAverage: {average}\nPrecision:\t{precision}\nRecall:\t\t{recall}\n--> F1-score:\t{f1}")
        return precision, recall, f1
    except Exception as err:
        logging.error(f"Có lỗi xảy ra trong quá trình đọc dữ liệu. Chi tiết lỗi: {err.__class__} " + str(err))
        traceback.print_exc()


def measure_score(estimator, X_train, y_train, average: str = None, labels=None, cv=10, save_figure : bool = True, fig_title="Untitled"):
    '''
    Đo đạc hiệu suất của mô hình bằng cách sử dụng ma trận nhầm lẫn và các điểm số Precision, Recall và F1.

    Hàm sẽ thực hiện vẽ một ma trận nhầm lẫn và hiển thị lên.
    
    Parameters
    ---
        
        estimator: mô hình cần đo đạc.  

        X_train: Tập dữ liệu kích thước (n_samples, n_features) gồm các đối tượng huấn luyện của mô hình.

        y_train: Tập nhãn (lớp) kích thước (n_samples, ) gồm các nhãn (n_classes) tương ứng với từng đối tượng huấn luyện.

        average: str. Phương pháp đánh giá điểm số của mô hình. average gồm các giá trị: None, `micro`, `macro`, `weighted` và `binary`.

            `average = None`. Tính điểm Precision, Recall và F1 theo từng lớp riêng biệt.

            `average = "binary"`. Tính điểm Precision, Recall và F1 theo lớp dương. Chỉ áp dụng khi tập lớp chỉ chứa 2 lớp.

            `average = "micro"`. Tính điểm Precision, Recall và F1 tổng quát theo tổng số lượng đối tượng `TP`, `FP` và `FN`.

            `average = "macro"`. Tính điểm Precision, Recall và F1 theo từng lớp riêng biệt, sau đó tính trung bình không trọng số (trung bình cộng) của các điểm lại và cho ra điểm cuối cùng.

            `average = "weight"`. Tính điểm Precision, Recall và F1 theo từng lớp riêng biệt, sau đó tính trung bình có trọng số của các điểm lại và cho ra điểm cuối cùng.

        labels: Sequense[str]. Tập các nhãn đại diện cho các nhãn trong tập, sử dụng để hiển thị khi vẽ ảnh ma trận nhầm lẫn.

        cv: int. Số folds sử dụng khi thực hiện cross-validation.

        save_figure: Có lưu lại figure thành một ảnh hay không.

        fig_title: str. Tên của file ảnh sau khi lưu. Kết hợp với `save_figure = True`.
    
    
    Return
    ---
        
        Trả về một danh sách các điểm số đo đạc được.
    '''
    try:
        y_train_pred = cross_val_predict(estimator, X_train, y_train, cv=cv)
        matrix = confusion_matrix(y_train, y_train_pred)
        matrix_display = ConfusionMatrixDisplay(
            confusion_matrix=matrix, display_labels=labels)
        matrix_display.plot()
        plt.title("Confusion Matrix: " + str(estimator))
        if save_figure:
            save_fig(image_path=IMAGES_PATH, fig_id=fig_title)
        plt.show()
        return get_reported_score(y_train, y_train_pred, average)
    except Exception as err:
        logging.error(f"Có lỗi xảy ra trong quá trình đọc dữ liệu. Chi tiết lỗi: {err.__class__} " + str(err))
        traceback.print_exc()
        
def read_dataset(dataset_path : str | None = None, file_type : str = "xlsx", **kwargs) -> pd.DataFrame | None:
    '''
    Đọc tập dữ liệu từ đường dẫn chỉ định sẵn. Nếu chưa chỉ định đọc mặc định theo hằng số DATASET_PATH.
    
    
    Parameters
    ---
        
        dataset_path: str. Đường dẫn đến file tập dữ liệu. Nếu dataset_path = None, đọc đường dẫn từ hằng số DATASET_PATH

        file_type: str. Kiểu file của tập dữ liệu.
    
    Return
    ---
        pandas.DataFrame:
            DataFrame là tập dữ liệu được đọc từ file.
        None:
            None nếu đọc thất bại.
    '''
    try:
        path = None
        if not dataset_path:
            path = DATASET_PATH
        else:
            path = dataset_path

        logging.info(f"Đang đọc dữ liệu từ file {dataset_path} ...")
        if file_type == "xlsx":
            data = pd.read_excel(path, **kwargs)
        
        if file_type == "csv":
            data = pd.read_csv(path, **kwargs)
        
        if not data.empty:
            logging.info(f"Đọc thành công! ")
            return data
        
        logging.error(f"Có lỗi xảy ra trong quá trình đọc dữ liệu. Chi tiết lỗi: file_type = {file_type} không hợp lệ.")
        return None

    except Exception as err:
        logging.error(f"Có lỗi xảy ra trong quá trình đọc dữ liệu. Chi tiết lỗi: {err.__class__} " + str(err))
        traceback.print_exc()
        return None

    

