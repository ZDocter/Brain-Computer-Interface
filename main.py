import mne
import matplotlib.pyplot as plt
import logging
import numpy as np


def read_data(fname):
    """
    读取并可视化脑电数据

    :param fname: 原始数据路径
    :return: 脑电数据
    """
    raw = mne.io.read_raw_curry(fname, preload=True)  # 读取原始数据
    print(raw.info)  # 显示数据信息
    raw.plot()  # 绘图
    plt.show()

    return raw.copy()


def annotations_and_events(raw_data, onset_1, duration_1, onset_2, duration_2):
    """
    为数据打标

    :param raw_data: 打标之前的原始数据
    :param onset_1: 第一类 event 起始时间
    :param duration_1: 第一类 event 每个时间间隔
    :param onset_2: 第二类 event 起始时间
    :param duration_2: 第二类 event 每个时间间隔
    :return:
    """
    event_list = mne.Annotations(onset=onset_1, duration=duration_1, description='001')  # 观察松手事件
    event_list.append(onset=onset_2, duration=duration_2, description='002')  # 想象松手事件
    raw_data.set_annotations(event_list)  # 第二类事件打标
    raw_data.plot()
    plt.show()
    raw_data.save("data_with_annotations_raw.fif", overwrite=True)  # 保存打标后的数据

    return raw_data.copy()


def data_preprocess(data, bads):
    """
    对数据进行电极定位、插值坏导、滤波、重参考、降采样、ICA、去伪迹

    :param bads: 坏导列表
    :param data: 待处理数据
    :return: 预处理后的数据
    """
    ica = mne.preprocessing.ICA(n_components=35)

    data.set_montage(montage="standard_1020")
    data.plot_sensors(show_names=True)  # 显示二维电极图
    plt.show()

    data.info["bads"] = bads
    data.load()
    data.interpolate_bads()  # 插值坏导

    data.filter(h_freq=30.)  # 低通滤波
    data.notch_filter(freqs=50.)  # 工频滤波

    data.set_eeg_reference()  # 重参考
    data.resample(sfreq=128.)  # 降采样

    ica.fit(data)  # 独立成分分析
    ica.plot_scores()  # 查看独立成分
    ica.plot_components()  # 查看头皮分布

    return data.copy()


if __name__ == '__main__':
    logging.getLogger("mne").setLevel(logging.WARNING)

    # 1、读取并显示原始数据
    fname = r"D:\PyTorch_Python\Sttudy\Acquisition 01.cdt.dpa"
    raw_copy = read_data(fname)

    # 2、数据打标
    onset_1 = np.arange(0., 30., 2.)
    duration_1 = 1.5
    onset_2 = np.arange(30., 60., 2.)
    duration_2 = 1.5
    data_events = annotations_and_events(raw_copy, onset_1, duration_1, onset_2, duration_2)

    print(data_events.info['ch_names'])
    # 3、预处理
    # bads = []
    # data_preprocess(data_events, bads)
