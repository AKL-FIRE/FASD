import numpy as np
# import cv2
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

CANVAS_HIGH = 1100
CANVAS_WIDTH = 1100


class DataShow(object):
    # filepath 填写数据文件路径， canvas为空时创建一个新画布，否则使用参数里的画布
    def __init__(self, filepath=None, canvas=None):
        if filepath is not None:
            self.filepath = filepath
        self.data = None
        if canvas is None:
            self.canvas = np.ones((CANVAS_HIGH, CANVAS_WIDTH, 3), np.uint8)
            self.canvas *= 255
        else:
            self.canvas = canvas

    # 读取数据, 必须先初始化
    def load_data(self):
        self.data = np.load(self.filepath, allow_pickle=True)
        length = self.data.shape[0]
        i = 0
        while i < length:
            if self.data[i][0].shape != self.data[0][0].shape:
                self.data = np.delete(self.data, i, axis=0)
                length = length - 1
                i = i - 1
            i = i + 1

    # 画出画布， name为标题的名字
    def show_img(self, name):
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.imshow(name, self.canvas)
        cv2.resizeWindow(name, 1024, 1024)

    def draw_data_person(self, i_th_person):
        data_slice_per_person = self.data[i_th_person][0]
        data_slice_per_person_length = data_slice_per_person.shape[0]
        for j in range(data_slice_per_person_length):
            data_slice_per_time = data_slice_per_person[j]
            x = data_slice_per_time[0]
            y = data_slice_per_time[1]
            cv2.circle(self.canvas, (y, x), 3, (255, 0, 0))

    # 用于画出第i_th_person个人的眼动轨迹
    def draw_data_trace(self, i_th_person):
        data_slice_per_person = self.data[i_th_person][0]
        data_slice_per_person_length = data_slice_per_person.shape[0]
        for j in range(1, data_slice_per_person_length):
            data_slice_per_time_prev = data_slice_per_person[j - 1]
            data_slice_per_time_current = data_slice_per_person[j]
            first_point = (data_slice_per_time_prev[1], data_slice_per_time_prev[0])
            second_point = (data_slice_per_time_current[1], data_slice_per_time_current[0])
            cv2.circle(self.canvas, first_point, 3, (255, 0, 0))
            cv2.circle(self.canvas, second_point, 3, (255, 0, 0))
            cv2.arrowedLine(self.canvas, first_point, second_point, (0, 0, 255), tipLength=0.6)

    # 用于画出第frame帧的眼睛的fixation
    def draw_data_per_frame(self, frame, radius, color=None):
        for i in range(self.data.shape[0]):
            data_slice_per_frame = self.data[i][0][frame]
            if color is None:
                cv2.circle(self.canvas, (data_slice_per_frame[1], data_slice_per_frame[0]), radius, (255, 0, 0))
            else:
                cv2.circle(self.canvas, (data_slice_per_frame[1], data_slice_per_frame[0]), radius, color)

    def clear_canvas(self):
        self.canvas = np.ones((CANVAS_HIGH, CANVAS_WIDTH, 3), np.uint8)
        self.canvas *= 255

    def down_sample(self, interval, result=None):
        len_frame = self.data[0][0].shape[0]
        index = math.floor(len_frame / interval)
        if result is None:
            result = []
        for i in range(self.data.shape[0]):
            for j in range(index):
                result.append([self.data[i][0][j * interval][0], self.data[i][0][j * interval][1]])
        return result, index


def draw_data_per_frame_two(control_show, fasd_show, frame, radius1, radius2, color1, color2):
    control_show.clear_canvas()
    fasd_show.canvas = control_show.canvas
    control_show.draw_data_per_frame(frame, radius1, color1)
    fasd_show.draw_data_per_frame(frame, radius2, color2)
    # fasd_show.show_img('fasd')
    return fasd_show.canvas


def generate_video(control_show, fasd_show, start_frame, end_frame):
    video = np.zeros((CANVAS_HIGH, CANVAS_WIDTH, 3, 100), np.uint8)
    for i in range(start_frame, end_frame):
        canvas_per_frame = draw_data_per_frame_two(control_show, fasd_show, i, 5, 10, (255, 0, 0), (0, 0, 255))
        video[:, :, :, i] = canvas_per_frame.copy()
    return video


def show_video(video, start_frame, end_frame, interval):
    cv2.namedWindow("video", cv2.WINDOW_NORMAL)
    for i in range(start_frame, end_frame):
        cv2.imshow("video", video[:, :, :, i])
        cv2.waitKey(interval)


def calculate_cluster_rate(coordinators, threshold):
    flag = np.zeros(len(coordinators), dtype=np.uint8)
    for i in range(len(coordinators)):
        for j in range(i + 1, len(coordinators)):
            if math.sqrt(math.pow(coordinators[i][0] - coordinators[j][0], 2) + math.pow(coordinators[i][1] - coordinators[j][1], 2)) <= threshold:
                flag[i] = 1
                flag[j] = 1
    overlap = np.sum(flag) / flag.shape
    not_overlap = 1 - overlap
    return overlap, not_overlap


def save_overlap_result(filepath, interval, threshold):
    with open("./result.txt", "a+") as file:
        control_data = DataShow(filepath + "control_rawdata.npy")
        control_data.load_data()
        coor, _ = control_data.down_sample(interval)
        overlap, not_overlap = calculate_cluster_rate(coor, threshold)
        overlap = overlap[0]
        not_overlap = not_overlap[0]
        file.write("([" + str(overlap) + ", " + str(not_overlap) + "], ")
        fasd_data = DataShow(filepath + "fasd_rawdata.npy")
        fasd_data.load_data()
        coor1, _ = fasd_data.down_sample(interval)
        overlap, not_overlap = calculate_cluster_rate(coor1, threshold)
        overlap = overlap[0]
        not_overlap = not_overlap[0]
        file.write("[" + str(overlap) + ", " + str(not_overlap) + "], ")
        coor2 = coor + coor1
        overlap, not_overlap = calculate_cluster_rate(coor2, threshold)
        overlap = overlap[0]
        not_overlap = not_overlap[0]
        file.write("[" + str(overlap) + ", " + str(not_overlap) + "])\n")


def load_all_snips(basepath):
    for i in range(1, 71):
        print("正在处理第%d帧， 总共%d帧" % (i, 71))
        dir_path = basepath + "\\snip" + str(i) + "\\"
        save_overlap_result(dir_path, 50, 10)


def generate_heatmap(coordinators):
    heatmap = np.zeros((1500, 1500), dtype=np.uint32)
    for ele in coordinators:
        x, y = ele
        x = int(x)
        y = int(y)
        heatmap[x][y] = heatmap[x][y] + 1
    result = np.zeros((150, 150), dtype=np.uint32)
    for i in range(150):
        for j in range(150):
            result[i][j] = np.sum(heatmap[i * 150 : (i + 1) * 150, j * 150 : (j + 1) * 150])
    return result


def show_heatmap(heatmap):
    plt.figure(0)
    plt.pcolor(heatmap)
    plt.colorbar()
    plt.show()


def draw_trace3d(data, ax=None, person=None, color='red'):
    if person is None:
        person = [0]
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        Axes3D.view_init(ax, 45, 135)
    times = np.array([i for i in range(2000)])
    for i in range(len(person)):
        draw_data = np.zeros((2000, 2), dtype=np.uint32)
        draw_data[:, 0] = data.data[person[i]][0][0:2000, 0]
        draw_data[:, 1] = data.data[person[i]][0][0:2000, 1]
        Axes3D.plot(ax, xs=draw_data[:, 0], ys=draw_data[:, 1], zs=times, color=color)
    return ax


def save_draw_trace3d(path):
    data = DataShow(path)
    data.load_data()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(10):
        draw_trace3d(data, color='green', person=[i])
        # filename = './result/img' + str(i) + '.jpg'
        # plt.imsave(filename, ax)
    #draw_trace3d(data, color='red', person=[30])


if __name__ == '__main__':
    # load_all_snips("E:\\DATA\\FASD eye dataset")
    # save_overlap_result(50, 10)
    # control_data = DataShow("./data/control_rawdata.npy")
    # control_data.load_data()
    # coor, _ = control_data.down_sample(30)
    # fasd_data = DataShow("./data/fasd_rawdata.npy")
    # fasd_data.load_data()
    # coor2, _ = fasd_data.down_sample(30, coor)
    # heatmap = generate_heatmap(coor2)
    # show_heatmap(heatmap)

    # control_data = DataShow("./data/control_rawdata.npy")
    # control_data.load_data()
    # fasd_data = DataShow("./data/fasd_rawdata.npy", control_data.canvas)
    # fasd_data.load_data()
    # video = generate_video(control_data, fasd_data, 0, 30)
    # show_video(video, 0, 30, 300)
    # control_data = DataShow("./data/fasd_rawdata.npy")
    # control_data.load_data()
    # control_data.draw_data_trace(0)
    # control_data.show_img("trace")
    # cv2.waitKey()

    # control_data = DataShow("./data/control_rawdata.npy")
    # control_data.load_data()
    # ax = draw_trace3d(control_data, color='green', person=[i for i in range(control_data.data.shape[0])])
    # # # ax = draw_trace3d(control_data, color='green', person=[0])
    # # draw_trace3d(control_data, color='green', person=[i for i in range(control_data.data.shape[0])])
    # fasd_data = DataShow("./data/fasd_rawdata.npy")
    # fasd_data.load_data()
    # draw_trace3d(fasd_data, ax=ax, color='red', person=[i for i in range(fasd_data.data.shape[0])])
    # # # draw_trace3d(fasd_data, ax=ax, color='red', person=[0])
    # # draw_trace3d(fasd_data, color='red', person=[i for i in range(fasd_data.data.shape[0])])
    # plt.show()

    save_draw_trace3d("./data/control_rawdata.npy")
    save_draw_trace3d("./data/fasd_rawdata.npy")
    plt.show()
