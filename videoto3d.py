import numpy as np
import cv2


class Videoto3D:

    def __init__(self, width, height, depth):
        self.width = width
        self.height = height
        self.depth = depth

    def video3d(self, filename, color=False, skip=True):
        cap = cv2.VideoCapture(filename) # capturing the video of filename
        nframe = cap.get(cv2.CAP_PROP_FRAME_COUNT) # count frame 
        if skip:
            # (the number of frame/depth)分間隔を空ける
            frames = [x * nframe / self.depth for x in range(self.depth)]
        else:
            # 連続して
            frames = [x for x in range(self.depth)]
        framearray = []

        for i in range(self.depth):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frames[i]) # 位置frame[i](=0~depth)のフレームをセット(プロパティをいじる宣言)
            ret, frame = cap.read() # ret：次のコマがあるか否か(T/F値), frame：次のフレーム画像
            frame = cv2.resize(frame, (self.height, self.width)) # フレーム画像のリサイズ(目的はわからない？)
            if color:
                framearray.append(frame)
            else:
                framearray.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

        cap.release()
        return np.array(framearray)



    def get_UCF_classname(self, filename):
        return filename[filename.find('_') + 1:filename.find('_', 2)] # ex. 'v_Diving_g01_c01.avi' => filename.find('_',2) = Diving

