import cv2
import numpy as np
from PIL import Image
import os
from sklearn import svm
import pickle

#顔判別器の作成
images = []
labels = []
path = "/Users/akira/Desktop/hackathon/OctoberNoir/face_ai/images/"

for f in os.listdir(path):
    #画像のパス
    image_path = os.path.join(path, f)
    #.DS_Storeファイルを読み込まないようにする
    if image_path == "/Users/akira/Desktop/hackathon/OctoberNoir/face_ai/images/.DS_Store":
        continue
    else:
        #グレースケールで読み込む(convert("L")でグレースケール)
        gray_image = Image.open(image_path).convert("L")
        #numpy配列に格納
        image = np.array(gray_image,"uint8")
        #umageを1次元配列に変換
        image = image.flatten()
        #images[]にimageを格納
        images.append(image)
        #ファイル名からラベルを取得
        labels.append(str(f[0:3]))
#行列に変換
labels = np.array(labels)
images = np.array(images)

#svmの変換器を作成
clf = svm.LinearSVC()
#学習
clf.fit(images,labels)

#学習モデルを保存する
filename = "face_model.sav"
pickle.dump(clf,open(filename,"wb"))

print("モデル保管完了")