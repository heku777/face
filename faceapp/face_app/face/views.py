from django.shortcuts import render
from django.views import generic

class Home(generic.TemplateView):
    template_name = 'face_attendance/zikkou.html'

    import cv2
    import numpy as np
    import pickle
    from sklearn import svm
    import openpyxl
    import pprint
    import datetime

    def resize_image(image, height, width):
        # 元々のサイズを取得
        org_height, org_width = image.shape[:2]
        # 大きい方のサイズに合わせて縮小
        if float(height) / org_height > float(width) / org_width:
            ratio = float(height) / org_height
        else:
            ratio = float(width) / org_width
        # リサイズ
        resized = cv2.resize(image, (
        int(org_height * ratio), int(org_width * ratio)))
        return resized

    def mag_change(bai, img):
        # グレースケール変換
        frame_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 顔認識処理を早くするために、画像の解像度を下げる。画像サイズを取得し、サイズを変更する
        orgHeight, orgWidth = frame_gray.shape[:2]
        frame_gray_size = (int(orgWidth / bai), int(orgHeight / bai))
        # リサイズする
        frame_gray_resize = cv2.resize(frame_gray, frame_gray_size,
                                       interpolation=cv2.INTER_AREA)
        return frame_gray_resize

    def export_exel(name):
        # エクセルを開く
        wb = openpyxl.load_workbook('.../face_ai/勤怠管理表.xlsx')
        # シートを選択
        sheet = wb['Attendance_management_table']
        # 今日の日付を取得
        now = datetime.datetime.now()
        day = now.day
        # 今日の時刻を取得
        hour = now.hour
        minute = now.minute
        # 出勤時刻
        my_time = str(hour) + ':' + str(minute)
        # 今日の日付のセルに記録
        today_cell = int(day) + 2
        # 人によって分岐
        with open('.../face_ai//name_list.txt') as nr:
            name_list = [s.strip() for s in nr.readlines()]
            print(name_list)
            if name == 'akira':
                if name_list[0] == 'notexists':
                    sheet['B' + str(today_cell)] = my_time
                    with open('./name_list.txt', mode='w') as nw:
                        nw.write('exists' + '\n' + name_list[1])
                    print('akiraさんが出勤しました')
                else:
                    sheet['C' + str(today_cell)] = my_time
                    with open('./name_list.txt', mode='w') as nw:
                        nw.write('notexists' + '\n' + name_list[1])
                    print('akiraさんが退勤しました')
            elif name == 'kazuto':
                if name_list[1] == 'notexists':
                    sheet['D' + str(today_cell)] = my_time
                    with open('./name_list.txt', mode='w') as nw:
                        nw.write(name_list[0] + '\n' + 'exists')
                    print('kazutoさんが出勤しました')
                else:
                    sheet['E' + str(today_cell)] = my_time
                    with open('./name_list.txt', mode='w') as nw:
                        nw.write(name_list[0] + '\n' + 'notexists')
                    print('kazutoさんが退勤しました')
        # 保存
        wb.save('.../face_ai//勤怠管理表.xlsx')

    if __name__ == "__main__":
        # 内蔵カメラを起動
        cap = cv2.VideoCapture(0)

        # OpenCVに用意されている顔認識するためのxmlファイルのパス
        cascade_path = "/Users/akira/Desktop/hackathon/OctoberNoir/face_ai/haarcascades/haarcascade_frontalface_alt.xml"
        # カスケード分類器の特徴量を取得する
        cascade = cv2.CascadeClassifier(cascade_path)

        # STEP2で保存したモデルのロード
        clf = pickle.load(open(".../face_ai//face_model.sav", "rb"))

        # 連続で表示されるかカウントする
        akira_counter = 0
        kazuto_counter = 0

        while True:
            # 内蔵カメラから読み込んだキャプチャデータを取得
            ret, frame = cap.read()

            # 結果を保存するための変数を用意しておく
            result = frame

            # 顔認識を低解像度で実施するための準備
            mag = 3  # 倍率
            resized = mag_change(mag, frame)

            # 顔認識の実行
            facerect = cascade.detectMultiScale(resized, scaleFactor=1.2,
                                                minNeighbors=3,
                                                minSize=(10, 10))

            # 顔が見つかったらfacerectには(四角の左上のx座標, 四角の左上のy座標, 四角の横の長さ, 四角の縦の長さ) が格納されている。
            if len(facerect) > 0:
                for x, y, w, h in facerect:
                    x = int(x * mag)
                    y = int(y * mag)
                    w = int(w * mag)
                    h = int(h * mag)

                    # 顔の部分だけ切り抜いてモザイク処理をする
                    cut_img = result[y:y + h, x:x + w]
                    cut_face = cut_img.shape[:2][::-1]

                    # 顔判別用の箱を作る
                    recog = []
                    # リアルタイムに認識した顔を50×50の解像度（clf作成時の解像度）に合わせる
                    recog_img = cv2.resize(cut_img, (50, 50),
                                           interpolation=cv2.INTER_LINEAR)
                    # グレースケールに変換
                    recog_gray = cv2.cvtColor(recog_img, cv2.COLOR_BGR2GRAY)
                    # 1次元の行列に変換
                    recog_gray = np.array(recog_gray, "uint8").flatten()
                    # 顔認識用の箱に入れる
                    recog.append(recog_gray)
                    # 行列に変換
                    recog = np.array(recog)

                    # 予測実行
                    pred = clf.predict(recog)
                    print(pred)

                    # 予測の結果、画像上に文字を表示
                    if pred == "akr":
                        name = "akira"
                        kazuto_counter = 0
                        akira_counter += 1
                        cv2.putText(result, name, (int(x), y - int(h / 5)),
                                    cv2.FONT_HERSHEY_PLAIN, int(w / 50),
                                    (0, 255, 0), 5, cv2.LINE_AA)

                    elif pred == "kzt":
                        name = "kazuto"
                        akira_counter = 0
                        kazuto_counter += 1
                        cv2.putText(result, name, (int(x), y - int(h / 5)),
                                    cv2.FONT_HERSHEY_PLAIN, int(w / 50),
                                    (0, 255, 0), 5, cv2.LINE_AA)

                    else:
                        result = result

            cv2.imshow("image", result)

            # qキーを押すとループ終了
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            elif akira_counter == 5:
                print('あなたはakiraさんですか？')
                agree = input('はい：1, いいえ：2 →')
                if agree == '1':
                    print('akira')
                    export_exel('akira')
                    akira_counter = 0
                    break
            elif kazuto_counter == 5:
                print('あなたはkazutoさんですか？')
                agree = input('はい：1, いいえ：2 →')
                if agree == '1':
                    print('kazuto')
                    export_exel('kazuto')
                    kazuto_counter = 0
                    break

        # 内蔵カメラを終了
        cap.release()
        cv2.destroyAllWindows()