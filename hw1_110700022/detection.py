import cv2
import matplotlib.pyplot as plt
import numpy as np

def crop(x1, y1, x2, y2, x3, y3, x4, y4, img) :
    """
    This function ouput the specified area (parking space image) of the input frame according to the input of four xy coordinates.
    
      Parameters:
        (x1, y1, x2, y2, x3, y3, x4, y4, frame)
        
        (x1, y1) is the lower left corner of the specified area
        (x2, y2) is the lower right corner of the specified area
        (x3, y3) is the upper left corner of the specified area
        (x4, y4) is the upper right corner of the specified area
        frame is the frame you want to get it's parking space image
        
      Returns:
        parking_space_image (image size = 360 x 160)
      
      Usage:
        parking_space_image = crop(x1, y1, x2, y2, x3, y3, x4, y4, img)
    """
    left_front = (x1, y1)
    right_front = (x2, y2)
    left_bottom = (x3, y3)
    right_bottom = (x4, y4)
    src_pts = np.array([left_front, right_front, left_bottom, right_bottom]).astype(np.float32)
    dst_pts = np.array([[0, 0], [0, 160], [360, 0], [360, 160]]).astype(np.float32)
    projective_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    croped = cv2.warpPerspective(img, projective_matrix, (360,160))
    return croped


def detect(data_path, clf):
    """
    Please read detectData.txt to understand the format. 
    Use cv2.VideoCapture() to load the video.gif.
    Use crop() to crop each frame (frame size = 1280 x 800) of video to get parking space images. (image size = 360 x 160) 
    Convert each parking space image into 36 x 16 and grayscale.
    Use clf.classify() function to detect car, If the result is True, draw the green box on the image like the example provided on the spec. 
    Then, you have to show the first frame with the bounding boxes in your report.
    Save the predictions as .txt file (ML_Models_pred.txt), the format is the same as Sample.txt.
    
      Parameters:
        dataPath: the path of detectData.txt
      Returns:
        No returns.
    """
    # Begin your code (Part 4)
    #raise NotImplementedError("To be implemented")

    f = open(data_path, "r")
    number = int(f.readline())
    line = f.readline()

    myList = []
    while number > 0:
        x1, y1, x2, y2, x3, y3, x4, y4 = line.split(' ', 7)
        templist = [x1, y1, x2, y2, x3, y3 , x4, y4]
        myList.append(templist)
        #print(str(x1) + " " + str(y1))
        line = f.readline()
        number -= 1

    f.close()

    cap = cv2.VideoCapture("data/detect/video.gif")

    ret,frame = cap.read()
    fw = open("ML_Models_pred.txt", "w")
    while cap.isOpened():
        for i in range(76):
            image = crop(myList[i][0], myList[i][1], myList[i][2], myList[i][3], myList[i][4], myList[i][5], myList[i][6],
                     myList[i][7], frame)
            image = cv2.resize(image, (36, 16), interpolation=cv2.INTER_AREA)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            x_test = np.array(image)
            x_test = np.array(x_test).flatten().reshape(1, -1)
            y_pred = clf.classify(x_test)
            if y_pred == 1:
                fw.write('1')
                cv2.rectangle(frame, (int(myList[i][4]), int(myList[i][5])), (int(myList[i][2]), int(myList[i][3])), (0, 255, 0), 1)
            else:
                fw.write('0')
        cv2.imshow('result', frame)
        ret, frame = cap.read()
        # 按下q後
        if cv2.waitKey(5) == ord('q'):
            break
    fw.close()


        #x_test = np.array([List_train])
        #x_test = np.array(x_test).flatten().reshape(76, -1)
        #print(image.shape)
       # result = clf.classify(image)



    # End your code (Part 4)
