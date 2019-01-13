import time

import tensorflow as tf
import numpy as np
import cv2
import imageclassifier as imclassify

cap = cv2.VideoCapture(-1)

count =0

if cap.isOpened():
    rval,frame=cap.read()
else :
    rval=False

while True:
    cv2.imshow("Mobilenets classifier", frame)
    # true or false for ret if the capture is there or not
    ret, frame = cap.read()  # read fram from the webcam


    #name_of_file="frames/frame%d.jpg" % count
    # cv2.imwrite(name_of_file,frame)
    # print(name_of_file)

    #time.sleep(1)
    graph, t, input_operation, output_operation, label_file = imclassify.classify(frame)

    start = time.time()
    results = tf.Session(graph=graph).run(output_operation.outputs[0],
                                          {input_operation.outputs[0]: t})
    end = time.time()
    results = np.squeeze(results)

    top_k = results.argsort()[-1:][::-1]
    labels = imclassify.load_labels(label_file)
    print('\nEvaluation time (1-image): {:.3f}s\n'.format(end - start))
    template = "Prediction {} (score={:0.5f})"
    for i in top_k:
        s = str(template.format(labels[i], results[i]))
        print(s)

        if results[i] < 0.5:
            text_color = (255, 0, 0)
            cv2.putText(frame, s, (33, 455), cv2.QT_FONT_NORMAL, 1.0, text_color, thickness=2)
        name_of_file = "framemobilenets/" + str(labels[i])+str(labels[i]) + "%d.jpg" % count
        cv2.imwrite(name_of_file, frame)

    count += 1
    key = cv2.waitKey(20)
    if key == 31:
        pass
    if key == 27:
        break
cv2.destroyWindow("Mobilenets classifier")
