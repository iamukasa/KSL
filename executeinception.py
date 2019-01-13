import tensorflow as tf
import cv2
import base64
import  time
cap = cv2.VideoCapture(-1)
count =0

if cap.isOpened():
    rval,frame=cap.read()
else :
    rval=False

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line
                           in tf.gfile.GFile("inception/output_labels.txt")]

# Unpersists graph from file
f = tf.gfile.FastGFile("inception/output_graph.pb", 'rb')
graph_def = tf.GraphDef()
graph_def.ParseFromString(f.read())
_ = tf.import_graph_def(graph_def, name='')
with tf.Session() as sess:
    while True:
        cv2.imshow("Inception Classifier", frame)
        # true or false for ret if the capture is there or not
        ret, frame = cap.read()  # read frame from the ]

        #time.sleep(1)
        buffer= cv2.imencode('.jpg',frame)[1].tostring()
        image_data=buffer
        # Feed the image_data as input to the graph and get first prediction
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        predictions = sess.run(softmax_tensor,{'DecodeJpeg/contents:0': image_data})
        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-1:][::-1]

        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            print('%s (score = %.5f)' % (human_string, score))
            s=str(human_string)+ " "+str(score * 100)+"%"

            if score > 0.4:
                text_color = (255, 0, 0)
                cv2.putText(frame, s, (33, 455), cv2.QT_FONT_NORMAL, 1.0, text_color, thickness=2)
            name_of_file="frames/"+str(human_string)+str(score)+".jpg"
            cv2.imwrite(name_of_file,frame)
            print(name_of_file)

            count= count+1
        key = cv2.waitKey(20)
        if key == 27:
            break
    #cv2.destroyWindow("Inception Classifier")


