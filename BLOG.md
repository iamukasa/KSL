#MNIST for Kenyan sign language


July 2016 I met Hudson, a brilliant deaf developer who had just built an app to teach his deaf community sexual health. We became fast friends.Since then my circle of deaf friends and acquaintances only grew.
I even got a sign name.In the deaf community its easier to assign someone a sign from a unique characteristic rather than finger spelling their name every time. But there was a problem, for a long time I couldn't communicate well enough with this growing circle of deaf friends ans acquaitances,I had not learned sign language.

Conversations with Hudson and his also deaf co founder,Alfred were first interceded by the sign language interpreters they had assign
ed for meetings. Hudson's  app and sophie bot were funded under the same the program so we had a lot of those.Our conversations grew to typing texts on each of our phones notes or messaging app.I was guilty of distracting him from paying attention to his intepreter in a couple of confrences and events. Soon we developed a crude list of gestures we both understood.These are known in the deaf community as "Village sign language".
 
Interactions with the deaf community only grew  after that. May last year at Microsoft's Build conference, on a special keynote session queue on the side , a crew of deaf devs joined us and I ached to introduce myself and show off my sign name.Lucky they had an interpreter, and he helped. June last year after an event in coastal Kenya, Hudson had me tag along and gate crash  his friends beauty pageant. She was the reigning queen of the pageant and was deaf too. We all went out for drinks later but I couldnt fully participate in the conversation.All I knew was how to finger spell "GIF", my sign name and how to say "hi".
October last year at a conference I  help organise, Droidcon Kenya, we had two deaf developers in attendance and they brought in two sign language intepreters .It was clear the universe wanted me to learn sign language.

Proud to say I am now fluent at finger spelling the Kenyan Sign Language alphabet, and I intentionally pick up a word a day. 
I know this learning curve isn't a solve everyone who wants to communicate better with deaf friends, family , workmates and acquaintances.We could use the advances ai technology has given us to build a tool to make this easier. Late last year Hudson , the brilliant dev he is, published an app to teach hearing guys sign language and that gave me the intuition for the right tool. What if we could build an image classifier sign language intepreter? What if we could build one end to end translation model for sign language just using photos of people signing? This blog investigates that.

## Image Classifier for the Kenyan sign Language Intepreter
The idea is to build a single an image classifier that recognises sign language not only by looking at photos of fingers,but through a real life feed webcam or phone camera.
Just like a human eye would. Here is a step by step  by step guide , explaining my process through the problem :

### STEP ONE : DATA COLLECTION AND LABELLING
First instinct was to head over to Kaggle to find out if a dataset existed. I found one but it was all white fingers, american sign language. The images were 
already vectorised and flattened to remove the rgb layer.It would not be suitable for my problem case. I wanted an end to end model that takes a half body image like in real world data vs having to build an object detection layer to detect hands.

Second instinct was to post on my social media accounts asking my friends to send images of them signing different letters of the alphabet. Either no one in my online circles isn't keen on sign language as I or I do not wield as much influence as I thought I did.

I had to get to the work myself collect my own data myself. Taking photos of me signing the entire KSL alphabet with different shirts and backgrounds. When I had ten photos for each letter I kept adding minute transformations on each image till I had 100+ photos. Hustle.

Labelling my dataset was the easy part.Putting every letter on its own directory and the name of the directory would be picked as the label as such 
```dir
~/data
~/data/A/A (1).jpg
~/data/A/A (2).jpg
...
~/data/Z/Z (1).jpg
~/data/Z/Z (2).jpg

```
### STEP TWO : CHOOSING AN AI MODEL
This community is big on fast ai  and ResNet but I learnt the tensorflow learning curve and doubled down on it. Convolutional networks are the go to networks for computer vision problems, but training one from scratch needs a lot of data and time. So lets stop doing this on keras codelabs.

We can retrain the last layer of an already pretrained CNN to do our image classification.I pick two pretrained Convolutional neural networks.Inception, larger,slower but high accuracy convolutional neural networks and Mobilenets, a lighter,faster ,lower accuracy neural network meant for running on mobile phones.
My background as a mobile developer has me instinctively trying to build mobile first.To my advantage I have experience with the two models and even have predefined scripts from an earlier 
"hotdog,not hotdog " codelab.

The predefined scripts do way more than just train the model.They have code to automatically download computational graphs of the pretrained weights.
Write tensorboard summary logs for evaluating and debugging the model, outputs the trained model as a frozen computational graph.
Lastly they also preprocess my trainning data,  vectorising the images and storing the vector values in a text file in a similar directory structure as the training data.This is saves a lot of time when retraining the model with tweaked hyperparameters.


[logo]:/home/iamukasa/PythonProjects/KSL/img/bottleneckmobilenets.png

Scripts are by the tensorflow team which then inspired their "Tensorflow for poets codelab".


### STEP THREE: TRAINING THE MODEL
Perks of the well written predefined scripts from the tensoflow team, we just run the scripts with the appropriate hyperparameters.
Sample run with hyperparameters

#####Mobilenets
```bash
python retrain.py \
  --bottleneck_dir= [path to bottleneck directory] \
  --how_many_training_steps= [number of trainnig steps you want] \
  --output_graph= [path and name of file you want to save .pb file]\
  --output_labels= [path and name of file to save list of labels] \
  --image_dir= [path to training data]

```
##### INCEPTION
``` bash
python retrain.py --model_dir ./inception --image_dir=data
```

To explain the different hyperparameters, the inception script has more default values defined in the scripts vs the mobilenets one.

Once everything is set , and the bottleneck code has run as illustrated above, you should expect an output like this on your terminal.

[logo]:/home/iamukasa/PythonProjects/KSL/img/bottleneckmobilenets.png

### EVALUATING AND DEBUGGING THE MODEL
The predefined scripts are so well written you can simply visualise accuracy and loss over training iterations with one simple command
``` bash
tensorboard --logdir /tmp/retrain_logs

```
If you passed a different path to write summary logs change that in the arguments of the command
When you click the link that generates you should see an output like this :

[logo]:/home/iamukasa/PythonProjects/KSL/img/tensorboardinception1500.png


Nothing is as satisfying to an AI developer as those two plotted graphs, accuracy headed towards 1 and loss (entropy) headed towards 0 each training iterations. Its the first indicator you are doing things right.
There are two different coloured plots,orange one is a plot on evaluation of  loss and accuracy with data we trained the model with and the blue one a plot that evaluates the model with data that was not used to train the model.


To build even better intuition you can visualise the computational graph of the neural network you just trained, to see what beauty is happening under the hood.

###### inception
[logo]:/home/iamukasa/PythonProjects/KSL/img/inception.png


###### Mobilenets
[logo]:/home/iamukasa/PythonProjects/KSL/img/mobilenets.png