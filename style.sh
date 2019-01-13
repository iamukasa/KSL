counter=1
while [ $counter -le 100000 ]
do
 python /home/iamukasa/PythonProjects/KSL/styletransfer.py
 ((counter++))
done
echo done
