# legorobot
This repository is a distilled version of https://github.com/SteffenCzolbe/LEGO-machine-learning, containing only the needed code for the swing robot.
If you are reading this it is probably because you have been given the task of setting the robot up to run on a new coputer or because the old one failed.

## General notes for the setup
The setup consists of 2 computers, the computer on which the computations are done and the computer inside the ev3 lego mindstorm. The first computer is nessecary because the computer inside the ev3 is so bad that it cannot do the computations needed for reinforcement learning, even tough this is a very small computational problem.

## Tips for setup
Setup a python environment on the 2nd computer per the requirements.txt file, note here that we use old versions of a lot of packages, this is especially important for the rpyc pacakge since the installed version on the robot is 3.3.0, so the connected computer needs to use the same version.
My recommendation is to use conda for this setup, just keep in mind that this will require at least more than 1Gb of memory
Then make sure that the computers are connected on the network, if you can ping ev3dev.local and ssh into it, you are golden. Otherwise you need to do some networking. The only solution i could make work is the command "sudo ip addr add 10.42.0.1/24 dev <usb_address>" where <usb_address> is the usb address on the local network and can be found using ifconfig.
From there on you should be golden :)

## How to make it run
From your computer, you ssh into the robot with "ssh robot@ev3dev.local" (or substitude ev3dev.local with whatever ip adress you made work). The password is "maker"
run the script "rpyc_server.sh" located in the home folder.
Then, from your computer, you can check if the connection using the check_con.py script.
Is the samme manner you can make the robot learn to swing using the swing_script.py file.
NOTE: Near the end of the swing_script.py file there are some saved vales from different time in training, you can use these as initial valuesif you don't want to wait for the robot to learn swinging from nothing.
