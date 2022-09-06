#!/bin/bash

sudo apt-get update
sudo apt-get -y install apt-transport-https ca-certificates curl criu software-properties-common
sudo apt-get -y install python3-pip virtualenv python3-setuptools linux-tools-generic sysbench ioping
sudo apt-get -y install linux-tools-4.15.0-72-generic
sudo apt-get -y install linux-cloud-tools-4.15.0-191-generic
sudo apt-get -y install linux-tools-4.15.0-191-generic
sudo apt-get -y install linux-tools-5.15.0-1011-aws
sudo apt-get -y install linux-tools-aws
sudo apt-get -y install linux-cloud-tools-5.15.0-1011-aws
sudo apt-get -y install linux-cloud-tools-aws
sudo python3 -m pip install flask-restful inotify Flask psutil docker
sudo chmod +x $HOME/agent/agent.py
sudo sed -i -e 's/\r$//' ./agent/scripts/calIPS_clock.sh
# Install Docker

sudo mkdir /etc/docker
sudo cp $HOME/agent/scripts/daemon.json /etc/docker
sudo apt-get -y install apt-transport-https ca-certificates curl gnupg-agent software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
sudo apt-get update
sudo apt-get -y install docker-ce docker-ce-cli containerd.io
sudo groupadd docker 
sudo usermod -aG docker $USER
newgrp docker

# Setup Flask 

#sudo cp ~/agent/scripts/flask.conf /etc/init.d/
#sudo cp ~/agent/scripts/flask.service /lib/systemd/system/flask.service
#sudo service flask start
sudo python3 ./agent/agent.py &
sudo chmod +x ~/agent/scripts/delete.sh

# Load Docker images

sudo docker pull shreshthtuli/yolo
sudo docker pull shreshthtuli/pocketsphinx
sudo docker pull shreshthtuli/aeneas
