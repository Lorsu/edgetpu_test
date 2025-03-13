FROM debian:10 
 
WORKDIR /home 
ENV HOME=/home 
RUN cd ~ 
RUN apt-get update 
RUN apt-get install -y git nano python3-pip python-dev pkg-config wget usbutils curl 
 
RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" \ 
| tee /etc/apt/sources.list.d/coral-edgetpu.list 
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - 

RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | tee /etc/apt/sources.list.d/coral-edgetpu.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -

#sudo sh -c "echo 'SUBSYSTEM==\"apex\", MODE=\"0660\", GROUP=\"apex\"' >> /etc/udev/rules.d/65-apex.rules"
#sudo groupadd apex
#sudo adduser $USER apex

RUN apt-get update 
RUN apt-get install -y edgetpu-examples
RUN apt-get install -y python3-opencv
#RUN apt-get install -y gasket-dkms
RUN apt-get install -y libedgetpu1-std
RUN apt-get install -y python3-pycoral