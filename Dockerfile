FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel

RUN apt update
RUN apt-get install -y git cmake qt5-default zip vim
RUN pip3 install --upgrade pip
WORKDIR /
RUN git clone https://github.com/JiaRenChang/PSMNet.git
Run mv /PSMNet /Ensembler
WORKDIR /Ensembler
COPY data_scene_flow.zip /Ensembler/data_scene_flow.zip
RUN unzip /Ensembler/data_scene_flow.zip
RUN pip install scikit-image
COPY disparity.py /Ensembler/disparity.py
COPY KITTILoader.py /Ensembler/dataloader/KITTILoader.py
COPY KITTIloader2015.py /Ensembler/dataloader/KITTIloader2015.py
COPY rundisparity.sh /Ensembler/rundisparity.sh
RUN chmod +x rundisparity.sh
CMD ./rundisparity.sh
