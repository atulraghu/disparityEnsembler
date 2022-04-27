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
COPY disparity.py /Ensembler/test.py
COPY stackhourglass.py /Ensembler/models/stackhourglass.py
COPY submodule.py /Ensembler/models/submodule.py
COPY KITTILoader.py /Ensembler/dataloader/KITTILoader.py
COPY rundisparity.sh /Ensembler/runtest.sh
RUN chmod +x rundisparity.sh
RUN chmod +x runtest.sh
#CMD ./rundisparity.sh