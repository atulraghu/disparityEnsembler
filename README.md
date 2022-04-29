# disparityEnsembler
1. Download the KITTI 2015 Stereo dataset, and include it in this main folder. Do not unzip. The zipped download of this dataset should be called data_scene_flow.zip. 
2. Run the following command to start docker container: docker build -t ens . && docker run -it --gpus all --shm-size=16G -v "$(pwd)"/mount:/Ensembler/mount ens
3. Modify Dockerfile to not run disparity.py on startup if desired. 
