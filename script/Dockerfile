FROM jupyter/tensorflow-notebook
USER root
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y
USER 1000
RUN pip install --upgrade pip numpy matplotlib opencv-python
