# a Dockerfile is a build specification for a Docker image
FROM continuumio/anaconda3:2020.11

# /code is copy of all content we have in the folder we are working in
ADD . /code
WORKDIR /code

ENTRYPOINT [ "python", "stroke_app.py" ]