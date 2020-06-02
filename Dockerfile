# Using lightweight alpine image
FROM ubuntu:18.04

# Updating and installing packages
RUN apt-get update
RUN apt-get install -y apt-utils
RUN apt-get install -y python3
RUN apt-get install -y python-pip
#RUN pip install virtualenv
#RUN virtualenv venv
#RUN source venv/bin/activate

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY bootstrap.sh ./
# Add application code.
COPY ./app /app

#WORKDIR /usr/src/app

# Installing requirements for venv
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Start app
EXPOSE 5000
ENTRYPOINT ["./bootstrap.sh"]
