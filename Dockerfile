# Using lightweight alpine image
FROM python:3.6-alpine

# Updating and installing packages
RUN apk update

# Installing and creating a virtual environment
RUN pip install virtualenv
RUN virtualenv venv
RUN source venv/bin/activate

COPY bootstrap.sh ./
# Add application code.
COPY ./app /app

WORKDIR /usr/src/app

# Installing requirements for venv
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Start app
EXPOSE 5000
ENTRYPOINT ["./bootstrap.sh"]
