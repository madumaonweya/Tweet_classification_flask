FROM python:3.8.18
COPY . /flask_app
WORKDIR /flask_app
RUN pip install -r requirement.txt
CMD python flask_app.py

