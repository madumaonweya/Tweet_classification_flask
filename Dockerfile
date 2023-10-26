FROM python:3.8.18
COPY . /tweet_app
WORKDIR /tweet_app
RUN pip install -r requirement.txt
CMD streamlit run tweet_app.py

