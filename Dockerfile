FROM ubuntu:22.04
RUN DEBIAN_FRONTEND=noninteractive \
  apt-get update \
  && apt-get install -y python3 \
  && rm -rf /var/lib/apt/lists/*
WORKDIR /app 
COPY . .
RUN pip install -r requirements.txt

ENTRYPOINT ["streamlit", "run", "streamlit_app.py",]