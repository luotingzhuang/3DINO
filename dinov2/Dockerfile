FROM python:3.9-slim

ARG USER_ID
ARG GROUP_ID

ENV TZ=America/Los_Angeles

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

RUN if [ ${USER_ID:-0} -ne 0 ] && [ ${GROUP_ID:-0} -ne 0 ]; then \
    groupadd -g ${GROUP_ID} luotingzhuang && \
    useradd -l -u ${USER_ID} -g luotingzhuang luotingzhuang && \
    install -d -m 0755 -o luotingzhuang -g luotingzhuang /home/luotingzhuang;fi


USER luotingzhuang