FROM huggingface/transformers-pytorch-gpu

WORKDIR /kirby

ADD . /kirby

# Workaround https://unix.stackexchange.com/questions/2544/how-to-work-around-release-file-expired-problem-on-a-local-mirror

RUN echo "Acquire::Check-Valid-Until \"false\";\nAcquire::Check-Date \"false\";" | cat > /etc/apt/apt.conf.d/10no--check-valid-until

ENV PACKAGES="\
    git \
    vim \
"

RUN apt update && apt -y install ${PACKAGES}

RUN pip install -r requirements.txt

RUN python3 setup.py develop

ENV NAME kirby 
