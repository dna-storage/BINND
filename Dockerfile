FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

ARG USERNAME=developer
ARG USER_UID=1000
ARG USER_GID=1000

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME

# Install dependencies
ENV CONDA_HOME=/opt/conda
ENV PATH=$CONDA_HOME/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
        wget git build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py312_24.1.2-0-Linux-x86_64.sh -O miniconda.sh \
    && /bin/bash miniconda.sh -b -p $CONDA_HOME \
    && rm miniconda.sh \
    && conda clean --all -f -y


# Use bash shell so conda works properly
SHELL ["/bin/bash", "-lc"]

# Create environment BEFORE switching user
COPY BINND.yml .
RUN conda env create -f BINND.yml --name BINND

ENV PATH=/opt/conda/envs/BINND/bin:$PATH

WORKDIR /home/$USERNAME/app

# Copy project files
COPY requirements.txt .
COPY setup.py .
COPY init.sh .
COPY Makefile .
COPY src/ ./src/
COPY main.py .
COPY inference_demo/ ./inference_demo/

ARG REQUIREMENTS_FILE="requirements.txt"

# Install PyTorch inside env
RUN conda run -n BINND pip install \
        torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
        --index-url https://download.pytorch.org/whl/cu126

# 3. Install other pip packages from requirements.txt
RUN if [ -f "$REQUIREMENTS_FILE" ]; then \
        echo "--- Installing other pip packages from $REQUIREMENTS_FILE ---" \
        && conda run -n BINND pip install -r $REQUIREMENTS_FILE ; \
    else \
        echo "WARNING: $REQUIREMENTS_FILE not found. Skipping." ; \
    fi

# 4. Install current project in standard mode (pip install .)
RUN echo "--- Installing current project in standard mode (pip install .) ---" \
    && conda run -n BINND pip install .

# Initialize conda for non-root user (critical!)
RUN echo "source /opt/conda/etc/profile.d/conda.sh" >> /home/$USERNAME/.bashrc

# Fix permissions on working directory
RUN chown -R $USERNAME:$USERNAME /home/$USERNAME

# Switch to non-root user
USER $USERNAME

RUN conda init bash
RUN echo "conda activate base" >> /home/$USERNAME/.bashrc