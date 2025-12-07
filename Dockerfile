FROM quay.io/jupyter/minimal-notebook:2024-10-14

USER root

WORKDIR /home/jovyan/work

# Install mamba + conda-lock
RUN conda install -n base -c conda-forge -y mamba conda-lock && \
    conda clean --all -y

# Install Quarto for ARM64
RUN apt-get update && \
    apt-get install -y wget && \
    wget https://quarto.org/download/latest/quarto-linux-arm64.deb && \
    apt-get install -y ./quarto-linux-arm64.deb && \
    rm quarto-linux-arm64.deb

# Copy environment + lock file
COPY environment.yml .
COPY conda-lock.yml .

# Create a NEW environment (NOT base)
RUN conda-lock install --prefix /opt/conda/envs/env conda-lock.yml && \
    conda clean --all -y

# Make the new environment the default
ENV PATH="/opt/conda/envs/env/bin:$PATH"
ENV CONDA_DEFAULT_ENV=env

USER ${NB_UID}

EXPOSE 8888

CMD ["start-notebook.sh"]
