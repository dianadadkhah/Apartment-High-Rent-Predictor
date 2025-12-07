FROM quay.io/jupyter/minimal-notebook:2024-10-14

USER root

WORKDIR /home/jovyan/work

# Install mamba (fast solver)
RUN conda install -n base -c conda-forge mamba && \
    conda clean --all -y

# Install Quarto for ARM64 (M1/M2 Macs + ARM Docker)
RUN apt-get update && \
    apt-get install -y wget && \
    wget https://quarto.org/download/latest/quarto-linux-arm64.deb && \
    apt-get install -y ./quarto-linux-arm64.deb && \
    rm quarto-linux-arm64.deb

# Copy environment file
COPY environment.yml .

# Install pinned environment (Python, pandas, sklearn, etc.)
RUN mamba env update -n base -f environment.yml && \
    conda clean --all -y

USER ${NB_UID}

EXPOSE 8888

CMD ["start-notebook.sh"]
