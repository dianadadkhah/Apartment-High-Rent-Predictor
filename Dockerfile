FROM quay.io/jupyter/minimal-notebook:2024-10-14

USER root

WORKDIR /home/jovyan/work

# Install micromamba manually
RUN apt-get update && apt-get install -y curl bzip2 && \
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C /usr/local/bin --strip-components=1 bin/micromamba && \
    chmod +x /usr/local/bin/micromamba

# Copy environment files
COPY environment.yml .
COPY conda-lock.yml .

# Use micromamba to install exact conda-lock environment
RUN micromamba install -y -n base -f conda-lock.yml && \
    micromamba clean --all --yes

USER ${NB_UID}

EXPOSE 8888

CMD ["start-notebook.sh"]
