# Base image recommended by MDS instructors
FROM quay.io/jupyter/minimal-notebook:2024-10-14

# The user in Jupyter Docker stacks
USER root

# Create working directory
WORKDIR /home/jovyan/work

# Copy environment and lock files
COPY environment.yml .
COPY conda-lock.yml .

# Install mamba to resolve environments faster
RUN micromamba install -y -n base -f conda-lock.yml && \
    micromamba clean --all --yes

# Change owner back to the notebook user
USER ${NB_UID}

# Expose port for Jupyter
EXPOSE 8888

# Launch Jupyter automatically
CMD ["start-notebook.sh"]
