FROM --platform=linux/amd64 pytorch/pytorch AS example-algorithm-amd64

# Use a 'large' base container to show-case how to load pytorch and use the GPU (when enabled)
RUN python --version
# Ensures that Python output to stdout/stderr is not buffered: prevents missing information when terminating
ENV PYTHONUNBUFFERED=1

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

WORKDIR /opt/app

COPY --chown=user:user requirements.txt /opt/app/
COPY --chown=user:user resources /opt/app/resources

# You can manually download the dependency packages to /opt/app/python_packages (If you are experiencing network issues)
# # Install Python dependencies
COPY resources/python_packages /opt/app/python_packages

RUN python -m pip install --no-cache-dir --find-links /opt/app/python_packages --no-index SimpleITK numpy Pillow

# # Or you can add any Python dependencies to requirements.txt
# RUN python -m pip install \
#     --user \
#     --no-cache-dir \
#     --no-color \
#     --requirement /opt/app/requirements.txt

COPY --chown=user:user inference.py /opt/app/

ENTRYPOINT ["python", "inference.py"]
