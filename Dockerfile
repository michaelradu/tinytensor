FROM ubuntu:latest

# Update apt-get and install necessary packages
RUN apt-get update && \
    apt-get install -y python3-full python3-pip neovim

# Set Python 3 as the default python interpreter
RUN ln -s /usr/bin/python3 /usr/bin/python

# Verify Python and pip installations
RUN python --version && \
    pip3 --version

# Create a Python 3 virtual environment
RUN python3 -m venv venv

# Activate the virtual environment and install pip packages
RUN /bin/bash -c "source venv/bin/activate && pip install --upgrade pip"

COPY ./requirements.txt /app/requirements.txt
COPY ./tinytensor /app/tinytensor


# Activate the virtual environment and install pip packages
RUN /bin/bash -c "source venv/bin/activate && pip install --no-cache-dir --upgrade -r /app/requirements.txt"

# RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# Activate the virtual environment when the container starts and open a bash session
CMD ["/bin/bash", "-c", "source venv/bin/activate && exec bash"]