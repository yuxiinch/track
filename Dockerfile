FROM osrf/ros:humble-desktop-full

ARG USERNAME=ros_dev
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && usermod -aG video $USERNAME \
    && usermod -aG dialout $USERNAME

RUN apt install -y vim wget curl git python3-pip gpg apt-transport-https

# # Realsense
# RUN mkdir -p /etc/apt/keyrings && curl -sSf https://librealsense.intel.com/Debian/librealsense.pgp | tee /etc/apt/keyrings/librealsense.pgp > /dev/null
# RUN echo "deb [signed-by=/etc/apt/keyrings/librealsense.pgp] https://librealsense.intel.com/Debian/apt-repo `lsb_release -cs` main" | \
#     tee /etc/apt/sources.list.d/librealsense.list && apt update
# RUN apt-get update && apt-get install -y build-essential dkms
# RUN apt-get install -y librealsense2-dkms librealsense2-utils librealsense2-dev librealsense2-dbg

# Realsense
RUN mkdir -p /etc/apt/keyrings && curl -sSf https://librealsense.intel.com/Debian/librealsense.pgp | gpg --dearmor -o /etc/apt/keyrings/librealsense.pgp
RUN echo "deb [signed-by=/etc/apt/keyrings/librealsense.pgp] https://librealsense.intel.com/Debian/apt-repo `lsb_release -cs` main" | \
    tee /etc/apt/sources.list.d/librealsense.list && apt update
# Install dependencies
RUN apt-get update && apt-get install -y build-essential dkms linux-headers-$(uname -r) cmake
# Install librealsense packages
RUN apt-get install -y librealsense2-utils librealsense2-dev librealsense2-dbg


# add microsoft gpg
RUN wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > packages.microsoft.gpg \
    && sudo install -D -o root -g root -m 644 packages.microsoft.gpg /etc/apt/keyrings/packages.microsoft.gpg \
    && echo "deb [arch=amd64,arm64,armhf signed-by=/etc/apt/keyrings/packages.microsoft.gpg] https://packages.microsoft.com/repos/code stable main" |sudo tee /etc/apt/sources.list.d/vscode.list > /dev/null \
    && rm -f packages.microsoft.gpg \
    && apt update

RUN apt install -y code

# install pack in pack dir
# RUN mkdir /root/pack
# COPY DockerBuild/pack/* /root/pack
# RUN apt install -y /root/pack/*
# COPY DockerBuild/scripts/entry.sh /entry.sh
USER $USERNAME

# install vscode extension
# RUN code --install-extension ms-iot.vscode-ros
# RUN code --install-extension ms-vscode.cpptools-extension-pack
# RUN code --install-extension ms-vscode.cmake-tools
# RUN code --install-extension donjayamanne.python-extension-pack
# RUN code --install-extension eamodio.gitlens

CMD ["/bin/bash"]
# CMD ["/bin/bash", "-c", "cd ~; bash"]
