# 「Dockerで学ぶROS2プログラミング入門」（科学情報出版）
# https://gitlab.com/it-book/docker-ros2-programming
# インストール環境
#    ubuntu:18.04
#    ROS2 Eloquent Elusor
#
# General guidelines and recommendations for writing 'Dockerfile':
# https://docs.docker.com/engine/userguide/eng-image/dockerfile_best-practices/
#
# @product   Dockerfile
# @author    Hiroyuki Okada <hiroyuki.okada@okadanet.org>
# @copyright 2020 Hiroyuki Okada
# @license   https://www.apache.org/licenses/LICENSE-2.0 Apache-2.0

# 基本となるイメージ
FROM ubuntu:18.04

# Dockerfileを作成したユーザの情報
LABEL maintainer="Hiroyuki Okada <hiroyuki.okada@okadanet.org>"
ENV TZ JST-9

# インストール時のインタラクティブな設定をスキップする
ENV DEBIAN_FRONTEND=noninteractive

# インストールに必要なパッケージのインストール
RUN apt-get update -q  && \
    apt-get upgrade -yq && \
    apt-get install -yq bash-completion build-essential curl gnupg2 lsb-release locales git tmux wget nano gedit x11-apps eog && \
    rm -rf /var/lib/apt/lists/*

# ROS2 Eloquentと演習に必要なパッケージのインストール
RUN curl -Ls https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | \
apt-key add -
RUN sh -c 'echo "deb http://packages.ros.org/ros2/ubuntu $(\
lsb_release -cs) main" > /etc/apt/sources.list.d/ros2-latest.list'              
RUN apt-get update -q  
# ROS2 Eloquentのインストール
RUN apt-get install -yq  ros-eloquent-desktop
# ROS2　ツールのインストール
RUN apt-get install -yq  python3-argcomplete python3-colcon-common-extensions python3-vcstool 
# Turtlesimのインストール
RUN apt-get install -yq ros-eloquent-turtlesim ros-eloquent-gazebo-ros-* 
# Turtlebot3のインストール
## Cartographer の依存パッケージ
RUN apt-get install -yq google-mock libceres-dev liblua5.3-dev libboost-dev libboost-iostreams-dev libprotobuf-dev protobuf-compiler libcairo2-dev libpcl-dev python3-sphinx
## Cartographer のインストール
RUN apt-get install -yq ros-eloquent-cartographer ros-eloquent-cartographer-ros ros-eloquent-navigation2 ros-eloquent-nav2-bringup

# ワークディレクトリの設定
WORKDIR /root
SHELL ["/bin/bash", "-c"]
RUN echo "source /opt/ros/eloquent/setup.bash" >> ~/.bashrc &&\
   : "commnet"
RUN source /opt/ros/eloquent/setup.bash   
RUN mkdir -p ~/ros2_ws/src

#　公式ROS2サンプルのダウンロード
RUN cd ~/ros2_ws/src &&\
    git clone -b eloquent https://github.com/ros2/examples ros2_examples

# Turtlebot3開発環境のダウンロード
RUN cd ~/ros2_ws &&\
    wget https://raw.githubusercontent.com/ROBOTIS-GIT/turtlebot3/ros2/turtlebot3.repos && vcs import src < turtlebot3.repos

RUN set -x
RUN echo "export ROS_DOMAIN_ID=30 #TURTLEBOT3" >> ~/.bashrc &&\
   : "commnet"
RUN echo "export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:~/ros2_ws/src/turtlebot3/turtlebot3_simulations/turtlebot3_gazebo/models" >> ~/.bashrc &&\
   : "commnet"
RUN echo "export TURTLEBOT3_MODEL=burger" >> ~/.bashrc &&\
   : "commnet"

# 付属サンプルのをイメージにリンクする
ADD ./tutrials/ ./tutrials

# ホストPCのディレクトリをワークスペースにシンボリックリンクする
# Dockerコンテナが終了しても変更内容が保存される
RUN cd ~/ros2_ws/src &&\
    ln -s ~/myProjects .

# ROS2ワークスペースのビルド
RUN set -x
RUN cd ~/ros2_ws &&\
    source /opt/ros/eloquent/setup.bash &&\
    MAKEFLAGS="-j1 -l1" colcon build --symlink-install --executor sequential
RUN echo "source ~/ros2_ws/install/setup.bash" >> ~/.bashrc &&\
   : "commnet"
RUN echo "source ~/ros2_ws/install/local_setup.bash" >> ~/.bashrc &&\
   : "commnet"

# コンテナ実行時にtmuxを実行する
#ENTRYPOINT ["tmux"]
