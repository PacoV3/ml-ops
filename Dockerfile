FROM tensorflow/tensorflow:latest-jupyter

WORKDIR "/tf/notebooks/local-dev"

RUN pip install sklearn pandas

CMD ["bash", "-c", "source /etc/bash.bashrc && jupyter notebook --ip 0.0.0.0 --no-browser --allow-root --notebook-dir=/tf/notebooks/local-dev"]
