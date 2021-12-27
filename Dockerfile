FROM tensorflow/tensorflow:latest-jupyter

WORKDIR "/tf/notebooks"

RUN pip install sklearn pandas azure-storage-blob seaborn matplotlib

CMD ["bash", "-c", "source /etc/bash.bashrc && jupyter notebook --ip 0.0.0.0 --no-browser --allow-root --notebook-dir=/tf/notebooks"]
