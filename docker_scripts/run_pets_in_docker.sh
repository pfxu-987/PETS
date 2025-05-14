docker run --gpus 1  -it   -v $PWD:/workspace -v /etc/passwd:/etc/passwd --name=pets_dev_2 eliaswyq/pets_gpu:4090
