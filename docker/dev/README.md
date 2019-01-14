# Develop with Theia

To develop ExaTN using the Eclipse Theia IDE and Docker

```bash
$ git clone --recursive https://code.ornl.gov/qci/exatn
$ cd exatn/docker/dev
$ nvidia-docker run -d -ti --ipc=host -p 3005:3005 -v ../../../:/home/project code.ornl.gov:4567/qci/exatn
```

Navigate to `http://localhost:3005` in your web browser.

To delete this development workspace
```bash
$ docker ps -a
```
find the correct container and run, it has name $NAME
```bash
$ docker stop $NAME && docker rm -v $NAME
```
