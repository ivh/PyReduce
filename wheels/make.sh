export DOCKER_IMAGE=quay.io/pypa/manylinux2010_x86_64
export PLAT=manylinux2010_x86_64

docker pull $DOCKER_IMAGE
docker run --rm -e PLAT=$PLAT -v `pwd`:/io $DOCKER_IMAGE $PRE_CMD /io/travis/build-wheels.sh
