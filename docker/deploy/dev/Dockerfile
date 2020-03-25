from xacc/deploy-base
workdir /home/dev

run cd /home/dev && git clone --recursive https://github.com/ornl-qci/exatn && cd exatn && mkdir build && cd build \
    && cmake .. -DEXATN_BUILD_TESTS=TRUE -DBLAS_LIB=ATLAS -DBLAS_PATH=/usr/lib/x86_64-linux-gnu -DCMAKE_BUILD_TYPE=Release \
    && make -j$(nproc) install 

# Theia application
workdir /home/dev
ARG version=latest
ADD $version.package.json ./package.json

RUN yarn --cache-folder ./ycache && rm -rf ./ycache && \
    NODE_OPTIONS="--max_old_space_size=4096" yarn theia build ;\
    yarn theia download:plugins
EXPOSE 3000
ENV SHELL=/bin/bash \
    THEIA_DEFAULT_PLUGINS=local-dir:/home/dev/plugins
ENV PYTHONPATH "${PYTHONPATH}:/root/.exatn:$(python3 -m site --user-site)/psi4/lib"
ENTRYPOINT [ "yarn", "theia", "start", "/home/dev", "--hostname=0.0.0.0" ]
