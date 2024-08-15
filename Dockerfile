FROM julia/1.10.4-bookworm

ENV USER widebanddoa
ENV USER_HOME_DIR /home/${USER}
ENV JULIA_DEPOT_PATH ${USER_HOME_DIR}/.julia

RUN useradd -m -d ${USER_HOME_DIR} ${USER}

# add this entire repository to the docker image
ADD *.toml ${USER_HOME_DIR}/

RUN julia -e "cd(\"${USER_HOME_DIR}\"); using Pkg; Pkg.add(url=\"https://github.com/Red-Portal/ReversibleJump.jl\"); Pkg.update(); Pkg.precompile(); Pkg.status(); println(pwd())"
 
ADD src     ${USER_HOME_DIR}/src
ADD scripts ${USER_HOME_DIR}/scripts

RUN chmod -R a+rwX ${USER_HOME_DIR}

USER ${USER}

# configure the script entry point
WORKDIR ${USER_HOME_DIR}

ENTRYPOINT ["julia", "-p", "20", "-e", "using Pkg; Pkg.activate(\"scripts\"); include(\"scripts/detection.jl\");  @everywhere using Pkg; @everywhere Pkg.activate(\"scripts\"); @everywhere include(\"scripts/detection.jl\"); @everywhere system_setup(is_hyper=true, start=0); main()"]
