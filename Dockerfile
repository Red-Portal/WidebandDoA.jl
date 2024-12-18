FROM julia:1.10.4-bookworm

ENV USER widebanddoa
ENV USER_HOME_DIR /home/${USER}
ENV JULIA_DEPOT_PATH ${USER_HOME_DIR}/.julia

RUN useradd -m -d ${USER_HOME_DIR} ${USER}

# Add unregistered dependencies
RUN julia -e "using Pkg; Pkg.develop(url=\"https://github.com/UBC-Stat-ML/mcmcse.jl\"); Pkg.develop(url=\"https://github.com/Red-Portal/ReversibleJump.jl\"); Pkg.develop(url=\"https://github.com/Red-Portal/WidebandDoA.jl\"); Pkg.add(\"SysInfo\")"

# Copy source files
ADD *.toml ${USER_HOME_DIR}/
ADD scripts ${USER_HOME_DIR}/scripts

RUN julia -e "cd(\"${USER_HOME_DIR}\"); using Pkg; Pkg.activate(\"scripts\"); Pkg.develop([\"mcmcse\", \"ReversibleJump\", \"WidebandDoA\"])"

RUN julia -e "cd(\"${USER_HOME_DIR}\"); using Pkg; Pkg.activate(\"scripts\"); Pkg.update(); Pkg.precompile(); Pkg.status(); println(pwd());"
 
RUN chmod -R a+rwX ${USER_HOME_DIR}

USER ${USER}

# configure the script entry point
WORKDIR ${USER_HOME_DIR}

ENTRYPOINT ["julia", "-e", "using Distributed, SysInfo; addprocs(SysInfo.ncores() > 80 ? div(SysInfo.ncores(), 2) : 40); @everywhere using Pkg; @everywhere Pkg.activate(\"scripts\"); @everywhere include(\"scripts/detection.jl\"); @everywhere system_setup(is_hyper=true, start=0); main()"]
