#!/bin/bash

# set -ex

BASE_DIR_SCOTTY="/mnt/scratch001/custeng-cn-scratch/yongxiy"
POP_BASE_DIR="/mnt/scratch001/custeng-cn-scratch/yongxiy/sdk"

export GCDA_MONITOR=1
export TF_CPP_VMODULE='poplar_compiler=1,poplar_executable=1'
# export TF_CPP_VMODULE='poplar_compiler=1'
export TF_POPLAR_FLAGS="--max_compilation_threads=40 --executable_cache_path=${BASE_DIR_SCOTTY}/cachedir --show_progress_bar=true"
export TMPDIR="${BASE_DIR_SCOTTY}/tmp"
export IPUOF_CONFIG_PATH="${POP_BASE_DIR}/pod16_ipuof.conf"

POP_VERSION="2.0.0+481-79b41f85d1"
POPART_VERSION="2.0.0+108156-165bbd8a64"
POPLAR_VERSION="2.0.0+108156-165bbd8a64"

POPART_ROOT="${POP_BASE_DIR}/poplar_sdk-ubuntu_18_04-${POP_VERSION}/popart-ubuntu_18_04-${POPART_VERSION}"

function echo_popart_info {
  echo "Please input the root path of popart:"
  echo -ne "[${1}] >> "
}

function echo_poplar_info {
  echo "Please input the root path of poplar:"
  echo -ne "[${1}] >> "
}

function gen_envfile(){
  local file_name=$1

  echo -ne "" > "./.vscode/${file_name}"

  echo "GCDA_MONITOR=$GCDA_MONITOR" >> "./.vscode/${file_name}"
  echo "TF_CPP_VMODULE='$TF_CPP_VMODULE'" >> "./.vscode/${file_name}"
  echo "TF_POPLAR_FLAGS='$TF_POPLAR_FLAGS'" >> "./.vscode/${file_name}"
  echo "TMPDIR=$TMPDIR" >> "./.vscode/${file_name}"
  echo "IPUOF_CONFIG_PATH=$IPUOF_CONFIG_PATH" >> "./.vscode/${file_name}"
  echo "CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH" >> "./.vscode/${file_name}"
  echo "PATH=$PATH" >> "./.vscode/${file_name}"
  echo "CPATH=$CPATH" >> "./.vscode/${file_name}"
  echo "LIBRARY_PATH=$LIBRARY_PATH" >> "./.vscode/${file_name}"
  echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH" >> "./.vscode/${file_name}"
  echo "OMPI_CPPFLAGS=$OMPI_CPPFLAGS" >> "./.vscode/${file_name}"
  echo "OPAL_PREFIX=$OPAL_PREFIX" >> "./.vscode/${file_name}"
  echo "PYTHONPATH=$PYTHONPATH" >> "./.vscode/${file_name}"
  echo "POPLAR_SDK_ENABLED=$POPLAR_SDK_ENABLED" >> "./.vscode/${file_name}"
}

function gen_terminial_init_script(){
  local file_name=$1 
  cat <<EOF > "./.vscode/${file_name}"
#!/bin/bash
# set -ex

export GCDA_MONITOR=$GCDA_MONITOR
export TF_CPP_VMODULE=$TF_CPP_VMODULE
export TF_POPLAR_FLAGS="'$TF_POPLAR_FLAGS'"
export TMPDIR=$TMPDIR
export IPUOF_CONFIG_PATH=$IPUOF_CONFIG_PATH

POPART_ROOT="${POP_BASE_DIR}/poplar_sdk-ubuntu_18_04-${POP_VERSION}/popart-ubuntu_18_04-${POPART_VERSION}"
export CMAKE_PREFIX_PATH=\${POPART_ROOT}\${CMAKE_PREFIX_PATH:+:\${CMAKE_PREFIX_PATH}}
export CPATH=\${POPART_ROOT}/include\${CPATH:+:\${CPATH}}
export LIBRARY_PATH=\${POPART_ROOT}/lib\${LIBRARY_PATH:+:\${LIBRARY_PATH}}
export LD_LIBRARY_PATH=\${POPART_ROOT}/lib\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}
export PYTHONPATH=\${POPART_ROOT}/python:\$PYTHONPATH

if [ -z \${POPLAR_SDK_ENABLED+x} ]
then
  POPLAR_ROOT="${POP_BASE_DIR}/poplar_sdk-ubuntu_18_04-${POP_VERSION}/poplar-ubuntu_18_04-${POPLAR_VERSION}"
  export CMAKE_PREFIX_PATH=\${POPLAR_ROOT}\${CMAKE_PREFIX_PATH:+:\${CMAKE_PREFIX_PATH}}
  export PATH=\${POPLAR_ROOT}/bin\${PATH:+:\${PATH}}
  export CPATH=\${POPLAR_ROOT}/include\${CPATH:+:\${CPATH}}
  export LIBRARY_PATH=\${POPLAR_ROOT}/lib\${LIBRARY_PATH:+:\${LIBRARY_PATH}}
  export LD_LIBRARY_PATH=\${POPLAR_ROOT}/lib\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}
  export OMPI_CPPFLAGS="-I\${POPLAR_ROOT}/include/openmpi"
  export OPAL_PREFIX=\${POPLAR_ROOT}
  export PYTHONPATH=\${POPLAR_ROOT}/python:\${POPLAR_ROOT}/lib/python\${PYTHONPATH:+:\${PYTHONPATH}}
  export POPLAR_SDK_ENABLED=\${POPLAR_ROOT}
else
  echo 'ERROR: A Poplar SDK has already been enabled.'
  echo "Path of enabled Poplar SDK: \${POPLAR_SDK_ENABLED}"
  echo 'If this is not wanted then please start a new shell.'
fi

# set +ex
EOF
}

# this for debugging and run
gen_envfile ".ipu.env"

# this is for terminal
gen_terminial_init_script "terminal.env.sh"

# set +ex
