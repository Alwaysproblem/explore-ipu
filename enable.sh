#!/bin/bash

# set -ex

export GCDA_MONITOR=1
export TF_CPP_VMODULE='poplar_compiler=1,poplar_executable=1'
# export TF_CPP_VMODULE='poplar_compiler=1'
export TF_POPLAR_FLAGS='--max_compilation_threads=40 --executable_cache_path=/localdata/yongxiy/cachedir'
export TMPDIR='/localdata/yongxiy/tmp'
export IPUOF_CONFIG_PATH=/localdata/yongxiy/tmp/pod16_ipuof.conf 


POPART_ROOT="$HOME/Desktop/poplar_sdk-ubuntu_18_04-2.0.0-EA.1+418-ec08eca1f6/popart-ubuntu_18_04-1.5.0+91590-712863139b"

function echo_popart_info {
  echo "Please input the root path of popart:"
  echo -ne "[${1}] >> "
}

function echo_poplar_info {
  echo "Please input the root path of poplar:"
  echo -ne "[${1}] >> "
}

# function is_generate_ipu_envfile(){
#   ;
# }
# function is_generate_term_ipu_envfile(){
#   ;
# }
# function is_generate_debug_ipu_envfile(){
#   ;
# }

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

export GCDA_MONITOR=1
export TF_CPP_VMODULE='poplar_compiler=1,poplar_executable=1'
# export TF_CPP_VMODULE='poplar_compiler=1'
export TF_POPLAR_FLAGS='--max_compilation_threads=40 --executable_cache_path=/localdata/yongxiy/cachedir'
export TMPDIR='/localdata/yongxiy/tmp'
export IPUOF_CONFIG_PATH=/localdata/yongxiy/tmp/pod16_ipuof.conf 

POPART_ROOT="\$HOME/Desktop/poplar_sdk-ubuntu_18_04-2.0.0-EA.1+418-ec08eca1f6/popart-ubuntu_18_04-1.5.0+91590-712863139b"
export CMAKE_PREFIX_PATH=\${POPART_ROOT}\${CMAKE_PREFIX_PATH:+:\${CMAKE_PREFIX_PATH}}
export CPATH=\${POPART_ROOT}/include\${CPATH:+:\${CPATH}}
export LIBRARY_PATH=\${POPART_ROOT}/lib\${LIBRARY_PATH:+:\${LIBRARY_PATH}}
export LD_LIBRARY_PATH=\${POPART_ROOT}/lib\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}
export PYTHONPATH=\${POPART_ROOT}/python:\$PYTHONPATH

if [ -z \${POPLAR_SDK_ENABLED+x} ]
then
  POPLAR_ROOT="\$HOME/Desktop/poplar_sdk-ubuntu_18_04-2.0.0-EA.1+418-ec08eca1f6/poplar-ubuntu_18_04-2.0.0+91590-712863139b"
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


# POPART_ROOT=$(cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd)
echo_popart_info $POPART_ROOT
read popart_root_path
POPART_ROOT="${popart_root_path:-${POPART_ROOT}}"
export CMAKE_PREFIX_PATH=${POPART_ROOT}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}
export CPATH=${POPART_ROOT}/include${CPATH:+:${CPATH}}
export LIBRARY_PATH=${POPART_ROOT}/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}
export LD_LIBRARY_PATH=${POPART_ROOT}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export PYTHONPATH=${POPART_ROOT}/python:$PYTHONPATH

if [ -z ${POPLAR_SDK_ENABLED+x} ]
then
#   POPLAR_ROOT=$(cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd)
  POPLAR_ROOT="$HOME/Desktop/poplar_sdk-ubuntu_18_04-2.0.0-EA.1+418-ec08eca1f6/poplar-ubuntu_18_04-2.0.0+91590-712863139b"

  echo_popart_info $POPLAR_ROOT
  read poplar_root_path
  POPART_ROOT="${poplar_root_path:-${POPLAR_ROOT}}"

  export CMAKE_PREFIX_PATH=${POPLAR_ROOT}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}
  export PATH=${POPLAR_ROOT}/bin${PATH:+:${PATH}}
  export CPATH=${POPLAR_ROOT}/include${CPATH:+:${CPATH}}
  export LIBRARY_PATH=${POPLAR_ROOT}/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}
  export LD_LIBRARY_PATH=${POPLAR_ROOT}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
  export OMPI_CPPFLAGS="-I${POPLAR_ROOT}/include/openmpi"
  export OPAL_PREFIX=${POPLAR_ROOT}
  export PYTHONPATH=${POPLAR_ROOT}/python:${POPLAR_ROOT}/lib/python${PYTHONPATH:+:${PYTHONPATH}}
  export POPLAR_SDK_ENABLED=${POPLAR_ROOT}
else
  echo 'ERROR: A Poplar SDK has already been enabled.'
  echo "Path of enabled Poplar SDK: ${POPLAR_SDK_ENABLED}"
  echo 'If this is not wanted then please start a new shell.'
fi

# this for debugging and run
gen_envfile ".inpu.env"


# this is for terminal
gen_terminial_init_script "terminal.env.sh"

# set +ex