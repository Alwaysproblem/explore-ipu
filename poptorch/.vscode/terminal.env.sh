#!/bin/bash
# set -ex

export GCDA_MONITOR=1
export TF_CPP_VMODULE="poplar_compiler=1,poplar_executable=1"
export TF_POPLAR_FLAGS="--max_compilation_threads=40 --executable_cache_path=/mnt/scratch001/custeng-cn-scratch/yongxiy/cachedir --show_progress_bar=true"
export TMPDIR="/mnt/scratch001/custeng-cn-scratch/yongxiy/tmp"
export IPUOF_CONFIG_PATH="/localdata/yongxiy/env/ipu-config/pod64/gbnwx-pod004-1.ipu.graphcore.ai/ipuof.conf"

function check_file_exist()
{
  _file=$1
  if [ ! -f $_file ];then
    printf "\e[93m%s is not file or file does not exist\n\e[0m" $_file
  fi	
}

check_file_exist "/mnt/scratch001/custeng-cn-scratch/yongxiy/Desktop/IPU-setup/test/sdk/poplar_sdk-ubuntu_18_04-2.0.0+481-79b41f85d1/popart-ubuntu_18_04-2.0.0+108156-165bbd8a64/enable.sh"
pushd /mnt/scratch001/custeng-cn-scratch/yongxiy/Desktop/IPU-setup/test/sdk/poplar_sdk-ubuntu_18_04-2.0.0+481-79b41f85d1/popart-ubuntu_18_04-2.0.0+108156-165bbd8a64
  source "/mnt/scratch001/custeng-cn-scratch/yongxiy/Desktop/IPU-setup/test/sdk/poplar_sdk-ubuntu_18_04-2.0.0+481-79b41f85d1/popart-ubuntu_18_04-2.0.0+108156-165bbd8a64/enable.sh"
popd

check_file_exist "/mnt/scratch001/custeng-cn-scratch/yongxiy/Desktop/IPU-setup/test/sdk/poplar_sdk-ubuntu_18_04-2.0.0+481-79b41f85d1/poplar-ubuntu_18_04-2.0.0+108156-165bbd8a64/enable.sh"
pushd /mnt/scratch001/custeng-cn-scratch/yongxiy/Desktop/IPU-setup/test/sdk/poplar_sdk-ubuntu_18_04-2.0.0+481-79b41f85d1/poplar-ubuntu_18_04-2.0.0+108156-165bbd8a64
  source "/mnt/scratch001/custeng-cn-scratch/yongxiy/Desktop/IPU-setup/test/sdk/poplar_sdk-ubuntu_18_04-2.0.0+481-79b41f85d1/poplar-ubuntu_18_04-2.0.0+108156-165bbd8a64/enable.sh"
popd

# set +ex
