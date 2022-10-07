import os
import re


def test_gpu(id=0, mem_collect='auto'):
    gpu_state = os.popen('nvidia-smi -i %s' % str(id)).read()
    gpu_type = re.findall('(?=TITAN\s).*?(?=\s+Off)', gpu_state)[0]
    usage = re.findall('(\d+)(?=\s*MiB)', gpu_state)[0]
    memory = re.findall('(\d+)(?=\s*MiB)', gpu_state)[1]
    idle = int(memory) - int(usage)
    if mem_collect == 'auto':  
        if int(usage) < 20:
            return 1, gpu_type, idle, int(memory)
        else:
            return 0, gpu_type, idle, int(memory)

    else:
        if idle > mem_collect:
            return 1, gpu_type, idle, int(memory)
        else:
            return 0, gpu_type, idle, int(memory)


def auto_gpu(gpu_num=1, gpu_mem='auto'):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
    gpu_id = []
    for i in [0, 1]:
        is_val, gpu, usa, mem = test_gpu(i, gpu_mem)
        if is_val == 1:
            gpu_id.append(str(i))
            print('{} gpu selected, type {}, {}MB/{}MB remaining'.format(i, gpu, usa, mem))
            if len(gpu_id) == gpu_num:
                break
    assert len(gpu_id) == gpu_num, 'The gpu requirements you require cannot be met'
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(gpu_id)
