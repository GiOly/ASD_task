import os
import re


def test_gpu(gpu_id=0):
    gpu_state = os.popen('nvidia-smi -i %s' % str(gpu_id)).read()
    usage = re.findall('(\d+)(?=\s*MiB)', gpu_state)[0]
    memory = re.findall('(\d+)(?=\s*MiB)', gpu_state)[1]
    idle = int(memory) - int(usage)
    return idle, usage, memory


def auto_gpu():
    mem_remain = []
    for i in [0, 1]:
        idle, _, _ = test_gpu(i)
        mem_remain.append(idle)
    selected_gpu_id = mem_remain.index(max(mem_remain))
    idle, usage, memory = test_gpu(gpu_id=selected_gpu_id)
    print('GPU_ID:{} selected, {}MB/{}MB remaining'.format(selected_gpu_id, usage, memory))
    os.environ['CUDA_VISIBLE_DEVICES'] = str(selected_gpu_id)
