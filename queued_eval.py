import argparse
import os
import sys
import re
from hparams import hparams, hparams_debug_string
from synthesizer import Synthesizer
from util import plot
from multiprocessing import Pool, Process, Queue
import random
import time

sentences = [
    ["Hallo, das ist ein Test vom T T S System", 1],
    ["Das ist wahrscheinlich laut und übersteuert", 2],
    ["Ich weiß nicht, was ich sagen soll", 3],
    ["Hallihallo.", 4],
    ["Jannik", 5],
    ["Fischers Fritz fischt frische Fische.", 6]
]

def get_output_base_path(checkpoint_path):
    base_dir = os.path.dirname(checkpoint_path)
    m = re.compile(r'.*?\.ckpt\-([0-9]+)').match(checkpoint_path)
    name = 'eval-%d' % int(m.group(1)) if m else 'eval'
    return os.path.join(base_dir, name)


def run_eval():
    args = parser.parse_args()
    hparams.parse(args.hparams)
    print(hparams_debug_string())
    synth = Synthesizer()
    synth.load(args.checkpoint)
    base_path = get_output_base_path(args.checkpoint)

    while True:
        sentence = process_queue.get()
        text = sentence[0]
        wav_path = str(sentence[1]) + '.wav'
        print('Synthesizing: %s' % wav_path)
        wav, alignment = synth.synthesize(text)
        with open(wav_path, 'wb') as f:
            f.write(wav)

def job_queuer():
    while True:
        if len(sentences) != 0:
            process_queue.put(sentences[0])
            del sentences[0]
        else:
            time.sleep(1)

parser = argparse.ArgumentParser()
parser.add_argument(
    '--checkpoint', required=True,
    help='Path to model checkpoint')
parser.add_argument(
    '--hparams', default='',
    help='Hyperparameter overrides as a comma-separated list of name=value pairs')
parser.add_argument(
    '--force_cpu', default=False,
    help='Force synthesize with cpu')
parser.add_argument(
    '--gpu_assignment', default='0',
    help='Set the gpu the model should run on')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_assignment

if args.force_cpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

process_queue = Queue()

writer_process = Process(target=job_queuer)
writer_process.start()

if sys.platform == 'win32':
    threads = int(os.environ['NUMBER_OF_PROCESSORS'])
else:
    threads = int(os.popen('grep -c cores /proc/cpuinfo').read())

processes = []

for i in range(threads):
    processes += [Process(target=run_eval)]
    processes[-1].start()
